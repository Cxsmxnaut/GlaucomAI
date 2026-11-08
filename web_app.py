"""Local research UI: browser-only webcam alignment and genuine fundus inference."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
import torch

from glaucoma.pipeline import InputError, MAX_BYTES, ModelError, decode_image, infer, load_model

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
SAMPLE = ROOT / "data/web_sample/example-retina.jpg"
SAMPLE_SHA = "000f552bce6ac5995b4f068cf0112359f295c51821e81c3171ee15882fdd460b"


def create_app(model_path=ROOT / "models/best_model.pth", sample_path=SAMPLE):
    model_path, sample_path = Path(model_path), Path(sample_path)

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(2)
        app.state.loaded = None
        app.state.error = None
        app.state.lock = asyncio.Lock()
        try:
            loaded = await run_in_threadpool(load_model, model_path)
            if loaded.metadata.get("label_semantics") == "mixed_dataset_binary_research_labels":
                raise ModelError("Mixed-label checkpoints cannot be presented as referral scores.")
            checksum = sha256(model_path.read_bytes()).hexdigest()
            with torch.inference_mode():
                loaded.model(torch.zeros(1, 3, 224, 224))
            app.state.checksum = checksum
            app.state.loaded = loaded
        except (ModelError, OSError, RuntimeError) as exc:
            app.state.error = str(exc)
        yield
        app.state.loaded = None

    app = FastAPI(title="GlaucomAI local research demo", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def local_headers(request: Request, call_next):
        response = None
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            try:
                parsed = urlsplit(origin) if origin else None
                valid_origin = parsed is None or (
                    parsed.scheme == request.url.scheme and parsed.netloc == request.headers.get("host")
                    and not parsed.path and not parsed.query and not parsed.fragment
                )
            except ValueError:
                valid_origin = False
            if not valid_origin:
                response = JSONResponse(
                    {"detail": "This local app only accepts same-origin requests."}, status_code=403
                )
        if response is None:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
            "style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "connect-src 'self'; worker-src 'self' blob:; object-src 'none'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    def sample_bytes():
        if not sample_path.is_file():
            raise HTTPException(404, "Example image is unavailable. Run scripts/setup_web_assets.py.")
        raw = sample_path.read_bytes()
        if sha256(raw).hexdigest() != SAMPLE_SHA:
            raise HTTPException(503, "The example image does not match its pinned source.")
        return raw

    @app.get("/api/model")
    async def model_info():
        loaded = app.state.loaded
        info = {"ready": loaded is not None, "error": app.state.error,
                "architecture": "EfficientNet-B0", "device": "CPU",
                "example_available": sample_path.is_file(),
                "camera_frames_uploaded": False, "clinical_validation": False,
                "tracker_assets_available": (WEB / "vendor/face_landmarker.task").is_file()}
        if loaded:
            checksum = app.state.checksum
            baseline = json.loads((ROOT / "experiments/baseline.json").read_text())
            published = checksum == baseline["checkpoint_sha256"]
            info.update({"version": "Original baseline · v0.1.0" if published else "User-supplied research checkpoint",
                         "sha256": checksum, "threshold": loaded.threshold,
                         "evaluation": baseline["test_evaluation"]["metrics"] if published else None})
        return info

    @app.get("/api/example")
    async def example_image():
        with decode_image(sample_bytes(), "example-retina.jpg") as image:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    async def analyze(raw, filename, source):
        if app.state.loaded is None:
            raise HTTPException(503, app.state.error or "No compatible checkpoint is loaded.")
        if app.state.lock.locked():
            raise HTTPException(429, "Another image is being analyzed. Try again in a moment.")

        def work():
            with decode_image(raw, filename) as image:
                result = infer(image, app.state.loaded)
                result.update({"width": image.width, "height": image.height,
                               "source": source, "model_sha256": app.state.checksum,
                               "clinical_validation": False})
                return result

        async with app.state.lock:
            try:
                return await run_in_threadpool(work)
            except InputError as exc:
                raise HTTPException(422, str(exc)) from exc
            except (ModelError, RuntimeError) as exc:
                raise HTTPException(503, "Analysis could not complete. No score is available.") from exc

    @app.post("/api/analyze-example")
    async def analyze_example():
        return await analyze(sample_bytes(), "example-retina.jpg", {
            "kind": "benchmark_sample", "label": "Example retina — not captured by the webcam",
            "publisher_label": "referable", "dataset": "EyePACS-AIROGS-light-V2 v4",
        })

    @app.post("/api/analyze")
    async def analyze_upload(request: Request, filename: str = "", source: str = ""):
        # Webcam images have no route from the UI to this model. The manual upload
        # declaration is provenance, not a trained retinal-content detector.
        if source != "fundus_upload":
            raise HTTPException(422, "Only a separately supplied fundus photograph can be analyzed. Webcam images are not retinal images.")
        if not filename or len(filename) > 255:
            raise HTTPException(422, "Choose a PNG or JPEG retinal photograph.")
        declared = request.headers.get("content-length")
        if declared:
            try:
                length = int(declared)
            except ValueError as exc:
                raise HTTPException(400, "Invalid upload length.") from exc
            if length < 1 or length >= MAX_BYTES:
                raise HTTPException(413, "Choose a non-empty image smaller than 10 MB.")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size >= MAX_BYTES:
                raise HTTPException(413, "Choose an image smaller than 10 MB.")
            chunks.append(chunk)
        return await analyze(b"".join(chunks), filename,
                             {"kind": "fundus_upload", "label": "Uploaded retinal photograph"})

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    app.mount("/assets", StaticFiles(directory=WEB), name="assets")
    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/best_model.pth")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host="127.0.0.1", port=args.port, access_log=False)
