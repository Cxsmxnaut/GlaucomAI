"""Local HTTP behavior with synthetic images and an untrained test checkpoint.

These tests establish no clinical accuracy and never require camera access.
"""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from threading import Event

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch

from glaucoma.pipeline import CLASSES, PREPROCESSING, LoadedModel, build_model
import web_app


def image_bytes(*, uniform=False, format="PNG"):
    image = Image.new("RGB", (32, 48), (120, 80, 40))
    if not uniform:
        image.putpixel((0, 0), (250, 220, 30))
    # The preview endpoint must strip this source metadata.
    exif = Image.Exif()
    exif[270] = "synthetic private test metadata"
    output = BytesIO()
    image.save(output, format=format, exif=exif)
    return output.getvalue()


@pytest.fixture(scope="module")
def web_checkpoint(tmp_path_factory):
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("web_checkpoint") / "untrained_test_fixture.pth"
    torch.save({"architecture": "efficientnet_b0", "classes": CLASSES,
                "preprocessing": PREPROCESSING, "threshold": .4,
                "state_dict": build_model(pretrained=False).state_dict()}, path)
    return path


@pytest.fixture
def sample(tmp_path, monkeypatch):
    raw = image_bytes()
    path = tmp_path / "synthetic-example.jpg"
    path.write_bytes(raw)
    monkeypatch.setattr(web_app, "SAMPLE_SHA", sha256(raw).hexdigest())
    return path


@pytest.fixture
def client(web_checkpoint, sample):
    app = web_app.create_app(web_checkpoint, sample)
    with TestClient(app, base_url="http://127.0.0.1") as connection:
        yield connection


def test_actual_cpu_inference_retains_explicit_provenance_and_saves_no_images(client, sample):
    before = set(sample.parent.rglob("*"))
    upload = client.post("/api/analyze", params={"filename": "retina.png", "source": "fundus_upload"},
                         content=image_bytes())
    example = client.post("/api/analyze-example")
    for response in (upload, example):
        assert response.status_code == 200
        result = response.json()
        assert result["referable"] + result["non_referable"] == pytest.approx(1)
        assert result["flagged"] == (result["referable"] >= .4)
        assert result["threshold"] == .4
        assert result["elapsed_ms"] > 0
        assert result["demo"] is False
        assert result["clinical_validation"] is False
        assert (result["width"], result["height"]) == (32, 48)
        assert len(result["model_sha256"]) == 64
    assert upload.json()["source"] == {"kind": "fundus_upload", "label": "Uploaded retinal photograph"}
    assert example.json()["source"] == {
        "kind": "benchmark_sample", "label": "Example retina — not captured by the webcam",
        "publisher_label": "referable", "dataset": "EyePACS-AIROGS-light-V2 v4",
    }
    assert set(sample.parent.rglob("*")) == before


def test_model_info_does_not_attribute_published_metrics_to_another_checkpoint(client, web_checkpoint):
    info = client.get("/api/model").json()
    assert info["ready"] is True
    assert info["device"] == "CPU"
    assert info["version"] == "User-supplied research checkpoint"
    assert info["sha256"] == sha256(web_checkpoint.read_bytes()).hexdigest()
    assert info["evaluation"] is None
    assert info["clinical_validation"] is False
    assert info["camera_frames_uploaded"] is False
    assert info["example_available"] is True


def test_example_preview_is_reencoded_and_metadata_free(client):
    response = client.get("/api/example")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    with Image.open(BytesIO(response.content)) as image:
        assert image.size == (32, 48)
        assert not image.getexif()
        assert "synthetic private test metadata" not in str(image.info)


@pytest.mark.parametrize("source", ["", "webcam", "benchmark_sample", "face_capture"])
def test_camera_or_unspecified_source_rejected_before_model_runs(client, monkeypatch, source):
    def must_not_run(*args):
        pytest.fail("A webcam or undeclared source reached inference")
    monkeypatch.setattr(web_app, "infer", must_not_run)
    response = client.post("/api/analyze", params={"filename": "face.jpg", "source": source},
                           content=image_bytes())
    assert response.status_code == 422
    assert "Webcam images are not retinal images" in response.json()["detail"]


@pytest.mark.parametrize("filename,raw", [
    ("retina.jpg", b""), ("retina.png", b"not an image"),
    ("retina.png", image_bytes(format="GIF")), ("retina.exe", image_bytes()),
    ("retina.png", image_bytes()[:30]), ("", image_bytes()), ("x" * 256, image_bytes()),
])
def test_invalid_uploads_do_not_return_scores(client, filename, raw):
    response = client.post("/api/analyze", params={"filename": filename, "source": "fundus_upload"},
                           content=raw)
    assert response.status_code in {413, 422}
    assert "referable" not in response.json()


def test_uniform_image_abstains(client):
    response = client.post("/api/analyze", params={"filename": "retina.png", "source": "fundus_upload"},
                           content=image_bytes(uniform=True))
    assert response.status_code == 422
    assert "No score was produced" in response.json()["detail"]


@pytest.mark.parametrize("length,status", [(str(web_app.MAX_BYTES), 413), ("-1", 413), ("invalid", 400)])
def test_declared_upload_limit(client, length, status):
    response = client.post("/api/analyze", params={"filename": "retina.png", "source": "fundus_upload"},
                           content=b"x", headers={"content-length": length})
    assert response.status_code == status
    assert "referable" not in response.json()


def test_chunked_upload_cannot_bypass_byte_limit(client, monkeypatch):
    monkeypatch.setattr(web_app, "MAX_BYTES", 100)
    response = client.post("/api/analyze", params={"filename": "retina.png", "source": "fundus_upload"},
                           content=iter([b"x" * 60, b"y" * 40]))
    assert response.request.headers.get("content-length") is None
    assert response.status_code == 413


@pytest.mark.parametrize("path", ["/api/example", "/api/analyze-example"])
def test_example_checksum_mismatch_never_falls_back(client, sample, path):
    sample.write_bytes(b"replaced image")
    response = client.post(path) if path.endswith("analyze-example") else client.get(path)
    assert response.status_code == 503
    assert "pinned source" in response.json()["detail"]
    assert "referable" not in response.json()


def test_missing_example_is_unavailable(client, sample):
    sample.unlink()
    assert client.get("/api/model").json()["example_available"] is False
    assert client.get("/api/example").status_code == 404
    assert client.post("/api/analyze-example").status_code == 404


@pytest.mark.parametrize("origin", ["https://example.com", "null", "http://localhost", "http://[broken",
                                   "http://127.0.0.1/path"])
def test_foreign_or_malformed_origin_is_rejected(client, origin):
    response = client.post("/api/analyze-example", headers={"origin": origin})
    assert response.status_code == 403
    assert response.headers["cache-control"] == "no-store"
    assert "referable" not in response.json()


def test_local_origin_and_security_headers(client):
    response = client.post("/api/analyze-example", headers={"origin": "http://127.0.0.1"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["permissions-policy"] == "camera=(self), microphone=(), geolocation=()"
    assert "connect-src 'self'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/api/model", headers={"host": "example.com"}).status_code == 400


def test_concurrent_inference_is_rejected_without_running_second_model(client, monkeypatch):
    started, release = Event(), Event()
    actual_infer = web_app.infer

    def paused_infer(*args):
        started.set()
        assert release.wait(timeout=10), "test failed to release its inference worker"
        return actual_infer(*args)

    monkeypatch.setattr(web_app, "infer", paused_infer)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.post, "/api/analyze-example")
        try:
            assert started.wait(timeout=10)
            response = client.post("/api/analyze-example")
            assert response.status_code == 429
        finally:
            release.set()
        assert pending.result(timeout=10).status_code == 200


def test_missing_checkpoint_keeps_page_available_without_fake_scores(tmp_path, sample):
    app = web_app.create_app(tmp_path / "missing.pth", sample)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/").status_code == 200
        info = client.get("/api/model").json()
        assert info["ready"] is False
        assert info["error"]
        assert "evaluation" not in info
        response = client.post("/api/analyze-example")
        assert response.status_code == 503
        assert "referable" not in response.json()


def test_mixed_label_checkpoint_cannot_be_presented_as_referral_score(web_checkpoint, sample, monkeypatch):
    class MustNotWarmUp(torch.nn.Module):
        def forward(self, image):
            pytest.fail("An incompatible checkpoint was warmed up")
    monkeypatch.setattr(web_app, "load_model", lambda path: LoadedModel(MustNotWarmUp(), .5, {
        "label_semantics": "mixed_dataset_binary_research_labels",
    }))
    with TestClient(web_app.create_app(web_checkpoint, sample), base_url="http://127.0.0.1") as client:
        info = client.get("/api/model").json()
        assert info["ready"] is False
        assert "Mixed-label" in info["error"]
        assert client.post("/api/analyze-example").status_code == 503


def test_failed_warmup_does_not_mark_model_ready(web_checkpoint, sample, monkeypatch):
    class BrokenModel(torch.nn.Module):
        def forward(self, image):
            raise RuntimeError("synthetic initialization failure")
    monkeypatch.setattr(web_app, "load_model", lambda path: LoadedModel(BrokenModel(), .5, {}))
    with TestClient(web_app.create_app(web_checkpoint, sample), base_url="http://127.0.0.1") as client:
        info = client.get("/api/model").json()
        assert info["ready"] is False
        assert "initialization failure" in info["error"]
        assert client.post("/api/analyze-example").status_code == 503
