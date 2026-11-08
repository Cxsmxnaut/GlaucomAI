"""Shared, deterministic preprocessing and CPU inference. Never stores images."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import math
import time
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from torchvision import models, transforms

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
CLASSES = ["non_referable", "referable"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PREPROCESSING = {"size": [224, 224], "mean": MEAN, "std": STD}


class InputError(ValueError):
    pass


class ModelError(ValueError):
    pass


def decode_image(data: bytes, filename: str) -> Image.Image:
    """Check the actual format, byte size and pixel count, then remove metadata."""
    if Path(filename).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise InputError("Choose a PNG or JPEG image.")
    if not data or len(data) >= MAX_BYTES:
        raise InputError("Choose a non-empty image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise InputError("The file contents must be PNG or JPEG.")
                if source.width * source.height > MAX_PIXELS:
                    raise InputError("The image exceeds the 25-megapixel safety limit.")
                source.verify()
            with Image.open(BytesIO(data)) as source:
                oriented = ImageOps.exif_transpose(source).convert("RGB")
                # A new image avoids retaining EXIF, names or other source metadata.
                return Image.frombytes("RGB", oriented.size, oriented.tobytes())
    except InputError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InputError("This image could not be decoded. Try another PNG or JPEG.") from exc


def preprocessing():
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def validate_image_signal(image: Image.Image):
    """Reject an exactly uniform input; this is not a retinal-quality classifier."""
    extrema = image.convert("RGB").getextrema()
    if all(low == high for low, high in extrema):
        raise InputError("This image is completely uniform and contains no usable image detail. Choose another research image. No score was produced.")


def build_model(pretrained: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


@dataclass
class LoadedModel:
    model: nn.Module
    threshold: float
    metadata: dict


def load_model(path: str | Path) -> LoadedModel:
    """Load only a local, trusted checkpoint. Never download inference weights."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ModelError("Checkpoint must be a dictionary exported by train.py.")
        if checkpoint.get("architecture") != "efficientnet_b0":
            raise ModelError("Checkpoint architecture must be efficientnet_b0.")
        if checkpoint.get("classes") != CLASSES:
            raise ModelError("Checkpoint class order must be non_referable=0, referable=1.")
        if checkpoint.get("preprocessing") != PREPROCESSING:
            raise ModelError("Checkpoint preprocessing does not match this app.")
        threshold = float(checkpoint["threshold"])
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ModelError("Checkpoint threshold must be between zero and one.")
        model = build_model()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        if any(not torch.isfinite(p).all() for p in model.state_dict().values()):
            raise ModelError("Checkpoint contains invalid model values.")
        return LoadedModel(model, threshold, {
            key: value for key, value in checkpoint.items() if key != "state_dict"
        })
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError("Cannot load this checkpoint. Export a compatible model with train.py.") from exc


def infer(image: Image.Image, loaded: LoadedModel) -> dict:
    start = time.perf_counter()
    validate_image_signal(image)
    with torch.inference_mode():
        tensor = preprocessing()(image.convert("RGB")).unsqueeze(0)
        probabilities = torch.softmax(loaded.model(tensor), dim=1)[0]
    values = probabilities.tolist()
    if len(values) != 2 or not all(math.isfinite(p) for p in values):
        raise ModelError("The model returned invalid scores. No result is available.")
    return {
        "non_referable": values[0],
        "referable": values[1],
        "threshold": loaded.threshold,
        "flagged": values[1] >= loaded.threshold,
        "elapsed_ms": (time.perf_counter() - start) * 1000,
        "demo": False,
    }
