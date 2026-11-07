from io import BytesIO

from PIL import Image
import pytest
import torch

from glaucoma.pipeline import (
    CLASSES, MAX_BYTES, MEAN, PREPROCESSING, STD, InputError, LoadedModel,
    ModelError, build_model, decode_image, infer, load_model, preprocessing,
)


def encoded_image(mode="RGB", format="PNG", color=128):
    output = BytesIO()
    Image.new(mode, (32, 48), color=color).save(output, format=format)
    return output.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_decode_and_exact_preprocessing(mode):
    image = decode_image(encoded_image(mode), "retina.png")
    assert image.mode == "RGB"
    assert image.size == (32, 48)
    tensor = preprocessing()(image)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert not image.info


def test_imagenet_normalization_values():
    tensor = preprocessing()(Image.new("RGB", (100, 100), (255, 0, 0)))
    expected = torch.tensor([(1 - MEAN[0]) / STD[0], -MEAN[1] / STD[1], -MEAN[2] / STD[2]])
    assert torch.allclose(tensor[:, 0, 0], expected)


@pytest.mark.parametrize("data,name", [
    (b"", "a.png"), (b"not an image", "a.jpg"),
    (encoded_image(format="GIF"), "a.png"), (encoded_image(), "a.exe"),
    (b"x" * MAX_BYTES, "a.jpg"), (encoded_image()[:30], "a.png"),
])
def test_rejects_invalid_uploads(data, name):
    with pytest.raises(InputError):
        decode_image(data, name)


def test_pixel_limit(monkeypatch):
    monkeypatch.setattr("glaucoma.pipeline.MAX_PIXELS", 10)
    with pytest.raises(InputError, match="megapixel"):
        decode_image(encoded_image(), "a.png")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("checkpoint") / "untrained_test_fixture.pth"
    model = build_model(pretrained=False)
    payload = {"architecture": "efficientnet_b0", "classes": CLASSES,
               "preprocessing": PREPROCESSING, "threshold": 0.4, "state_dict": model.state_dict()}
    torch.save(payload, path)
    return path


def test_real_checkpoint_roundtrip_and_cpu_inference(checkpoint):
    loaded = load_model(checkpoint)
    image = decode_image(encoded_image(), "retina.png")
    result = infer(image, loaded)
    assert result["non_referable"] + result["referable"] == pytest.approx(1)
    assert result["flagged"] == (result["referable"] >= 0.4)
    assert result["elapsed_ms"] > 0
    assert result["demo"] is False
    assert not loaded.model.training
    assert next(loaded.model.parameters()).device.type == "cpu"


@pytest.mark.parametrize("key,value", [
    ("classes", list(reversed(CLASSES))), ("architecture", "resnet50"),
    ("threshold", 0), ("threshold", float("nan")), ("preprocessing", {}),
])
def test_rejects_incompatible_checkpoint(checkpoint, tmp_path, key, value):
    payload = torch.load(checkpoint, weights_only=True)
    payload[key] = value
    path = tmp_path / "bad.pth"
    torch.save(payload, path)
    with pytest.raises(ModelError):
        load_model(path)


def test_no_random_fallback_for_missing_checkpoint(tmp_path):
    with pytest.raises(ModelError):
        load_model(tmp_path / "missing.pth")


def test_rejects_nonfinite_model_output():
    class BadModel(torch.nn.Module):
        def forward(self, image):
            return torch.tensor([[float("nan"), 0.0]])

    with pytest.raises(ModelError):
        infer(Image.new("RGB", (224, 224)), LoadedModel(BadModel(), 0.5, {}))
