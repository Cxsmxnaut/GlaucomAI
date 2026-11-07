from hashlib import sha256
from io import BytesIO

import pytest

from glaucoma.download import download_checkpoint

URL = "https://github.com/Cxsmxnaut/GlaucomAI/releases/download/v0.1.0/best_model.pth"


def test_download_verifies_exact_hash_and_reuses_existing(monkeypatch, tmp_path):
    content = b"test checkpoint bytes"
    checksum = sha256(content).hexdigest()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: BytesIO(content))
    destination = tmp_path / "models/test.pth"
    assert download_checkpoint(URL, checksum, destination) == destination
    assert destination.read_bytes() == content
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("should not download twice"))
    assert download_checkpoint(URL, checksum, destination) == destination


def test_bad_checksum_removes_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: BytesIO(b"wrong bytes"))
    destination = tmp_path / "bad.pth"
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_checkpoint(URL, "0" * 64, destination)
    assert not destination.exists()
    assert not destination.with_suffix(".pth.part").exists()


def test_does_not_overwrite_existing_model(tmp_path):
    destination = tmp_path / "existing.pth"
    destination.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="different checkpoint"):
        download_checkpoint(URL, "0" * 64, destination)
    assert destination.read_bytes() == b"keep me"


def test_rejects_external_download_url(tmp_path):
    with pytest.raises(ValueError):
        download_checkpoint("https://example.com/model.pth", "0" * 64, tmp_path / "bad.pth")
