"""Fetch a published checkpoint and verify its exact SHA-256 before loading."""

from hashlib import sha256
from pathlib import Path
import urllib.request


def download_checkpoint(url: str, expected_sha256: str, destination="models/best_model.pth"):
    if not url.startswith("https://github.com/Cxsmxnaut/GlaucomAI/releases/download/"):
        raise ValueError("Only this project's HTTPS release URLs are supported.")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("Expected a lowercase SHA-256 checksum.")
    destination = Path(destination)
    if destination.exists():
        if sha256(destination.read_bytes()).hexdigest() == expected_sha256:
            return destination
        raise ValueError("A different checkpoint already exists. Choose a new path or remove it explicitly.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "GlaucomAI-research-demo"})
        total = 0
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 100 * 1024 * 1024:
                    raise ValueError("Unexpected checkpoint size. Download stopped.")
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Checkpoint checksum mismatch. The downloaded file was discarded.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
