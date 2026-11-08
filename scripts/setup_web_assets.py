"""Fetch pinned browser tracker assets and one attributed local benchmark sample."""

import argparse
import base64
from hashlib import sha256, sha512
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def setup(download_baseline=False):
    config = json.loads((ROOT / "web/assets.lock.json").read_text())
    output = ROOT / "web/vendor"
    archive = None
    for name, metadata in config["files"].items():
        destination = output / name
        if destination.is_file() and sha256(destination.read_bytes()).hexdigest() == metadata["sha256"]:
            continue
        print("Preparing browser asset:", name, flush=True)
        if "archive_path" in metadata:
            if archive is None:
                raw = urllib.request.urlopen(config["npm_tarball"], timeout=60).read()
                if base64.b64encode(sha512(raw).digest()).decode() != config["npm_sha512"]:
                    raise ValueError("Browser SDK archive integrity check failed.")
                archive = tarfile.open(fileobj=BytesIO(raw), mode="r:gz")
            content = archive.extractfile(metadata["archive_path"]).read()
        else:
            content = urllib.request.urlopen(metadata["url"], timeout=60).read()
        if len(content) != metadata["bytes"] or sha256(content).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Asset integrity check failed: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    if archive:
        archive.close()
    sample = config["sample"]
    destination = ROOT / "data/web_sample/example-retina.jpg"
    if not destination.exists():
        prepared = ROOT / sample["prepared_path"]
        if prepared.is_file():
            source = prepared
        else:
            import kagglehub
            source = Path(kagglehub.dataset_download(sample["dataset"], path=sample["source_path"]))
            if source.is_dir():
                source = source / sample["source_path"]
        if sha256(source.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError("Benchmark example integrity check failed.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if sha256(destination.read_bytes()).hexdigest() != sample["sha256"]:
        raise ValueError("Existing benchmark example differs from its pinned source.")
    if download_baseline:
        sys.path.insert(0, str(ROOT))
        from glaucoma.download import download_checkpoint
        baseline = json.loads((ROOT / "experiments/baseline.json").read_text())
        download_checkpoint(baseline["checkpoint_url"], baseline["checkpoint_sha256"],
                            ROOT / "models/best_model.pth")
        print("Published v0.1.0 checkpoint verified.")
    print("Browser assets and benchmark example verified. Camera processing stays in the browser.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-baseline", action="store_true",
                        help="Fetch and verify the published checkpoint; never overwrite a different model.")
    setup(parser.parse_args().download_baseline)
