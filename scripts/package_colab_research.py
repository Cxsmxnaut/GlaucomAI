"""Package permitted research artifacts in Colab, excluding images and manifests."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("GlaucomAI_research_evidence.zip"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        raise ValueError("Archive exists. Choose a new output path; do not replace evidence.")
    paths = []
    for folder in ("sequential_run", "multitask_run"):
        for path in sorted((root / folder).iterdir()):
            if path.is_file() and (path.suffix == ".pth" or path.name in {
                "results.json", "stages.json", "clinical_audit.json", "input_guard_audit.json", "environment.txt"
            } or path.name.endswith(".history.json")):
                paths.append(path)
    paths.extend(sorted((root / "glaucoma").glob("*.py")))
    paths.extend(root / name for name in ("train.py", "evaluate.py", "prepare_data.py", "sequential_train.py",
                 "clinical_audit.py", "input_guard_audit.py", "multitask_train.py"))
    paths.extend(sorted((root / "docs/healthcare").glob("*.md")))
    for name in ("README.md", "DATASETS.md", "MODEL_CARD.md", "reports/evaluation.json"):
        if (root / name).is_file():
            paths.append(root / name)
    if not all(path.is_file() for path in paths) or len([p for p in paths if p.suffix == ".pth"]) != 5:
        raise ValueError("Expected all five S1/P2 research checkpoints and source files.")
    inventory = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                  "sha256": sha256(path.read_bytes()).hexdigest()} for path in paths]
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, arcname=path.relative_to(root).as_posix())
        bundle.writestr("artifact_inventory.json", json.dumps(inventory, indent=2))
        bundle.writestr("RESEARCH_NOTICE.md", "Research only. Not clinically validated or approved for patient care. "
                         "S1 mixes source label meanings; P2 separates them and was not promoted. "
                         "Original dataset terms apply, including PAPILA GPL-3.0+. "
                         "Dataset images, clinical spreadsheets, image manifests and credentials are excluded. "
                         "See the included DATASETS.md and healthcare documentation.")
    with zipfile.ZipFile(args.output) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive integrity check failed.")
        if any(Path(name).name in {"manifest.json", "kaggle.json"} for name in bundle.namelist()):
            raise ValueError("Excluded file unexpectedly entered the archive.")
    result = {"archive": str(args.output.resolve()), "bytes": args.output.stat().st_size,
              "sha256": sha256(args.output.read_bytes()).hexdigest(), "files": len(inventory),
              "checkpoint_count": 5, "images_included": False, "image_manifests_included": False,
              "clinical_release_allowed": False}
    args.output.with_suffix(".inventory.json").write_text(json.dumps(result, indent=2))
    print("RESEARCH ARCHIVE VERIFIED", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
