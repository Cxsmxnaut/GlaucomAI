"""Download the pinned Kaggle dataset and preserve its supplied splits."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil

from glaucoma.data import FundusDataset, check_split_overlap
from glaucoma.pipeline import decode_image

DATASET = "deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4"
SOURCE = "https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4"
SPLITS = {"train": "train", "validation": "val", "test": "test"}
LABELS = {"NRG": "non_referable", "RG": "referable"}


def prepare(source: Path, output: Path):
    """Validate every file before copying; no guessed labels or random re-split."""
    roots = [p for p in source.rglob("*") if p.is_dir()
             and all((p / name).is_dir() for name in SPLITS)]
    if all((source / name).is_dir() for name in SPLITS):
        roots.insert(0, source)
    if len(roots) != 1:
        raise ValueError("Expected one dataset root with train/, validation/ and test/ folders.")
    root = roots[0]
    if output.exists():
        raise ValueError(f"{output} already exists. Choose a new output directory to avoid mixing splits.")
    rows = []
    seen = {}
    for source_split, split in SPLITS.items():
        for source_class, label in LABELS.items():
            folder = root / source_split / source_class
            paths = sorted(p for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not paths:
                raise ValueError(f"No images found in {folder}.")
            for path in paths:
                raw = path.read_bytes()
                decode_image(raw, path.name).close()
                digest = sha256(raw).hexdigest()
                if digest in seen:
                    raise ValueError(f"Duplicate image: {path} and {seen[digest]}. Review source data.")
                seen[digest] = path
                rows.append({
                    "source": path.relative_to(root).as_posix(), "split": split,
                    "label": label, "sha256": digest,
                    "destination": f"{split}/{label}/{digest[:16]}{path.suffix.lower()}",
                })
    # Validation succeeds before creating the output directory.
    output.mkdir(parents=True)
    for row in rows:
        destination = output / row["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row["source"], destination)
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    check_split_overlap(*[FundusDataset(output / split) for split in SPLITS.values()])
    provenance = {
        "dataset": DATASET, "source_url": SOURCE,
        "counts": dict(Counter(f"{r['split']}/{r['label']}" for r in rows)),
        "split_policy": "Preserve the dataset publisher's train/validation/test splits.",
        "patient_separation": "Not independently verified. File-hash disjointness does not establish patient disjointness.",
        "label_mapping": LABELS,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Use an already downloaded Kaggle dataset root.")
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    args = parser.parse_args()
    if args.source:
        source = args.source
    else:
        import kagglehub
        source = Path(kagglehub.dataset_download(DATASET))
    print(json.dumps(prepare(source, args.output), indent=2))


if __name__ == "__main__":
    main()
