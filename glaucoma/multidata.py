"""Explicit adapters and locked splits for the sequential research experiment."""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import random
import re

from torch.utils.data import Dataset

from glaucoma.pipeline import CLASSES, decode_image, preprocessing

SEED = 20261107
HANDLES = {
    "eyepacs": "deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4",
    "acrima": "orvile/acrima-glaucoma-assessment-using-fundus-images/versions/1",
    "papila": "orvile/papila-retinal-fundus-images/versions/1",
}


def locked_group_split(rows, seed=SEED):
    """Stratify groups by maximum label; no patient can cross splits."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    strata = defaultdict(list)
    for group, members in groups.items():
        strata[max(row["label"] for row in members)].append(group)
    rng = random.Random(seed)
    for label in sorted(strata):
        keys = sorted(strata[label])
        rng.shuffle(keys)
        n_test = max(1, round(len(keys) * .15))
        n_val = max(1, round(len(keys) * .15))
        if len(keys) - n_test - n_val < 1:
            raise ValueError("Too few groups for train/validation/test.")
        for i, group in enumerate(keys):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for row in groups[group]:
                row["split"] = split
    return rows


def acrima_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = re.fullmatch(r"Im(\d{3})(_g)?_ACRIMA", path.stem)
        if not match:
            raise ValueError(f"Unrecognized ACRIMA image: {path.name}")
        rows.append({"path": str(path), "label": int(bool(match[2])),
                     "group": match[1], "dataset": "acrima"})
    if Counter(r["label"] for r in rows) != {0: 309, 1: 396}:
        raise ValueError("ACRIMA must contain 309 normal and 396 glaucoma images.")
    return locked_group_split(rows)


def papila_rows(root):
    # Original paper, Data Records: 0 healthy, 1 glaucoma, 2 suspicious.
    # Suspects are excluded, never relabeled as healthy.
    import openpyxl

    root = Path(root)
    images = {p.name: p for p in root.rglob("*.jpg") if p.parent.name == "FundusImages"}
    rows, counts = [], Counter()
    for eye in ("od", "os"):
        files = list(root.rglob(f"patient_data_{eye}.xlsx"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one PAPILA {eye} clinical table.")
        workbook = openpyxl.load_workbook(files[0], read_only=True, data_only=True)
        try:
            table = list(workbook.active.values)
        finally:
            workbook.close()
        # Header names and diagnosis column are checked, not inferred from counts.
        header_index = next((i for i, row in enumerate(table)
                             if any(str(v).strip().lower() == "diagnosis" for v in row)), None)
        if header_index is None:
            raise ValueError("PAPILA diagnosis header not found.")
        header = [str(v).strip().lower() for v in table[header_index]]
        diagnosis = header.index("diagnosis")
        for row in table[header_index + 1:]:
            identifier = row[0]
            if identifier is None:
                continue
            if str(identifier).strip() == "ID" and row[diagnosis] is None:
                continue
            match = re.fullmatch(r"(?:#|RET)?0*(\d+)", str(identifier).strip())
            if not match or row[diagnosis] not in (0, 1, 2):
                raise ValueError("Unexpected PAPILA patient identifier or diagnosis.")
            number = int(match[1])
            name = f"RET{number:03d}{eye.upper()}.jpg"
            if name not in images:
                raise ValueError(f"Missing PAPILA image {name}.")
            label = int(row[diagnosis])
            counts[label] += 1
            if label == 2:
                continue
            rows.append({"path": str(images[name]), "label": label,
                         "group": f"RET{number:03d}", "dataset": "papila"})
    if counts != {0: 333, 1: 87, 2: 68}:
        raise ValueError(f"Unexpected PAPILA label counts: {counts}")
    return locked_group_split(rows)


def eyepacs_rows(prepared):
    rows = []
    for split in ("train", "val", "test"):
        for label, name in enumerate(CLASSES):
            for path in sorted((Path(prepared) / split / name).glob("*")):
                if path.is_file():
                    rows.append({"path": str(path), "label": label, "split": split,
                                 "group": path.stem, "dataset": "eyepacs"})
    if Counter(r["split"] for r in rows) != {"train": 8000, "val": 770, "test": 770}:
        raise ValueError("Unexpected EyePACS publisher split counts.")
    return rows


def audit_rows(rows):
    """Reject byte or decoded-pixel duplicates across splits or source datasets."""
    seen, groups = {}, {}
    for i, row in enumerate(rows):
        path = Path(row["path"])
        raw = path.read_bytes()
        with decode_image(raw, path.name) as image:
            pixel_hash = sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        for kind, digest in (("file", sha256(raw).hexdigest()), ("pixels", pixel_hash)):
            key = (kind, digest)
            if key in seen and seen[key]["path"] != row["path"]:
                raise ValueError(f"Duplicate image in experiment: {path.name} / {Path(seen[key]['path']).name}")
            seen[key] = row
        row["sha256"] = sha256(raw).hexdigest()
        key = (row["dataset"], row["group"])
        if key in groups and groups[key] != row["split"]:
            raise ValueError("Patient/group crosses data splits.")
        groups[key] = row["split"]
        if (i + 1) % 1000 == 0:
            print(f"Audited {i + 1}/{len(rows)} images", flush=True)
    return dict(Counter(f"{r['dataset']}/{r['split']}/{r['label']}" for r in rows))


class RecordDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = preprocessing()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["path"])
        with decode_image(path.read_bytes(), path.name) as image:
            return self.transform(image), row["label"]


def replay_indices(rows, current, previous, seed=SEED):
    """All new training data plus at most its size in earlier training examples."""
    new = [i for i, row in enumerate(rows) if row["dataset"] == current and row["split"] == "train"]
    rng = random.Random(seed)
    replay = []
    if previous:
        quota = max(1, len(new) // (len(previous) * 2))
        for dataset in previous:
            for label in (0, 1):
                pool = [i for i, row in enumerate(rows) if row["dataset"] == dataset
                        and row["split"] == "train" and row["label"] == label]
                rng.shuffle(pool)
                replay.extend(pool[:quota])
    return new + replay
