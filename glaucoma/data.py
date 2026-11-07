"""Explicit class order; identical files cannot cross dataset splits."""

from hashlib import sha256
from pathlib import Path

from torch.utils.data import Dataset

from glaucoma.pipeline import CLASSES, decode_image, preprocessing


class FundusDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.transform = preprocessing()
        for label, name in enumerate(CLASSES):
            folder = Path(root) / name
            paths = sorted(p for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            if not paths:
                raise ValueError(f"No PNG/JPEG images found in {folder}.")
            self.samples.extend((p, label) for p in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(decode_image(path.read_bytes(), path.name)), label


def check_split_overlap(*datasets):
    seen = {}
    for split, dataset in enumerate(datasets):
        for path, _ in dataset.samples:
            digest = sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError(f"Duplicate image crosses dataset splits: {path} and {seen[digest][1]}")
            seen[digest] = (split, path)
