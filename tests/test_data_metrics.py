import json

from PIL import Image
import pytest

from glaucoma.data import FundusDataset, check_split_overlap
from glaucoma.metrics import binary_metrics, select_threshold
from prepare_data import prepare


def make_dataset(root):
    for split_index, split in enumerate(("train", "validation", "test")):
        for label_index, label in enumerate(("NRG", "RG")):
            folder = root / "eyepac-light-v2-512-jpg" / split / label
            folder.mkdir(parents=True)
            Image.new("RGB", (24, 30), (split_index * 40, label_index * 100, 20)).save(folder / "same_name.jpg")


def test_preserves_publisher_splits_and_class_order(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "prepared"
    make_dataset(source)
    result = prepare(source, destination)
    assert len(result["counts"]) == 6
    assert sum(result["counts"].values()) == 6
    for split in ("train", "val", "test"):
        dataset = FundusDataset(destination / split)
        assert [label for _, label in dataset.samples] == [0, 1]
        assert "non_referable" in str(dataset.samples[0][0])
        assert "referable" in str(dataset.samples[1][0])
    assert "Not independently verified" in result["patient_separation"]
    assert json.loads((destination / "provenance.json").read_text())["label_mapping"]["NRG"] == "non_referable"
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, destination)


def test_detects_duplicate_across_splits_before_output(tmp_path):
    source = tmp_path / "source"
    make_dataset(source)
    root = source / "eyepac-light-v2-512-jpg"
    (root / "test/NRG/same_name.jpg").write_bytes((root / "train/NRG/same_name.jpg").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_overlap_check_for_user_supplied_dataset(tmp_path):
    for split in ("train", "val"):
        for label in ("non_referable", "referable"):
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            Image.new("RGB", (20, 20)).save(folder / "a.png")
    with pytest.raises(ValueError, match="crosses dataset splits"):
        check_split_overlap(FundusDataset(tmp_path / "train"), FundusDataset(tmp_path / "val"))


def test_metrics_and_validation_threshold():
    labels, scores = [0, 0, 1, 1], [0.1, 0.3, 0.2, 0.9]
    result = binary_metrics(labels, scores, 0.5)
    assert result["true_positive"] == 1
    assert result["false_negative"] == 1
    assert result["specificity"] == 1
    threshold = select_threshold(labels, scores, target=0.88)
    assert threshold == 0.2
    selected = binary_metrics(labels, scores, threshold)
    assert selected["sensitivity"] == 1
    assert selected["specificity"] == 0.5


def test_missing_classes_and_invalid_metrics_are_explicit():
    assert binary_metrics([0, 0], [0.1, 0.2])["sensitivity"] is None
    with pytest.raises(ValueError):
        select_threshold([0, 0], [0.1, 0.2])
    for labels, scores in [([], []), ([1], []), ([3], [0.2]), ([1], [float("nan")])]:
        with pytest.raises(ValueError):
            binary_metrics(labels, scores)
