from collections import Counter

from PIL import Image
import pytest

from glaucoma.multidata import acrima_rows, audit_rows, locked_group_split, papila_rows, replay_indices
from sequential_train import domain_threshold


def test_patient_splits_are_deterministic_and_keep_both_eyes_together():
    rows = [{"group": f"p{i}", "label": i % 2, "eye": eye} for i in range(60) for eye in ("L", "R")]
    a = locked_group_split([dict(r) for r in rows])
    b = locked_group_split([dict(r) for r in reversed(rows)])
    assert {r["group"]: r["split"] for r in a} == {r["group"]: r["split"] for r in b}
    for group in {r["group"] for r in a}:
        assert len({r["split"] for r in a if r["group"] == group}) == 1
    assert {r["split"] for r in a} == {"train", "val", "test"}


def test_acrima_mapping_uses_explicit_g_marker_and_checks_counts(tmp_path):
    for i in range(1, 706):
        marker = "_g" if i > 309 else ""
        (tmp_path / f"Im{i:03d}{marker}_ACRIMA.jpg").touch()
    rows = acrima_rows(tmp_path)
    assert Counter(r["label"] for r in rows) == {0: 309, 1: 396}
    (tmp_path / "Im705_g_ACRIMA.jpg").unlink()
    with pytest.raises(ValueError, match="309 normal"):
        acrima_rows(tmp_path)


def test_papila_excludes_suspects_and_parses_three_header_rows(tmp_path):
    import openpyxl
    clinical = tmp_path / "ClinicalData"
    images = tmp_path / "FundusImages"
    clinical.mkdir()
    images.mkdir()
    labels = [0] * 333 + [1] * 87 + [2] * 68
    for eye_index, eye in enumerate(("od", "os")):
        workbook = openpyxl.Workbook()
        ws = workbook.active
        ws.append([None, "Age", "Gender", "Diagnosis"])
        ws.append([None, "Age", "Gender", "Diagnosis"])
        ws.append(["ID", None, None, None])
        for patient in range(1, 245):
            ws.append([f"#{patient:03d}", 65, 0, labels[(patient - 1) * 2 + eye_index]])
            (images / f"RET{patient:03d}{eye.upper()}.jpg").touch()
        workbook.save(clinical / f"patient_data_{eye}.xlsx")
    rows = papila_rows(tmp_path)
    assert Counter(r["label"] for r in rows) == {0: 333, 1: 87}
    assert len(rows) == 420
    for patient in {r["group"] for r in rows}:
        assert len({r["split"] for r in rows if r["group"] == patient}) == 1


def test_replay_never_uses_validation_test_or_future_domains():
    rows = [{"dataset": d, "split": s, "label": label} for d in ("eyepacs", "acrima", "papila")
            for s in ("train", "val", "test") for label in (0, 1) for _ in range(10)]
    indices = replay_indices(rows, "acrima", ["eyepacs"])
    assert all(rows[i]["split"] == "train" and rows[i]["dataset"] != "papila" for i in indices)
    assert Counter(rows[i]["dataset"] for i in indices) == {"acrima": 20, "eyepacs": 20}
    assert len(indices) == len(set(indices))


def test_duplicate_pixels_rejected_even_with_different_file_bytes(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    image = Image.new("RGB", (32, 32), "green")
    image.save(a, compress_level=0)
    image.save(b, compress_level=9)
    assert a.read_bytes() != b.read_bytes()
    rows = [{"path": str(p), "dataset": "source", "group": p.stem, "split": s}
            for p, s in ((a, "train"), (b, "test"))]
    with pytest.raises(ValueError, match="Duplicate image"):
        audit_rows(rows)


def test_threshold_weights_sources_equally_instead_of_large_source_dominating():
    values = {"large": {"labels": [0, 1] * 50, "scores": [.1, .9] * 50},
              "small": {"labels": [0, 1], "scores": [.2, .3]}}
    assert domain_threshold(values) == .3


def test_adaptation_continues_parent_head_preserves_backbone_and_records_lineage(tmp_path, monkeypatch):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset
    from glaucoma.pipeline import LoadedModel
    import sequential_train

    torch.manual_seed(4)
    model = nn.Module()
    model.features = nn.Linear(3, 3)
    model.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(3, 2))
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    parent = tmp_path / "parent.pth"
    parent.write_bytes(b"synthetic-test-parent")
    monkeypatch.setattr(sequential_train, "load_model", lambda _: LoadedModel(model, .5, {}))
    rows = [{"dataset": d, "split": split, "label": label}
            for d in ("eyepacs", "acrima") for split in ("train", "val") for label in (0, 1)]
    features = TensorDataset(torch.randn(8, 3), torch.tensor([r["label"] for r in rows]))
    output = tmp_path / "adapted.pth"
    record = sequential_train.adapt(parent, output, features, rows, "acrima", ["eyepacs"], 2, torch.device("cpu"))
    saved = torch.load(output, weights_only=True)
    assert record["parent_checkpoint_sha256"] == sequential_train.digest(parent)
    assert record["checkpoint_sha256"] == sequential_train.digest(output)
    assert record["training_domain_counts"] == {"acrima": 2, "eyepacs": 2}
    assert torch.equal(saved["state_dict"]["features.weight"], original["features.weight"])
    assert not torch.equal(saved["state_dict"]["classifier.1.weight"], original["classifier.1.weight"])
    assert record["epochs_run"] == 2


def test_sequential_notebook_matches_source_and_is_clean():
    import ast
    import json
    from pathlib import Path
    import runpy
    import nbformat
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/GlaucomAI_Sequential.ipynb"
    nbformat.validate(nbformat.read(path, as_version=4))
    notebook = json.loads(path.read_text())
    assert notebook == runpy.run_path(str(root / "scripts/build_sequential_notebook.py"))["build"]()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


def test_referral_dashboard_rejects_mixed_clinical_label_meanings(monkeypatch):
    from glaucoma import notebook_ui
    from glaucoma.pipeline import LoadedModel, ModelError
    monkeypatch.setattr(notebook_ui, "load_model", lambda _: LoadedModel(
        None, .5, {"label_semantics": "mixed_dataset_binary_research_labels"}))
    with pytest.raises(ModelError, match="cannot be displayed as a referral score"):
        notebook_ui.display_assistant("synthetic-mixed-checkpoint.pth")
