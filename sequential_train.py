"""Colab-only sequential adaptation: EyePACS -> ACRIMA -> PAPILA, with replay."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from glaucoma.metrics import binary_metrics
from glaucoma.multidata import (HANDLES, SEED, RecordDataset, acrima_rows, audit_rows,
                               eyepacs_rows, papila_rows, replay_indices)
from glaucoma.pipeline import load_model
from glaucoma.runtime import run_logged
from train import extract_features


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validation(model, features, rows, domains, device):
    model.eval()
    result = {}
    with torch.inference_mode():
        for domain in domains:
            indices = [i for i, r in enumerate(rows) if r["dataset"] == domain and r["split"] == "val"]
            x, y = (v[indices].to(device) for v in features.tensors)
            logits = model.classifier(x)
            result[domain] = {"loss": nn.functional.cross_entropy(logits, y).item(),
                              "labels": y.cpu().tolist(), "scores": logits.softmax(1)[:, 1].cpu().tolist()}
    return result


def domain_threshold(predictions, target=.88):
    """Give each source equal weight when selecting the validation threshold."""
    candidates = sorted({.5, *(p for v in predictions.values() for p in v["scores"] if 0 < p < 1)})
    best, best_specificity = None, -1.
    for threshold in candidates:
        metrics = [binary_metrics(v["labels"], v["scores"], threshold) for v in predictions.values()]
        recall = sum(m["sensitivity"] for m in metrics) / len(metrics)
        specificity = sum(m["specificity"] for m in metrics) / len(metrics)
        if recall >= target and specificity > best_specificity:
            best, best_specificity = threshold, specificity
    if best is None:
        raise ValueError("No valid validation threshold reaches the requested macro sensitivity.")
    return best


def adapt(parent, output, features, rows, domain, previous, epochs, device):
    torch.manual_seed(SEED)
    loaded = load_model(parent)
    model = loaded.model.to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    indices = replay_indices(rows, domain, previous)
    loader = DataLoader(Subset(features, indices), batch_size=32, shuffle=True)
    domains = previous + [domain]
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=3e-4)
    best_loss, best_state, best_epoch = float("inf"), None, None
    history = []
    print(f"STAGE {domain}: continue {digest(parent)}; {len(indices)} new/replay training examples", flush=True)
    for epoch in range(1, epochs + 1):
        model.classifier.train()
        loss_sum = 0.
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model.classifier(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        values = validation(model, features, rows, domains, device)
        val_loss = sum(v["loss"] for v in values.values()) / len(values)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(indices),
                        "validation_macro_loss": val_loss,
                        "validation_domain_loss": {k: v["loss"] for k, v in values.items()}})
        print(f"  Epoch {epoch:02d}/{epochs}: train={loss_sum / len(indices):.4f}, macro val={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    values = validation(model, features, rows, domains, device)
    threshold = domain_threshold(values)
    metadata = dict(loaded.metadata)
    metadata.update({"state_dict": best_state, "threshold": threshold, "epoch": best_epoch,
                     "seed": SEED, "parent_checkpoint_sha256": digest(parent),
                     "stage": domain, "training_domains": domains,
                     "training_samples": len(indices),
                     "training_domain_counts": dict(Counter(rows[i]["dataset"] for i in indices)),
                     "dataset_provenance": {k: HANDLES[k] for k in domains},
                     "label_semantics": "mixed_dataset_binary_research_labels",
                     "label_note": "EyePACS referral labels differ from ACRIMA/PAPILA glaucoma labels; scores are not clinical referral risks.",
                     "validation_loss": best_loss,
                     "validation_metrics": {k: binary_metrics(v["labels"], v["scores"], threshold) for k, v in values.items()},
                     "status": "research_only_unvalidated"})
    torch.save(metadata, output)
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    record = {k: v for k, v in metadata.items() if k != "state_dict"}
    record.update({"checkpoint": output.name, "checkpoint_sha256": digest(output), "epochs_run": epochs})
    print(f"SAVED {output.name} sha256={record['checkpoint_sha256']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("sequential_run"))
    parser.add_argument("--baseline", type=Path, help="Reuse a trusted EyePACS-only checkpoint trained in this Colab runtime.")
    parser.add_argument("--eyepacs-prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This workflow requires a Colab GPU; choose a T4 runtime before running.")
    if args.work.exists():
        raise ValueError("Output directory already exists; choose a new --work to preserve prior runs.")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    print("TRAINING IN COLAB:", torch.cuda.get_device_name(0), flush=True)
    import kagglehub
    from prepare_data import prepare

    if not args.eyepacs_prepared.exists():
        prepare(Path(kagglehub.dataset_download(HANDLES["eyepacs"])), args.eyepacs_prepared)
    rows = eyepacs_rows(args.eyepacs_prepared)
    for domain, adapter in (("acrima", acrima_rows), ("papila", papila_rows)):
        source = Path(kagglehub.dataset_download(HANDLES[domain]))
        rows.extend(adapter(source))
    counts = audit_rows(rows)
    print("LOCKED SPLITS", json.dumps(counts, sort_keys=True), flush=True)
    args.work.mkdir(parents=True)
    (args.work / "manifest.json").write_text(json.dumps(rows, indent=2))
    baseline = args.work / "01_eyepacs.pth"
    if args.baseline:
        source = load_model(args.baseline)
        if source.metadata.get("label_semantics") or source.metadata.get("training_domains"):
            raise ValueError("Baseline must be EyePACS-only, not an adapted checkpoint.")
        provenance = source.metadata.get("dataset_provenance", {})
        if provenance.get("dataset") != HANDLES["eyepacs"] or source.metadata.get("training_samples") != 8000:
            raise ValueError("Baseline provenance must identify the pinned 8,000-image EyePACS training split.")
        shutil.copyfile(args.baseline, baseline)
    else:
        run_logged([sys.executable, "-u", "train.py", "--data", str(args.eyepacs_prepared),
                    "--epochs", str(args.epochs), "--batch-size", "32", "--device", "cuda", "--output", str(baseline)])
    loaded = load_model(baseline)
    record = {k: v for k, v in loaded.metadata.items()}
    record.update({"stage": "eyepacs", "checkpoint": baseline.name, "checkpoint_sha256": digest(baseline),
                   "reused_existing_colab_stage": bool(args.baseline)})
    stages = [record]
    feature_rows = [r for r in rows if r["split"] != "test"]
    print("Caching frozen features for training and validation only.", flush=True)
    features = extract_features(loaded.model.to(device), RecordDataset(feature_rows), device, 32)
    parent = baseline
    for index, domain in enumerate(("acrima", "papila"), 2):
        output = args.work / f"{index:02d}_{domain}.pth"
        stages.append(adapt(parent, output, features, feature_rows, domain,
                            ["eyepacs"] + (["acrima"] if domain == "papila" else []), args.epochs, device))
        parent = output
        (args.work / "stages.json").write_text(json.dumps(stages, indent=2))
    # Test extraction and every test evaluation happen only after all stages are locked.
    print("ALL STAGES LOCKED. Evaluating withheld test images once per checkpoint.", flush=True)
    test_rows = [r for r in rows if r["split"] == "test"]
    test_features = extract_features(load_model(parent).model.to(device), RecordDataset(test_rows), device, 32)
    for stage in stages:
        model = load_model(args.work / stage["checkpoint"]).model.to(device).eval()
        with torch.inference_mode():
            scores = model.classifier(test_features.tensors[0].to(device)).softmax(1)[:, 1].cpu().tolist()
        stage["test_metrics"] = {}
        for domain in HANDLES:
            indices = [i for i, r in enumerate(test_rows) if r["dataset"] == domain]
            stage["test_metrics"][domain] = binary_metrics(
                [test_rows[i]["label"] for i in indices], [scores[i] for i in indices], stage["threshold"])
    report = {"training_device": torch.cuda.get_device_name(0), "seed": SEED,
              "datasets": HANDLES, "counts": counts, "stages": stages,
              "label_semantics": "Dataset-specific binary labels; not interchangeable clinical diagnoses.",
              "test_policy": "All checkpoints locked before test evaluation. EyePACS test was inspected in the earlier baseline run; new ACRIMA/PAPILA test splits were withheld throughout adaptation.",
              "split_limitations": "PAPILA patient-group split; ACRIMA image split (patient IDs unavailable); EyePACS publisher splits, patient separation unverified. Byte/pixel hashes reject exact duplicates, not all near-duplicates.",
              "method": "Frozen ImageNet EfficientNet-B0; sequential head adaptation with bounded replay. Macro source validation loss chooses epochs; equal-source validation sensitivity >=88% chooses thresholds.",
              "status": "research_only_unvalidated"}
    (args.work / "results.json").write_text(json.dumps(report, indent=2))
    versions = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (args.work / "environment.txt").write_text(versions)
    archive = args.work.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in args.work.iterdir():
            # The full image manifest stays private to the runtime; no images or clinical tables are exported.
            if path.name != "manifest.json":
                bundle.write(path, arcname=path.name)
    print("SEQUENTIAL TRAINING COMPLETE", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print("Archive:", archive, flush=True)


if __name__ == "__main__":
    main()
