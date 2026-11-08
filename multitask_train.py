"""Execute preregistered P2.0 in Colab; separate labels and tune suffix blocks."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from glaucoma.metrics import binary_metrics, select_threshold
from glaucoma.multidata import RecordDataset, replay_indices
from glaucoma.multitask import ARCHITECTURE, TASKS, TASK_CLASSES, SeparateTaskModel, load_separate_tasks
from glaucoma.pipeline import PREPROCESSING, load_model
from glaucoma.statistics import score_statistics

SEED = 20261108


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def cache_prefix(model, rows, device):
    model.eval()
    tensors, labels = [], []
    loader = DataLoader(RecordDataset(rows), batch_size=32)
    with torch.inference_mode():
        for i, (images, y) in enumerate(loader):
            tensors.append(model.prefix(images.to(device)).cpu())
            labels.append(y)
            if (i + 1) % 25 == 0 or i + 1 == len(loader):
                print(f"  Frozen prefix: {min((i + 1) * 32, len(rows))}/{len(rows)}", flush=True)
    return TensorDataset(torch.cat(tensors).clone(), torch.cat(labels).clone(),
                         torch.tensor([TASKS.index(row["dataset"]) for row in rows]))


def routed_loss(model, prefix, labels, tasks):
    embedding = model.embedding(prefix)
    loss = embedding.new_zeros(())
    for task_index, task in enumerate(TASKS):
        mask = tasks == task_index
        if mask.any():
            loss = loss + F.cross_entropy(model.task_logits(embedding[mask], task), labels[mask], reduction="sum")
    return loss / len(labels)


def predictions(model, cache, indices, task, device):
    model.eval()
    labels, scores, loss_sum = [], [], 0.
    with torch.inference_mode():
        for prefix, y, _ in DataLoader(Subset(cache, indices), batch_size=32):
            y = y.to(device)
            logits = model.task_logits(model.embedding(prefix.to(device)), task)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            labels.extend(y.cpu().tolist())
            scores.extend(logits.softmax(1)[:, 1].cpu().tolist())
    if not labels or set(labels) != {0, 1}:
        raise ValueError("Every native-task validation/test population needs both classes.")
    return {"loss": loss_sum / len(labels), "labels": labels, "scores": scores}


def validation(model, cache, rows, tasks, device):
    return {task: predictions(model, cache,
        [i for i, row in enumerate(rows) if row["dataset"] == task and row["split"] == "val"], task, device)
        for task in tasks}


def train_stage(model, cache, rows, current, previous, parent, output, epochs, device, frozen_hash):
    active = previous + [current]
    indices = replay_indices(rows, current, previous)
    optimizer = torch.optim.AdamW([
        {"params": model.features[7:].parameters(), "lr": 1e-5},
        {"params": [p for task in active for p in model.heads[task].parameters()], "lr": 1e-3},
    ])
    loader = DataLoader(Subset(cache, indices), batch_size=32, shuffle=True)
    best_loss, best_state, best_epoch = float("inf"), None, None
    history = []
    print(f"P2 STAGE {current}: {len(indices)} training/replay images; separate task labels", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for prefix, labels, tasks in loader:
            optimizer.zero_grad()
            loss = routed_loss(model, prefix.to(device), labels.to(device), tasks.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; stopping without promotion.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        values = validation(model, cache, rows, active, device)
        val_loss = sum(value["loss"] for value in values.values()) / len(values)
        if model.prefix_sha256() != frozen_hash:
            raise ValueError("Frozen prefix changed; cached features are invalid.")
        history.append({"epoch": epoch, "training_loss": total / len(indices), "validation_macro_loss": val_loss,
                        "validation_task_loss": {key: value["loss"] for key, value in values.items()}})
        print(f"  Epoch {epoch:02d}/{epochs}: train {total/len(indices):.4f}, native-task macro val {val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None or any(not torch.isfinite(value).all() for value in best_state.values()):
        raise ValueError("No finite checkpoint available.")
    model.load_state_dict(best_state)
    values = validation(model, cache, rows, active, device)
    thresholds = {key: select_threshold(value["labels"], value["scores"]) for key, value in values.items()}
    metadata = {"architecture": ARCHITECTURE, "tasks": list(TASKS), "trained_tasks": active,
                "task_classes": TASK_CLASSES,
                "state_dict": best_state, "preprocessing": PREPROCESSING,
                "thresholds": thresholds, "seed": SEED, "epoch": best_epoch, "epochs_run": epochs,
                "frozen_prefix_sha256": frozen_hash, "parent_checkpoint_sha256": digest(parent),
                "training_domain_counts": dict(Counter(rows[i]["dataset"] for i in indices)),
                "protocol": "P2.0", "status": "research_only_unvalidated",
                "label_meanings": {"eyepacs": "dataset referral label", "acrima": "expert image label",
                                   "papila": "dataset clinical glaucoma label"},
                "validation_metrics": {key: binary_metrics(value["labels"], value["scores"], thresholds[key])
                                       for key, value in values.items()},
                "validation_macro_loss": best_loss}
    torch.save(metadata, output)
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    result = {key: value for key, value in metadata.items() if key != "state_dict"}
    result.update({"checkpoint": output.name, "checkpoint_sha256": digest(output)})
    print("SAVED", output.name, result["checkpoint_sha256"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("sequential_run/manifest.json"))
    parser.add_argument("--baseline", type=Path, default=Path("sequential_run/01_eyepacs.pth"))
    parser.add_argument("--work", type=Path, default=Path("multitask_run"))
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2 requires a Colab GPU; no local/CPU fallback.")
    if args.epochs != 12:
        raise ValueError("P2.0 specifies 12 epochs; revise the protocol before changing this.")
    if args.work.exists():
        raise ValueError("Choose a new output folder; prior runs must remain intact.")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    loaded = load_model(args.baseline)
    if loaded.metadata.get("dataset_provenance", {}).get("dataset") != "deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4":
        raise ValueError("Initialization must be the original EyePACS-only checkpoint.")
    rows = json.loads(args.manifest.read_text())
    original_run = json.loads((args.manifest.parent / "results.json").read_text())
    if digest(args.baseline) != original_run["stages"][0]["checkpoint_sha256"]:
        raise ValueError("Baseline is not the audited first-stage checkpoint.")
    counts = dict(Counter(f"{r['dataset']}/{r['split']}/{r['label']}" for r in rows))
    if counts != original_run["counts"]:
        raise ValueError("Manifest counts differ from the audited initialization record.")
    seen_hashes, groups = set(), {}
    for row in rows:
        if row["dataset"] not in TASKS or row["label"] not in (0, 1) or row["split"] not in ("train", "val", "test"):
            raise ValueError("Manifest has unsupported task/label/split.")
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("An input changed since the audited S1 manifest.")
        if row["sha256"] in seen_hashes:
            raise ValueError("Duplicate image in the manifest.")
        seen_hashes.add(row["sha256"])
        group = (row["dataset"], row["group"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Patient/group crosses splits.")
        groups[group] = row["split"]
    args.work.mkdir(parents=True)
    shutil.copyfile(args.manifest, args.work / "manifest.json")
    model = SeparateTaskModel(loaded.model).to(device)
    frozen_hash = model.prefix_sha256()
    print("P2 TRAINING IN COLAB", torch.cuda.get_device_name(0), "frozen prefix", frozen_hash, flush=True)
    feature_rows = [row for row in rows if row["split"] != "test"]
    cache = cache_prefix(model, feature_rows, device)
    stages, parent = [], args.baseline
    for index, current in enumerate(("acrima", "papila"), 2):
        output = args.work / f"{index:02d}_{current}_separate_tasks.pth"
        previous = ["eyepacs"] + (["acrima"] if current == "papila" else [])
        record = train_stage(model, cache, feature_rows, current, previous, parent, output,
                             args.epochs, device, frozen_hash)
        stages.append(record)
        parent = output
        (args.work / "stages.json").write_text(json.dumps(stages, indent=2))
    print("P2 CHECKPOINTS LOCKED. Existing test partitions are inspected development benchmarks.", flush=True)
    test_rows = [row for row in rows if row["split"] == "test"]
    test_cache = cache_prefix(model, test_rows, device)
    for stage in stages:
        candidate, metadata = load_separate_tasks(args.work / stage["checkpoint"])
        candidate = candidate.to(device)
        stage["development_benchmark_metrics"] = {}
        for task in metadata["trained_tasks"]:
            values = predictions(candidate, test_cache,
                [i for i, row in enumerate(test_rows) if row["dataset"] == task], task, device)
            stage["development_benchmark_metrics"][task] = score_statistics(values["labels"], values["scores"], metadata["thresholds"][task])
    report = {"protocol": "P2.0", "device": torch.cuda.get_device_name(0), "seed": SEED,
              "source_run_sha256": digest(args.manifest.parent / "results.json"),
              "initialization_sha256": digest(args.baseline), "manifest_sha256": digest(args.manifest),
              "frozen_prefix_sha256": frozen_hash, "stages": stages, "clinical_release_allowed": False,
              "test_status": "Previously inspected development benchmarks; not independent clinical confirmation.",
              "scope": "Three separate source-specific targets; no generic clinical output or automated source selection."}
    report["source_sha256"] = {name: digest(name) for name in
        ("multitask_train.py", "glaucoma/multitask.py", "glaucoma/multidata.py", "glaucoma/pipeline.py")}
    (args.work / "results.json").write_text(json.dumps(report, indent=2))
    print("P2 RESEARCH TRAINING COMPLETE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
