"""Evaluate a locked model on held-out images; report real CPU latency."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from glaucoma.data import FundusDataset, check_split_overlap
from glaucoma.metrics import binary_metrics
from glaucoma.pipeline import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    train, val, test = [FundusDataset(args.data / split) for split in ("train", "val", "test")]
    check_split_overlap(train, val, test)
    loaded = load_model(args.model)
    labels, scores, latencies = [], [], []
    with torch.inference_mode():
        # Report cold model execution separately from warmed per-image pipeline.
        warmup_start = time.perf_counter()
        loaded.model(test[0][0].unsqueeze(0))
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        for index in range(len(test)):
            start = time.perf_counter()
            image, label = test[index]
            score = loaded.model(image.unsqueeze(0)).softmax(dim=1)[0, 1].item()
            latencies.append((time.perf_counter() - start) * 1000)
            labels.append(label)
            scores.append(score)
    metrics = binary_metrics(labels, scores, loaded.threshold)
    sorted_times = sorted(latencies)
    p95 = sorted_times[min(len(sorted_times) - 1, int(0.95 * len(sorted_times)))]
    report = {
        "split": "test", "threshold": loaded.threshold, "metrics": metrics,
        "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "warmup_ms": warmup_ms, "median_pipeline_ms": statistics.median(latencies),
        "p95_pipeline_ms": p95,
        "sensitivity_target_met": metrics["sensitivity"] >= 0.88,
        "latency_target_met": p95 <= 2000,
        "note": "Latency includes file decoding, preprocessing and CPU forward pass, not browser transport or UI rendering. Patient separation must be verified by the dataset owner. Model scores are not calibrated clinical probabilities.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
