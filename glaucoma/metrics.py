"""Small, auditable metrics without an additional ML dependency."""

import math


def binary_metrics(labels, probabilities, threshold=0.5):
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("Provide equally sized, non-empty labels and scores.")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one.")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Scores must be finite probabilities.")
    tp = sum(y == 1 and p >= threshold for y, p in zip(labels, probabilities))
    tn = sum(y == 0 and p < threshold for y, p in zip(labels, probabilities))
    fp = sum(y == 0 and p >= threshold for y, p in zip(labels, probabilities))
    fn = sum(y == 1 and p < threshold for y, p in zip(labels, probabilities))
    return {
        "samples": len(labels), "true_positive": tp, "true_negative": tn,
        "false_positive": fp, "false_negative": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "accuracy": (tp + tn) / len(labels),
    }


def select_threshold(labels, probabilities, target=0.88):
    """Choose on validation only; maximize specificity subject to target recall."""
    if set(labels) != {0, 1} or not 0 < target <= 1:
        raise ValueError("Validation needs both classes and a target in (0, 1].")
    binary_metrics(labels, probabilities)
    candidates = sorted({0.001, *[p for p in probabilities if 0 < p < 1]})
    feasible = [t for t in candidates
                if binary_metrics(labels, probabilities, t)["sensitivity"] >= target]
    if not feasible:
        raise ValueError("No usable threshold meets the validation recall target.")
    return max(feasible, key=lambda t: (binary_metrics(labels, probabilities, t)["specificity"], t))
