"""Research statistics with explicit uncertainty; no clinical release decision."""

from collections import defaultdict
import math
import random

from glaucoma.metrics import binary_metrics


def wilson_interval(successes, total, z=1.959963984540054):
    """95% Wilson score interval; independence must be assessed separately."""
    if isinstance(total, bool) or isinstance(successes, bool) or not isinstance(total, int) or not isinstance(successes, int):
        raise ValueError("Counts must be integers.")
    if not 0 <= successes <= total or total < 0:
        raise ValueError("Invalid binomial counts.")
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def roc_auc(labels, scores):
    """Pair-ranking AUROC with half credit for ties."""
    binary_metrics(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(scores, labels))
    rank_sum, i = 0., 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += sum(label for _, label in pairs[i:j]) * ((i + 1 + j) / 2)
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def score_statistics(labels, scores, threshold):
    metrics = binary_metrics(labels, scores, threshold)
    tp, tn = metrics["true_positive"], metrics["true_negative"]
    fp, fn = metrics["false_positive"], metrics["false_negative"]
    metrics.update({"roc_auc": roc_auc(labels, scores),
                    "brier_score": sum((p - y) ** 2 for y, p in zip(labels, scores)) / len(labels),
                    "sample_ppv": tp / (tp + fp) if tp + fp else None,
                    "sample_npv": tn / (tn + fn) if tn + fn else None,
                    "sensitivity_wilson_95": wilson_interval(tp, tp + fn),
                    "specificity_wilson_95": wilson_interval(tn, tn + fp),
                    "interval_note": "Wilson intervals assume independent observations; eye-level intervals are descriptive when both eyes or repeated patients occur. PPV/NPV describe this sample, not population prevalence."})
    return metrics


def cluster_bootstrap(labels, scores, groups, threshold, repetitions=2000, seed=20261107):
    """Resample whole patients, preserving correlated eyes within each patient."""
    binary_metrics(labels, scores, threshold)
    if len(groups) != len(labels) or repetitions < 100:
        raise ValueError("Provide one group per image and at least 100 replicates.")
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    keys = sorted(members)
    if len(keys) < 2:
        raise ValueError("At least two groups are required.")
    rng = random.Random(seed)
    samples = {"sensitivity": [], "specificity": [], "accuracy": []}
    for _ in range(repetitions):
        indices = [i for key in rng.choices(keys, k=len(keys)) for i in members[key]]
        metrics = binary_metrics([labels[i] for i in indices], [scores[i] for i in indices], threshold)
        for key in samples:
            if metrics[key] is not None:
                samples[key].append(metrics[key])

    def percentile(values, q):
        values = sorted(values)
        position = (len(values) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {"groups": len(keys), "replicates": repetitions, "seed": seed,
            "method": "Percentile 95% bootstrap, sampling whole patient groups with replacement.",
            "intervals": {key: {"lower": percentile(values, .025), "upper": percentile(values, .975),
                                "valid_replicates": len(values)} if values else None
                          for key, values in samples.items()},
            "limitation": "Small cohorts and rare outcomes can produce unstable intervals. Grouping correctness, site independence, and sampling representativeness are not established by bootstrap."}
