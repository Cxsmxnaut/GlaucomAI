"""Check research evidence integrity; this does not certify clinical readiness."""

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    run_path = root / "experiments/sequential_colab_t4.json"
    run = json.loads(run_path.read_text())
    audit = json.loads((root / "experiments/clinical_audit_colab_t4.json").read_text())
    guard = json.loads((root / "experiments/input_guard_colab_t4.json").read_text())
    if sha256(run_path.read_bytes()).hexdigest() != audit["training_run_sha256"]:
        raise ValueError("Clinical audit refers to a different sequential results file.")
    if sha256((root / "glaucoma/pipeline.py").read_bytes()).hexdigest() != guard["pipeline_source_sha256"]:
        raise ValueError("Input pipeline changed after the recorded Colab guard audit. Reverify affected behavior and refresh evidence.")
    if [s["stage"] for s in run["stages"]] != ["eyepacs", "acrima", "papila"]:
        raise ValueError("Missing or reordered stages.")
    for index, stage in enumerate(run["stages"]):
        if index and stage["parent_checkpoint_sha256"] != run["stages"][index - 1]["checkpoint_sha256"]:
            raise ValueError("Broken checkpoint continuation lineage.")
        reviewed = audit["stages"][index]
        if stage["checkpoint_sha256"] != reviewed["checkpoint_sha256"]:
            raise ValueError("Audit/checkpoint mismatch.")
        for source, metrics in stage["test_metrics"].items():
            for key, value in metrics.items():
                if reviewed["datasets"][source][key] != value:
                    raise ValueError(f"Inconsistent recorded outcome: {source}/{key}")
    if audit["clinical_release_allowed"] is not False or guard["clinical_release_allowed"] is not False:
        raise ValueError("These research audits cannot authorize clinical release.")
    if len(guard["controls"]) != 15 or not all(c["abstained"] for c in guard["controls"]):
        raise ValueError("A known uniform-input failure is not closed by recorded evidence.")
    if guard["model_forward_calls_for_uniform_controls"] != 0:
        raise ValueError("Uniform input reached model execution.")
    if guard["test_images_checked"] != {"eyepacs": 770, "acrima": 105, "papila": 62}:
        raise ValueError("Guard audit population differs from the locked test set.")
    p2 = json.loads((root / "experiments/separate_tasks_colab_t4.json").read_text())
    if p2["source_run_sha256"] != sha256(run_path.read_bytes()).hexdigest():
        raise ValueError("P2 used a different source run.")
    if p2["initialization_sha256"] != run["stages"][0]["checkpoint_sha256"]:
        raise ValueError("P2 initialization differs from the recorded Colab baseline.")
    parent = p2["initialization_sha256"]
    for stage in p2["stages"]:
        if stage["parent_checkpoint_sha256"] != parent or stage["frozen_prefix_sha256"] != p2["frozen_prefix_sha256"]:
            raise ValueError("P2 checkpoint lineage or frozen-prefix identity differs.")
        parent = stage["checkpoint_sha256"]
        for metrics in stage["development_benchmark_metrics"].values():
            tp, tn, fp, fn = (metrics[key] for key in ("true_positive", "true_negative", "false_positive", "false_negative"))
            if tp + tn + fp + fn != metrics["samples"]:
                raise ValueError("P2 confusion counts do not sum to its sample count.")
            if metrics["sensitivity"] != tp / (tp + fn) or metrics["specificity"] != tn / (tn + fp):
                raise ValueError("P2 reported rates disagree with confusion counts.")
    provenance = json.loads((root / "experiments/separate_tasks_colab_t4.provenance.json").read_text())
    revision = provenance["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("P2 evidence requires an immutable source commit.")
    for name, expected in p2["source_sha256"].items():
        original = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root,
                                  check=True, capture_output=True).stdout
        if sha256(original).hexdigest() != expected:
            raise ValueError(f"P2 source hash does not match its recorded revision: {name}.")
    if p2["clinical_release_allowed"] is not False:
        raise ValueError("P2 research benchmarks cannot authorize clinical release.")
    return {"evidence_integrity": "verified", "clinical_release_allowed": False,
            "note": "Hashes and internally consistent research records do not establish clinical validity."}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
