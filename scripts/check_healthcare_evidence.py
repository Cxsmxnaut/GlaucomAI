"""Check research evidence integrity; this does not certify clinical readiness."""

from hashlib import sha256
import json
from pathlib import Path

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
    return {"evidence_integrity": "verified", "clinical_release_allowed": False,
            "note": "Hashes and internally consistent research records do not establish clinical validity."}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
