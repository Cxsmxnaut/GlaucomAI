"""Package an actual completed run and record its reproducible baseline metadata."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glaucoma.pipeline import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v0.1.0")
    parser.add_argument("--training-device", required=True)
    parser.add_argument("--date", required=True, help="User-local run date, YYYY-MM-DD")
    parser.add_argument("--environment-file", type=Path,
                        help="Use the training runtime's exported pip freeze instead of the local environment.")
    parser.add_argument("--notebook-url", help="Exact Colab notebook URL used for this run.")
    args = parser.parse_args()
    model = ROOT / "models/best_model.pth"
    checkpoint_hash = sha256(model.read_bytes()).hexdigest()
    loaded = load_model(model)
    report = json.loads((ROOT / "reports/evaluation.json").read_text())
    if report["checkpoint_sha256"] != checkpoint_hash:
        raise SystemExit("Evaluation report does not match the checkpoint.")
    baseline = {
        "release_tag": args.tag,
        "checkpoint_url": f"https://github.com/Cxsmxnaut/GlaucomAI/releases/download/{args.tag}/best_model.pth",
        "checkpoint_sha256": checkpoint_hash,
        "run_date": args.date, "training_device": args.training_device,
        "architecture": loaded.metadata["architecture"], "classes": loaded.metadata["classes"],
        "selected_epoch": loaded.metadata["epoch"], "threshold": loaded.threshold,
        "training_samples": loaded.metadata["training_samples"],
        "validation_metrics": loaded.metadata["validation_metrics"],
        "dataset_provenance": loaded.metadata["dataset_provenance"],
        "test_evaluation": report,
        "clinical_validation": "Not performed. Research prototype only.",
    }
    if args.notebook_url:
        baseline["training_notebook_url"] = args.notebook_url
    (ROOT / "experiments").mkdir(exist_ok=True)
    (ROOT / "experiments/baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    versions = (args.environment_file.read_text() if args.environment_file else
                subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                               capture_output=True, text=True).stdout)
    archive = ROOT / "GlaucomAI_trained_artifacts.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ["models/best_model.pth", "models/best_model.history.json", "reports/evaluation.json",
                     "data/prepared/provenance.json", "experiments/baseline.json"]:
            bundle.write(ROOT / name, arcname=name)
        bundle.writestr("environment.txt", versions)
    print("Recorded baseline:", checkpoint_hash)
    print("Archive:", archive.name)


if __name__ == "__main__":
    main()
