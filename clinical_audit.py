"""Audit locked research checkpoints in Colab without selecting or tuning them."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import torch

from glaucoma.multidata import HANDLES, RecordDataset
from glaucoma.pipeline import decode_image, load_model, preprocessing
from glaucoma.statistics import cluster_bootstrap, score_statistics
from train import extract_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("sequential_run"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this audit in the connected Colab GPU runtime.")
    torch.set_num_threads(2)
    device = torch.device("cuda")
    source = json.loads((args.run / "results.json").read_text())
    rows = json.loads((args.run / "manifest.json").read_text())
    rows = [row for row in rows if row["split"] == "test"]
    base = load_model(args.run / source["stages"][0]["checkpoint"]).model.to(device)
    features = extract_features(base, RecordDataset(rows), device, 32)
    backbone = {key: value.cpu().clone() for key, value in base.features.state_dict().items()}
    report = {"training_run_sha256": sha256((args.run / "results.json").read_bytes()).hexdigest(),
              "audit_device": torch.cuda.get_device_name(0), "stages": [],
              "clinical_release_allowed": False,
              "clinical_release_reason": "Research data and engineering tests cannot establish clinical safety. Intended-use approval, independent multisite clinical evidence, image-quality validation, workflow validation, and jurisdiction review remain absent.",
              "test_reuse_note": "These test results were already inspected after the locked sequential run. They are development benchmarks for future experiments; they cannot remain an unseen confirmatory test.",
              "negative_control_note": "Solid-color images are raw-model negative controls, bypassing the application's input guard; they are not a validated retinal image-quality test. See the separate input guard audit for application abstention."}
    for stage in source["stages"]:
        path = args.run / stage["checkpoint"]
        if sha256(path.read_bytes()).hexdigest() != stage["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after training results were locked.")
        loaded = load_model(path)
        model = loaded.model.to(device).eval()
        if any(not torch.equal(value.cpu(), backbone[key]) for key, value in model.features.state_dict().items()):
            raise ValueError("Shared feature audit requires identical frozen backbones.")
        with torch.inference_mode():
            scores = model.classifier(features.tensors[0].to(device)).softmax(1)[:, 1].cpu().tolist()
        entry = {"stage": stage["stage"], "checkpoint_sha256": stage["checkpoint_sha256"], "datasets": {}}
        for domain in HANDLES:
            indices = [i for i, row in enumerate(rows) if row["dataset"] == domain]
            labels = [rows[i]["label"] for i in indices]
            probabilities = [scores[i] for i in indices]
            metrics = score_statistics(labels, probabilities, stage["threshold"])
            # Recomputed outcomes must agree exactly with the locked confusion counts.
            for key in ("samples", "true_positive", "true_negative", "false_positive", "false_negative"):
                if metrics[key] != stage["test_metrics"][domain][key]:
                    raise ValueError("Audit disagrees with a locked test outcome.")
            if domain == "papila":
                metrics["patient_bootstrap_95"] = cluster_bootstrap(labels, probabilities,
                    [rows[i]["group"] for i in indices], stage["threshold"])
            else:
                metrics["patient_bootstrap_95"] = None
                metrics["patient_interval_status"] = "Unavailable: no independently verified patient identifiers."
            entry["datasets"][domain] = metrics
        controls = []
        for color in ((0, 0, 0), (255, 255, 255), (128, 128, 128), (220, 40, 20), (0, 180, 70)):
            image = Image.new("RGB", (512, 512), color)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            with decode_image(buffer.getvalue(), "negative_control.png") as decoded:
                tensor = preprocessing()(decoded).unsqueeze(0).to(device)
            with torch.inference_mode():
                score = model(tensor).softmax(1)[0, 1].item()
            controls.append({"control": f"solid_rgb{color}", "positive_score": score,
                             "format_validation_passed": True,
                             "above_threshold": score >= stage["threshold"], "input_abstention": False})
        entry["non_retinal_negative_controls"] = controls
        report["stages"].append(entry)
        print("Audited", stage["stage"], flush=True)
    (args.run / "clinical_audit.json").write_text(json.dumps(report, indent=2))
    print("CLINICAL AUDIT COMPLETE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
