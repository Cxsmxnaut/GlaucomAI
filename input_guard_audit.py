"""Verify the exact-uniform input guard against real checkpoints and test images."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image

from glaucoma.pipeline import InputError, decode_image, infer, load_model, validate_image_signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("sequential_run"))
    args = parser.parse_args()
    source = json.loads((args.run / "results.json").read_text())
    rows = json.loads((args.run / "manifest.json").read_text())
    report = {"scope": "Exactly uniform RGB images only; not a validated retinal/gradable classifier.",
              "pipeline_source_sha256": sha256(Path("glaucoma/pipeline.py").read_bytes()).hexdigest(),
              "controls": [], "test_images_checked": {}, "test_images_rejected": {},
              "clinical_release_allowed": False}
    checked, rejected = Counter(), Counter()
    for row in rows:
        if row["split"] != "test":
            continue
        path = Path(row["path"])
        checked[row["dataset"]] += 1
        with decode_image(path.read_bytes(), path.name) as image:
            try:
                validate_image_signal(image)
            except InputError:
                rejected[row["dataset"]] += 1
    report["test_images_checked"] = dict(checked)
    report["test_images_rejected"] = {key: rejected[key] for key in checked}
    for stage in source["stages"]:
        path = args.run / stage["checkpoint"]
        if sha256(path.read_bytes()).hexdigest() != stage["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from locked research run.")
        loaded = load_model(path)
        calls = []
        hook = loaded.model.register_forward_pre_hook(lambda *args: calls.append(True))
        for color in ((0, 0, 0), (255, 255, 255), (128, 128, 128), (220, 40, 20), (0, 180, 70)):
            try:
                infer(Image.new("RGB", (512, 512), color), loaded)
            except InputError:
                report["controls"].append({"stage": stage["stage"], "color": list(color), "abstained": True})
            else:
                raise AssertionError("A uniform control incorrectly produced a score.")
        hook.remove()
        if calls:
            raise AssertionError("The model executed before input rejection.")
    report["model_forward_calls_for_uniform_controls"] = 0
    (args.run / "input_guard_audit.json").write_text(json.dumps(report, indent=2))
    print("INPUT GUARD AUDIT COMPLETE", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
