"""Generate the reproducible Colab workflow for all three training stages."""

import copy
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/GlaucomAI_Sequential.ipynb"


def build():
    notebook = copy.deepcopy(runpy.run_path(str(ROOT / "scripts/build_notebook.py"))["build"]())
    notebook["cells"] = notebook["cells"][:5]
    notebook["cells"][0]["source"] = [
        "# GlaucomAI: train across three datasets\n",
        "**EyePACS → ACRIMA → PAPILA**, one model, one stage at a time, inside Google Colab.\n\n",
        "Choose **Runtime → Change runtime type → T4 GPU**, then **Run all**. "
        "The script refuses CPU training. Downloads and training may take several minutes.\n\n",
        "All three Kaggle listings had **10/10 usability** on 2026-11-07. This measures dataset usability, "
        "not clinical quality. The source datasets have different labels: EyePACS uses referral labels; "
        "ACRIMA and PAPILA use glaucoma/healthy labels. This is an experimental binary task, "
        "not a calibrated diagnosis or hospital-ready system.\n\n",
        "Source terms: [EyePACS](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2), "
        "[ACRIMA](https://www.kaggle.com/datasets/orvile/acrima-glaucoma-assessment-using-fundus-images), "
        "[PAPILA](https://www.kaggle.com/datasets/orvile/papila-retinal-fundus-images). "
        "PAPILA is GPL-3.0+; do not assume the dataset or resulting artifacts have unrestricted commercial rights.\n",
    ]
    notebook["cells"][2]["source"] = [line.replace('"ipywidgets>=7.7,<9"',
        '"openpyxl>=3.1,<4", "ipywidgets>=7.7,<9"') for line in notebook["cells"][2]["source"]]

    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        notebook["cells"].append(cell)

    add("markdown", """## Train the stages and evaluate
1. EyePACS: preserve the publisher's 8,000/770/770 train/validation/test images and train for 12 epochs.
2. ACRIMA: continue the same head on new training images plus a bounded replay sample from EyePACS training images.
3. PAPILA: continue from ACRIMA with new training images and replay from both earlier sources.

The EfficientNet-B0 backbone stays frozen. Each new stage trains the classifier head for 12 epochs.
Earlier data is replayed to reduce forgetting. Each source contributes equally to validation loss and threshold selection.
The 88% macro validation sensitivity target is not a guarantee on test images.

Splits are fixed before training (seed 20261107). PAPILA splits by patient, keeping both eyes together;
68 suspect eyes are excluded. ACRIMA has no usable patient identifiers, so its split is image-based.
EyePACS patient separation is unverified. Byte and decoded-pixel hashes reject exact duplicates;
they do not rule out every near-duplicate or repeated patient.

Test images do not choose epochs, thresholds, or replay samples. All stages finish before test evaluation.
Every stage is then tested on every dataset so forgetting is visible. EyePACS results are a previously inspected benchmark;
the new ACRIMA/PAPILA test splits remain withheld during adaptation. Small test sets have substantial uncertainty.
""")
    add("code", """from glaucoma.runtime import run_logged
run_logged([sys.executable, "-u", "sequential_train.py", "--epochs", "12",
            "--work", "sequential_run"])
""")
    add("markdown", "## Compare the locked checkpoints\nScores below use each checkpoint's validation-selected threshold. No test score selects a winner.\n")
    add("code", """from IPython.display import HTML, display
from html import escape
report = json.loads(Path("sequential_run/results.json").read_text())
rows = []
for stage in report["stages"]:
    for dataset, m in stage["test_metrics"].items():
        rows.append(f"<tr><td>{escape(stage['stage'])}</td><td>{escape(dataset)}</td>"
                    f"<td>{m['samples']}</td><td>{m['sensitivity']:.1%}</td>"
                    f"<td>{m['specificity']:.1%}</td><td>{m['accuracy']:.1%}</td>"
                    f"<td>{m['false_negative']}</td><td>{m['false_positive']}</td></tr>")
display(HTML("<h2>Sequential training completed in Colab</h2><table cellpadding='8'><tr>"
             "<th>Training stage</th><th>Test dataset</th><th>Images</th><th>Sensitivity</th>"
             "<th>Specificity</th><th>Accuracy</th><th>Missed positives</th><th>False alarms</th>"
             "</tr>" + "".join(rows) + "</table><p>Research only. Not clinically validated.</p>"))
""")
    add("markdown", """## Save the trained artifacts
The archive contains all three checkpoints, stage histories, test results, source versions, and package versions.
It contains no images, clinical tables, credentials, or image manifest. Colab files are temporary.
Run the next cell to download through Colab's supported download helper.

The original demo uses the separately published EyePACS checkpoint. These experimental mixed-label checkpoints
must not be presented as validated referral scores. More datasets do not establish clinical reliability.
""")
    add("code", """from google.colab import files
files.download("sequential_run.zip")
""")
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"glaucomai-sequential-{index:02d}"
    return notebook


if __name__ == "__main__":
    import sys
    content = json.dumps(build(), indent=1) + "\n"
    if "--check" in sys.argv:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            raise SystemExit("Sequential notebook is stale. Run python scripts/build_sequential_notebook.py")
    else:
        OUTPUT.write_text(content)
        print(OUTPUT)
