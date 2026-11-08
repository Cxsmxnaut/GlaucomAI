"""Generate a fresh-runtime Colab reproduction of the P2 research protocol."""

import copy
import json
from pathlib import Path
import runpy
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/GlaucomAI_Separate_Tasks.ipynb"


def build():
    notebook = copy.deepcopy(runpy.run_path(str(ROOT / "scripts/build_notebook.py"))["build"]())
    notebook["cells"] = notebook["cells"][:5]
    notebook["cells"][0]["source"] = [
        "# GlaucomAI P2: separate research tasks\n",
        "Train in a **Colab T4 GPU** runtime. Select T4, then **Run all**. "
        "This notebook prepares three pinned Kaggle datasets, trains an EyePACS initialization, "
        "then adapts the last feature blocks using separate task-specific outputs.\n\n",
        "**Research only.** EyePACS referral, ACRIMA image labels, and PAPILA clinical labels stay separate. "
        "These are previously inspected development benchmarks, not an independent hospital study. "
        "No generic clinical output or referral demo is offered for this experimental architecture.\n\n",
        "[Protocol](https://github.com/Cxsmxnaut/GlaucomAI/blob/main/docs/healthcare/14-next-experiment-protocol.md) · "
        "[Data terms and labels](https://github.com/Cxsmxnaut/GlaucomAI/blob/main/DATASETS.md) · "
        "[Healthcare playbook](https://github.com/Cxsmxnaut/GlaucomAI/tree/main/docs/healthcare)\n\n",
        "The original P2 run reused an existing Colab EyePACS checkpoint. "
        "A fresh reproduction trains that initialization again and records its actual hash.\n",
    ]
    notebook["cells"][2]["source"] = [line.replace('"ipywidgets>=7.7,<9"',
        '"openpyxl>=3.1,<4", "ipywidgets>=7.7,<9"') for line in notebook["cells"][2]["source"]]

    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        notebook["cells"].append(cell)

    add("markdown", """
    ## Prepare audited data and a Colab initialization
    Preserve EyePACS splits; reproduce ACRIMA image splits and PAPILA patient splits from S1.
    Exclude PAPILA suspects. Decode/hash every image and reject exact duplicates before training.
    The initialization uses only EyePACS training/validation data; its test set does not choose weights.
    This notebook refuses a CPU runtime and refuses to overwrite initialization or result folders.
    """)
    add("code", """
    import hashlib
    import kagglehub
    from glaucoma.runtime import run_logged
    from glaucoma.multidata import HANDLES, acrima_rows, papila_rows, eyepacs_rows, audit_rows
    from prepare_data import prepare
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab T4 GPU before running this workflow.")
    initial = Path("p2_initialization")
    if initial.exists() or Path("multitask_run").exists():
        raise RuntimeError("Existing evidence found. Use a fresh runtime rather than overwriting it.")
    prepared = Path("data/prepared")
    if not prepared.exists():
        prepare(Path(kagglehub.dataset_download(HANDLES["eyepacs"])), prepared)
    rows = eyepacs_rows(prepared)
    for source, adapter in (("acrima", acrima_rows), ("papila", papila_rows)):
        rows.extend(adapter(Path(kagglehub.dataset_download(HANDLES[source]))))
    counts = audit_rows(rows)
    initial.mkdir()
    (initial / "manifest.json").write_text(json.dumps(rows, indent=2))
    baseline = initial / "eyepacs.pth"
    run_logged([sys.executable, "-u", "train.py", "--device", "cuda", "--seed", "42",
                "--epochs", "12", "--batch-size", "32", "--output", str(baseline)])
    initialization = {"protocol": "Fresh Colab EyePACS initialization for a P2.0 reproduction",
                      "counts": counts, "stages": [{"checkpoint_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}]}
    (initial / "results.json").write_text(json.dumps(initialization, indent=2))
    print("Initialization recorded:", initialization["stages"][0]["checkpoint_sha256"])
    """)
    add("markdown", """
    ## Train P2 stages without merging label meanings
    Stage ACRIMA then PAPILA, each with earlier training replay. Frozen-prefix features are cached;
    trainable suffix features are recomputed. Batch-normalization statistics stay frozen.
    Each source uses its own head and validation threshold. Native-task mean validation loss selects weights.
    After all stages lock, evaluate native heads on the existing development benchmarks.
    """)
    add("code", """
    run_logged([sys.executable, "-u", "multitask_train.py", "--manifest", str(initial / "manifest.json"),
                "--baseline", str(baseline), "--work", "multitask_run"])
    """)
    add("markdown", "## Inspect all results\nNo benchmark score authorizes clinical use or chooses a release.\n")
    add("code", """
    report = json.loads(Path("multitask_run/results.json").read_text())
    for stage in report["stages"]:
        print("Stage:", stage["trained_tasks"], "Selected epoch:", stage["epoch"])
        for task, m in stage["development_benchmark_metrics"].items():
            print(f"  {task}: sensitivity {m['sensitivity']:.1%}, specificity {m['specificity']:.1%}; "
                  f"missed {m['false_negative']}, false alarms {m['false_positive']}")
    print(report["test_status"])
    print("Clinical release allowed:", report["clinical_release_allowed"])
    """)
    add("markdown", """
    ## Package and save research evidence
    The archive contains model checkpoints, histories, results, source, initialization provenance,
    and environment versions. It excludes photographs, clinical spreadsheets, credentials, and image manifests.
    Review source-specific data terms, including PAPILA's GPL-3.0+ dataset terms.
    Colab storage is temporary. Run the supported download helper to save artifacts.
    """)
    add("code", """
    import zipfile
    environment = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    with zipfile.ZipFile("multitask_run.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in Path("multitask_run").iterdir():
            if path.name != "manifest.json":
                bundle.write(path, arcname=path.name)
        bundle.write(baseline, arcname="initialization/eyepacs.pth")
        bundle.write(initial / "results.json", arcname="initialization/provenance.json")
        for name, content in SOURCES.items():
            bundle.writestr("source/" + name, content)
        bundle.writestr("environment.txt", environment)
        bundle.writestr("RESEARCH_NOTICE.md", "Research only; not clinically validated. Separate source labels. "
                         "Review original dataset terms at https://github.com/Cxsmxnaut/GlaucomAI/blob/main/DATASETS.md")
    from google.colab import files
    files.download("multitask_run.zip")
    """)
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"glaucomai-separate-{index:02d}"
    return notebook


if __name__ == "__main__":
    import sys
    content = json.dumps(build(), indent=1) + "\n"
    if "--check" in sys.argv:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            raise SystemExit("Separate-task notebook is stale. Run python scripts/build_multitask_notebook.py")
    else:
        OUTPUT.write_text(content)
        print(OUTPUT)
