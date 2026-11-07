"""Build a self-contained Colab notebook from the tested Python source files."""

import argparse
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/GlaucomAI_Colab.ipynb"


def build():
    cells = []

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)})

    def code(source):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": textwrap.dedent(source).strip().splitlines(True)})

    markdown("""
    # GlaucomAI
    ### A clearer view. An earlier signal.
    **Train on Kaggle data → evaluate on held-out images → try the model here.**

    [GitHub source](https://github.com/Cxsmxnaut/GlaucomAI) ·
    [Kaggle dataset](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4)

    This notebook includes all project source code. You do not need to clone a repository or run a web server.
    In Colab, choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
    GPU availability depends on your Colab account. Download and training time depend on the assigned runtime.

    **Research prototype, not a diagnosis.** The classes are **non-referable** and **referable glaucoma**.
    A non-referable result does not mean the eye is healthy. No clinical performance is promised.
    """)
    markdown("""
    ## 1. Install the notebook tools
    Keep Colab's preinstalled GPU-enabled PyTorch and Torchvision. This installs dataset access,
    the upload controls, and plotting tools. No API keys belong in notebook cells or GitHub.
    """)
    code("""
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "kagglehub>=0.3.13,<1", "ipywidgets>=7.7,<9", "matplotlib>=3.9,<4", "Pillow>=11,<13"], check=True)
    import torch
    import torchvision
    print("PyTorch:", torch.__version__, "| Torchvision:", torchvision.__version__)
    print("Training device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slower)")
    """)
    markdown("""
    ## 2. Load the project
    The next cell writes the same Python files tested in GitHub into this runtime's working directory.
    The notebook is generated from those files, so the notebook and repository stay in sync.
    """)
    sources = {p.relative_to(ROOT).as_posix(): p.read_text()
               for p in sorted((ROOT / "glaucoma").glob("*.py"))}
    for name in ("train.py", "evaluate.py", "prepare_data.py"):
        sources[name] = (ROOT / name).read_text()
    code("# @title Initialize the GlaucomAI project\nimport json\nimport os\nfrom pathlib import Path\n"
         "PROJECT = Path('/content/GlaucomAI') if Path('/content').exists() else Path.cwd() / '.glaucomai-notebook'\n"
         "PROJECT.mkdir(parents=True, exist_ok=True)\n"
         + "SOURCES = json.loads(" + repr(json.dumps(sources, sort_keys=True)) + ")\n"
         + "for name, content in SOURCES.items():\n"
         + "    destination = PROJECT / name\n    destination.parent.mkdir(parents=True, exist_ok=True)\n"
         + "    destination.write_text(content)\n"
         + "os.chdir(PROJECT)\n"
         + "if str(PROJECT) not in sys.path:\n    sys.path.insert(0, str(PROJECT))\n"
         + "print('Project ready:', PROJECT)")
    cells[-1]["metadata"] = {"cellView": "form", "jupyter": {"source_hidden": True}}
    markdown("""
    ## 3. Download the Kaggle dataset
    We use **EyePACS-AIROGS-light-V2, version 4**. Its publisher supplies train, validation, and test folders.
    We preserve those splits and map `NRG → non_referable (0)` and `RG → referable (1)` explicitly.
    This is a balanced subset, so its class mix does not represent disease prevalence in the general population.

    Public downloads may work without signing in. If Kaggle requests authentication, add a Colab secret
    named `KAGGLE_API_TOKEN` using the key icon in the sidebar, grant this notebook access, then rerun this cell.
    You can create the token in [Kaggle settings](https://www.kaggle.com/settings/api).
    Read the dataset page and its linked source terms before use or redistribution.
    """)
    code("""
    import kagglehub
    from prepare_data import DATASET
    raw_path = Path(kagglehub.dataset_download(DATASET))
    print("Downloaded dataset:", DATASET)
    print("Dataset root:", raw_path)
    """)
    markdown("""
    ## 4. Validate images and preserve the splits
    This checks file contents, rejects broken or oversized images, and checks for byte-identical duplicates.
    It writes a manifest with source paths and hashes. File-hash checks cannot prove that different photographs
    come from different patients; the publisher's patient separation has not been independently verified.
    The prepared data stays in the temporary runtime and is not uploaded to GitHub.
    """)
    code("""
    from prepare_data import prepare
    data_path = PROJECT / "data/prepared"
    if data_path.exists():
        raise RuntimeError("Prepared data already exists. To reuse it, skip this cell. For a fresh run, delete the runtime first.")
    provenance = prepare(raw_path, data_path)
    print(json.dumps(provenance, indent=2))
    """)
    markdown("""
    ## 5. Train EfficientNet-B0
    We start from ImageNet weights, freeze the feature extractor, and train a two-class head for 12 epochs.
    Every image is resized to **224 × 224**, converted to RGB, and normalized using ImageNet mean and standard deviation.
    Frozen batch-normalization statistics stay frozen, too.
    Because preprocessing is deterministic and the backbone is frozen, its features are extracted once
    into memory. The classification head then trains on those features for all 12 epochs.

    The best checkpoint is chosen using **validation loss**. Its routing threshold is chosen on validation
    images to target at least 88% recall while maximizing specificity. Test images do not choose weights or thresholds.
    Meeting the validation target does **not** mean the model will meet it on the test set or on new patients.
    """)
    code("""
    from glaucoma.runtime import run_logged
    EPOCHS = 12
    BATCH_SIZE = 32
    run_logged([sys.executable, "-u", "train.py", "--data", str(data_path),
                    "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE),
                    "--output", "models/best_model.pth"])
    """)
    markdown("""
    ## 6. Inspect training
    A falling training loss with rising validation loss can be a sign of overfitting:
    the model is learning the training examples more than useful patterns.
    """)
    code("""
    import matplotlib.pyplot as plt
    history = json.loads(Path("models/best_model.history.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Training", color="#147d70")
    ax.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], label="Validation", color="#c99341")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and validation loss")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
    """)
    markdown("""
    ## 7. Evaluate once on the held-out test set
    This uses the saved threshold without tuning it on test images. The report includes sensitivity
    (how many referable images it catches), specificity (how many non-referable images it leaves unflagged),
    and false positives/negatives. We also time the real CPU pipeline after one warmup.
    The 2-second target is measured for decoding, preprocessing, and inference; browser transfer and rendering
    need a separate end-to-end check. CPU performance here describes this Colab machine, not every laptop.
    """)
    code("""
    run_logged([sys.executable, "-u", "evaluate.py", "--data", str(data_path),
                "--model", "models/best_model.pth", "--output", "reports/evaluation.json"])
    report = json.loads(Path("reports/evaluation.json").read_text())
    metrics = report["metrics"]
    print(f"TEST sensitivity: {metrics['sensitivity']:.1%} | specificity: {metrics['specificity']:.1%}")
    print(f"CPU pipeline p95: {report['p95_pipeline_ms']:.0f} ms")
    print("88% sensitivity target:", "MET on this test set" if report["sensitivity_target_met"] else "NOT MET")
    print("2-second pipeline target:", "MET on this CPU" if report["latency_target_met"] else "NOT MET")
    matrix = [[metrics["true_negative"], metrics["false_positive"]],
              [metrics["false_negative"], metrics["true_positive"]]]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Greens", vmin=0)
    for y in range(2):
        for x in range(2):
            ax.text(x, y, str(matrix[y][x]), ha="center", va="center", fontsize=20,
                    color="white" if matrix[y][x] > max(map(max, matrix)) / 2 else "#193e43")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-referable", "Referable"],
           yticklabels=["Non-referable", "Referable"], xlabel="Predicted", ylabel="Dataset label",
           title="Held-out test confusion matrix")
    plt.tight_layout()
    plt.show()
    """)
    markdown("""
    ## 8. Try the trained model
    Upload one PNG/JPEG under 10 MB, then select **Analyze image**. This uses the trained checkpoint;
    it does not generate sample scores. It runs on CPU, even if training used a GPU.
    Format validation does not detect whether an image is a retina or whether it is suitable for diagnosis.

    Uploads are held in this runtime's memory. Use **Clear image** when done. Clear cell outputs before
    sharing the notebook, and keep **Edit → Notebook settings → Omit code cell output when saving** enabled.
    Notebook outputs may contain the uploaded image. Use de-identified research images for the demo.
    """)
    code("""
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except ImportError:
        pass
    from glaucoma.notebook_ui import display_assistant
    dashboard = display_assistant("models/best_model.pth")
    """)
    markdown("""
    ## 9. Export the trained model and evidence
    Colab runtimes are temporary. Download this archive before disconnecting.
    It includes the model, its metadata, training history, evaluation report, data provenance, and exact installed package versions.
    It does **not** include the dataset, your API token, or uploaded demo images.
    The repository contains code and a notebook; trained artifacts are separate downloads.
    """)
    code("""
    import zipfile
    archive = PROJECT / "GlaucomAI_trained_artifacts.zip"
    versions = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ["models/best_model.pth", "models/best_model.history.json", "reports/evaluation.json",
                     "data/prepared/provenance.json"]:
            bundle.write(name, arcname=name)
        bundle.writestr("environment.txt", versions)
    print("Artifacts ready:", archive.name)
    from IPython.display import FileLink, display
    display(FileLink(archive.name))
    """)
    markdown("""
    ### Download to your computer (run this cell when ready)
    Colab's download helper opens a browser download. Outside Colab, use the archive link above.
    """)
    code("""
    # Uncomment these two lines in Colab when you want to download the archive:
    # from google.colab import files
    # files.download(str(archive))
    """)
    markdown("""
    ## Reopen a trained model without retraining
    In a fresh runtime, run steps 1–2. Upload your trusted `best_model.pth` with the Colab Files sidebar
    into `/content/GlaucomAI/models/` (create the `models` folder if needed), then run step 8.
    Only load checkpoints you trust. Do not rerun steps 3–7 just to show the demo.

    ## What this prototype proves—and what it does not
    This is a binary image classifier. It does **not** segment the optic cup/disc, measure cup-to-disc ratio,
    establish the cause of a prediction, or diagnose glaucoma. Softmax scores are not calibrated clinical risks.
    Publisher splits and exact-duplicate checks are useful, but patient separation and cross-dataset performance
    still need independent verification. The 88% recall and 2-second latency values are targets until measured.

    **Sources:** [dataset and citations](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4),
    [KaggleHub authentication](https://github.com/Kaggle/kagglehub#authenticate),
    [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html),
    [Colab limits and sharing](https://research.google.com/colaboratory/faq.html).
    """)
    for index, cell in enumerate(cells):
        cell["id"] = f"glaucomai-{index:02d}"
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": "GlaucomAI_Colab.ipynb", "provenance": [], "private_outputs": True},
                         "accelerator": "GPU", "gpuClass": "standard",
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    notebooks = {NOTEBOOK: build(), ROOT / "notebooks/GlaucomAI_Demo.ipynb": build_demo()}
    for path, notebook in notebooks.items():
        content = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"{path.name} is stale. Run python scripts/build_notebook.py")
            print(f"{path.name} matches current Python source.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print(f"Built {path.relative_to(ROOT)}")


def build_demo():
    baseline = json.loads((ROOT / "experiments/baseline.json").read_text())
    notebook = build()
    notebook["metadata"]["colab"]["name"] = "GlaucomAI_Demo.ipynb"
    notebook["metadata"].pop("accelerator", None)
    notebook["metadata"].pop("gpuClass", None)
    notebook["cells"] = notebook["cells"][:5]
    notebook["cells"][0]["source"] = [
        "# GlaucomAI · Try the trained model\n",
        "### A clearer view. An earlier signal.\n",
        "Run **Runtime → Run all**, then upload a PNG/JPEG and select **Analyze image**. A CPU runtime is enough.\n\n",
        "This notebook downloads our trained EfficientNet-B0 checkpoint from GitHub and verifies its SHA-256 checksum.\n",
        "No Kaggle account, dataset download, or retraining is needed for this demo.\n\n",
        f"**Recorded baseline:** {baseline['test_evaluation']['metrics']['sensitivity']:.1%} sensitivity, "
        f"{baseline['test_evaluation']['metrics']['specificity']:.1%} specificity on 770 held-out images.\n",
        "These results come from one dataset and have not been clinically validated. Non-referable does not mean healthy.\n\n",
        "[GitHub](https://github.com/Cxsmxnaut/GlaucomAI) · "
        "[Train your own model](https://colab.research.google.com/github/Cxsmxnaut/GlaucomAI/blob/main/notebooks/GlaucomAI_Colab.ipynb)\n",
    ]

    def add(cell_type, source):
        cell = {"cell_type": cell_type, "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}
        if cell_type == "code":
            cell.update({"execution_count": None, "outputs": []})
        notebook["cells"].append(cell)

    add("markdown", """
    ## 3. Load the published checkpoint
    This is a trained model, not random demonstration scores. Its exact file checksum is pinned below.
    The published run used Kaggle data on the training device recorded below.
    """)
    add("markdown", f"**Training device:** {baseline['training_device']}. "
        "The training notebook contains the complete workflow.")
    add("code", f"""
    from glaucoma.download import download_checkpoint
    checkpoint = download_checkpoint(
        {baseline['checkpoint_url']!r},
        {baseline['checkpoint_sha256']!r},
        "models/published_{baseline['release_tag']}.pth",
    )
    print("Published checkpoint verified:", checkpoint.name)
    """)
    add("markdown", """
    ## 4. Upload and analyze
    Choose one fundus photograph under 10 MB. Format validation does not determine whether an image is a retina
    or suitable for diagnosis. Use de-identified research images. Select **Clear image** when finished.
    Images stay in runtime memory, but saved notebook outputs can contain them. Clear cell outputs before sharing,
    and keep **Edit → Notebook settings → Omit code cell output when saving** enabled.
    """)
    add("code", """
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except ImportError:
        pass
    from glaucoma.notebook_ui import display_assistant
    dashboard = display_assistant(checkpoint)
    """)
    add("markdown", """
    ## Understand the result
    The two model scores sum to one. The referable score is compared with a threshold selected on validation data.
    It is **not** a calibrated probability that a person has glaucoma. A lower score does not rule out disease.
    The CPU timer covers preprocessing and inference, excluding upload transfer and display time.

    This prototype does not segment the optic disc/cup, measure cup-to-disc ratio, or diagnose glaucoma.
    A qualified eye-care professional must interpret retinal findings. Dataset patient separation and
    performance on other populations have not been independently verified.

    [Recorded metrics and checkpoint hash](https://github.com/Cxsmxnaut/GlaucomAI/blob/main/experiments/baseline.json) ·
    [Model card](https://github.com/Cxsmxnaut/GlaucomAI/blob/main/MODEL_CARD.md)
    """)
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"glaucomai-demo-{index:02d}"
    return notebook


if __name__ == "__main__":
    main()
