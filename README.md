# GlaucomAI

**Train a glaucoma image classifier on Kaggle data, then demonstrate it in Google Colab.**

**Try the trained model:** [![Open demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cxsmxnaut/GlaucomAI/blob/main/notebooks/GlaucomAI_Demo.ipynb)

**Train your own model:** [![Open training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cxsmxnaut/GlaucomAI/blob/main/notebooks/GlaucomAI_Colab.ipynb)

GlaucomAI is an AI + Health hackathon research prototype. It uses an ImageNet-pretrained EfficientNet-B0 with a two-class head to learn **non-referable vs. referable glaucoma** from retinal photographs. It includes training, held-out evaluation, a side-by-side image/results dashboard, and export of trained artifacts.

**Not a diagnosis.** This prototype is not clinically validated. Model scores are not calibrated disease probabilities. Non-referable does not mean healthy. A qualified eye-care professional must interpret retinal findings. The classifier does not measure optic disc cupping or cup-to-disc ratio.

## Run in Google Colab

For the **fast demo**, open the first notebook and choose **Runtime → Run all**. A CPU runtime is enough. It downloads the published trained checkpoint, verifies its checksum, and opens the upload dashboard. You do not need a Kaggle account or training data to try this checkpoint.

For **training**:

1. Select **Open training in Colab** above. Save a copy to your Drive if you want to keep your own notebook.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Choose **Runtime → Run all**. The notebook installs tools, downloads the Kaggle dataset, validates images, trains for 12 epochs, evaluates, and opens the upload dashboard.
4. If Kaggle requests authentication, create an API token in [Kaggle settings](https://www.kaggle.com/settings/api). Add it through Colab's **Secrets** sidebar as `KAGGLE_API_TOKEN`, allow notebook access, and rerun the download cell. Do not paste it into code or GitHub.
5. At the dashboard, upload a PNG/JPEG under 10 MB and select **Analyze image**.
6. Run the final download cell to save `GlaucomAI_trained_artifacts.zip` before the runtime disconnects. It contains the model checkpoint, training history, evaluation, data provenance, and installed package versions.

The notebook contains the project source, so it also works when uploaded directly through **File → Upload notebook**. It needs no web server, tunnel, or separate hosting account. The dashboard runs inside the notebook. [Colab GPU availability and runtime lifetimes vary](https://research.google.com/colaboratory/faq.html); Colab is interactive compute, not permanent app hosting.

To show an already-trained model in a new runtime, run notebook steps **1–2**, upload your trusted `best_model.pth` to `/content/GlaucomAI/models/` using the Files sidebar, and run **step 8**. You can skip downloading the dataset and retraining.

## Dataset and labels

Default: [EyePACS-AIROGS-light-V2, version 4](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4), published by Riley Kiefer. The publisher describes 4,000 training, 385 validation, and 385 test images **per class**. The code counts and records the files actually downloaded.

| Kaggle folder | Model class | Meaning |
| --- | --- | --- |
| `NRG` | `non_referable` / 0 | Non-referable glaucoma label |
| `RG` | `referable` / 1 | Referable glaucoma label |

The code preserves the publisher's train/validation/test folders. It never guesses labels from filename numbers and never relies on alphabetical folder order. It validates every image and rejects exact duplicates. **File disjointness does not establish patient disjointness.** Patient identities and the publisher's split process have not been independently audited. This balanced subset also does not represent population disease prevalence.

Read the dataset's terms and linked original-source conditions before use or redistribution. Data is downloaded at runtime, excluded from Git, and not included in the artifact archive. Please cite the dataset and the related works requested on its Kaggle page. Dataset citation: Riley Kiefer, *EyePACS-AIROGS-light-V2*, Kaggle, 2024, DOI: `10.34740/KAGGLE/DSV/7802508`.

## Training and evaluation

- RGB images → exact 224 × 224 resize → tensor → ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- ImageNet-pretrained EfficientNet-B0 with a trainable two-class head; frozen backbone weights and batch-normalization statistics. Features are extracted once into memory to avoid repeating the frozen backbone on every epoch. AdamW, learning rate `0.001`, 12 epochs, seed 42.
- Lowest validation loss selects the checkpoint. Validation images select the referral-score threshold, targeting ≥88% sensitivity while maximizing specificity. The test split is not used to select either.
- Held-out evaluation reports the confusion matrix, sensitivity, specificity, accuracy, model hash, and warmed CPU pipeline median/p95 latency. Cold warmup is reported separately. Latency includes decoding/preprocessing/inference, and excludes browser transfer/rendering.
- The dashboard uses real softmax outputs from a loaded checkpoint. A missing or incompatible checkpoint stops inference; there is no random-score fallback.

**≥88% sensitivity and ≤2-second latency are goals, not established performance claims.** The notebook reports whether each target is met in that run. A high sensitivity alone can hide excessive false positives; inspect specificity and the confusion matrix too. See [MODEL_CARD.md](MODEL_CARD.md) for limitations.

## Recorded baseline

A complete 12-epoch run on November 7, 2026 trained on **8,000 Kaggle images** using a local Mac GPU (Apple MPS). Validation selected epoch 11 and threshold `0.3982966840`. The supplied Colab notebook runs the same training workflow. This recorded run was not performed in Colab.

| Held-out test result | Measured value |
| --- | ---: |
| Test images | 770 (385 per class) |
| Sensitivity | 91.95% (354/385 referable images caught) |
| Specificity | 65.71% (253/385 non-referable images left unflagged) |
| Accuracy | 78.83% |
| False negatives | 31 |
| False positives | 132 |
| CPU pipeline median / p95 | 68.4 / 72.9 ms on the local Mac |

This checkpoint exceeded the sensitivity target **on this test split**, with substantial false positives. The latency result covers decoding, preprocessing, and inference; the full browser click-to-result requirement is not established. Patient separation and cross-dataset performance remain unverified. See [the complete baseline record](experiments/baseline.json) for the exact checkpoint hash and data provenance.

The trained checkpoint and artifact archive are separate downloads in [release v0.1.0](https://github.com/Cxsmxnaut/GlaucomAI/releases/tag/v0.1.0). The demo notebook verifies the checkpoint's SHA-256 before loading it.

## Run locally

Python 3.11 or newer is recommended. CUDA is used automatically when available; CPU training is slower. Apple Silicon can use `--device mps` explicitly.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py
python train.py --data data/prepared --epochs 12 --batch-size 32
python evaluate.py --data data/prepared
```

To use data already downloaded from Kaggle:

```bash
python prepare_data.py --source /path/to/downloaded/dataset --output data/prepared
```

The source must contain `train/`, `validation/`, and `test/`, each containing `NRG/` and `RG/`. Reusing an existing prepared-data destination is blocked to avoid mixing runs. Skip preparation if reusing the same verified data, or choose a fresh output directory.

## Project files

| File | Purpose |
| --- | --- |
| `notebooks/GlaucomAI_Colab.ipynb` | Self-contained training, evaluation, upload demo, export |
| `notebooks/GlaucomAI_Demo.ipynb` | Download the verified trained checkpoint and try uploads |
| `prepare_data.py` | Pinned Kaggle download, explicit label mapping, split validation |
| `train.py` | Frozen-backbone transfer learning and checkpoint selection |
| `evaluate.py` | Held-out metrics and CPU latency |
| `glaucoma/pipeline.py` | Safe decoding, shared preprocessing, checkpoint validation, inference |
| `glaucoma/notebook_ui.py` | In-notebook image/results dashboard |
| `scripts/build_notebook.py` | Rebuild notebook from Python source |
| `experiments/baseline.json` | Recorded baseline metrics and release checkpoint checksum |
| `tests/` | Image, class-order, split, checkpoint, inference, metrics, and notebook checks |

Uploaded images are not written to disk by the dashboard. They remain in session memory until cleared or the runtime ends. **Saved notebook outputs can contain uploaded images.** Use **Clear image**, clear cell outputs before sharing, and keep Colab's **Omit code cell output when saving** setting enabled. The generated notebook has no outputs or saved widget state. No external database is required.

## Development checks

```bash
python -m pip install -r requirements-dev.txt
python scripts/build_notebook.py
ruff check .
pytest -q
python scripts/build_notebook.py --check
```

Rebuild the notebook whenever its bundled Python files change. GitHub Actions checks that generated notebook code matches the tested source. Tests use synthetic images and untrained test checkpoints; they do not download patient data or claim model accuracy. There is no Graphify graph or managed index configured in this repository.

## Technical references

[KaggleHub download and authentication](https://github.com/Kaggle/kagglehub) · [Torchvision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html) · [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
