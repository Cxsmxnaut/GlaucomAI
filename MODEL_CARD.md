# GlaucomAI model card

## Purpose

Research-only, hackathon demonstration of classifying retinal fundus photographs into **non-referable** and **referable glaucoma** labels. The training notebook produces a new checkpoint. The demo notebook loads the separate trained checkpoint from release v0.1.0 and verifies its checksum.

## Architecture and inputs

ImageNet-pretrained EfficientNet-B0, frozen feature extractor and a two-output linear head. Images are oriented using EXIF before metadata is removed, converted to RGB, resized directly to 224 × 224, and normalized using ImageNet channel statistics. Direct resizing can alter image proportions. Inputs must decode as PNG/JPEG, be smaller than 10 MiB, and contain no more than 25 million pixels.

## Training data

Default source: [EyePACS-AIROGS-light-V2, version 4](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4). Its `NRG` and `RG` folders map explicitly to indices 0 and 1. The publisher's train, validation, and test splits are preserved. Every image is decoded before preparation and exact duplicates are rejected.

Patient disjointness, label correctness, demographic coverage, and suitability for a clinical population have not been independently established by this project. Different pictures of the same patient can evade byte-hash duplicate checks. No cross-dataset validation is included.

## Outputs

Two softmax scores that sum to one. The referable score is compared with a validation-selected routing threshold. These scores are **not calibrated clinical probabilities**. A lower score does not establish a healthy eye, rule out glaucoma, or replace a clinical examination.

No segmentation, optic disc/cup localization, cup-to-disc ratio, clinical staging, image quality grading, or fundus-image detection is performed. Unrelated or low-quality uploads can receive confident scores.

## Evaluation and claims

The best validation loss selects weights. Validation scores select a threshold targeting 88% sensitivity while maximizing specificity. Held-out test data is then evaluated with the locked threshold. The report includes false positives and false negatives alongside sensitivity and specificity.

The ≥88% sensitivity and ≤2-second inference values in the PRD are **targets**. Do not present them as achieved without a real evaluation report and its checkpoint hash. File hashes establish exact file separation, not patient separation. Repeat experiments on the test set can turn it into a development set; reserve a new external set for final evidence after further tuning.

The initial recorded run trained on 8,000 images for 12 epochs using Apple MPS on November 7, 2026. Validation selected epoch 11 at threshold 0.3982966840. On 770 held-out test images, sensitivity was **91.95%**, specificity **65.71%**, and accuracy **78.83%**. There were **31 false negatives** and **132 false positives**. CPU pipeline median/p95 were **68.4/72.9 ms** on the local Mac. The test result meets the recall target on this split, while showing a substantial false-positive rate. This does not establish clinical usefulness or end-to-end browser latency. Exact metrics, counts, source provenance, and checkpoint hash are in [experiments/baseline.json](experiments/baseline.json).

A subsequent 12-epoch run completed inside Google Colab on a Tesla T4 GPU. On the same held-out split, sensitivity was **91.69%**, specificity **65.19%**, and accuracy **78.44%**, with **32 false negatives** and **134 false positives**. Validation selected epoch 11 and threshold 0.4288072586. Its checkpoint hash and measured CPU latency are in [experiments/colab_t4.json](experiments/colab_t4.json). This repeat run did not tune weights or thresholds on test images. Its archive has been created in Colab and is awaiting a supported export; the released v0.1.0 checkpoint remains the earlier local baseline.

The CPU timing report covers image decoding, preprocessing, and a forward pass after warmup. It excludes model-loading time, browser transfer and rendering. It cannot establish the full click-to-result requirement on another machine. Small datasets and domain shifts can cause large performance changes.

## Intended use and limitations

Use with de-identified research images for interactive demonstrations and education. Do not use for diagnosis, treatment, ruling out disease, or autonomous patient triage. Clinical use would require independent validation, calibration, image-quality handling, patient-level split verification, and review by qualified specialists.

## Provenance and persistence

Checkpoints contain architecture, class order, preprocessing, threshold, selected epoch, training sample count, seed, validation metrics, and data provenance when available. Evaluation records the checkpoint SHA-256 hash. The export archive also includes installed package versions.

The dashboard writes no uploaded files, uses no database, and keeps images in runtime memory until cleared. Notebook outputs may include images if saved. Clear uploads and outputs before sharing. Dataset use and redistribution remain subject to the Kaggle page and original-source terms; no dataset or credentials belong in GitHub.
