# Experiment protocol and test-set discipline

Status: experiment S1 completed in Colab; protocol for further work is draft.

## Completed S1 protocol

- ImageNet EfficientNet-B0 frozen backbone; deterministic 224×224 preprocessing; trainable two-class head.
- Stage 1: reuse the verified 12-epoch Colab T4 EyePACS checkpoint. Its selected epoch was 11, seed 42.
- Stages 2 and 3: ACRIMA then PAPILA; 12 head-training epochs each; AdamW `0.0003`, batch 32, seed `20261107`.
- All current-domain training images plus a deterministic bounded replay sample from earlier training images. No validation/test examples enter replay.
- Equal-source validation loss selects each stage's checkpoint. Equal-source validation recall ≥88% with maximum mean specificity selects its threshold. A mean target does not guarantee that each source reaches the target.
- Lock all checkpoints before evaluating any new test scores; evaluate all stages on all three sources to expose forgetting.

Source: [training implementation](../../sequential_train.py), [adapters](../../glaucoma/multidata.py), [run record](../../experiments/sequential_colab_t4.json). The run's source corresponds to commit `933b60654f90335f50cc8363d2bee2c23f4a0692`. Weights stayed in Colab; no local dataset training was used for S1 adaptation.

## Rules for the next experiment

These test scores have now been inspected. Use these partitions as development benchmarks if changing the model, loss, replay, preprocessing, or threshold rule. Do not report a subsequent improvement on the same partitions as independent confirmation. Reserve a genuinely new site/time/patient cohort before the next confirmatory study; keep its labels and results inaccessible to model selection.

Write a protocol revision before running: hypothesis, endpoint, data version, permitted labels, split units, architecture, initialization lineage, trainable layers, augmentations, optimizer, epoch limit, early stopping, metric/threshold selection, randomness, compute budget, and stopping rules. Record every attempted run, including failures. Never tune on the final test or quietly replace it.

## Recommended next model direction

The observed source shift and label mismatch support investigating a shared image encoder with **separate task heads**, or returning to a single clinician-defined referral endpoint. ACRIMA image appearance and PAPILA diagnosis should not silently supervise a referral score. A trained image-quality/gradable gate also needs its own labeled data and independent test. These are hypotheses and missing evidence, not completed improvements.

Any fine-tuned backbone invalidates the shared frozen-feature cache. Recompute features or train end-to-end; do not reuse stale cached features. Record pretrained-weight sources and possible overlap with evaluation datasets. Clinical claims need independent clinical evidence even if every engineering benchmark improves.
