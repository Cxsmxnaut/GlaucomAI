# P2: separate task outputs and limited backbone adaptation

Protocol revision: P2.0, 2026-11-07. Status: **preregistered at commit de48812 before implementation; original Colab run completed**. Research only. This is a model-development hypothesis, not a clinical-study protocol or approval. The fresh-reproduction initialization note was added afterward for reproducibility; it does not change the initialization used by the original run. Results are recorded in [separate_tasks_colab_t4.json](../../experiments/separate_tasks_colab_t4.json).

## Problem and hypothesis

S1 mixed referral and disease labels into one head and selected a mean-source recall threshold. The final model improved detection of new-source positives but produced 206 false alarms among 385 EyePACS negatives. PAPILA sensitivity remained uncertain. These results do not identify a clinically acceptable operating point.

Hypothesis: share an image representation while keeping each source's label meaning in a separate classifier head. Allow limited representation learning in the last two EfficientNet blocks instead of only changing a frozen classifier. Preserve earlier training information through replay. This may reduce target confusion and improve validation discrimination; it may also overfit small datasets or forget earlier tasks.

## Fixed data and interpretation

Use exactly the S1 manifests/splits and explicit labels. Never silently change patient groups or exclude mistakes after seeing scores. EyePACS head predicts referral labels; ACRIMA head predicts expert image labels; PAPILA head predicts its clinical labels. A source-selected head is a research interface, not an automatic clinical choice or proof of cross-site generalization.

All three existing test partitions are now inspected **development benchmarks**. They may describe P2 after freezing but cannot support a new unseen-test or hospital validation claim. A new independently governed clinical cohort is still required.

## Training specification

- Start from the verified Colab EyePACS checkpoint `c68e06ad913da75f5b0630be5ea81d8335fc3b62cf7362fdc6d773e39fe64750`. Preserve its referral head.
- A fresh-runtime reproduction first repeats the original EyePACS initialization recipe (12 epochs, seed 42, batch 32, frozen ImageNet backbone) in Colab and records its actual checkpoint hash. This is recorded as a new reproduction, not claimed to be byte-identical to the original artifact.
- Add separate ACRIMA and PAPILA two-class heads, initialized with a fixed seed `20261108`. Use a distinct architecture/checkpoint schema; do not load through the original referral dashboard.
- Freeze EfficientNet feature blocks `0..6`. Cache only their outputs. Blocks `7..8` and active classifier heads train; do not cache outputs from these trainable blocks. Verify frozen-prefix identity in the evidence.
- Keep all batch-normalization running statistics frozen. Use deterministic 224×224 preprocessing for direct comparability; no unreviewed augmentation in this first comparison.
- Stage ACRIMA: all 495 ACRIMA training images plus the S1 deterministic bounded EyePACS replay sample. Stage PAPILA: all 296 PAPILA training images plus bounded replay from both earlier sources. Validation/test images never enter replay.
- Cross-entropy routed only to the matching task head. AdamW learning rates: `0.00001` for trainable backbone blocks, `0.001` for heads; batch 32; 12 epochs per adaptation stage. Record all randomness and actual hardware.
- Equal-source mean validation loss over each source's **own head** selects the checkpoint after each stage. At that locked checkpoint, choose each head's threshold separately on its own validation data, using the existing ≥88% validation recall rule. This is still a development rule, not clinical acceptance.
- Record per-task losses, confusion counts, AUROC, threshold, selected epoch, train/replay counts, and all checkpoint parent/child hashes. Do not choose or discard a run using development-test improvements.

## Checks and stopping rules

Before executing on data: verify task routing, initialization continuity, prefix cache/full-forward equivalence, no gradients in the frozen prefix, gradients in the suffix and active heads, finite outputs, correct class ordering, and loss/threshold selection without test access.

Stop and record failure on nonfinite loss/weights, checkpoint/schema mismatch, duplicate/split inconsistency, changed frozen-prefix parameters, incompatible cache dimensions, accidental validation/test training, missing class labels, or exhausted Colab resources. Do not silently fall back to local training. Keep S1 artifacts and failed-run logs intact.

After both stages are locked, evaluate each native head on its source's development benchmark. Report all outcomes and tradeoffs, including regressions. Do not promote a clinical model or change the public referral demo based on these results.

## External-cohort preparation

G1020 is a possible later external source: its original paper describes 1,020 images from 432 patients, with repeated images per patient. The paper's linked DFKI data URL returned 404 during this check. Original access/usage terms and a trustworthy patient mapping must be verified before onboarding; a Kaggle mirror or aggregate's rating alone is insufficient. No G1020 images or labels have been inspected or used in P2.

The clinical partner, intended-use sign-off, new confirmatory cohort, gradability dataset, clinical acceptance bounds, and global regulatory/institutional review remain missing. P2 cannot close those gaps by itself.

## Recorded outcome and decision

Both adaptation stages ran 12 epochs on a Colab Tesla T4. Validation selected ACRIMA-stage epoch 12 and PAPILA-stage epoch 3. The original frozen prefix remained identical throughout. Every recorded source hash matches the committed implementation.

Final native-task development benchmarks:

| Task | Sensitivity | Specificity | AUROC | Missed positives | False alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| EyePACS referral | 91.2% | 57.4% | 0.8680 | 34/385 | 164/385 |
| ACRIMA image label | 83.1% | 97.8% | 0.9882 | 10/59 | 1/46 |
| PAPILA clinical label | 78.6% | 52.1% | 0.7798 | 3/14 | 23/48 |

The architecture separates target meanings and improves ACRIMA ranking relative to S1, but this run does not improve the PAPILA benchmark or establish acceptable clinical performance. No checkpoint is promoted and no threshold is retuned on these benchmark results. The next priority is source/label/quality and independent clinical evidence, not a sequence of test-driven hyperparameter searches.
