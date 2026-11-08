# Process journal

Date: 2026-11-07. Repository identity: **Cxsmxnaut**. All values below describe completed research runs, not hospital performance.

## Initial implementation and baseline

The hackathon PRD requested a small retinal classifier, 224×224 ImageNet preprocessing, ≥88% sensitivity, and ≤2-second interaction. The implementation uses PyTorch EfficientNet-B0 and an in-notebook upload dashboard. It does not implement optic-cup/disc segmentation, structural measurement, or a calibrated disease risk.

An initial local Apple MPS run used 8,000 EyePACS training images, 770 validation and 770 test images, 12 epochs. It caught 354/385 positives with 132/385 false alarms. Its checkpoint SHA-256 is `fcbcc55c7349cf602757972dbf69fadf2cd76dd58b169c3ab0731aea8fc11ce6`. This is the separately published v0.1.0 demo artifact. [Record](../../experiments/baseline.json).

The user clarified that training must happen in Colab and browser work must not interrupt the desktop. A complete 12-epoch **Colab Tesla T4** EyePACS run then caught 353/385 positives with 134/385 false alarms. Selected epoch 11, threshold `0.42880725860595703`. SHA-256 `c68e06ad913da75f5b0630be5ea81d8335fc3b62cf7362fdc6d773e39fe64750`. [Record](../../experiments/colab_t4.json).

The Colab archive was created. The normal download helper did not yield a local file, and browser policy explicitly blocked a data-URL download. No workaround bypass was used. The published demo therefore remains v0.1.0; cloud training and published weights are distinct facts.

## Sequential run S1

The user requested multiple highly rated Kaggle datasets trained one after another. Kaggle metadata and original papers/terms were checked. EyePACS v4, ACRIMA v1, and PAPILA v1 were selected. SMDG overlap and RIM-ONE source conditions were documented as exclusions.

The hidden Colab session remained connected to a T4. ACRIMA and PAPILA were downloaded there. The audit decoded and checked **10,665 eligible images**. PAPILA's 68 suspect eyes were excluded, with paired-eye patient grouping. New locked test sets contain 105 ACRIMA images and 62 PAPILA images, including 14 PAPILA glaucoma eyes.

Stage 1 reused the previously completed T4 EyePACS checkpoint. Stage 2 trained on 495 ACRIMA + 494 replay EyePACS images for 12 epochs. Stage 3 trained on 296 PAPILA + 148 EyePACS + 148 ACRIMA images for 12 epochs. All checkpoints were saved before new test evaluation.

| Checkpoint stage | Selected epoch | Threshold | SHA-256 |
| --- | ---: | ---: | --- |
| EyePACS | 11 | 0.4288072586 | `c68e06ad913da75f5b0630be5ea81d8335fc3b62cf7362fdc6d773e39fe64750` |
| + ACRIMA | 12 | 0.4658272266 | `9f2df71fba73ec13fd2a6eb0eaab34b8cf186ebd9f71eb741e3350cbe233aae1` |
| + PAPILA | 12 | 0.1890892982 | `f2daa2899f32ec27b3814a8e550d0a49a9612bf067addd4d79bbd75a47260dc0` |

| Stage | EyePACS sensitivity / specificity | ACRIMA sensitivity / specificity | PAPILA sensitivity / specificity |
| --- | --- | --- | --- |
| EyePACS | 91.7% / 65.2% | 6.8% / 100.0% | 35.7% / 87.5% |
| + ACRIMA | 82.3% / 75.6% | 91.5% / 78.3% | 71.4% / 68.8% |
| + PAPILA | 94.8% / 46.5% | 96.6% / 69.6% | 85.7% / 64.6% |

The model learned the new source labels, but false alarms increased at the final selected threshold. Different endpoints and small samples limit interpretation. No stage was promoted on these test scores. Full result: [sequential Colab record](../../experiments/sequential_colab_t4.json). Source commit: `933b60654f90335f50cc8363d2bee2c23f4a0692`.

## Verification and documentation work

The sequential source/notebooks were published using Cxsmxnaut. GitHub Actions run [42426884276](https://github.com/Cxsmxnaut/GlaucomAI/actions/runs/42426884276) passed both widget-version jobs, with 43 tests at that revision. Tests use synthetic data and random fixture checkpoints; they do not train on patient data locally. Generated notebooks are checked against source. No Graphify graph is configured.

A further Colab audit completed and verified checkpoint hashes/backbone identity, reproduced all locked confusion counts, calculated AUROC/Brier/sample predictive values, estimated patient-group bootstrap uncertainty for PAPILA, and tested nonretinal solid-color controls. Final-stage PAPILA AUROC was 0.8690, with a 31-patient-group sensitivity interval of 66.7%–100% (1,998 valid of 2,000 replicates). Every raw model flagged every solid-color control. [Audit record](../../experiments/clinical_audit_colab_t4.json).

That observed failure led to an exact-uniform-image guard in `infer` and the upload dashboard. Uniform images now raise an input error before model execution; upload rejection clears the decoded image and disables analysis. This is a narrow engineering control, not a trained retina/quality classifier. Synthetic checks expanded to 52 passing tests. A Colab guard audit verified all **15/15 controls abstained**, with **zero model executions** and **zero guard rejections among all 937 benchmark test images**. [Guard record](../../experiments/input_guard_colab_t4.json). This confirms only the narrow behavior on these inputs, not clinical gradability.

The user then requested continuing toward hospital use and documenting the whole process for reuse. A 13-document playbook was created. Desired institutions are examples; no partner, clinical approval, deployment, or regulatory authorization is claimed. Intended use and study materials remain drafts for the relevant experts.

The playbook and verified S1/audit/guard evidence were published at `2d8ef3c142614e1f3c1bb13b5749efe1f7d46463`. A subsequent P2.0 protocol was written before implementation, specifying source-specific task heads and limited suffix adaptation. Existing test sets are explicitly development benchmarks for P2. A possible G1020 external cohort was researched from its original paper; its data URL returned 404 and original usage/patient-mapping verification remains pending. No G1020 images were downloaded or used.

P2 protocol commit: `de488122d2a0720bcce6a921b8b4568a8259061c`. Implementation commit: `23f461eeaa161e8142b731463f220aa947c65c0a`. Its 56 engineering tests and both widget-version CI jobs passed. The hidden Colab T4 run started with the original cloud EyePACS checkpoint and verified S1 manifest; it reported frozen-prefix hash `30c85a287da6168820236744569d2574d3c568c2771cca9ace13ebf7881708c2`. No P2 outcome is claimed until training completes. A separate fresh-runtime notebook was then added and brought local checks to 57; it repeats the initialization recipe and records a new artifact hash rather than claiming byte identity with the original run.
