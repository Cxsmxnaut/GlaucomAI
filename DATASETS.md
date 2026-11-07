# Sequential glaucoma dataset experiment

The same EfficientNet-B0 head trains in order **EyePACS → ACRIMA → PAPILA** in a Colab GPU runtime. This is sequential adaptation with bounded replay of earlier **training** examples. The ImageNet backbone stays frozen. This is not full-backbone fine-tuning.

## Selected Kaggle datasets

Kaggle API metadata checked **2026-11-07**. The usability score measures documentation and packaging, not clinical reliability. Votes are separate community signals; these are not claimed to be the most popular datasets.

| Order | Dataset and pinned version | Kaggle usability | Votes at check | Eligible images | Labels |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | [EyePACS-AIROGS-light-V2 v4](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/versions/4) | 10/10 | 43 | 9,540 | 4,770 non-referable; 4,770 referable |
| 2 | [ACRIMA v1 mirror](https://www.kaggle.com/datasets/orvile/acrima-glaucoma-assessment-using-fundus-images/versions/1) | 10/10 | 11 | 705 | 309 normal; 396 glaucoma |
| 3 | [PAPILA v1 mirror](https://www.kaggle.com/datasets/orvile/papila-retinal-fundus-images/versions/1) | 10/10 | 13 | 420 of 488 | 333 healthy; 87 glaucoma; 68 suspect excluded |

EyePACS labels describe referral, while ACRIMA/PAPILA describe glaucoma. These are different targets. The mixed-source experiment uses binary source labels, retaining legacy internal class indices for checkpoint compatibility. Its scores must not be interpreted as clinical referral probabilities. The original referral dashboard rejects mixed-label checkpoints.

## Original sources and terms

- **EyePACS:** Riley Kiefer's listing links the AIROGS/EyePACS sources and source-specific conditions; Kaggle lists “Other (specified in description).” Preserve those conditions. Cite the Kaggle dataset and the works requested by its author.
- **ACRIMA:** Díaz-Pinto et al., [*CNNs for automatic glaucoma assessment using fundus images: an extensive validation*](https://link.springer.com/article/10.1186/s12938-019-0649-y), 2019. The Kaggle mirror lists CC BY 4.0. Its image annotations come from glaucoma experts; the labels were not based on a complete clinical assessment. Filenames explicitly encode `_g_` for glaucoma. The adapter checks the published class counts.
- **PAPILA:** Kovalyk et al., [*PAPILA: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184612/), 2022, corrected 2024. The original [Figshare dataset](https://figshare.com/articles/dataset/PAPILA/14798004) lists **GPL 3.0+**, distinct from the article's CC BY license. The paper's Data Records section defines diagnosis `0=healthy`, `1=glaucoma`, `2=suspicious`. Both eye tables must reproduce 333/87/68 labels. Only fundus photographs enter the model; age, gender, pressures, and other clinical measurements are not features.

Data remains in Colab and is not committed or bundled with model artifacts. The archive includes source provenance and research results. Do not assume mixed-source weights have unrestricted commercial rights; review source terms for the intended use.

## Fixed experiment rules

- EyePACS keeps its publisher's 8,000 training / 770 validation / 770 test images. Patient separation is unverified. Its test results were already inspected during the earlier baseline run, so this is a historical benchmark rather than a newly unseen test.
- ACRIMA uses a seeded 70/15/15 image split stratified by label. Patient identifiers are unavailable; different images of a patient could cross splits.
- PAPILA uses a seeded 70/15/15 **patient group** split, stratified by the highest retained eye label. Both eyes stay in the same split; suspect eyes are excluded.
- Seed `20261107` locks new splits and replay samples. Every image is decoded and hashed. Byte-identical and decoded-pixel-identical duplicates are rejected across the whole experiment, including within a dataset. These checks do not establish the absence of all near-duplicates or unknown shared patients.
- Each adaptation stage uses all current training images plus up to the same number of earlier training images, divided across earlier datasets and classes. Earlier validation/test images never enter replay.
- Each stage trains its head for 12 epochs at AdamW learning rate `0.0003`. Equal-source mean validation loss selects its epoch. Equal-source mean validation sensitivity of at least 88%, maximizing mean specificity, selects its threshold. This is an experimental validation target, not a test guarantee.
- All checkpoints are locked before any new test results are computed. Then every checkpoint is tested against every source. No test score selects a winner. Small ACRIMA/PAPILA test sets have substantial statistical uncertainty.
- Each stage records its parent checkpoint SHA-256 and its own SHA-256, so continuation is auditable. Fresh runs train EyePACS first; the live November 7 run reused its previously completed Colab T4 EyePACS checkpoint.

## Candidates deliberately excluded

- **SMDG-19:** its high-usability listing aggregates existing sources, including EyePACS and PAPILA. Adding the whole collection would count copies as new independent data and risk leakage.
- **RIM-ONE DL:** the [authors' README](https://github.com/miag-ull/rim-one-dl) prescribes its original partitions and single-dataset training for comparable benchmarks, with research/education restrictions. A Kaggle mirror's more permissive license label is insufficient evidence to ignore the original conditions.
- Synthetic/generated glaucoma collections and poorly documented mirrors were not chosen as independent clinical evidence.

## Reproduce

Open [GlaucomAI Sequential in Colab](https://colab.research.google.com/github/Cxsmxnaut/GlaucomAI/blob/main/notebooks/GlaucomAI_Sequential.ipynb), select a T4 GPU, and run all cells. A fresh run trains all three stages. The final archive contains all checkpoints, histories, results, source versions, and installed package versions, excluding images, clinical tables, and the full image manifest.

The existing published demo continues to use its separately verified EyePACS release. A completed multi-dataset experiment does not establish hospital readiness.
