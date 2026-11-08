# MSHF quality-data onboarding review

Status: **header/path inspection completed; patient mapping unresolved; no quality model trained or promoted**. Date: 2026-11-07.

## Original sources

[Dataset version 1](https://figshare.com/articles/figure/MSHF_A_Multi-Source_Heterogeneous_Fundus_MSHF_Dataset_for_Image_Quality_Assessment/21507564/1), [data paper](https://www.nature.com/articles/s41597-023-02188-x), and its [public index](https://pubmed.ncbi.nlm.nih.gov/37198230/). The versioned Figshare API lists CC BY 4.0, no embargo, archive file ID `38115186`, size 1,285,292,595 bytes, and publisher/computed MD5 `c4f7bbb7979ae918691c35e543900f03`. A SHA-256 will be recorded after transfer; MD5 is a publisher integrity cross-check, not a modern authenticity guarantee.

The authors describe 1,302 images from 952 subjects, with CFP, portable-camera and UWF inputs. Quality traits are illumination, clarity, contrast and overall quality; original labels are `0=poor`, `1=good`, with three-rater majority voting. All reported subjects are Asian. The original 1,042/260 image split does not by itself establish patient independence. Original images and AI-use copies are the same observations.

## Why inspect it

The exact-uniform guard closes only the observed flat-image failure. A separate quality dataset could support research on poor-image detection. It would not establish whether an arbitrary image is a retina or whether it is suitable for a specific glaucoma referral decision. The latter requires a clinically reviewed reference standard.

## Required checks before a training protocol

1. Verify the pinned archive size/digest; inspect paths and spreadsheet **headers only** before locking any new test access.
2. Identify the explicit overall-quality column and image join keys. Never infer quality from a disease folder, filename order, visual guess, or model score.
3. Count unique originals separately from analysis copies. Validate decoded content and compare exact hashes with all existing glaucoma sources; review flagged near-duplicates separately.
4. Determine whether supplied identifiers support patient grouping. Do not manufacture patient IDs by an undocumented filename rule. Record missing identity evidence honestly.
5. Identify supplied train/test membership without reading held-out scores. Freeze the cohort and split/threshold policy before model selection.
6. Predefine per-device/source reporting, class counts, uncertainty, rejected-image coverage, and any disease-related selection analysis.

## Clinical risks and scope

A quality model can reject sick patients disproportionately if quality and disease/source are correlated. Rejected or uncertain images must never be treated as a negative glaucoma result. A future clinical workflow would need a defined path to reacquisition or professional review, evaluated with clinicians. Do not silently drop rejected images from performance denominators.

Any experimental quality model must use a distinct checkpoint schema and label names. A poor-quality label is not a referable-glaucoma label. No new quality threshold will be applied to the current demo or presented as clinically validated on the basis of this source review.

## Current execution record

`scripts/inspect_quality_source.py` was written to download the original public archive in Colab, verify its pinned size/digest, record SHA-256, count paths, and read only spreadsheet header rows. The script does not extract/execute source code, read held-out quality labels, or train a model. Its actual completion/error must be recorded before inspection is marked complete. Data stays in Colab and is excluded from research model archives and Git.

The initial inspection completed. SHA-256 is `87c9dad6bc850d9362714d121dc4f6d8666b27bfa4684dfc02fb0debe9cabb07`; counts confirm 1,302 originals and 1,302 AI-use copies. The inspected workbook exposes only image name, illumination, clarity, contrast and Overall columns. Patient-group mapping is not established. The `Local1` sheet declares 1,048,575 rows although its original image folder has 199 images; a training loader must not materialize rows from that declared range. [Public inspection record](../../experiments/mshf_source_inspection.json). An extended header/path inspection was prepared to enumerate any other metadata files without reading quality labels.

The [extended inspection](../../experiments/mshf_source_inspection_extended.json) also completed. The pinned archive contains only one non-image file, `MSHF_quality_scores.xlsx`; no separate patient-group mapping file was found. Filename-to-patient semantics remain undocumented in the inspected evidence. The oversized Local1 dimension is recorded explicitly. No held-out quality labels were inspected and no quality model was trained.

## Decision after source inspection

The source can support a carefully scoped image-level research experiment after a protocol is written. It cannot currently establish patient-independent clinical quality validation. First obtain a documented non-identifying subject grouping or appropriate independent cohort/reference evidence; never manufacture a patient mapping from image numbering. A quality benchmark would also need a separate review of glaucoma-specific gradability and disease-related rejection. Current models remain research-only, and no quality gate is promoted from this inspection.
