# Next evidence acquisition plan

Status: research planning after S1 and P2; no new clinical cohort or quality-model training is claimed. Checked 2026-11-07.

## What the completed experiments establish

We can reproduce model training, maintain explicit labels/splits, detect some leakage, preserve lineage, measure native-source errors, and close a demonstrated exact-uniform-input failure. The experiments do not establish hospital reliability. P2's PAPILA result worsened, and existing benchmarks have already influenced development decisions.

The next useful evidence must address input quality, clinically consistent targets, independent patients/sites/devices, and realistic workflow. More epochs on the same inspected partitions would not supply that evidence.

## Candidate sources under review

| Candidate | Original-source evidence | Possible use | Unresolved items / current decision |
| --- | --- | --- | --- |
| [EyeQ](https://github.com/HzFu/EyeQ) | Authors describe 28,792 EyePACS images with Good/Usable/Reject quality labels; original images come from the EyePACS competition | Research quality/abstention model | Repository code license is CC BY-NC-SA 4.0 and requires permission for commercial use; do not assume this grants image/label or hospital-use rights. Verify original image/label terms, patient IDs, and possible overlap with existing EyePACS cohorts. No images downloaded. |
| [G1020](https://arxiv.org/abs/2006.09158) | Original paper describes 1,020 images from 432 patients, including repeated images | Potential new-site clinical-label research benchmark | Original linked data URL returned 404. Verify original access/usage terms and trustworthy patient mapping before onboarding. No images downloaded. |
| [REFUGE](https://refuge.grand-challenge.org/Rules/) | Challenge authors publish terms governing registration/data participation | Potential external clinical-label research benchmark | Review applicable original data conditions and whether access requires registration/agreements. Do not treat a mirror's license as replacement permission. No new images downloaded. |
| [MSHF original data](https://figshare.com/articles/figure/MSHF_A_Multi-Source_Heterogeneous_Fundus_MSHF_Dataset_for_Image_Quality_Assessment/21507564/1) | Pinned Figshare version 1 lists CC BY 4.0; original quality-study data | Separate research quality model | Initial header-only transfer/inspection verified; patient mapping, copy deduplication, cross-source overlap and test policy remain under review. One oversized spreadsheet dimension requires bounded joins. No quality model trained. See [review](16-quality-source-review.md). |

Being listed here is not acceptance. A quality label from diabetic-retinopathy images is not automatically a glaucoma-specific gradability standard. An image can be adequate for one task and inadequate for another; the clinical lead must define what anatomy and detail the intended task requires.

## Data onboarding work before a new training run

1. Record original owner, access route, terms, exact version, task definition, image and patient counts, acquisition sites/devices, label process, and exclusions.
2. Verify patient/group mapping and audit potential byte/pixel/near-duplicate overlap against **every** existing training, validation and benchmark source. Shared source names indicate a risk, not proof of shared patients; resolve rather than assume.
3. Reserve new source/site/time patients for a future evaluation before looking at results. Establish a test custodian or access boundary appropriate to the study.
4. Obtain a clinician-reviewed target and reference standard. Define supported cameras, ungradable handling, output meaning, false-alarm burden, and independent acceptance criteria.
5. Preregister the next experiment and analysis, including the status of every already-inspected benchmark. Record failures and avoid benchmark-driven threshold searches.

## Clinical evidence that the agent cannot supply by self-assessment

A confirmed clinical partner, institutional study decisions, appropriate patient-data permissions, a new representative cohort, expert reference adjudication, clinician–AI workflow evaluation, and applicable regulatory/institutional review remain absent. Named health systems are desired settings, not partners. No patient information has been solicited for upload to this research repository.

An illustrative mathematical limit shows why small validation cohorts are insufficient: even 14/14 successes has an independent-binomial Wilson lower 95% bound near 78.5%. This is not a clinical sample-size recommendation; a study must account for clinically justified bounds, prevalence, clustering, subgroups, missingness and workflow. See [statistical evaluation](05-statistical-evaluation.md) and the [study draft](06-clinical-study-draft.md).

## Advancement decision

Continue only with documented, permitted evidence acquisition and preregistered research work. Keep clinical release closed. Do not convert confidence scores, a passing checklist, more dataset names, or a self-issued “hospital grade” label into a claim of clinical trust.
