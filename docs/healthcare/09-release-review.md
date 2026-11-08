# Release evidence and advancement review

Current decision: **research only; do not promote for clinical use.** “Hospital grade” is not a numerical model score or a status this repository can self-certify.

## Evidence levels

| Level | Meaning | GlaucomAI status |
| --- | --- | --- |
| Engineering prototype | Tested software executes a defined pipeline | Present, with documented limits |
| Research benchmark | Locked experiments and reproducible source-label metrics | Present for the recorded runs |
| Independent clinical validation | Appropriate untouched cohort, reference standard, statistical plan, clinical oversight | Missing |
| Workflow validation | Intended clinicians use the system safely and effectively in the intended setting | Missing |
| Clinical release | Applicable institutional, regulatory, security, quality and operational reviews complete | Missing |

## Release dossier fields

For every candidate, record immutable code/model/environment versions, intended use, patient/site/device scope, training/validation/test provenance, label definition, threshold-selection history, all failures, uncertainty/subgroups, quality/abstention behavior, independent study results, risk controls, data permissions, privacy/security review, usability results, monitoring/rollback plan, and actual named reviewer decisions.

Unknown, missing, failed, or out-of-scope evidence keeps clinical release closed. A passing test suite, high AUROC, disclaimer, Markdown checklist, “human in the loop” label, or self-filled approval field cannot replace independent clinical review.

`python scripts/check_healthcare_evidence.py` checks the exact sequential-report hash, parent/child checkpoint lineage, audit outcome agreement, current pipeline hash against the Colab guard audit, and the recorded narrow input-control behavior. CI runs this check. It deliberately reports clinical release as false. It verifies consistency of recorded evidence, not external truth, clinical validity, or approval. The exact JSON bytes are retained so the recorded audit hash remains reproducible.

## Current reasons to remain research-only

1. No confirmed clinical partner, agreed endpoint, or approved study.
2. Mixed referral/disease labels in the sequential model; source shifts and poor specificity.
3. Patient independence unverified for two sources; small single-source PAPILA cohort.
4. No validated input-quality/abstention system or demonstrated clinician–AI workflow benefit.
5. No reviewed clinical deployment, privacy/data agreements, regulatory determination, or operated monitoring/support.
6. Colab weights remain temporary; a supported binary export/retention path is unresolved.

The [FDA GMLP overview](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles) points to international principles spanning the medical-device lifecycle. This project uses them as a planning reference, not as proof of compliance or approval.
