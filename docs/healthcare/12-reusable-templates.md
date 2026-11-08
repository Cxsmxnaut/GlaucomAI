# Reusable templates for future healthcare AI projects

Copy these into a new project and replace every placeholder. Keep unknowns explicit. A completed form is not evidence of clinical validity.

## A. Project and intended-use brief

```markdown
# [Project]
Status: [research / study candidate / other reviewed status]
Clinical lead: [name or missing]
Intended patient population:
Clinical condition and endpoint:
Intended user, setting, workflow, and decision supported:
Input device/acquisition requirements:
Output meaning and action; ungradable/uncertain/error handling:
Excluded uses and populations:
Jurisdictions and institutional reviewers:
Expected benefit and possible harms:
Clinically approved acceptance criteria and evidence:
Open decisions:
```

## B. Dataset acceptance record

```markdown
# [Dataset/version]
Original owner / publication / source URL:
Mirror and pinned version / retrieval date / hashes:
Original terms and allowed purpose; reviewer:
Patient/site/device/time identifiers and independence checks:
Label definition, annotators, clinical reference and adjudication:
Class counts, prevalence, missing/suspect/ungradable categories:
Population/device/site coverage and known exclusions:
Patient consent/data agreement and processing environment:
Overlap with other sources or pretraining:
Split policy fixed before training:
Privacy, retention/deletion, and access owners:
Decision: [accept / reject / pending] with evidence:
```

## C. Preregistered experiment record

```markdown
# [Experiment ID / protocol revision]
Hypothesis and clinical target:
Data/labels/splits; final test custodian and access rule:
Code/model/environment/initialization versions and hashes:
Trainable layers, preprocessing, augmentation, optimizer:
Epoch/budget/seed, checkpoint selection, threshold/calibration rule:
Validation criteria and stopping rules fixed before execution:
Start/end times, hardware, source revision:
All attempts/failures and outputs:
Results with counts, uncertainty, subgroup limitations:
Protocol deviations and their impact:
Decision and reviewer; next experiment:
```

## D. Clinical evidence and release record

```markdown
# [Candidate release]
Intended use and supported jurisdictions/sites/devices:
Immutable artifact hashes and reproducible build:
Independent clinical validation protocol and actual approvals:
Reference standard, population, sites and patient counts:
Prespecified outcomes, uncertainty, subgroup and failure results:
Image quality/abstention and clinician-workflow validation:
Risk controls and residual-risk review:
Privacy/security/data agreements and operational readiness:
Applicable regulatory determination and actual authorization:
Monitoring, incident support, rollback and change control:
Missing/failed/out-of-scope evidence:
Named independent reviewers, decisions and dates:
Release decision and permitted use:
```

## E. Journal entry and change review

```markdown
Date / author / change ID:
Problem or question:
Confirmed evidence versus assumptions:
What changed and why:
Files/artifacts/versions:
Checks run and exact scope:
Results, failures, limitations:
Effect on intended use, data, risk, test independence, approvals:
Reviewer and required follow-up:
```

Never copy another project's clinical acceptance bounds, regulatory status, licenses, cohort representativeness, or approvals. Reassess them for the new intended use.
