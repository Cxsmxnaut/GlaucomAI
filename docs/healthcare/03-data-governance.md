# Data governance and provenance

Status: public research datasets only. See [the dataset catalog](../../DATASETS.md) for pinned Kaggle versions, ratings, original papers, and terms.

## Current inventory

| Source | Images used | Label meaning | Split control | Main limitation |
| --- | ---: | --- | --- | --- |
| EyePACS light V2 v4 | 9,540 | Non-referable / referable | Supplied 8,000/770/770 | Patient separation unverified; balanced sample |
| ACRIMA mirror v1 | 705 | Expert image-based normal / glaucoma | Seeded image split | Patient identifiers unavailable; optic-disc crops differ from full fundus |
| PAPILA mirror v1 | 420 | Clinical healthy / glaucoma | Patient group split | One source site; small positive sample; 68 suspect eyes excluded |

Kaggle usability is 10/10 for each selected listing at the recorded check. It measures usability/documentation, not a medical quality score. A mirror's license is not authority to override original source restrictions. PAPILA's original dataset lists GPL-3.0+; its article has a different license.

## Current checks

Explicit mappings are source-specific. PAPILA diagnosis is `0 healthy / 1 glaucoma / 2 suspect`; suspect eyes are excluded. Both retained eyes of a patient share one split. Every eligible image is decoded. Byte hashes and decoded-RGB pixel hashes reject exact duplicates across the experiment. They cannot prove absence of lossy near-duplicates, shared patients with different IDs, or hidden training-set overlap in pretrained weights.

The full manifest stays in Colab. Dataset images, clinical spreadsheets, credentials, and the manifest are excluded from published model archives. Only aggregate results and provenance are committed. The model uses images, not PAPILA age, gender, pressures, or other clinical measurements.

## Before institutional data

Obtain a named data controller/owner, approved research purpose, lawful data-access process, applicable patient consent or institutional determination, data-use agreement, retention/deletion schedule, and a reviewed processing environment. Establish site/patient/device/date identifiers inside the approved environment so independence can be audited without publishing identifiers. Restrict access to the minimum necessary data; review image borders and embedded metadata for identifiers.

No real hospital patient upload is authorized by this project. A consumer Colab account, public GitHub repository, or disclaimer does not establish a compliant clinical environment. In the U.S., cloud processing of ePHI has specific safeguards and business-associate-agreement considerations described by [HHS](https://www.hhs.gov/hipaa/for-professionals/faq/may-a-hipaa-covered-entity-or-business-associate-use-cloud-service-to-store-or-process-ephi/index.html). Other jurisdictions require their own review.

## Source onboarding record for future work

Record original owner, mirror, version, hashes, acquisition sites/devices/dates, patient/group IDs, label procedure, adjudication, excluded categories, original terms, consent scope, permitted processing, known overlaps, prevalence, demographics availability, and a decision to accept/reject. Missing patient IDs must remain a limitation; never invent them from image order.
