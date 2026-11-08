# Security, privacy, and operations

Status: research notebook; hospital service design is **not implemented or approved**.

## Current boundaries

GitHub contains source, notebooks, and aggregate experiment evidence. Data downloads and model execution occur in Colab. The dashboard retains uploaded images in session memory and exposes a clear control; saved notebook outputs can still contain images. No patient records, credentials, or dataset images belong in Git. Model loading uses a restrictive weights-only loader with schema and finite-value checks, but users must still load only trusted checkpoints.

Colab is temporary interactive compute. It has no demonstrated clinical availability commitment, institutional identity/access control design, approved PHI contract, clinical audit trail, disaster-recovery process, or operated patient-safety support workflow in this project. Public/de-identified research data is the current boundary. Do not infer institutional approval from using a major cloud provider.

## Deployment design needed before patient use

- Institution-approved processing and data storage, reviewed data flow and contracts, access by role, strong authentication, least privilege, encryption and managed keys.
- Secrets in an approved secret store; no tokens in notebooks, URLs, logs, model metadata, commits, or chat output.
- Logs limited to operationally necessary fields. Avoid images, direct identifiers, and full model inputs. Define retention, deletion, access review, and incident investigation procedures.
- Versioned immutable model, preprocessing, threshold, environment, dependencies, and release manifest; authenticated delivery and checksum/signature verification.
- Deployment rollback, output suspension, outage fallback to normal clinical workflow, support ownership, backup/restore verification, and tested monitoring.
- Resource limits and robust parsers for uploads; dependency/vulnerability review and adversarial-input tests. Existing image tests are only part of this work.
- Drift monitoring by device/site, gradeability, abstention, referral burden and delayed outcome labels. A distribution shift flag is not proof that accuracy changed; investigate with clinical evidence.

No clinical model updates should run automatically from production data. A candidate change needs a documented purpose, permitted data, impact analysis, independent validation, review, and a controlled release.

For U.S. cloud handling of ePHI, review the [HHS cloud guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/health-information-technology/cloud-computing/index.html) with institutional privacy/security teams. This checklist is a project plan, not a HIPAA or global compliance determination.
