# Healthcare AI development playbook

Last updated: 2026-11-07. Live example: GlaucomAI. **Current status: research prototype; clinical use is not authorized or validated.**

This folder records what happened, why decisions were made, what the evidence supports, and what remains. It is also a starting point for future healthcare projects. Copy the templates, not GlaucomAI's performance numbers or unverified assumptions.

## Start here

| Document | Purpose |
| --- | --- |
| [01 Intended use](01-intended-use.md) | Define the clinical question before choosing labels or metrics |
| [02 Requirements and evidence](02-requirements-and-evidence.md) | Connect each claim to a check and identify missing proof |
| [03 Data governance](03-data-governance.md) | Sources, licenses, patient grouping, privacy, and data permissions |
| [04 Experiment protocol](04-experiment-protocol.md) | Reproduce training without contaminating confirmatory tests |
| [05 Statistical evaluation](05-statistical-evaluation.md) | Errors, uncertainty, calibration, subgroups, and robustness |
| [06 Clinical study draft](06-clinical-study-draft.md) | A study outline for a clinical partner and statistician to review |
| [07 Risk register](07-risk-register.md) | Known hazards, controls, evidence, and unresolved risk |
| [08 Security and operations](08-security-and-operations.md) | The gap between a Colab demo and a hospital service |
| [09 Release review](09-release-review.md) | Evidence needed before advancing beyond research |
| [10 Decision log](10-decision-log.md) | Reasons for important design choices |
| [11 Process journal](11-process-journal.md) | Actual runs, hashes, outcomes, checks, and limitations |
| [12 Reusable templates](12-reusable-templates.md) | Blank forms for new healthcare projects |
| [13 References](13-references.md) | Primary sources and what they support |

## Evidence language

- **Verified:** a reproducible artifact or observed result exists, with its scope stated.
- **Draft:** a proposal awaiting the relevant clinical, statistical, security, or regulatory review.
- **Missing:** evidence has not been obtained. A passing unit test cannot fill this gap.
- **Failed:** a required behavior was tested and did not meet its predefined criterion.
- **Not applicable:** requires a written justification; absence of evidence is not a justification.

Every experiment must update the journal and decision log. Every new source must update data governance. A changed target, population, camera, threshold, or model requires an impact review and an appropriate fresh evaluation. Never overwrite a failed experiment with a more flattering result.

These are project records and draft engineering/study materials. They are not a clinical endorsement, compliance certification, or completed regulatory submission.
