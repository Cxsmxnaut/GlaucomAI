# Clinical validation study outline

Status: **draft only; no institution, ethics approval, recruitment, or clinical trial is in place.** This document supports review with a clinical lead and biostatistician. It does not authorize a study or patient-data collection.

## Questions to settle first

Agree on intended use, eligible patients, supported cameras, referral endpoint, unit of analysis (person or eye), clinical reference standard, acceptable missed-case and false-alarm burden, ungradable workflow, and each jurisdiction's requirements. A glaucoma diagnosis and a referral recommendation are not interchangeable outcomes.

## Proposed staged evidence

1. **Independent retrospective validation:** a locked model on previously untouched patients/sites/time periods, with an expert-defined reference and no model-driven patient care. Audit patient/site/device independence before unblinding results.
2. **Prospective silent-mode study:** collect representative routine inputs in an approved environment; record model outputs without changing care. Compare against the agreed clinical reference and existing workflow.
3. **Human–AI workflow evaluation:** after prior evidence and approvals justify it, assess clinician decisions, usability, workload, missed cases, referral burden, and relevant patient outcomes. Comparison design and escalation procedures require clinical review.

IMDRF distinguishes clinical association, analytical validation, and clinical validation; successful software tests alone do not cover all three. [Clinical evaluation framework](https://www.imdrf.org/sites/default/files/docs/imdrf/final/technical/imdrf-tech-170921-samd-n41-clinical-evaluation_1.pdf).

## Protocol fields to complete before enrollment

- Named sponsor, principal investigator, sites, independent monitor/adjudicator, and statistician.
- Intended endpoint and reference-standard procedure; masking and disagreement adjudication. Do not let the model score define its own ground truth.
- Consecutive or otherwise justified recruitment; explicit inclusions/exclusions; handling of prior disease, surgery, comorbidities, urgent findings, and ungradable images.
- A priori sample-size calculation from clinically justified sensitivity/specificity lower bounds, disease prevalence, site/patient clustering, subgroup precision, missingness, and planned comparisons. No arbitrary image count can guarantee clinical validity.
- Model/checkpoint hash, preprocessing and threshold freeze; test-label access controls and unblinding record.
- Required device/site/demographic strata, analysis population, missing-data policy, abstention coverage, and prespecified statistical methods.
- Privacy/data agreements, consent or institutional determination, retention, incident handling, and study registration/publication requirements as applicable.
- Safety review and stop/escalation criteria; routine clinical care remains available regardless of AI output.

## Global use

A study in one system or country does not authorize use worldwide. Maintain a jurisdiction/site/device evidence matrix and obtain local institutional and regulatory advice. Specific FDA classification, pathway, exemptions, CE/UK claims, or approvals have not been determined for GlaucomAI.
