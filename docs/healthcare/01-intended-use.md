# Intended use and claims

Status: **draft for clinical review**. No hospital or ophthalmologist is currently confirmed as a validation partner. Kaiser, Sutter Health, and John Muir were supplied as examples of desired settings, not endorsements or partnerships.

## Proposed future use

A clinician-supervised aid that analyzes retinal fundus photographs from adults and helps an eye-care professional decide whether additional glaucoma assessment or referral is needed. The clinician retains responsibility for care. The age range, acquisition setting, camera list, eligible population, handling of both eyes, workflow position, and reference standard remain to be agreed with clinical partners.

This proposal is narrower than a worldwide diagnostic claim. Evidence from a few public datasets does not establish performance at a named health system, in every country, or on every camera. Each intended environment needs relevant evidence and jurisdiction review.

## What exists today

- An EyePACS binary image classifier that predicts **dataset referral labels**.
- A separate sequential research experiment that mixes EyePACS referral labels with ACRIMA/PAPILA glaucoma labels.
- A notebook dashboard with real model outputs, strict image decoding, and clear research limitations.

The mixed-label experiment is not an implementation of an agreed clinical endpoint. It must not become the default clinical model because its aggregate score looks better. A future clinical model should use a consistent, clinician-approved endpoint or separate task-specific heads and reports.

## Claims currently unsupported

Diagnosis, exclusion of glaucoma, assessment of disease stage, measurement of optic cup/disc geometry, identification of why a score was produced, treatment recommendation, safe autonomous referral decisions, verified image gradability, worldwide generalization, and validated benefit to patients or clinicians.

The original hackathon PRD proposed a healthy/pathological output and a structural cupping explanation. The implemented classifier does not segment or measure these structures. “Non-referable” must not be renamed “healthy,” and softmax output must not be labeled a person's disease probability.

## Decisions required before a clinical protocol is final

1. Clinical lead, target population, referral endpoint, and reference-standard procedure.
2. Consequences and acceptable rates of missed cases, false alarms, and abstentions.
3. Supported imaging devices and sites; who resolves ungradable or conflicting results.
4. Study jurisdiction, institutional approvals, data agreements, and intended regulatory pathway.

No answers or approvals are implied by this draft.
