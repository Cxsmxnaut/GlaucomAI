# Statistical evaluation plan

Status: implemented research audit plus draft clinical analysis plan. Code: [statistics](../../glaucoma/statistics.py), [Colab audit](../../clinical_audit.py).

## Always report denominators and errors

For every source, stage, site, and supported subgroup, report TP/TN/FP/FN, positive/negative counts, sensitivity, specificity, and accuracy. Keep referral and disease endpoints separate. Report the threshold and exactly which validation data chose it. Avoid a pooled accuracy dominated by the largest or easiest source.

The S1 final stage achieved:

| Dataset benchmark | Caught positives | Sensitivity | Correctly unflagged negatives | Specificity |
| --- | ---: | ---: | ---: | ---: |
| EyePACS | 365/385 | 94.8% | 179/385 | 46.5% |
| ACRIMA | 57/59 | 96.6% | 32/46 | 69.6% |
| PAPILA | 12/14 | 85.7% | 31/48 | 64.6% |

These are source-label results at different acquisition conditions, not a common clinical diagnostic endpoint. They do not establish benefit or reliability in a hospital.

## Uncertainty

The audit computes 95% Wilson intervals for binomial rates and labels their independence assumption. For example, even a hypothetical perfect 14/14 has a Wilson lower bound near 78.5%. The observed PAPILA 12/14 interval is about 60.1%–96.0% **under an independent-image assumption**. Both eyes can be correlated, so this is descriptive rather than a justified patient-level interval.

For PAPILA, resample whole patient groups with replacement (2,000 replicates, fixed seed) and report percentile intervals plus the number of valid replicates. For ACRIMA/EyePACS, do not claim patient-level intervals without verified patient IDs. For a future multisite cohort, account for patient/site sampling in a statistician-reviewed analysis. Bootstrap does not fix bias, tiny cohorts, label error, or nonrepresentative recruitment. [NIST describes the Wilson score method](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

## Additional audit outputs

AUROC handles score ties. Brier score describes squared error against the dataset labels. Test-sample PPV/NPV must not be advertised as population predictive values, especially on balanced EyePACS data. A calibration study requires a relevant calibration cohort, validation-only fitting, and independently assessed reliability; a softmax sum of one is not calibration.

Solid-color PNG controls pass the decoder but are not fundus images. Their outputs expose the absence of a retinal-suitability gate; they are not a validated image-quality benchmark. Further studies need ungradable photographs, nonretinal inputs, camera changes, illumination/blur/compression, and realistic clinical workflow errors. Define changes and expected behavior before testing; do not claim clinical sensitivity from synthetic corruptions.

Clinical acceptance bounds, subgroup rules, multiplicity handling, missing/ungradable accounting, and sample size remain for clinical/statistical review. The legacy 88% threshold is not a substitute.

## Completed audit findings

The independent Colab audit reproduced every locked confusion count and confirmed the three backbones were identical. For final-stage PAPILA, AUROC was **0.8690**. Its 31 test patient groups produced a **66.7%–100%** bootstrap sensitivity interval, with 1,998 valid sensitivity replicates of 2,000 (two sampled cohorts had no positive labels). This wide interval and the one-site sample prevent a strong reliability claim.

All five solid-color PNGs generated above-threshold raw model scores at every stage. The final black image score was about **78.3%**, showing that a model score is not proof of retinal content. A subsequent exact-uniform input rejection control addresses these specific inputs before inference; it does not validate general gradability or retinal suitability. [Original audit record](../../experiments/clinical_audit_colab_t4.json).

The subsequent Colab guard audit rejected 15/15 uniform controls before any model forward call. None of the 937 existing benchmark test images was rejected by this exact-uniform check. [Guard audit](../../experiments/input_guard_colab_t4.json). The original audit intentionally retains the unprotected raw-model outputs as evidence of the discovered failure; it was not rewritten to hide them.
