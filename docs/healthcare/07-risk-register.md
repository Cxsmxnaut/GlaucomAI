# Risk register

Status: initial engineering risk analysis; **not a completed clinical risk-management file**. Severity/acceptability ratings require clinical review. No numeric risk score is invented here.

| ID | Hazard and possible harm | Current evidence/control | Residual gap and next action |
| --- | --- | --- | --- |
| R01 | Missed glaucoma/referral leads to delayed assessment | Real false negatives reported; research limitation shown | Agree endpoint; independent clinical evidence and safe negative-result workflow |
| R02 | False alarms overload referral service and cause anxiety | Final EyePACS stage produces 206/385 false positives | Define acceptable service burden; prospectively measure it |
| R03 | Dataset label mismatch creates an ambiguous clinical output | Mixed checkpoint metadata; referral dashboard rejects it | Separate heads or consistent clinically adjudicated target |
| R04 | Nonretinal/poor-quality image receives a confident score | Negative-control audit found all flat controls positive; exact-uniform guard added before inference | Narrow guard only; obtain quality/gradable labels and independently validate broader abstention |
| R05 | Site/camera/patient shift hides poor performance | Original model catches only 4/59 ACRIMA positives | New-site/device study; training changes alone cannot close evidence gap |
| R06 | Patient or near-duplicate leakage inflates scores | PAPILA grouped; byte/pixel duplicate audit | ACRIMA/EyePACS identities unresolved; near-duplicate/source-overlap review |
| R07 | Tiny or unrepresentative cohorts hide subgroup failures | Counts and uncertainty reported | Prespecified representative multisite sample and subgroup precision |
| R08 | Softmax interpreted as calibrated disease risk | UI limitations; mixed-label guard | Calibration and human-factors testing on intended workflow |
| R09 | Wrong/corrupt/stale checkpoint changes behavior | Strict class/preprocessing/finite checks and hashes | Signed release provenance, controlled deployment and rollback |
| R10 | Patient identifiers leak through images or notebook outputs | Demo memory clearing and no disk upload; private-output metadata | Institution-approved environment, full privacy assessment and operational controls |
| R11 | Runtime loss destroys evidence or model | Source and aggregate results in Git; weights in temporary Colab | Supported artifact retention/export; current binary export limitation unresolved |
| R12 | Automation bias or inappropriate autonomous care | Research-only positioning; no care integration | Clinician–AI workflow validation, training and escalation design |
| R13 | Uncontrolled retraining silently changes clinical behavior | Each research stage saved with parent hash | Clinical change control, impact analysis, revalidation and approval |

For each risk, assign a named owner, review date, concrete control verification, residual-risk assessment, and clinical/security acceptance evidence before any clinical release. A disclaimer does not substitute for a working risk control.
