# Requirements and evidence map

Status: live engineering record, 2026-11-07. The hackathon's **88% sensitivity** and **2-second latency** targets are development targets, not clinician-approved acceptance criteria.

| ID | Requirement or claim | Current evidence | Status / next proof |
| --- | --- | --- | --- |
| ING-01 | Decode PNG/JPEG safely under stated limits | Image pipeline tests; RGB/EXIF handling, size and pixel limits | Verified for tested cases |
| PRE-01 | Exact 224×224 and ImageNet normalization | Shared preprocessing and numeric tests | Verified; clinical impact of resizing unassessed |
| MOD-01 | Real trained inference, no synthetic scores | Strict checkpoint loader and real baseline inference | Verified engineering behavior |
| MOD-02 | Detect structural cupping | No segmentation or measurement module | Missing; remove structural claim |
| LAB-01 | Output has a clinically consistent meaning | EyePACS referral labels differ from other sources | Failed for a single mixed clinical endpoint; mixed checkpoints barred from referral dashboard |
| DAT-01 | Patients independent across partitions | PAPILA grouping verified by supplied IDs | Partial; ACRIMA/EyePACS patient independence unverified |
| DAT-02 | Independent test for future changes | New test sets were withheld until sequential stages locked | Completed experiment; now inspected development benchmarks |
| PERF-01 | Catch at least 88% of positives | Final PAPILA stage catches 12/14 PAPILA positives (85.7%) | Not met on that small test; not a clinical pass elsewhere |
| PERF-02 | Acceptable false-alarm burden | Final EyePACS specificity 179/385 (46.5%) | Clinical limit undefined; major concern |
| TIME-01 | Click-to-result within 2 seconds | Baseline CPU pipeline latency measured | Partial; browser/network end-to-end performance unverified |
| QLT-01 | Reject ungradable/nonretinal images | Format validation and exact-uniform-image guard only | Partial narrow control; quality gate needs labeled independent validation |
| CLI-01 | Improve clinician/patient outcomes | No institutional clinical study | Missing |
| SEC-01 | Suitable for hospital patient data | No reviewed hospital deployment or data agreement | Missing; public/de-identified research data only |
| OPS-01 | Monitoring, support, rollback, change approval | Prototype only | Missing for clinical operation |

Engineering tests show code behavior on their fixtures. They do not demonstrate clinical accuracy, dataset representativeness, security certification, or hospital approval. Each future requirement needs an owner, acceptance criterion, evidence location, and independent reviewer before it can be marked verified.
