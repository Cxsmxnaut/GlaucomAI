# Decision log

| Date / ID | Decision | Evidence and reason | Consequence |
| --- | --- | --- | --- |
| 2026-11-07 D01 | Train in Google Colab after initial local baseline | User explicitly requested Colab | Later training uses a verified T4; local tests are synthetic engineering checks |
| 2026-11-07 D02 | Keep browser work hidden and isolated | User reported desktop interruption | Use DOM controls in hidden in-app browser; no native desktop control |
| 2026-11-07 D03 | Preserve exact label order and preprocessing in checkpoints | Alphabetical ordering or transformations can invert/change results | Loader rejects incompatible metadata |
| 2026-11-07 D04 | Select three well-documented Kaggle sources | EyePACS, ACRIMA, PAPILA all show 10/10 usability | Original labels/terms still govern interpretation; votes are not clinical validation |
| 2026-11-07 D05 | Exclude entire SMDG aggregate and RIM-ONE from this sequence | Aggregate overlaps existing sources; RIM-ONE original benchmark/usage conditions differ from mirrors | Avoid counting copied sources as independent data |
| 2026-11-07 D06 | Exclude PAPILA suspects and group by patient | Original paper defines 0/1/2 diagnoses and paired eyes | Use 420/488 images; retain both eligible eyes in same split |
| 2026-11-07 D07 | Continue head weights with bounded training replay | User asked to train one dataset after another; earlier-source forgetting is plausible | Save every parent/child checkpoint and report every stage |
| 2026-11-07 D08 | Lock stages before evaluating new test sets | Prevent test-driven checkpoint/threshold selection | Scores now inspected; future experiments treat these as development benchmarks |
| 2026-11-07 D09 | Do not replace referral demo with mixed-label model | Different clinical targets and specificity tradeoff | Dashboard rejects mixed-label checkpoint; original release stays separate |
| 2026-11-07 D10 | Add uncertainty and patient bootstrap audit | PAPILA has only 14 positive test eyes; eye correlation matters | No hospital-readiness claim from point sensitivity |
| 2026-11-07 D11 | Keep clinical advancement open but evidence-based | User requested hospital-grade work globally; no partner yet | Prepare reusable records/study draft; do not invent approvals or imply named-system partnership |
| 2026-11-07 D12 | Respect blocked artifact download | Browser policy rejected data-URL binary download; normal Colab download produced no file | Do not bypass the restriction; preserve source and aggregate evidence, report temporary cloud weights honestly |
| 2026-11-07 D13 | Reject exactly uniform images before model execution | All five flat-color controls exceeded every checkpoint threshold | Implement narrow deterministic guard; retain general quality/retinal recognition as an unresolved clinical gap |

Next architecture decision remains open: separate task heads versus one newly defined clinical endpoint. Decide before the next protocol, not after viewing its confirmatory results.
