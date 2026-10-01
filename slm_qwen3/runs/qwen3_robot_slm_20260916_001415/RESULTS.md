# SLM-only final run results

- Run: `qwen3_robot_slm_20260916_001415`
- Base model: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`
- Training: 3 epochs, QLoRA rank 8, learning rate 5e-5, seed 42, A100 40 GB
- Best checkpoint: `checkpoint-204` with evaluation loss `0.0219620019`

## Primary real held-out evaluation

The real held-out evaluation contains 22 records. It is development evidence: participants represented in the broader project had already informed system development, so it is not a new untouched publication test.

| Metric | Base | Fine-tuned | Change |
|---|---:|---:|---:|
| Schema-valid output | 45.5% | 100.0% | +54.5 pp |
| Validator acceptance | 45.5% | 95.5% | +50.0 pp |
| Action accuracy | 45.5% | 100.0% | +54.5 pp |
| Joint goal accuracy | 0.0% | 100.0% | +100.0 pp |
| Structured task success | 0.0% | 54.5% | +54.5 pp |
| Failure macro-F1 | 0.123 | 0.867 | +0.744 |
| Failure-detection F1 | 0.625 | 0.970 | +0.345 |
| Recovery-strategy agreement@1 | 17.6% | 47.1% | +29.4 pp |
| Median generation latency | 6.11 s | 7.63 s | +1.53 s |
| P95 generation latency | 7.42 s | 9.36 s | +1.93 s |

The fine-tuned model produced valid JSON decisions on all 22 real held-out records and selected the correct action on all 22. The stricter structured-task score was 54.5%, because it additionally requires all scored labels and grounded fields to agree. Recovery-strategy agreement improved from 17.6% to 47.1%. Median generation latency increased by 1.53 seconds.

## Separate supplemental behavior test

The 145-record behavior test is a separate supplement and must not be merged with the real held-out score.

| Metric | Base | Fine-tuned | Change |
|---|---:|---:|---:|
| Action accuracy | 19.3% | 75.9% | +56.6 pp |
| Structured task success | 4.8% | 70.3% | +65.5 pp |
| Failure macro-F1 | 0.111 | 0.947 | +0.836 |
| Failure-detection F1 | 0.263 | 0.822 | +0.559 |
| Recovery-strategy agreement@1 | 33.3% | 97.8% | +64.4 pp |
| Validator acceptance | 19.3% | 88.3% | +69.0 pp |
| Unsupported-ID proposal rate | 4.1% | 5.5% | +1.4 pp |

## Interpretation boundary

- These results evaluate the SLM decision component, not the complete FSM/SLM router or the live participant study.
- The real held-out set is development evidence, not a newly collected untouched publication test.
- The review and synthetic smoke evaluations remain separate in `evaluation_summary.json`.
- The base Qwen3 weights are not included. Runtime loading requires the pinned public base-model revision plus the LoRA adapter.
- The validator blocked some otherwise well-formed fine-tuned outputs: acceptance was 95.5% on the real held-out set and 88.3% on the behavior supplement.
