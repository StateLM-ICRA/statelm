from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

summaries = {
    row["model"]: row
    for row in json.loads((RESULTS / "evaluation_summary.json").read_text())
}

pairs = {
    "real held-out": ("base_real_test", "fine_tuned_real_test"),
    "development review": ("base_review_development", "fine_tuned_review_development"),
    "synthetic smoke": ("base_template_smoke", "fine_tuned_template_smoke"),
    "supplement behavior": ("base_behavior_test", "fine_tuned_behavior_test"),
}

metrics = [
    ("Schema valid", "schema_valid_rate"),
    ("Validator accepted", "validator_acceptance_rate"),
    ("Action accuracy", "action_accuracy"),
    ("Structured task success", "structured_task_success_rate"),
    ("Failure macro-F1", "failure_macro_f1"),
    ("Failure detection F1", "failure_detection_f1"),
    ("Recovery agreement@1", "recovery_strategy_agreement_at_1"),
]


def percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def points(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:+.1f} pp"


with (RESULTS / "paper_metrics.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["evaluation_set", "n", "metric", "base", "fine_tuned", "absolute_change"])
    for set_name, (base_name, tuned_name) in pairs.items():
        base = summaries[base_name]
        tuned = summaries[tuned_name]
        for label, key in metrics:
            before = base.get(key)
            after = tuned.get(key)
            change = None if before is None or after is None else after - before
            writer.writerow([set_name, int(tuned["n"]), label, before, after, change])


def comparison_plot(set_name: str, output_name: str) -> None:
    base_name, tuned_name = pairs[set_name]
    base, tuned = summaries[base_name], summaries[tuned_name]
    labels = [label for label, _ in metrics]
    base_values = [base[key] for _, key in metrics]
    tuned_values = [tuned[key] for _, key in metrics]
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar([i - width / 2 for i in x], base_values, width, label="Base Qwen3-4B", color="#9AA6B2")
    ax.bar([i + width / 2 for i in x], tuned_values, width, label="Fine-tuned SLM", color="#2563EB")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Base vs. fine-tuned SLM - {set_name} (n={int(tuned['n'])})")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{100 * v:.0f}%" for v in container.datavalues], padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / output_name, dpi=200)
    plt.close(fig)


comparison_plot("real held-out", "real_heldout_comparison.png")
comparison_plot("supplement behavior", "supplement_behavior_comparison.png")

base_real = summaries["base_real_test"]
tuned_real = summaries["fine_tuned_real_test"]
base_behavior = summaries["base_behavior_test"]
tuned_behavior = summaries["fine_tuned_behavior_test"]

report = f"""# SLM-only final run results

- Run: `qwen3_robot_slm_20260916_001415`
- Base model: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`
- Training: 3 epochs, QLoRA rank 8, learning rate 5e-5, seed 42, A100 40 GB
- Best checkpoint: `checkpoint-204` with evaluation loss `0.0219620019`

## Primary real held-out evaluation

The real held-out evaluation contains 22 records. It is development evidence: participants represented in the broader project had already informed system development, so it is not a new untouched publication test.

| Metric | Base | Fine-tuned | Change |
|---|---:|---:|---:|
| Schema-valid output | {percent(base_real['schema_valid_rate'])} | {percent(tuned_real['schema_valid_rate'])} | {points(tuned_real['schema_valid_rate'] - base_real['schema_valid_rate'])} |
| Validator acceptance | {percent(base_real['validator_acceptance_rate'])} | {percent(tuned_real['validator_acceptance_rate'])} | {points(tuned_real['validator_acceptance_rate'] - base_real['validator_acceptance_rate'])} |
| Action accuracy | {percent(base_real['action_accuracy'])} | {percent(tuned_real['action_accuracy'])} | {points(tuned_real['action_accuracy'] - base_real['action_accuracy'])} |
| Joint goal accuracy | {percent(base_real['joint_goal_accuracy'])} | {percent(tuned_real['joint_goal_accuracy'])} | {points(tuned_real['joint_goal_accuracy'] - base_real['joint_goal_accuracy'])} |
| Structured task success | {percent(base_real['structured_task_success_rate'])} | {percent(tuned_real['structured_task_success_rate'])} | {points(tuned_real['structured_task_success_rate'] - base_real['structured_task_success_rate'])} |
| Failure macro-F1 | {base_real['failure_macro_f1']:.3f} | {tuned_real['failure_macro_f1']:.3f} | {tuned_real['failure_macro_f1'] - base_real['failure_macro_f1']:+.3f} |
| Failure-detection F1 | {base_real['failure_detection_f1']:.3f} | {tuned_real['failure_detection_f1']:.3f} | {tuned_real['failure_detection_f1'] - base_real['failure_detection_f1']:+.3f} |
| Recovery-strategy agreement@1 | {percent(base_real['recovery_strategy_agreement_at_1'])} | {percent(tuned_real['recovery_strategy_agreement_at_1'])} | {points(tuned_real['recovery_strategy_agreement_at_1'] - base_real['recovery_strategy_agreement_at_1'])} |
| Median generation latency | {base_real['generation_p50_ms'] / 1000:.2f} s | {tuned_real['generation_p50_ms'] / 1000:.2f} s | {(tuned_real['generation_p50_ms'] - base_real['generation_p50_ms']) / 1000:+.2f} s |
| P95 generation latency | {base_real['generation_p95_ms'] / 1000:.2f} s | {tuned_real['generation_p95_ms'] / 1000:.2f} s | {(tuned_real['generation_p95_ms'] - base_real['generation_p95_ms']) / 1000:+.2f} s |

The fine-tuned model produced valid JSON decisions on all 22 real held-out records and selected the correct action on all 22. The stricter structured-task score was 54.5%, because it additionally requires all scored labels and grounded fields to agree. Recovery-strategy agreement improved from 17.6% to 47.1%. Median generation latency increased by {(tuned_real['generation_p50_ms'] - base_real['generation_p50_ms']) / 1000:.2f} seconds.

## Separate supplemental behavior test

The 145-record behavior test is a separate supplement and must not be merged with the real held-out score.

| Metric | Base | Fine-tuned | Change |
|---|---:|---:|---:|
| Action accuracy | {percent(base_behavior['action_accuracy'])} | {percent(tuned_behavior['action_accuracy'])} | {points(tuned_behavior['action_accuracy'] - base_behavior['action_accuracy'])} |
| Structured task success | {percent(base_behavior['structured_task_success_rate'])} | {percent(tuned_behavior['structured_task_success_rate'])} | {points(tuned_behavior['structured_task_success_rate'] - base_behavior['structured_task_success_rate'])} |
| Failure macro-F1 | {base_behavior['failure_macro_f1']:.3f} | {tuned_behavior['failure_macro_f1']:.3f} | {tuned_behavior['failure_macro_f1'] - base_behavior['failure_macro_f1']:+.3f} |
| Failure-detection F1 | {base_behavior['failure_detection_f1']:.3f} | {tuned_behavior['failure_detection_f1']:.3f} | {tuned_behavior['failure_detection_f1'] - base_behavior['failure_detection_f1']:+.3f} |
| Recovery-strategy agreement@1 | {percent(base_behavior['recovery_strategy_agreement_at_1'])} | {percent(tuned_behavior['recovery_strategy_agreement_at_1'])} | {points(tuned_behavior['recovery_strategy_agreement_at_1'] - base_behavior['recovery_strategy_agreement_at_1'])} |
| Validator acceptance | {percent(base_behavior['validator_acceptance_rate'])} | {percent(tuned_behavior['validator_acceptance_rate'])} | {points(tuned_behavior['validator_acceptance_rate'] - base_behavior['validator_acceptance_rate'])} |
| Unsupported-ID proposal rate | {percent(base_behavior['unsupported_id_proposal_rate'])} | {percent(tuned_behavior['unsupported_id_proposal_rate'])} | {points(tuned_behavior['unsupported_id_proposal_rate'] - base_behavior['unsupported_id_proposal_rate'])} |

## Interpretation boundary

- These results evaluate the SLM decision component, not the complete FSM/SLM router or the live participant study.
- The real held-out set is development evidence, not a newly collected untouched publication test.
- The review and synthetic smoke evaluations remain separate in `evaluation_summary.json`.
- The base Qwen3 weights are not included. Runtime loading requires the pinned public base-model revision plus the LoRA adapter.
- The validator blocked some otherwise well-formed fine-tuned outputs: acceptance was 95.5% on the real held-out set and 88.3% on the behavior supplement.
"""

(ROOT / "RESULTS.md").write_text(report)

manifest = []
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and path.name not in {"MANIFEST.sha256"}:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f"{digest}  {path.relative_to(ROOT)}")
(ROOT / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n")

print(ROOT / "RESULTS.md")
print(RESULTS / "paper_metrics.csv")
print(FIGURES)
