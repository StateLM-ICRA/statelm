"""Completion-only tokenization and raw model evaluation for the Colab notebook."""
import copy
import json
import time
from collections import defaultdict

from slm_runtime import (Decision, apply_chat_template, parse_decision, prompt_messages,
                         validate_decision, render_response)


def tokenize_record(record, tokenizer, max_length, max_input_tokens):
    messages = prompt_messages(record["input"])
    prompt_ids = apply_chat_template(tokenizer, messages, tokenize=True, add_generation_prompt=True)
    target = json.dumps(record["target"], ensure_ascii=False, separators=(",", ":"))
    # Build the completion directly after Qwen's generation prompt. This avoids a subtle mismatch between
    # Qwen3's generation-time no-thinking prefix and its full-conversation rendering.
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    full_ids = prompt_ids + target_ids
    if len(full_ids) > max_length or len(prompt_ids) > max_input_tokens:
        raise ValueError(f"{record['id']}: prompt={len(prompt_ids)}, full={len(full_ids)} exceeds budget. Shorten context or increase the configured budget; targets are never truncated.")
    labels = [-100] * len(prompt_ids) + target_ids
    if not target_ids:
        raise ValueError("No supervised answer tokens")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class CompletionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch
        # Right-pad TRAINING examples; label padding must be ignored, not learned.
        width = ((max(len(x["input_ids"]) for x in features) + 7) // 8) * 8
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            n = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * n)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * n)
            batch["labels"].append(feature["labels"] + [-100] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def balanced_subset(rows, limit):
    # Round-robin by scenario. Same selected record IDs for baseline and fine-tuned model.
    if not limit or limit >= len(rows):
        return list(rows)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["scenario"]].append(row)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    return selected


def evaluate_model(engine, rows, label):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    # Exclude one warm-up generation from latency statistics.
    try:
        engine(rows[0]["input"])
    except Exception:
        pass
    logs = []
    for i, row in enumerate(rows):
        started = time.perf_counter()
        raw, d, accepted, error = None, None, False, None
        generation_ms = None
        try:
            raw = engine(row["input"])
            generation_ms = 1000 * (time.perf_counter() - started)
            d = parse_decision(raw)
            validate_decision(d, row["input"], row["inventory"])
            render_response(d, row["inventory"])
            accepted = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        pred = d.model_dump() if d is not None else {}
        target = row["target"]
        goal_keys = ["item_family", "resolved_item_id", "attributes", "missing_attributes"]
        score_item_slots = row.get("evaluation_mask", {}).get("item_slots", True)
        goal = bool(pred) and all(pred.get(k) == target[k] for k in goal_keys)
        strategy_ok = pred.get("recovery_strategy") in (
            row.get("acceptable_recovery_strategies") or [target["recovery_strategy"]])
        label_core_correct = pred.get("action") == target["action"] and \
            pred.get("failure_type") == target["failure_type"] and strategy_ok
        scored_label_correct = label_core_correct and (goal if score_item_slots else True)
        conservative_target_correct = label_core_correct and goal
        # Proposed action/ID violations are counted even if the full output fails schema validation.
        try:
            proposed = json.loads(raw)
            proposed = proposed if isinstance(proposed, dict) else {}
        except Exception:
            proposed = {}
        proposed_id = proposed.get("resolved_item_id")
        known_ids = {x["id"] for x in row["inventory"]}
        unsupported_id = proposed_id is not None and (not isinstance(proposed_id, str) or proposed_id not in known_ids)
        disallowed = "action" in proposed and proposed["action"] not in row["input"]["allowed_actions"]
        predicted_failure = bool(pred.get("failure_detected", False)) if pred else False
        # A rejected output triggers the runtime fallback and must count as a detection error.
        metric_predicted_failure = predicted_failure if accepted else not bool(target["failure_detected"])
        operational_review_alarm = (not accepted) or predicted_failure
        logs.append({"id": row["id"], "scenario": row["scenario"], "source": row.get("source"),
                     "decision_mode": row["input"].get("decision_mode", "dialogue_decision"),
                     "schema_valid": d is not None,
                     "validator_accepted": accepted, "action_correct": pred.get("action") == target["action"],
                     "item_slots_scored": score_item_slots,
                     "joint_goal_correct": goal if score_item_slots else None,
                     "scored_label_success": bool(accepted and scored_label_correct),
                     "structured_task_success": bool(accepted and conservative_target_correct),
                     "unsupported_id_proposal": bool(unsupported_id), "disallowed_action_proposal": bool(disallowed),
                     "gold_failure": target["failure_type"], "pred_failure": pred.get("failure_type", "invalid"),
                     "is_recovery": target["failure_detected"], "predicted_failure": predicted_failure,
                     "scored_pred_failure": pred.get("failure_type", "invalid") if accepted else "invalid",
                     "metric_predicted_failure": metric_predicted_failure,
                     "failure_detection_correct": bool(accepted and predicted_failure == target["failure_detected"]),
                     "operational_review_alarm": operational_review_alarm,
                     "recovery_strategy_accepted": bool(strategy_ok),
                     "generation_ms": generation_ms, "decision_pipeline_ms": 1000 * (time.perf_counter() - started),
                     "raw_output": raw, "target": target, "error": error})
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"{label}: {i+1}/{len(rows)}")
    df = pd.DataFrame(logs)
    recoveries = df[df.is_recovery]
    contexts = df[df.scenario.str.startswith("context_")]
    labels = sorted(set(df.gold_failure))
    detect_precision, detect_recall, detect_f1, _ = precision_recall_fscore_support(
        df.is_recovery.astype(bool), df.metric_predicted_failure.astype(bool), average="binary", zero_division=0)
    review = df[df.decision_mode == "pre_send_review"]
    review_failures = review[review.is_recovery]
    review_controls = review[~review.is_recovery]
    summary = {"model": label, "n": len(df), "schema_valid_rate": float(df.schema_valid.mean()),
               "validator_acceptance_rate": float(df.validator_accepted.mean()),
               "action_accuracy": float(df.action_correct.mean()),
               "joint_goal_accuracy": float(df.joint_goal_correct.mean()) if df.joint_goal_correct.notna().any() else None,
               "scored_label_success_rate": float(df.scored_label_success.mean()),
               "structured_task_success_rate": float(df.structured_task_success.mean()),
               "failure_macro_f1": float(f1_score(df.gold_failure, df.scored_pred_failure, labels=labels,
                                                   average="macro", zero_division=0)),
               "failure_detection_precision": float(detect_precision),
               "failure_detection_recall": float(detect_recall),
               "failure_detection_f1": float(detect_f1),
               "failure_detection_accuracy": float(df.failure_detection_correct.mean()),
               "review_failure_recall": float((review_failures.validator_accepted &
                   review_failures.predicted_failure).mean()) if len(review_failures) else None,
               "review_false_alarm_rate": float(review_controls.operational_review_alarm.mean()) if len(review_controls) else None,
               "recovery_strategy_agreement_at_1": float(recoveries.recovery_strategy_accepted.mean()) if len(recoveries) else None,
               "recovery_acceptance_at_1": float((recoveries.validator_accepted &
                   recoveries.recovery_strategy_accepted).mean()) if len(recoveries) else None,
               "context_goal_accuracy": float(contexts.joint_goal_correct.mean()) if len(contexts) else None,
               "unsupported_id_proposal_rate": float(df.unsupported_id_proposal.mean()),
               "disallowed_action_proposal_rate": float(df.disallowed_action_proposal.mean()),
               "generation_p50_ms": float(df.generation_ms.median()) if df.generation_ms.notna().any() else None,
               "generation_p95_ms": float(df.generation_ms.quantile(.95)) if df.generation_ms.notna().any() else None}
    return summary, logs
