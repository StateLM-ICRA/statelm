#!/usr/bin/env python3
"""Replay the real deployment stream through the FSM/SLM router.

The SLM's decision for every stream payload must already be in the cache
(``teacher_pass.py`` fills it once on a GPU).  Routing then runs offline and
deterministically, so threshold settings can be compared without further
SLM calls.

Outputs in ``--out``:
  turn_log.jsonl   one record per turn (route, similarity, decisions, checks)
  metrics.json     summary metrics (overall, by stream third, by turn kind)
  snapshot.json    the learned FSM cache after the last turn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statelm_realstream import (  # noqa: E402
    HashingEmbedder,
    PaperAlignedRouter,
    RouterConfig,
    SentenceTransformerEmbedder,
    decision_dict,
)


def payload_key(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CachedTeacher:
    """Serves the trained SLM's cached raw output for a payload."""

    def __init__(self, records: list[dict[str, Any]]):
        self.cache = {record["key"]: record["raw"] for record in records}
        self.calls = 0
        self.misses = 0

    def __call__(self, payload: dict[str, Any]) -> Any:
        self.calls += 1
        key = payload_key(payload)
        if key not in self.cache:
            self.misses += 1
            raise KeyError(f"SLM cache has no answer for payload {key}")
        return self.cache[key]

    def has(self, payload: dict[str, Any]) -> bool:
        return payload_key(payload) in self.cache


def make_embedder(name: str, model: str, device: str | None):
    if name == "hash":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model, device=device)


def build_router(seed_rows, inventory, embedder, config, teacher):
    router = PaperAlignedRouter(teacher, inventory, embedder=embedder, config=config)
    seeded = 0
    refused = 0
    for row in seed_rows:
        pattern = router.seed_example(row["input"], row["decision"], row["id"])
        if pattern is None:
            refused += 1
        else:
            seeded += 1
    return router, {"seed_examples": len(seed_rows), "seeded": seeded, "refused": refused,
                    "seed_patterns": sum(p.origin == "seed" for p in router.patterns)}


def slm_reference(router, teacher, payload):
    """The decision the SLM path would have served (checked, guard-repaired)."""

    features = router._features(payload)
    raw = teacher.cache.get(payload_key(payload))
    raw_decision = None
    if raw is not None:
        try:
            raw_decision = decision_dict(raw)
        except Exception:
            raw_decision = None
    try:
        checked = router._task_valid(raw, features, payload) if raw is not None else None
        repaired = False
    except Exception:
        checked = router._guard_decision(features, payload)
        repaired = True
    return raw_decision, checked, repaired


def rate(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def summarize(logs, window):
    by_kind = defaultdict(list)
    for log in logs:
        by_kind[log["kind"] if log["origin"] != "augmented_v1_llm_written" else "augmented"].append(log)
    n = len(logs)
    thirds = [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
    out = {
        "turns": n,
        "fsm_share_overall": rate([l["source"] == "fsm" for l in logs]),
        "fsm_share_last_window": rate([l["source"] == "fsm" for l in logs[-window:]]),
        "fsm_share_by_third": [rate([l["source"] == "fsm" for l in logs[a:b]]) for a, b in thirds],
        "fsm_share_by_kind": {k: rate([l["source"] == "fsm" for l in v]) for k, v in by_kind.items()},
        "fsm_share_by_slm_split": {
            split: rate([l["source"] == "fsm" for l in logs if l["slm_split"] == split])
            for split in sorted({l["slm_split"] for l in logs if l["slm_split"]})},
        "served_action_ok_by_slm_split": {
            split: rate([l["served_action_ok"] for l in logs if l["slm_split"] == split])
            for split in sorted({l["slm_split"] for l in logs if l["slm_split"]})},
        "augmented_action_ok_by_category": {
            cat: rate([l["served_action_ok"] for l in logs if l.get("category") == cat])
            for cat in sorted({l["category"] for l in logs if l.get("category")})},
        "fsm_served_exact_agreement_with_checked_slm": rate(
            [l["fsm_matches_checked_slm"] for l in logs if l["source"] == "fsm"]),
        "fsm_served_exact_agreement_with_raw_slm": rate(
            [l["fsm_matches_raw_slm"] for l in logs if l["source"] == "fsm"]),
        "fsm_served_task_agreement_item_and_action": rate(
            [l["fsm_task_agreement"] for l in logs if l["source"] == "fsm"]),
        "served_action_matches_reference": rate([l["served_action_ok"] for l in logs]),
        "served_item_matches_reference": rate([l["served_item_ok"] for l in logs]),
        "review_failure_type_correct_served": rate(
            [l["served_failure_type_ok"] for l in logs if l["kind"] == "review"]),
        "review_failure_type_correct_slm": rate(
            [l["slm_failure_type_ok"] for l in logs if l["kind"] == "review"]),
        "review_strategy_in_preferences_served": rate(
            [l["served_strategy_ok"] for l in logs if l["kind"] == "review"]),
        "guard_repairs_on_slm_path": sum(1 for l in logs if l["source"] == "slm" and l["guard_repaired"]),
        "wrong_item_served": sum(1 for l in logs if l["served_item_ok"] is False),
        "errors": sum(1 for l in logs if l["error"]),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--slm-cache", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=ROOT / "data" / "inventory_by_setting.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embedder", choices=("hash", "sentence-transformer"), default="sentence-transformer")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--merge-similarity", type=float, default=0.78)
    parser.add_argument("--routing-similarity", type=float, default=0.82)
    parser.add_argument("--admission-threshold", type=float, default=0.85)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--no-seed", action="store_true", help="Start from an empty FSM (ablation)")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    seed_rows = [] if args.no_seed else read_jsonl(args.seed)
    stream = read_jsonl(args.stream)
    cache_records = read_jsonl(args.slm_cache)
    inventories = json.loads(args.inventory.read_text())
    teacher = CachedTeacher(cache_records)
    missing = [row["id"] for row in stream if not teacher.has(row["input"])]
    if missing:
        raise SystemExit(f"{len(missing)} stream payloads have no cached SLM decision; run teacher_pass.py first "
                         f"(first missing: {missing[:3]})")
    embedder = make_embedder(args.embedder, args.embedding_model, args.device)
    config = RouterConfig(
        merge_similarity=args.merge_similarity,
        routing_similarity=args.routing_similarity,
        admission_threshold=args.admission_threshold,
        minimum_observations=args.minimum_observations,
        audit_probability=0.0,
    )
    router, seed_info = build_router(seed_rows, inventories["lab"], embedder, config, teacher)

    logs: list[dict[str, Any]] = []
    for row in stream:
        payload = row["input"]
        raw_ref, checked_ref, repaired = slm_reference(router, teacher, payload)
        try:
            result = router.route(payload, evidence_id=row["id"], session_id=row["id"])
            decision = result.decision
            error = None
        except Exception as exc:  # should not happen; logged, never hidden
            result = None
            decision = None
            error = f"{type(exc).__name__}: {exc}"
        ref_item = row.get("reference_item_id")
        ref_action = row.get("reference_action")
        served_item = decision.get("resolved_item_id") if decision else None
        served_action = decision.get("action") if decision else None
        served_item_ok = None
        if decision is not None and ref_item is not None and row["origin"] == "real_failure_trial":
            served_item_ok = served_item in (ref_item, None) if served_action in {"clarify", "not_found", "redirect", "cancel"} else served_item == ref_item
        elif decision is not None and served_item is not None and ref_item is None and row["origin"] == "real_failure_trial":
            served_item_ok = False  # an item was bound although the request names none
        served_action_ok = (served_action == ref_action) if (decision is not None and ref_action) else None
        failure_types = row.get("failure_types") or []
        served_failure_ok = None
        slm_failure_ok = None
        if row["kind"] == "review" and failure_types:
            served_failure_ok = bool(decision) and decision.get("failure_type") in failure_types
            slm_failure_ok = bool(checked_ref) and checked_ref.get("failure_type") in failure_types
        preferences = row.get("preferences") or []
        served_strategy_ok = None
        if row["kind"] == "review" and preferences and decision is not None:
            served_strategy_ok = decision.get("recovery_strategy") in preferences
        active = sum(p.status == "active" for p in router.patterns)
        logs.append({
            "position": row["position"],
            "id": row["id"],
            "kind": row["kind"],
            "origin": row["origin"],
            "category": row.get("category"),
            "participant_id": row.get("participant_id"),
            "slm_split": row.get("slm_split"),
            "setting": row.get("setting"),
            "failure_types": failure_types,
            "source": result.source if result else "error",
            "similarity": result.similarity if result else None,
            "pattern_id": result.pattern_id if result else None,
            "typed_text": result.typed_text if result else None,
            "reason": result.reason if result else None,
            "served_decision": decision,
            "served_response": result.response if result else None,
            "slm_raw_decision": raw_ref,
            "slm_checked_decision": checked_ref,
            "guard_repaired": repaired,
            "fsm_matches_checked_slm": (decision_dict(decision) == decision_dict(checked_ref))
            if (result and result.source == "fsm" and decision and checked_ref) else None,
            "fsm_matches_raw_slm": (decision_dict(decision) == raw_ref)
            if (result and result.source == "fsm" and decision and raw_ref) else None,
            "fsm_task_agreement": (
                decision.get("action") == checked_ref.get("action")
                and decision.get("resolved_item_id") == checked_ref.get("resolved_item_id")
            ) if (result and result.source == "fsm" and decision and checked_ref) else None,
            "reference_item_id": ref_item,
            "reference_action": ref_action,
            "served_item_ok": served_item_ok,
            "served_action_ok": served_action_ok,
            "served_failure_type_ok": served_failure_ok,
            "slm_failure_type_ok": slm_failure_ok,
            "served_strategy_ok": served_strategy_ok,
            "active_patterns": active,
            "candidate_patterns": len(router.patterns) - active,
            "error": error,
        })
        if not args.quiet and row["position"] % 25 == 0:
            recent = logs[-args.window:]
            share = sum(l["source"] == "fsm" for l in recent) / len(recent)
            print(f"turn {row['position']:4d}  rolling FSM share {share:5.1%}  active patterns {active}")

    metrics = {
        "config": {
            "embedder": embedder.name,
            "merge_similarity": args.merge_similarity,
            "routing_similarity": args.routing_similarity,
            "admission_threshold": args.admission_threshold,
            "minimum_observations": args.minimum_observations,
            "seeded": not args.no_seed,
            "window": args.window,
        },
        "seed": seed_info,
        "summary": summarize(logs, args.window),
        "router": router.summary(),
        "teacher_cache_records": len(cache_records),
        "teacher_cache_misses": teacher.misses,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "turn_log.jsonl").open("w", encoding="utf-8") as handle:
        for log in logs:
            handle.write(json.dumps(log, ensure_ascii=False) + "\n")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.out / "snapshot.json").write_text(json.dumps(router.snapshot(), indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
    print(json.dumps(metrics["summary"], indent=2))


if __name__ == "__main__":
    main()
