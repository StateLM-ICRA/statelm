#!/usr/bin/env python3
"""Build the seed set (FSM_0 data) and the deployment stream from real trials.

Inputs (all in ``data/``):
  original_cases.json      118 reviewed real cases (25 retrieval, 93 recovery),
                           sha256 3122ac1c..., the file the SLM was trained from
  success_corpus.json      41 success trials with speaker roles (FSM_0 snapshot)
  inventory_by_setting.json lab (24 items) and hospital (21 items) inventories,
                           identical to the SLM training inventory
  augmented_requests_v1.jsonl 36 LLM-written requests for the three behaviours
                           the transcripts do not contain (clarify, absent item,
                           irrelevant); kept separate and labelled
  slm_folds.json           participant folds of the SLM run (fold 0 deployed)

Outputs (``--out`` directory):
  seed.jsonl     success-labelled examples that build FSM_0 (never streamed)
  stream.jsonl   the ordered deployment stream (request + review turn per
                 failure trial, augmented rows interleaved)
  build_summary.json

Every payload is built with the SLM bundle's own ``make_payload`` semantics
(empty memory, inventory candidates = full cart inventory, review mode with
``allowed_actions = [locate, recover]``), exactly as in SLM training.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statelm_realstream.contract import empty_memory, make_payload  # noqa: E402
from statelm_realstream.features import extract_text_features, normal_text  # noqa: E402

DRAWER_RE = re.compile(r"\bdrawer\b", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def trial_number(trial_id: str) -> int:
    match = re.search(r"trial_(\d+)", trial_id)
    return int(match.group(1)) if match else 0


def map_roles(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """participant -> user; operator and robot -> robot (as in SLM training)."""

    return [
        {"role": "user" if turn["role"] == "participant" else "robot", "text": turn["text"]}
        for turn in history
    ]


def resolve_item(text: str, inventory: list[dict[str, Any]]) -> str | None:
    features = extract_text_features(text, inventory)
    return features.item_ids[0] if len(features.item_ids) == 1 else None


def review_payload(history, inventory, proposal, wait_seconds=None):
    timing = {"last_response_latency_ms": None}
    if isinstance(wait_seconds, (int, float)):
        timing["last_response_latency_ms"] = round(float(wait_seconds) * 1000, 3)
    return make_payload(history, empty_memory(), inventory, allowed_actions=["locate", "recover"],
                        proposed_robot_answer=proposal, timing=timing)


def request_payload(history, inventory):
    return make_payload(history, empty_memory(), inventory)


def locate_decision(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "locate",
        "item_family": item["family"],
        "resolved_item_id": item["id"],
        "attributes": copy.deepcopy(item["attributes"]),
        "missing_attributes": [],
        "failure_detected": False,
        "failure_type": "none",
        "recovery_strategy": "none",
    }


def build_seed(cases, corpus, inventories) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Success-labelled examples for FSM_0: request -> locate, correct proposal -> approve."""

    retrieval = {c["trial_id"]: c for c in cases if c["task"] == "retrieval"}
    alias_to_trial = {}
    for trial in corpus:
        for alias in trial.get("sample_aliases", []):
            alias_to_trial[alias] = trial
    seeds: list[dict[str, Any]] = []
    audit = Counter()
    for trial in sorted(corpus, key=lambda t: (int(t["participant_id"]), t["trial_index"])):
        setting = trial["setting"]
        if setting not in inventories:
            audit["unknown_setting"] += 1
            continue
        inventory = inventories[setting]
        by_id = {item["id"]: item for item in inventory}
        # Prefer the reviewed retrieval case when it exists.
        reviewed = None
        for case in retrieval.values():
            if set(case["provenance"].get("sample_aliases", [])) & set(trial.get("sample_aliases", [])):
                reviewed = case
                break
        request_text = None
        proposal = None
        item_id = None
        if reviewed is not None:
            history = map_roles(reviewed["input"]["history"])
            request_text = history[-1]["text"] if history and history[-1]["role"] == "user" else None
            item_id = reviewed["targets"]["item_id"]
        if request_text is None:
            candidates = trial.get("candidates", [])
            preferred = [c for c in candidates if c.get("kind") == "provided_role_archive"] or candidates
            for candidate in preferred:
                turns = candidate["turns"]
                for index, turn in enumerate(turns):
                    if turn["role"] == "robot" and DRAWER_RE.search(turn["text"]):
                        previous = [t for t in turns[:index] if t["role"] == "participant"]
                        if previous:
                            request_text = previous[-1]["text"]
                            proposal = turn["text"]
                        break
                if request_text:
                    break
            if request_text and item_id is None:
                item_id = resolve_item(request_text, inventory)
        if not request_text or item_id not in by_id:
            audit["no_resolved_request"] += 1
            continue
        if resolve_item(request_text, inventory) != item_id:
            # The reviewed target must be supported by the request wording,
            # otherwise the seed could not be instantiated by the FSM anyway.
            audit["request_does_not_resolve_target"] += 1
            continue
        item = by_id[item_id]
        history = [{"role": "user", "text": request_text}]
        seeds.append({
            "id": f"seed:{trial['trial_id']}:request",
            "trial_id": trial["trial_id"],
            "participant_id": str(trial["participant_id"]),
            "setting": setting,
            "kind": "request",
            "input": request_payload(history, inventory),
            "decision": locate_decision(item),
        })
        audit["request_seeds"] += 1
        if proposal is None:
            # Take the robot's drawer line from the role-tagged archive.
            for candidate in trial.get("candidates", []):
                for turn in candidate["turns"]:
                    if turn["role"] == "robot" and DRAWER_RE.search(turn["text"]):
                        proposal = turn["text"]
                        break
                if proposal:
                    break
        if proposal:
            seeds.append({
                "id": f"seed:{trial['trial_id']}:review",
                "trial_id": trial["trial_id"],
                "participant_id": str(trial["participant_id"]),
                "setting": setting,
                "kind": "review",
                "input": review_payload(history, inventory, proposal),
                "decision": locate_decision(item),
            })
            audit["review_seeds_offered"] += 1
    return seeds, dict(audit)


def build_stream(cases, inventories, augmented, folds) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold0 = folds[0]
    held_out = set(fold0["test_participants"])
    validation = set(fold0["validation_participants"])
    recovery = [c for c in cases if c["task"] == "recovery"]
    recovery.sort(key=lambda c: (int(c["participant_id"]), trial_number(c["trial_id"])))
    real_rows: list[list[dict[str, Any]]] = []  # one list (request, review) per trial
    audit = Counter()
    for case in recovery:
        setting = case["input"]["cart"]
        inventory = inventories[setting]
        history = map_roles(case["input"]["history"])
        proposed_turns = []
        while history and history[-1]["role"] == "robot":
            proposed_turns.append(history.pop()["text"])
        proposal = " ".join(reversed(proposed_turns)).strip() or None
        if not history or history[-1]["role"] != "user" or proposal is None:
            audit["no_user_decision_boundary"] += 1
            continue
        pid = str(case["participant_id"])
        split = "slm_test" if pid in held_out else "slm_validation" if pid in validation else "slm_train"
        item_id = resolve_item(history[-1]["text"], inventory)
        base = {
            "trial_id": case["trial_id"],
            "participant_id": pid,
            "slm_split": split,
            "setting": setting,
            "origin": "real_failure_trial",
            "failure_types": list(case["targets"].get("failure_types", [])),
            "preferences": list(case["targets"].get("preferences", [])),
            "reference_item_id": item_id,
            "proposal_text": proposal,
        }
        request_row = {
            **base,
            "id": f"{case['case_id']}:request",
            "kind": "request",
            "input": request_payload(history, inventory),
            "reference_action": "locate" if item_id else None,
        }
        review_row = {
            **base,
            "id": f"{case['case_id']}:review",
            "kind": "review",
            "input": review_payload(history, inventory, proposal, case["input"].get("observed_wait_seconds")),
            "reference_action": "recover",
        }
        real_rows.append([request_row, review_row])
        audit["trials"] += 1
    # Interleave the augmented rows evenly across the trial sequence.
    aug_rows = []
    for row in augmented:
        inventory = inventories[row["setting"]]
        aug_rows.append({
            "id": row["id"],
            "trial_id": None,
            "participant_id": None,
            "slm_split": "augmented",
            "setting": row["setting"],
            "origin": "augmented_v1_llm_written",
            "category": row["category"],
            "failure_types": [],
            "preferences": [],
            "reference_item_id": None,
            "proposal_text": None,
            "kind": "request",
            "input": request_payload([{"role": "user", "text": row["text"]}], inventory),
            "reference_action": row["expected_action"],
            "reference_family": row.get("expected_family"),
        })
    stream: list[dict[str, Any]] = []
    n_trials = len(real_rows)
    n_aug = len(aug_rows)
    aug_index = 0
    for trial_index, pair in enumerate(real_rows):
        stream.extend(pair)
        # place augmented row k after trial floor((k+1) * n_trials / (n_aug+1))
        while aug_index < n_aug and trial_index + 1 >= ((aug_index + 1) * n_trials) // (n_aug + 1):
            stream.append(aug_rows[aug_index])
            aug_index += 1
    while aug_index < n_aug:
        stream.append(aug_rows[aug_index])
        aug_index += 1
    for position, row in enumerate(stream, start=1):
        row["position"] = position
    audit["stream_rows"] = len(stream)
    audit["augmented_rows"] = n_aug
    return stream, dict(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    cases = json.loads((args.data / "original_cases.json").read_text())["cases"]
    corpus = json.loads((args.data / "success_corpus.json").read_text())
    inventories = json.loads((args.data / "inventory_by_setting.json").read_text())
    augmented = read_jsonl(args.data / "augmented_requests_v1.jsonl")
    folds = json.loads((args.data / "slm_folds.json").read_text())
    seeds, seed_audit = build_seed(cases, corpus, inventories)
    stream, stream_audit = build_stream(cases, inventories, augmented, folds)
    write_jsonl(args.out / "seed.jsonl", seeds)
    write_jsonl(args.out / "stream.jsonl", stream)
    summary = {
        "seed": seed_audit,
        "stream": stream_audit,
        "stream_by_kind": dict(Counter(r["kind"] for r in stream)),
        "stream_by_origin": dict(Counter(r["origin"] for r in stream)),
        "stream_by_slm_split": dict(Counter(r["slm_split"] for r in stream)),
        "review_failure_types": dict(Counter(
            "+".join(r["failure_types"]) for r in stream if r["kind"] == "review")),
        "unresolved_request_items": sum(
            1 for r in stream if r["origin"] == "real_failure_trial" and r["kind"] == "request"
            and r["reference_item_id"] is None),
        "participants_in_stream": len({r["participant_id"] for r in stream if r["participant_id"]}),
    }
    (args.out / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
