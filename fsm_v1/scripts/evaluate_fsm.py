#!/usr/bin/env python3
"""Run the frozen FSM against the quarantined held-out conversations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fsm_only import CompiledFSM, Inventory, InventorySession


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _inventory_records(path: Path, setting: str, overrides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = document["items"] if isinstance(document, dict) else document
    selected = [copy.deepcopy(row) for row in records if row.get("setting") == setting]
    for override in overrides:
        matches = [
            row for row in selected
            if row.get("setting") == override.get("setting")
            and row.get("source_item_id") == override.get("item_id")
        ]
        if len(matches) != 1:
            raise ValueError(f"override did not identify one runtime inventory row: {override!r}")
        field = override.get("field")
        if field in {"drawer", "location"}:
            matches[0]["location"] = override.get("value")
        else:
            matches[0][str(field)] = override.get("value")
    return selected


def evaluate(
    model_path: str | Path,
    inventory_path: str | Path,
    heldout_path: str | Path,
) -> dict[str, Any]:
    model = CompiledFSM.load(model_path)
    inventory_file = Path(inventory_path)
    case_file = Path(heldout_path)
    cases = _read_jsonl(case_file)
    results: list[dict[str, Any]] = []
    failure_count = 0

    for case in cases:
        records = _inventory_records(
            inventory_file,
            str(case["setting"]),
            list(case.get("inventory_overrides", [])),
        )
        session = InventorySession(model, Inventory.from_records(records))
        turn_results: list[dict[str, Any]] = []
        for expected in case["turns"]:
            actual = session.handle(expected["utterance"])
            errors: list[str] = []
            if actual.action != expected["target_action"]:
                errors.append(f"action: expected {expected['target_action']}, got {actual.action}")
            expected_names = tuple(item["name"] for item in expected.get("candidate_items", []))
            if actual.matches != expected_names:
                errors.append(f"matches: expected {expected_names!r}, got {actual.matches!r}")
            expected_location = expected.get("target_location")
            if expected_location and expected_location.casefold() not in actual.text.casefold():
                errors.append(f"response omits expected location {expected_location!r}")
            if errors:
                failure_count += 1
            turn_results.append({
                "utterance": expected["utterance"],
                "expected_action": expected["target_action"],
                "actual": actual.to_dict(),
                "passed": not errors,
                "errors": errors,
            })
        results.append({
            "id": case.get("id", case.get("record_id")),
            "passed": all(turn["passed"] for turn in turn_results),
            "turns": turn_results,
        })

    turn_count = sum(len(case["turns"]) for case in results)
    parent_name = case_file.parent.name
    suite_name = f"fsm_v1_{case_file.stem}" if parent_name == "fsm_v1" else f"fsm_v1_{parent_name}_{case_file.stem}"
    return {
        "suite": suite_name,
        "cases": len(results),
        "turns": turn_count,
        "passed_cases": sum(case["passed"] for case in results),
        "passed_turns": turn_count - failure_count,
        "failed_turns": failure_count,
        "success": failure_count == 0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/fsm_v1/model.json"))
    parser.add_argument("--inventory", type=Path, default=Path("datasets/fsm_v1/inventory.json"))
    parser.add_argument("--heldout", type=Path, default=Path("datasets/fsm_v1/heldout.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = evaluate(args.model, args.inventory, args.heldout)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
