#!/usr/bin/env python3
"""Build the deterministic FSM-v1 dataset from the reviewed embedded snapshot.

This script uses only the Python standard library.  It parses (but never
executes) the notebook cell containing EMBEDDED_DATA, verifies the three
reviewed payloads against both the notebook hashes and pinned hashes, and
writes the versioned dataset under datasets/fsm_v1.
"""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
import zlib


DATASET_VERSION = "1.0.0"
RECORD_SCHEMA = "fsm-dialogue-v1"
INVENTORY_SCHEMA = "fsm-inventory-v1"
MANIFEST_SCHEMA = "fsm-manifest-v1"
SOURCE_NOTEBOOK = Path("notebooks/Robot_SLM_Adaptive_FSM_Inventory_Query(AAA).ipynb")
OUTPUT_DIRECTORY = Path("datasets/fsm_v1")
INVENTORY_REFERENCE = "datasets/fsm_v1/inventory/original_inventory.json"

PINNED_SOURCE_SHA256 = {
    "original_cases.json": "3122ac1c2190aea3ac514c6e1ea73f257c718067b9cc56ee8735620d46847410",
    "inventory.json": "2e78f7dad1115239b835aeed254e5c9c93b7bcf176edfd8acc95fa56e10ea68b",
    "folds.json": "222e8313c1fc0f5de2935698cdd7a1303e84186b696ee78c1e6097abf895abd9",
}

WAITING_FOR_QUERY = "WAITING_FOR_QUERY"
WAITING_FOR_CLARIFICATION = "WAITING_FOR_CLARIFICATION"
DONE = "DONE"

RETURN_LOCATION = "RETURN_LOCATION"
ASK_WHICH_ONE = "ASK_WHICH_ONE"
OBJECT_MISSING = "OBJECT_MISSING"
LOCATION_MISSING = "LOCATION_MISSING"
UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"

ONE_WITH_LOCATION = "ONE_WITH_LOCATION"
ONE_WITHOUT_LOCATION = "ONE_WITHOUT_LOCATION"
MULTIPLE_MATCHES = "MULTIPLE_MATCHES"
ZERO_MATCHES = "ZERO_MATCHES"
NOT_APPLICABLE = "NOT_APPLICABLE"

SCENARIOS = (
    "unique",
    "ambiguous",
    "missing_object",
    "missing_location",
    "clarification",
    "unsupported",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assignment(tree: ast.Module, name: str) -> Any:
    matches: list[Any] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(ast.literal_eval(node.value))
    _require(len(matches) == 1, f"Expected exactly one literal {name} assignment")
    return matches[0]


def load_embedded_snapshot(notebook_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Find the embedded-data cell by content, decode it, and verify all hashes."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    found: list[tuple[dict[str, str], dict[str, str]]] = []
    for cell in notebook.get("cells", []):
        source_value = cell.get("source", [])
        source = "".join(source_value) if isinstance(source_value, list) else str(source_value)
        if "EMBEDDED_DATA" not in source or "EXPECTED_SHA256" not in source:
            continue
        try:
            tree = ast.parse(source)
            embedded = _assignment(tree, "EMBEDDED_DATA")
            expected = _assignment(tree, "EXPECTED_SHA256")
        except (SyntaxError, ValueError):
            continue
        found.append((embedded, expected))

    _require(len(found) == 1, "Could not uniquely locate the EMBEDDED_DATA notebook cell")
    embedded, expected = found[0]
    _require(isinstance(embedded, dict), "EMBEDDED_DATA must be a mapping")
    _require(isinstance(expected, dict), "EXPECTED_SHA256 must be a mapping")
    _require(set(embedded) == set(PINNED_SOURCE_SHA256), "Unexpected embedded payload set")
    _require(expected == PINNED_SOURCE_SHA256, "Notebook hash pins differ from the reviewed snapshot")

    decoded: dict[str, Any] = {}
    for filename in sorted(embedded):
        payload = embedded[filename]
        _require(isinstance(payload, str), f"Embedded payload {filename} is not text")
        try:
            raw = zlib.decompress(base64.b64decode(payload, validate=True))
        except Exception as exc:  # zlib/base64 expose several exception types.
            raise ValueError(f"Cannot decode embedded payload {filename}") from exc
        actual = _sha256(raw)
        _require(actual == expected[filename], f"SHA-256 mismatch for {filename}: {actual}")
        try:
            decoded[filename] = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Embedded payload {filename} is not UTF-8 JSON") from exc
    return decoded, dict(expected)


def validate_inventory(document: Any) -> list[dict[str, Any]]:
    _require(isinstance(document, dict), "inventory.json must contain an object")
    items = document.get("items")
    _require(isinstance(items, list) and len(items) == 45, "Reviewed inventory must contain 45 items")
    seen: set[tuple[str, str]] = set()
    required = {
        "setting",
        "item_id",
        "name",
        "drawer",
        "aliases",
        "reference_drawer",
        "evidence_status",
        "success_transcript_evidence",
    }
    for number, item in enumerate(items, 1):
        _require(isinstance(item, dict), f"Inventory row {number} is not an object")
        _require(required <= set(item), f"Inventory row {number} is missing required fields")
        key = (item["setting"], item["item_id"])
        _require(all(isinstance(value, str) and value for value in key), f"Invalid inventory key at row {number}")
        _require(key not in seen, f"Duplicate inventory key: {key}")
        seen.add(key)
        _require(isinstance(item["drawer"], int) and item["drawer"] > 0, f"Invalid drawer for {key}")
        _require(item["reference_drawer"] == item["drawer"], f"Drawer evidence mismatch for {key}")
        _require(isinstance(item["aliases"], list) and item["aliases"], f"Missing aliases for {key}")
        _require(all(isinstance(alias, str) and alias for alias in item["aliases"]), f"Invalid alias for {key}")
    return items


def _runtime_family_and_attributes(item: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Derive finite lookup selectors without changing source inventory facts."""
    item_id = item["item_id"]
    name = item["name"].casefold()
    attributes: dict[str, str] = {}
    if "pencil" in name:
        family = "pencil"
        color = name.split()[0]
        if color in {"blue", "orange", "red", "yellow"}:
            attributes["color"] = color
    elif item_id in {"blue_tape", "white_tape", "medical_tape"}:
        family = "tape"
        color = name.split()[0]
        if color in {"blue", "white"}:
            attributes["color"] = color
        elif "medical" in name:
            attributes["type"] = "medical"
    elif "syringe" in name:
        family = "syringe"
        if item_id == "6_ml_syringe":
            attributes["size"] = "6 ml"
        elif item_id == "1_ml_tuberculin_syringe":
            attributes.update({"size": "1 ml", "type": "tuberculin"})
        elif item_id == "line_draw_arterial_blood_sample_syringe":
            attributes["type"] = "blood sample"
    elif "marker" in name:
        family = "marker"
        if name.startswith("blue "):
            attributes["color"] = "blue"
    elif "multimeter" in name:
        family = "multimeter"
        if name.startswith("orange "):
            attributes["color"] = "orange"
    elif "glove" in name:
        family = "gloves"
        if "surgical" in name:
            attributes["type"] = "surgical"
    elif "mask" in name:
        family = "mask"
    else:
        family = item_id.replace("_", " ")
    return family, attributes


def build_runtime_inventory(items: list[dict[str, Any]], source_hash: str) -> dict[str, Any]:
    """Create runtime rows with globally unique IDs and retained source provenance."""
    runtime_items: list[dict[str, Any]] = []
    for item in items:
        family, attributes = _runtime_family_and_attributes(item)
        aliases = list(item["aliases"])
        if family in {"pencil", "syringe", "tape"}:
            aliases.extend((family, f"{family}s"))
        aliases = list(dict.fromkeys(aliases))
        runtime_items.append({
            "id": f"{item['setting']}:{item['item_id']}",
            "source_item_id": item["item_id"],
            "setting": item["setting"],
            "name": item["name"],
            "aliases": aliases,
            "family": family,
            "attributes": attributes,
            "location": _location(item),
            "evidence_status": item["evidence_status"],
            "source_inventory_sha256": source_hash,
        })
    ids = [row["id"] for row in runtime_items]
    _require(len(ids) == len(set(ids)), "Runtime inventory IDs are not globally unique")
    return {
        "schema_version": "fsm-runtime-inventory-v1",
        "source_sha256": source_hash,
        "items": runtime_items,
    }


def _item_ref(item: dict[str, Any]) -> dict[str, str]:
    return {"setting": item["setting"], "item_id": item["item_id"], "name": item["name"]}


def _location(item: dict[str, Any]) -> str:
    return f"drawer {item['drawer']}"


def _turn(
    *,
    index: int,
    state_before: str,
    utterance: str,
    symbols: list[str],
    observation: str,
    candidates: Iterable[dict[str, Any]],
    action: str,
    state_after: str,
    target_item: dict[str, Any] | None = None,
    target_location: str | None = None,
    options: Iterable[dict[str, Any]] = (),
    missing_field: str | None = None,
    requested_object: str | None = None,
) -> dict[str, Any]:
    return {
        "turn_index": index,
        "state_before": state_before,
        "utterance": utterance,
        "symbols": symbols,
        "lookup_observation": observation,
        "candidate_items": [_item_ref(item) for item in candidates],
        "target_action": action,
        "state_after": state_after,
        "target_item": _item_ref(target_item) if target_item else None,
        "target_location": target_location,
        "clarification_options": [_item_ref(item) for item in options],
        "missing_field": missing_field,
        "requested_object": requested_object,
    }


def _record(
    *,
    record_id: str,
    source_kind: str,
    split: str,
    scenario: str,
    setting: str,
    turns: list[dict[str, Any]],
    inventory_overrides: list[dict[str, Any]] | None = None,
    source_context: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA,
        "record_id": record_id,
        "source_kind": source_kind,
        "split": split,
        "scenario": scenario,
        "setting": setting,
        "inventory_ref": INVENTORY_REFERENCE,
        "inventory_overrides": inventory_overrides or [],
        "turns": turns,
        "source_context": source_context,
        "provenance": provenance or {},
    }


def build_original_records(
    cases_document: Any,
    folds_document: Any,
    inventory_index: dict[tuple[str, str], dict[str, Any]],
    source_hash: str,
) -> dict[str, list[dict[str, Any]]]:
    _require(isinstance(cases_document, dict), "original_cases.json must contain an object")
    cases = cases_document.get("cases")
    _require(isinstance(cases, list) and len(cases) == 118, "Expected 118 reviewed cases")
    retrieval = [case for case in cases if case.get("task") == "retrieval"]
    _require(len(retrieval) == 25, "Expected exactly 25 reviewed retrieval cases")
    _require(all(case.get("suite") == "original" and case.get("synthetic") is False for case in retrieval),
             "Retrieval subset contains a non-original case")

    _require(isinstance(folds_document, list), "folds.json must contain a list")
    fold_matches = [fold for fold in folds_document if fold.get("fold") == 0]
    _require(len(fold_matches) == 1, "Expected exactly one fold 0")
    fold = fold_matches[0]
    participant_to_split: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        participants = fold.get(f"{split}_participants")
        _require(isinstance(participants, list), f"Fold 0 has no {split} participant list")
        for participant in participants:
            _require(participant not in participant_to_split, f"Participant {participant} crosses fold splits")
            participant_to_split[participant] = split

    rows = {split: [] for split in ("train", "validation", "test")}
    for case in retrieval:
        participant = case.get("participant_id")
        _require(participant in participant_to_split, f"Participant {participant} is absent from fold 0")
        split = participant_to_split[participant]
        input_value = case.get("input", {})
        history = input_value.get("history")
        _require(isinstance(history, list) and history, f"Retrieval case {case.get('case_id')} has no history")
        _require(all(turn.get("role") == "participant" and isinstance(turn.get("text"), str) for turn in history),
                 f"Retrieval case {case.get('case_id')} has invalid source turns")
        target = case.get("targets", {})
        _require(target.get("acceptable_actions") == ["locate"], f"Unexpected action target in {case.get('case_id')}")
        key = (input_value.get("cart"), target.get("item_id"))
        _require(key in inventory_index, f"Retrieval target is absent from inventory: {key}")
        item = inventory_index[key]
        _require(target.get("drawer") == item["drawer"], f"Retrieval drawer disagrees with inventory: {key}")
        turn = _turn(
            index=0,
            state_before=WAITING_FOR_QUERY,
            utterance=history[-1]["text"],
            symbols=["FIND_LOCATION", "ITEM_MENTION"],
            observation=ONE_WITH_LOCATION,
            candidates=[item],
            action=RETURN_LOCATION,
            state_after=DONE,
            target_item=item,
            target_location=_location(item),
        )
        rows[split].append(_record(
            record_id=f"original-derived:{case['case_id']}",
            source_kind="original_derived",
            split=split,
            scenario="unique",
            setting=item["setting"],
            turns=[turn],
            source_context={
                "case_id": case["case_id"],
                "trial_id": case["trial_id"],
                "participant_id": participant,
                "history": history,
                "source_provenance": case.get("provenance", {}),
            },
            provenance={
                "embedded_file": "original_cases.json",
                "embedded_sha256": source_hash,
                "derivation": "reviewed retrieval target mapped to a single FSM location transition; source text unchanged",
                "fold": 0,
            },
        ))

    for split in rows:
        rows[split].sort(key=lambda row: row["record_id"])
    counts = {split: len(split_rows) for split, split_rows in rows.items()}
    _require(counts == {"train": 16, "validation": 4, "test": 5}, f"Unexpected retrieval split counts: {counts}")
    return rows


AMBIGUITY_GROUPS = {
    "tape": ("lab", ("blue_tape", "white_tape")),
    "pencil": ("lab", ("orange_pencils", "yellow_pencils", "red_pencils", "blue_pencils")),
    "syringe": ("hospital", ("line_draw_arterial_blood_sample_syringe", "1_ml_tuberculin_syringe", "6_ml_syringe")),
}

UNIQUE_SPECS = {
    "train": [
        ("lab", "measuring_tape", "Where can I find the measuring tape?"),
        ("lab", "circuit_board", "Please tell me where the circuit board is."),
        ("lab", "hand_sanitizer", "Locate the hand sanitizer for me."),
        ("hospital", "adenosine", "Which drawer contains Adenosine?"),
        ("hospital", "face_masks", "Please find the face masks."),
        ("hospital", "medical_tape", "Show me the location of the medical tape."),
    ],
    "validation": [
        ("lab", "wires", "Please point me to the wires."),
        ("lab", "book", "Which drawer stores the book?"),
        ("hospital", "blanket", "Can you locate the blanket?"),
    ],
    "test": [
        ("lab", "playstation_controller", "I am looking for the PlayStation controller."),
        ("hospital", "naloxone", "Tell me where Naloxone is."),
        ("hospital", "razor", "Please direct me to the razor."),
    ],
}

AMBIGUOUS_SPECS = {
    "train": [("tape", "Which drawer has the tape?"), ("pencil", "Please find a pencil for me."),
              ("syringe", "Where is the syringe?"), ("tape", "Can you locate some tape?"),
              ("pencil", "I need a pencil."), ("syringe", "Please point me to a syringe.")],
    "validation": [("tape", "Help me find tape."), ("pencil", "Where do you keep pencils?"),
                   ("syringe", "Which drawer contains a syringe?")],
    "test": [("tape", "Locate the tape."), ("pencil", "Can you find a pencil?"),
             ("syringe", "I am looking for a syringe.")],
}

CLARIFICATION_SPECS = {
    "train": [
        ("tape", "Could you find tape?", "White", "white_tape"),
        ("pencil", "I am looking for a pencil.", "Red", "red_pencils"),
        ("syringe", "Can you locate a syringe?", "6 mL", "6_ml_syringe"),
        ("tape", "Tell me where tape is.", "Blue tape", "blue_tape"),
        ("pencil", "Which drawer has a pencil?", "Orange", "orange_pencils"),
        ("syringe", "Please find a syringe.", "Tuberculin", "1_ml_tuberculin_syringe"),
    ],
    "validation": [
        ("tape", "Can you guide me to tape?", "White tape", "white_tape"),
        ("pencil", "Help me locate a pencil.", "Yellow", "yellow_pencils"),
        ("syringe", "I need a syringe.", "Blood sample", "line_draw_arterial_blood_sample_syringe"),
    ],
    "test": [
        ("tape", "Where do you store tape?", "The blue tape", "blue_tape"),
        ("pencil", "Point me to a pencil.", "The red one", "red_pencils"),
        ("syringe", "Where do you keep syringes?", "The 6 mL one", "6_ml_syringe"),
    ],
}

MISSING_OBJECT_SPECS = {
    "train": [("lab", "stapler"), ("lab", "hammer"), ("lab", "flashlight"),
              ("hospital", "thermometer"), ("hospital", "notebook"), ("hospital", "gauze")],
    "validation": [("lab", "ruler"), ("lab", "wrench"), ("hospital", "clipboard")],
    "test": [("hospital", "forceps"), ("hospital", "bandage"), ("lab", "stethoscope")],
}

MISSING_LOCATION_SPECS = {
    "train": [("lab", "gloves"), ("lab", "calculator"), ("hospital", "lidocaine"),
              ("hospital", "face_masks"), ("lab", "wipes"), ("hospital", "blanket")],
    "validation": [("lab", "leds"), ("hospital", "metoprolol"), ("lab", "pliers")],
    "test": [("lab", "mouse"), ("hospital", "scissors"), ("hospital", "nitroglycerin")],
}

UNSUPPORTED_SPECS = {
    "train": ["What time is it?", "Tell me a joke.", "Play some music.", "What is the weather?",
              "Count to ten.", "Who won the game?"],
    "validation": ["Set a timer for five minutes.", "Translate hello into French.", "Write me a poem."],
    "test": ["Call my phone.", "What is two plus two?", "Turn off the lights."],
}


def _group_items(name: str, index: dict[tuple[str, str], dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    setting, item_ids = AMBIGUITY_GROUPS[name]
    return setting, [index[(setting, item_id)] for item_id in item_ids]


def build_generated_records(index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {split: [] for split in ("train", "validation", "test")}
    for split in result:
        scenario_number = Counter()

        def add(scenario: str, setting: str, turns: list[dict[str, Any]], overrides: list[dict[str, Any]] | None = None) -> None:
            scenario_number[scenario] += 1
            result[split].append(_record(
                record_id=f"generated:{split}:{scenario}:{scenario_number[scenario]:03d}",
                source_kind="generated",
                split=split,
                scenario=scenario,
                setting=setting,
                turns=turns,
                inventory_overrides=overrides,
                provenance={
                    "generator": "scripts/build_fsm_dataset.py",
                    "generator_version": DATASET_VERSION,
                    "counterfactual": scenario in {"missing_object", "missing_location"},
                    "original_participant_text_used": False,
                },
            ))

        for setting, item_id, utterance in UNIQUE_SPECS[split]:
            item = index[(setting, item_id)]
            add("unique", setting, [_turn(index=0, state_before=WAITING_FOR_QUERY, utterance=utterance,
                symbols=["FIND_LOCATION", "ITEM_MENTION"], observation=ONE_WITH_LOCATION,
                candidates=[item], action=RETURN_LOCATION, state_after=DONE,
                target_item=item, target_location=_location(item))])

        for group_name, utterance in AMBIGUOUS_SPECS[split]:
            setting, items = _group_items(group_name, index)
            add("ambiguous", setting, [_turn(index=0, state_before=WAITING_FOR_QUERY, utterance=utterance,
                symbols=["FIND_LOCATION", "FAMILY_MENTION"], observation=MULTIPLE_MATCHES,
                candidates=items, action=ASK_WHICH_ONE, state_after=WAITING_FOR_CLARIFICATION,
                options=items, requested_object=group_name)])

        for position, (setting, object_name) in enumerate(MISSING_OBJECT_SPECS[split], 1):
            templates = ("Where can I find the {name}?", "Please locate the {name}.", "Which drawer has the {name}?")
            utterance = templates[(position - 1) % len(templates)].format(name=object_name)
            add("missing_object", setting, [_turn(index=0, state_before=WAITING_FOR_QUERY, utterance=utterance,
                symbols=["FIND_LOCATION", "UNKNOWN_OBJECT_MENTION"], observation=ZERO_MATCHES,
                candidates=[], action=OBJECT_MISSING, state_after=DONE, requested_object=object_name)])

        for setting, item_id in MISSING_LOCATION_SPECS[split]:
            item = index[(setting, item_id)]
            utterance = f"Where is the {item['name']}?"
            override = {"setting": setting, "item_id": item_id, "field": "drawer", "value": None}
            add("missing_location", setting, [_turn(index=0, state_before=WAITING_FOR_QUERY,
                utterance=utterance, symbols=["FIND_LOCATION", "ITEM_MENTION"],
                observation=ONE_WITHOUT_LOCATION, candidates=[item], action=LOCATION_MISSING,
                state_after=DONE, target_item=item, missing_field="location")], [override])

        for group_name, query, clarification, target_id in CLARIFICATION_SPECS[split]:
            setting, items = _group_items(group_name, index)
            target_item = index[(setting, target_id)]
            add("clarification", setting, [
                _turn(index=0, state_before=WAITING_FOR_QUERY, utterance=query,
                      symbols=["FIND_LOCATION", "FAMILY_MENTION"], observation=MULTIPLE_MATCHES,
                      candidates=items, action=ASK_WHICH_ONE, state_after=WAITING_FOR_CLARIFICATION,
                      options=items, requested_object=group_name),
                _turn(index=1, state_before=WAITING_FOR_CLARIFICATION, utterance=clarification,
                      symbols=["CLARIFICATION_VALUE"], observation=ONE_WITH_LOCATION,
                      candidates=[target_item], action=RETURN_LOCATION, state_after=DONE,
                      target_item=target_item, target_location=_location(target_item), requested_object=group_name),
            ])

        for utterance in UNSUPPORTED_SPECS[split]:
            add("unsupported", "lab", [_turn(index=0, state_before=WAITING_FOR_QUERY, utterance=utterance,
                symbols=["UNSUPPORTED_INTENT"], observation=NOT_APPLICABLE, candidates=[],
                action=UNSUPPORTED_REQUEST, state_after=DONE)])

        result[split].sort(key=lambda row: row["record_id"])
        counts = Counter(row["scenario"] for row in result[split])
        expected_per_scenario = 6 if split == "train" else 3
        _require(counts == Counter({scenario: expected_per_scenario for scenario in SCENARIOS}),
                 f"Generated {split} scenarios are unbalanced: {dict(counts)}")
    return result


def build_heldout(index: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    _, tape_items = _group_items("tape", index)
    blue_tape = index[("lab", "blue_tape")]
    sd_cards = index[("lab", "sd_cards")]
    multimeter = index[("lab", "orange_multimeter")]
    return [
        _record(
            record_id="heldout:clarification:tape-blue",
            source_kind="heldout",
            split="heldout",
            scenario="clarification",
            setting="lab",
            turns=[
                _turn(index=0, state_before=WAITING_FOR_QUERY, utterance="I need tape",
                      symbols=["FIND_LOCATION", "FAMILY_MENTION"], observation=MULTIPLE_MATCHES,
                      candidates=tape_items, action=ASK_WHICH_ONE, state_after=WAITING_FOR_CLARIFICATION,
                      options=tape_items, requested_object="tape"),
                _turn(index=1, state_before=WAITING_FOR_CLARIFICATION, utterance="Blue",
                      symbols=["CLARIFICATION_VALUE"], observation=ONE_WITH_LOCATION,
                      candidates=[blue_tape], action=RETURN_LOCATION, state_after=DONE,
                      target_item=blue_tape, target_location=_location(blue_tape), requested_object="tape"),
            ],
            provenance={"purpose": "untouched acceptance example", "excluded_from_splits": True},
        ),
        _record(
            record_id="heldout:missing-object:pen",
            source_kind="heldout",
            split="heldout",
            scenario="missing_object",
            setting="lab",
            turns=[_turn(index=0, state_before=WAITING_FOR_QUERY, utterance="I need a pen",
                         symbols=["FIND_LOCATION", "UNKNOWN_OBJECT_MENTION"], observation=ZERO_MATCHES,
                         candidates=[], action=OBJECT_MISSING, state_after=DONE, requested_object="pen")],
            provenance={"purpose": "untouched acceptance example", "excluded_from_splits": True},
        ),
        _record(
            record_id="heldout:unique:sd-card",
            source_kind="heldout",
            split="heldout",
            scenario="unique",
            setting="lab",
            turns=[_turn(index=0, state_before=WAITING_FOR_QUERY,
                         utterance="Where is the SD card?",
                         symbols=["FIND_LOCATION", "ITEM_MENTION"], observation=ONE_WITH_LOCATION,
                         candidates=[sd_cards], action=RETURN_LOCATION, state_after=DONE,
                         target_item=sd_cards, target_location=_location(sd_cards))],
            provenance={"purpose": "untouched acceptance example", "excluded_from_splits": True},
        ),
        _record(
            record_id="heldout:missing-location:multimeter",
            source_kind="heldout",
            split="heldout",
            scenario="missing_location",
            setting="lab",
            inventory_overrides=[{"setting": "lab", "item_id": "orange_multimeter", "field": "drawer", "value": None}],
            turns=[_turn(index=0, state_before=WAITING_FOR_QUERY,
                         utterance="Where can I find the orange multimeter?",
                         symbols=["FIND_LOCATION", "ITEM_MENTION"], observation=ONE_WITHOUT_LOCATION,
                         candidates=[multimeter], action=LOCATION_MISSING, state_after=DONE,
                         target_item=multimeter, missing_field="location")],
            provenance={"purpose": "untouched acceptance example", "excluded_from_splits": True},
        ),
    ]


def build_similarity_evaluation(
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Create generated, non-training cases for the cosine fallback."""

    sd_cards = index[("lab", "sd_cards")]
    wires = index[("lab", "wires")]
    _, tape_items = _group_items("tape", index)
    _, pencil_items = _group_items("pencil", index)

    def unique(record_id: str, split: str, utterance: str, item: dict[str, Any]) -> dict[str, Any]:
        return _record(
            record_id=record_id,
            source_kind="generated_similarity",
            split=split,
            scenario="unique",
            setting="lab",
            turns=[_turn(
                index=0,
                state_before=WAITING_FOR_QUERY,
                utterance=utterance,
                symbols=["SIMILAR_QUERY", "ITEM_MENTION"],
                observation=ONE_WITH_LOCATION,
                candidates=[item],
                action=RETURN_LOCATION,
                state_after=DONE,
                target_item=item,
                target_location=_location(item),
            )],
            provenance={
                "generator": "scripts/build_fsm_dataset.py",
                "purpose": "cosine-similarity evaluation only",
                "excluded_from_training": True,
            },
        )

    def ambiguous(
        record_id: str,
        split: str,
        utterance: str,
        family: str,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return _record(
            record_id=record_id,
            source_kind="generated_similarity",
            split=split,
            scenario="ambiguous",
            setting="lab",
            turns=[_turn(
                index=0,
                state_before=WAITING_FOR_QUERY,
                utterance=utterance,
                symbols=["SIMILAR_QUERY", "FAMILY_MENTION"],
                observation=MULTIPLE_MATCHES,
                candidates=items,
                action=ASK_WHICH_ONE,
                state_after=WAITING_FOR_CLARIFICATION,
                options=items,
                requested_object=family,
            )],
            provenance={
                "generator": "scripts/build_fsm_dataset.py",
                "purpose": "cosine-similarity evaluation only",
                "excluded_from_training": True,
            },
        )

    def missing(record_id: str, split: str, utterance: str, name: str) -> dict[str, Any]:
        return _record(
            record_id=record_id,
            source_kind="generated_similarity",
            split=split,
            scenario="missing_object",
            setting="lab",
            turns=[_turn(
                index=0,
                state_before=WAITING_FOR_QUERY,
                utterance=utterance,
                symbols=["SIMILAR_QUERY", "UNKNOWN_OBJECT_MENTION"],
                observation=ZERO_MATCHES,
                candidates=[],
                action=OBJECT_MISSING,
                state_after=DONE,
                requested_object=name,
            )],
            provenance={
                "generator": "scripts/build_fsm_dataset.py",
                "purpose": "cosine-similarity evaluation only",
                "excluded_from_training": True,
            },
        )

    def unsupported(record_id: str, split: str, utterance: str) -> dict[str, Any]:
        return _record(
            record_id=record_id,
            source_kind="generated_similarity",
            split=split,
            scenario="unsupported",
            setting="lab",
            turns=[_turn(
                index=0,
                state_before=WAITING_FOR_QUERY,
                utterance=utterance,
                symbols=["UNSUPPORTED_INTENT"],
                observation=NOT_APPLICABLE,
                candidates=[],
                action=UNSUPPORTED_REQUEST,
                state_after=DONE,
            )],
            provenance={
                "generator": "scripts/build_fsm_dataset.py",
                "purpose": "cosine-similarity evaluation only",
                "excluded_from_training": True,
            },
        )

    return {
        "validation": [
            unique("similarity:validation:article-omission", "validation", "Where is SD card?", sd_cards),
            unique("similarity:validation:entity-typo", "validation", "Where is the sd crad?", sd_cards),
            ambiguous("similarity:validation:ambiguous-family", "validation", "Where is tape?", "tape", tape_items),
            missing("similarity:validation:missing-object", "validation", "Where is item x?", "x"),
            unsupported("similarity:validation:unsupported", "validation", "What is the capital of France?"),
        ],
        "test": [
            unique("similarity:test:plural-question", "test", "Where are wires?", wires),
            unique("similarity:test:item-prefix", "test", "Where is item SD card?", sd_cards),
            ambiguous("similarity:test:ambiguous-plural", "test", "Where is item pencils?", "pencil", pencil_items),
            missing("similarity:test:missing-object", "test", "Please locate stapler for me.", "stapler"),
            unsupported("similarity:test:unsupported", "test", "Book me a flight."),
        ],
    }


def _normalized_utterance(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9%]+", text.casefold()))


def validate_no_heldout_leakage(
    original: dict[str, list[dict[str, Any]]],
    generated: dict[str, list[dict[str, Any]]],
    similarity: dict[str, list[dict[str, Any]]],
    heldout: list[dict[str, Any]],
) -> None:
    heldout_ids = {row["record_id"] for row in heldout}
    heldout_utterances = {
        _normalized_utterance(turn["utterance"])
        for row in heldout
        for turn in row["turns"]
    }
    heldout_fingerprints = {
        _canonical_json({
            "scenario": row["scenario"],
            "setting": row["setting"],
            "inventory_overrides": row["inventory_overrides"],
            "turns": row["turns"],
        })
        for row in heldout
    }
    for collection in (original, generated, similarity):
        for rows in collection.values():
            for row in rows:
                _require(row["record_id"] not in heldout_ids, f"Held-out ID leaked: {row['record_id']}")
                for turn in row["turns"]:
                    normalized = _normalized_utterance(turn["utterance"])
                    _require(normalized not in heldout_utterances,
                             f"Held-out utterance leaked into {row['record_id']}: {turn['utterance']}")
                fingerprint = _canonical_json({
                    "scenario": row["scenario"],
                    "setting": row["setting"],
                    "inventory_overrides": row["inventory_overrides"],
                    "turns": row["turns"],
                })
                _require(fingerprint not in heldout_fingerprints, f"Held-out row leaked: {row['record_id']}")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['\u2019-][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def _find_slot_value(text: str, candidates: Iterable[str]) -> str | None:
    """Return the longest candidate occurring as a complete token span."""
    haystack = _tokens(text)
    matches: list[tuple[int, str]] = []
    for candidate in candidates:
        needle = _tokens(candidate)
        if not needle or len(needle) > len(haystack):
            continue
        if any(haystack[index:index + len(needle)] == needle
               for index in range(len(haystack) - len(needle) + 1)):
            matches.append((len(needle), " ".join(needle)))
    return max(matches, default=(0, ""))[1] or None


def _query_record(
    *,
    text: str,
    object_value: str,
    split: str,
    source_kind: str,
    provenance: dict[str, Any],
    participant_id: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "query",
        "text": text,
        "slots": {"object": object_value},
        "intent": "FIND_LOCATION",
        "split": split,
        "source": source_kind,
        "source_kind": source_kind,
        "provenance": provenance,
    }
    if participant_id is not None:
        row["participant_id"] = participant_id
    return row


def build_compiler_splits(
    original: dict[str, list[dict[str, Any]]],
    generated: dict[str, list[dict[str, Any]]],
    inventory_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Adapt audit dialogues into the compiler's query/transition JSONL schema."""
    output = {split: [] for split in ("train", "validation", "test")}
    output["train"].append({
        "type": "config",
        "start_state": WAITING_FOR_QUERY,
        "entity_slot": "object",
        "templates": {
            "RETURN_LOCATION": "You can find {name} in {location}.",
            "UNSUPPORTED_REQUEST": "I can only help locate objects in the inventory.",
        },
        "similarity": {
            "enabled": True,
            "minimum": 0.72,
            "minimum_margin": 0.015,
            "slot_margin": 0.03,
            "max_slot_tokens": 12,
            "inventory_minimum": 0.55,
            "inventory_margin": 0.04,
        },
        "split": "train",
        "source": "generated_configuration",
    })

    # These labeled finite patterns form the generated language-coverage
    # portion of the dataset. They contain no inventory facts or held-out
    # utterances; the object slot is supplied only at runtime.
    grammar_patterns = [
        ["locate", "the", "{object}"],
        ["can", "you", "find", "a", "{object}"],
        ["could", "you", "please", "find", "a", "{object}"],
        ["where", "do", "you", "store", "{object}"],
        ["point", "me", "to", "a", "{object}"],
        ["where", "do", "you", "keep", "{object}"],
        ["i", "am", "looking", "for", "the", "{object}"],
        ["please", "direct", "me", "to", "the", "{object}"],
        ["please", "point", "me", "to", "the", "{object}"],
        ["where's", "the", "{object}"],
        ["find", "{object}"],
        ["can", "you", "give", "me", "the", "{object}"],
        ["can", "you", "guide", "me", "to", "{object}"],
        ["can", "you", "locate", "the", "{object}"],
        ["oh", "which", "door", "stores", "the", "{object}"],
        ["help", "me", "find", "a", "{object}"],
        ["help", "me", "find", "{object}"],
        ["help", "me", "locate", "a", "{object}"],
        ["which", "drawer", "contains", "a", "{object}"],
        ["which", "drawer", "stores", "the", "{object}"],
        ["{object}", "lead", "me", "to", "the", "{object}", "please"],
        ["okay", "{object}", "where", "are", "the", "{object}"],
    ]
    for number, pattern in enumerate(grammar_patterns, 1):
        output["train"].append({
            "type": "query",
            "pattern": pattern,
            "intent": "FIND_LOCATION",
            "split": "train",
            "source": "generated",
            "source_kind": "generated",
            "provenance": {
                "record_id": f"generated:language-pattern:{number:03d}",
                "generator": "scripts/build_fsm_dataset.py",
                "purpose": "finite query grammar coverage",
            },
        })

    # Original reviewed retrieval cases teach real wording. A bare slot-only
    # utterance is preserved in the audit data but excluded here because it
    # would accept every possible sentence and make unsupported requests unsafe.
    for split, rows in original.items():
        for row in rows:
            turn = row["turns"][0]
            target = turn["target_item"]
            source_item = inventory_index[(target["setting"], target["item_id"])]
            candidates = [source_item["name"], *source_item["aliases"]]
            object_value = _find_slot_value(turn["utterance"], candidates)
            _require(object_value is not None, f"No source item span in {row['record_id']}")
            if len(_tokens(turn["utterance"])) == len(_tokens(object_value)):
                continue
            context = row["source_context"]
            output[split].append(_query_record(
                text=turn["utterance"],
                object_value=object_value,
                split=split,
                source_kind="original_derived",
                participant_id=str(context["participant_id"]),
                provenance={
                    "case_id": context["case_id"],
                    "trial_id": context["trial_id"],
                    "embedded_file": row["provenance"]["embedded_file"],
                    "embedded_sha256": row["provenance"]["embedded_sha256"],
                },
            ))

    # Generated rows supply deliberately separated behavior coverage. Only the
    # first location-query turn becomes a query pattern; clarification text is
    # resolved against the finite candidate set by the inventory transducer.
    for split, rows in generated.items():
        for row in rows:
            if row["scenario"] == "unsupported":
                continue
            turn = row["turns"][0]
            requested = turn.get("requested_object")
            if requested:
                candidates = [requested, f"{requested}s"]
            else:
                target = turn.get("target_item")
                _require(target is not None, f"Generated query lacks an item: {row['record_id']}")
                source_item = inventory_index[(target["setting"], target["item_id"])]
                candidates = [source_item["name"], *source_item["aliases"]]
            object_value = _find_slot_value(turn["utterance"], candidates)
            _require(object_value is not None, f"No generated item span in {row['record_id']}")
            output[split].append(_query_record(
                text=turn["utterance"],
                object_value=object_value,
                split=split,
                source_kind="generated",
                provenance={
                    "record_id": row["record_id"],
                    "generator": "scripts/build_fsm_dataset.py",
                    "scenario": row["scenario"],
                },
            ))

    def transition(state: str, event: str, action: str, next_state: str, source: str) -> dict[str, Any]:
        return {
            "type": "transition",
            "state": state,
            "event": event,
            "action": action,
            "next_state": next_state,
            "split": "train",
            "source": "generated",
            "source_kind": "generated",
            "provenance": {"record_id": source, "generator": "scripts/build_fsm_dataset.py"},
        }

    # Learn controller support counts from the generated training dialogues.
    for row in generated["train"]:
        if row["scenario"] == "unsupported":
            output["train"].append(transition(
                WAITING_FOR_QUERY, "UNRECOGNIZED_QUERY", UNSUPPORTED_REQUEST,
                WAITING_FOR_QUERY, row["record_id"],
            ))
            continue
        output["train"].append(transition(
            WAITING_FOR_QUERY, "FIND_LOCATION", "LOOKUP", "LOOKUP",
            row["record_id"],
        ))
        first = row["turns"][0]
        first_next = WAITING_FOR_CLARIFICATION if first["target_action"] == ASK_WHICH_ONE else WAITING_FOR_QUERY
        output["train"].append(transition(
            "LOOKUP", first["lookup_observation"], first["target_action"],
            first_next, row["record_id"],
        ))
        if len(row["turns"]) > 1:
            second = row["turns"][1]
            second_next = WAITING_FOR_CLARIFICATION if second["target_action"] == ASK_WHICH_ONE else WAITING_FOR_QUERY
            output["train"].append(transition(
                WAITING_FOR_CLARIFICATION, second["lookup_observation"], second["target_action"],
                second_next, row["record_id"],
            ))

    # The generated training dialogues do not naturally exercise every second
    # clarification outcome, so these explicit labeled controller observations
    # make the learned table total and independently auditable.
    clarification_actions = {
        ZERO_MATCHES: OBJECT_MISSING,
        ONE_WITH_LOCATION: RETURN_LOCATION,
        ONE_WITHOUT_LOCATION: LOCATION_MISSING,
        MULTIPLE_MATCHES: ASK_WHICH_ONE,
    }
    for event, action in clarification_actions.items():
        next_state = WAITING_FOR_CLARIFICATION if action == ASK_WHICH_ONE else WAITING_FOR_QUERY
        output["train"].append(transition(
            WAITING_FOR_CLARIFICATION, event, action, next_state,
            f"generated:controller-coverage:{event.casefold()}",
        ))

    for split in output:
        # Config remains first; every other row is sorted deterministically.
        prefix = output[split][:1] if split == "train" else []
        rest = output[split][1:] if split == "train" else output[split]
        output[split] = prefix + sorted(rest, key=_canonical_json)
    return output


def flatten_heldout(heldout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in heldout:
        first = row["turns"][0]
        rows.append({
            "type": "test_case",
            "id": row["record_id"],
            "text": first["utterance"],
            "expected_action": first["target_action"],
            "setting": row["setting"],
            "turns": row["turns"],
            "inventory_overrides": row["inventory_overrides"],
            "source": "heldout",
            "provenance": row["provenance"],
        })
    return rows


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _artifact_metadata(data: bytes, rows: list[dict[str, Any]] | None = None, item_count: int | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"sha256": _sha256(data), "bytes": len(data)}
    if rows is not None:
        metadata["records"] = len(rows)
        metadata["scenario_counts"] = dict(sorted(Counter(row["scenario"] for row in rows).items()))
    if item_count is not None:
        metadata["items"] = item_count
    return metadata


def build(repo_root: Path, notebook_path: Path, output_dir: Path) -> dict[str, Any]:
    decoded, source_hashes = load_embedded_snapshot(notebook_path)
    items = validate_inventory(decoded["inventory.json"])
    inventory_index = {(item["setting"], item["item_id"]): item for item in items}
    original = build_original_records(
        decoded["original_cases.json"], decoded["folds.json"], inventory_index,
        source_hashes["original_cases.json"],
    )
    generated = build_generated_records(inventory_index)
    heldout = build_heldout(inventory_index)
    similarity_evaluation = build_similarity_evaluation(inventory_index)
    validate_no_heldout_leakage(original, generated, similarity_evaluation, heldout)
    compiler_splits = build_compiler_splits(original, generated, inventory_index)
    heldout_compiler = flatten_heldout(heldout)
    runtime_inventory = build_runtime_inventory(items, source_hashes["inventory.json"])

    inventory_export = {
        "schema_version": INVENTORY_SCHEMA,
        "source_embedded_file": "inventory.json",
        "source_sha256": source_hashes["inventory.json"],
        "source": decoded["inventory.json"].get("source"),
        "item_count": len(items),
        "items": items,
    }

    artifacts: dict[str, dict[str, Any]] = {}
    for source_name, collection in (("original_derived", original), ("generated", generated)):
        for split, rows in collection.items():
            relative = Path(source_name) / f"{split}.jsonl"
            data = _jsonl_bytes(rows)
            _atomic_write(output_dir / relative, data)
            artifacts[relative.as_posix()] = _artifact_metadata(data, rows=rows)

    for split, rows in similarity_evaluation.items():
        relative = Path("similarity") / f"{split}.jsonl"
        data = _jsonl_bytes(rows)
        _atomic_write(output_dir / relative, data)
        artifacts[relative.as_posix()] = _artifact_metadata(data, rows=rows)

    heldout_relative = Path("heldout/examples.jsonl")
    heldout_data = _jsonl_bytes(heldout)
    _atomic_write(output_dir / heldout_relative, heldout_data)
    artifacts[heldout_relative.as_posix()] = _artifact_metadata(heldout_data, rows=heldout)

    inventory_relative = Path("inventory/original_inventory.json")
    inventory_data = _json_bytes(inventory_export)
    _atomic_write(output_dir / inventory_relative, inventory_data)
    artifacts[inventory_relative.as_posix()] = _artifact_metadata(inventory_data, item_count=len(items))

    # Compiler-ready consolidated files are intentionally separate from the
    # richer audit dialogues above. The compiler consumes train.jsonl only.
    for split, rows in compiler_splits.items():
        relative = Path(f"{split}.jsonl")
        data = _jsonl_bytes(rows)
        _atomic_write(output_dir / relative, data)
        artifacts[relative.as_posix()] = {
            "sha256": _sha256(data),
            "bytes": len(data),
            "records": len(rows),
            "record_type_counts": dict(sorted(Counter(row["type"] for row in rows).items())),
        }

    heldout_compiler_relative = Path("heldout.jsonl")
    heldout_compiler_data = _jsonl_bytes(heldout_compiler)
    _atomic_write(output_dir / heldout_compiler_relative, heldout_compiler_data)
    artifacts[heldout_compiler_relative.as_posix()] = {
        "sha256": _sha256(heldout_compiler_data),
        "bytes": len(heldout_compiler_data),
        "records": len(heldout_compiler),
    }

    runtime_inventory_relative = Path("inventory.json")
    runtime_inventory_data = _json_bytes(runtime_inventory)
    _atomic_write(output_dir / runtime_inventory_relative, runtime_inventory_data)
    artifacts[runtime_inventory_relative.as_posix()] = _artifact_metadata(
        runtime_inventory_data, item_count=len(runtime_inventory["items"])
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_name": "fsm_v1",
        "dataset_version": DATASET_VERSION,
        "builder": "scripts/build_fsm_dataset.py",
        "deterministic": True,
        "source": {
            "notebook": notebook_path.relative_to(repo_root).as_posix(),
            "embedded_sha256": source_hashes,
            "fold": 0,
            "reviewed_cases": 118,
            "reviewed_retrieval_cases": 25,
            "reviewed_inventory_items": 45,
        },
        "split_policy": {
            "original_derived": "fold 0 participant split",
            "counts": {split: len(rows) for split, rows in original.items()},
            "generated_counts": {split: len(rows) for split, rows in generated.items()},
            "similarity_evaluation_counts": {
                split: len(rows) for split, rows in similarity_evaluation.items()
            },
            "compiler_counts": {split: len(rows) for split, rows in compiler_splits.items()},
            "compiler_train_only": True,
        },
        "heldout_policy": {
            "records": len(heldout),
            "record_ids": [row["record_id"] for row in heldout],
            "excluded_from_train_validation_test": True,
            "exact_utterance_leakage_check": "passed",
            "compiler_file": "heldout.jsonl",
        },
        "original_dataset_policy": {
            "datasets/original": "not read or modified",
            "original_participant_rows_generated": False,
            "only_reviewed_retrieval_cases_extracted": True,
        },
        "artifacts": dict(sorted(artifacts.items())),
    }
    manifest_data = _json_bytes(manifest)
    _atomic_write(output_dir / "manifest.json", manifest_data)
    return manifest


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--notebook", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    notebook_path = (args.notebook or (repo_root / SOURCE_NOTEBOOK)).resolve()
    output_dir = (args.output_dir or (repo_root / OUTPUT_DIRECTORY)).resolve()
    manifest = build(repo_root, notebook_path, output_dir)
    print(json.dumps({
        "output_dir": str(output_dir),
        "source_hashes_verified": manifest["source"]["embedded_sha256"],
        "original_derived_counts": manifest["split_policy"]["counts"],
        "generated_counts": manifest["split_policy"]["generated_counts"],
        "heldout_records": manifest["heldout_policy"]["records"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
