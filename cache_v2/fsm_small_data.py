"""Small synthetic request stream for the FSM-convergence demo.

Every row uses the same schema as the SLM training data: ``input`` is built with
the bundle's own ``make_payload`` and ``target`` is a decision that passes the
bundle's ``validate_decision``. All utterances are newly written here.

Pattern groups
  routine      12 repeated non-risky patterns (clarify / not_found / redirect / cancel)
  risky         5 repeated locate patterns with an item slot
  unlearnable   3 patterns the FSM cannot generalize by construction
                (two-item ambiguity, answer that depends on memory, pre-send review)
  novel         one-off phrasings that rarely repeat
"""
from __future__ import annotations

import copy
import hashlib
import json
import random

DATASET_VERSION = "fsm_small_v1"
SOURCE = "synthetic_fsm_small_v1"
BATCHES = 3

ITEM_PHRASES = {
    "blue_tape": "blue tape",
    "white_tape": "white tape",
    "sd_cards": "SD cards",
    "leds": "LEDs",
    "orange_pencils": "orange pencils",
    "yellow_pencils": "yellow pencils",
    "batteries": "batteries",
    "measuring_tape": "measuring tape",
    "red_pencils": "red pencils",
    "wires": "wires",
    "blue_markers": "blue markers",
    "blue_pencils": "blue pencils",
    "white_board_cleaner": "whiteboard cleaner",
    "circuit_board": "circuit board",
    "calculator": "calculator",
    "gloves": "gloves",
    "playstation_controller": "PlayStation controller",
    "mouse": "mouse",
    "scissors": "scissors",
    "pliers": "pliers",
    "book": "book",
    "wipes": "wipes",
    "orange_multimeter": "multimeter",
    "hand_sanitizer": "hand sanitizer",
}
MULTI_ITEM_FAMILIES = {"tape": "tape", "pencil": "pencil"}  # family id -> spoken word

# (pattern_id, group, scenario, per-batch count, spec)
ROUTINE = [
    ("R01", "ambiguity_missing_attribute", 22, {"kind": "family", "text": "Where is the {family}?"}),
    ("R02", "missing_unknown_item", 18, {"kind": "literal", "text": "Do you have a stapler?", "action": "not_found"}),
    ("R03", "out_of_domain", 16, {"kind": "literal", "text": "What time is it?", "action": "redirect"}),
    ("R04", "ambiguity_missing_attribute", 14, {"kind": "family", "text": "I need a {family} please."}),
    ("R05", "cancel", 12, {"kind": "literal", "text": "Never mind, I don't need anything.", "action": "cancel"}),
    ("R06", "missing_unknown_item", 11, {"kind": "literal", "text": "Where can I find a hammer?", "action": "not_found"}),
    ("R07", "out_of_domain", 10, {"kind": "literal", "text": "Can you tell me a joke?", "action": "redirect"}),
    ("R08", "ambiguity_missing_attribute", 9, {"kind": "family", "text": "{family}"}),
    ("R09", "missing_unknown_item", 8, {"kind": "literal", "text": "Is there any glue?", "action": "not_found"}),
    ("R10", "cancel", 7, {"kind": "literal", "text": "Stop, cancel the request.", "action": "cancel"}),
    ("R11", "out_of_domain", 6, {"kind": "literal", "text": "What's the weather like today?", "action": "redirect"}),
    ("R12", "missing_unknown_item", 5, {"kind": "literal", "text": "I need a screwdriver.", "action": "not_found"}),
]
RISKY = [
    ("K01", "direct_specific_item", 30, "Where can I find the {item}?"),
    ("K02", "direct_specific_item", 26, "I need the {item}."),
    ("K03", "direct_specific_item", 24, "Can you get me the {item}?"),
    ("K04", "direct_specific_item", 20, "Which drawer has the {item}?"),
    ("K05", "direct_specific_item", 12, "Please show me where the {item} is."),
]
UNLEARNABLE = [
    ("U01", "ambiguity", 10, "two_items"),
    ("U02", "context_attribute_answer", 12, "attribute_answer"),
    ("U03", "review_correct_item_and_drawer", 12, "review"),
]
NOVEL_PER_BATCH = 40

NOVEL_PREFIXES = ["hey robot, ", "hi, ", "excuse me, ", "um, ", "okay so ", "quick question, ", "robot, ", "sorry, "]
NOVEL_VERBS = [
    "i'm looking for the", "could you point me to the", "i was hoping to grab the",
    "help me locate the", "do you know which drawer holds the", "i could really use the",
    "any idea where you keep the", "where did someone put the",
]
NOVEL_SUFFIXES = ["", " please", " for my project", " thanks", " right now", " if you have them", " real quick"]
NOVEL_OFF_TOPIC = [
    "what's your name", "who built you", "how far is the cafeteria", "can you play some music",
    "what day is it", "are you a real robot", "how do i get to the elevator", "can you charge my phone",
]
REVIEW_OPENERS = ["Where can I find the {item}?"]
ATTRIBUTE_ANSWERS = ["The {color} one.", "{color} please."]


def _decision(Runtime, **fields):
    base = {
        "action": None, "item_family": None, "resolved_item_id": None, "attributes": {},
        "missing_attributes": [], "failure_detected": False, "failure_type": "none",
        "recovery_strategy": "none",
    }
    base.update(fields)
    return Runtime.Decision.model_validate(base)


def _locate(Runtime, item):
    return _decision(Runtime, action="locate", item_family=item["family"],
                     resolved_item_id=item["id"], attributes=dict(item["attributes"]))


def _row(Runtime, inventory, *, rid, batch, pattern, group, scenario, history,
         memory=None, timing=None, allowed=None, proposal=None, target):
    payload = Runtime.make_payload(
        history, memory or Runtime.empty_memory(), inventory, timing, allowed,
        proposed_robot_answer=proposal,
        decision_mode="pre_send_review" if proposal is not None else "dialogue_decision",
    )
    decision = Runtime.validate_decision(target, payload, inventory)  # gold must be valid
    return {
        "id": rid,
        "batch": batch,
        "pattern_id": pattern,
        "pattern_group": group,
        "scenario": scenario,
        "source": SOURCE,
        "input": payload,
        "target": decision.model_dump(),
        "inventory": copy.deepcopy(inventory),
    }


def build_dataset(Runtime, inventory, seed=20260916):
    rng = random.Random(seed)
    by_id = {item["id"]: item for item in inventory}
    missing = sorted(set(ITEM_PHRASES) - set(by_id))
    if missing:
        raise ValueError(f"Bundle inventory lacks items used by the generator: {missing}")
    item_ids = sorted(ITEM_PHRASES)
    rows, counter = [], 0

    def next_id():
        nonlocal counter
        counter += 1
        return f"fsm_small_{counter:05d}"

    for batch in range(1, BATCHES + 1):
        batch_rows = []
        for pid, scenario, count, spec in ROUTINE:
            for _ in range(count):
                if spec["kind"] == "family":
                    family = rng.choice(sorted(MULTI_ITEM_FAMILIES))
                    text = spec["text"].format(family=MULTI_ITEM_FAMILIES[family])
                    if text == MULTI_ITEM_FAMILIES[family]:
                        text = text.capitalize() + "."
                    target = _decision(Runtime, action="clarify", item_family=family,
                                       missing_attributes=["color"])
                else:
                    text = spec["text"]
                    target = _decision(Runtime, action=spec["action"])
                batch_rows.append(dict(pattern=pid, group="routine", scenario=scenario,
                                       history=[{"role": "user", "text": text}], target=target))
        for pid, scenario, count, template in RISKY:
            for _ in range(count):
                item = by_id[rng.choice(item_ids)]
                text = template.format(item=ITEM_PHRASES[item["id"]])
                batch_rows.append(dict(pattern=pid, group="risky", scenario=scenario,
                                       history=[{"role": "user", "text": text}],
                                       target=_locate(Runtime, item)))
        for pid, scenario, count, kind in UNLEARNABLE:
            for _ in range(count):
                if kind == "two_items":
                    first, second = rng.sample(item_ids, 2)
                    while by_id[first]["family"] == by_id[second]["family"]:
                        first, second = rng.sample(item_ids, 2)
                    text = f"I need the {ITEM_PHRASES[first]} or the {ITEM_PHRASES[second]}."
                    batch_rows.append(dict(pattern=pid, group="unlearnable", scenario=scenario,
                                           history=[{"role": "user", "text": text}],
                                           target=_decision(Runtime, action="clarify")))
                elif kind == "attribute_answer":
                    family = rng.choice(sorted(MULTI_ITEM_FAMILIES))
                    options = [x for x in inventory if x["family"] == family]
                    item = rng.choice(options)
                    ask = _decision(Runtime, action="clarify", item_family=family,
                                    missing_attributes=["color"])
                    history = [
                        {"role": "user", "text": f"Where is the {MULTI_ITEM_FAMILIES[family]}?"},
                        {"role": "robot", "text": Runtime.render_response(ask, inventory)},
                        {"role": "user", "text": rng.choice(ATTRIBUTE_ANSWERS).format(
                            color=item["attributes"]["color"]).capitalize()},
                    ]
                    memory = Runtime.next_memory(Runtime.empty_memory(), ask)
                    batch_rows.append(dict(pattern=pid, group="unlearnable", scenario=scenario,
                                           history=history, memory=memory,
                                           target=_locate(Runtime, item)))
                else:
                    item = by_id[rng.choice(item_ids)]
                    locate = _locate(Runtime, item)
                    text = rng.choice(REVIEW_OPENERS).format(item=ITEM_PHRASES[item["id"]])
                    batch_rows.append(dict(
                        pattern=pid, group="unlearnable", scenario=scenario,
                        history=[{"role": "user", "text": text}],
                        timing={"last_response_latency_ms": 900.0},
                        allowed=["locate", "recover"],
                        proposal=Runtime.render_response(locate, inventory),
                        target=locate,
                    ))
        for index in range(NOVEL_PER_BATCH):
            if index % 5 == 4:
                text = rng.choice(NOVEL_PREFIXES) + rng.choice(NOVEL_OFF_TOPIC) + "?"
                target = _decision(Runtime, action="redirect")
                scenario = "out_of_domain"
            else:
                item = by_id[rng.choice(item_ids)]
                text = (rng.choice(NOVEL_PREFIXES) + rng.choice(NOVEL_VERBS) + " "
                        + ITEM_PHRASES[item["id"]] + rng.choice(NOVEL_SUFFIXES) + ".")
                target = _locate(Runtime, item)
                scenario = "direct_specific_item"
            text = text[0].upper() + text[1:]
            batch_rows.append(dict(pattern=f"N{batch}{index:02d}", group="novel", scenario=scenario,
                                   history=[{"role": "user", "text": text}], target=target))

        rng.shuffle(batch_rows)
        for spec in batch_rows:
            rows.append(_row(
                Runtime, inventory, rid=next_id(), batch=batch,
                pattern=spec["pattern"], group=spec["group"], scenario=spec["scenario"],
                history=spec["history"], memory=spec.get("memory"), timing=spec.get("timing"),
                allowed=spec.get("allowed"), proposal=spec.get("proposal"), target=spec["target"],
            ))
    return rows


def dataset_card(rows):
    groups, patterns = {}, {}
    for row in rows:
        groups[row["pattern_group"]] = groups.get(row["pattern_group"], 0) + 1
        if row["pattern_group"] != "novel":
            patterns.setdefault(row["pattern_id"], {"group": row["pattern_group"],
                                                    "scenario": row["scenario"], "rows": 0})
            patterns[row["pattern_id"]]["rows"] += 1
    unique_inputs = len({json.dumps(r["input"], sort_keys=True) for r in rows})
    return {
        "version": DATASET_VERSION,
        "source": SOURCE,
        "rows": len(rows),
        "batches": BATCHES,
        "unique_inputs": unique_inputs,
        "rows_by_group": groups,
        "patterns": patterns,
        "note": "Newly written requests in the SLM training-data schema; not copied from existing data files.",
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(rows, path):
    # Keep key order: the SLM prompt serializes the payload in make_payload's order.
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
