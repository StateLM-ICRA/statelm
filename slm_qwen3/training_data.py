"""Canonical StateLM adapter, synthetic behavior cases, and strict grouped JSONL loading."""
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from slm_runtime import (ACTIONS, STRATEGIES, Decision, check_inventory, empty_memory, make_payload,
                         parse_decision, validate_decision)

DEMO_INVENTORY = [
    {"id": "blue_tape", "name": "blue tape", "family": "tape", "attributes": {"color": "blue"},
     "aliases": ["blue adhesive tape"], "drawer": "drawer 1"},
    {"id": "white_tape", "name": "white tape", "family": "tape", "attributes": {"color": "white"},
     "aliases": ["white adhesive tape"], "drawer": "drawer 2"},
    {"id": "pen", "name": "pen", "family": "pen", "attributes": {},
     "aliases": ["ballpoint", "writing pen"], "drawer": "drawer 4"},
    {"id": "pencil", "name": "pencil", "family": "pencil", "attributes": {},
     "aliases": ["graphite pencil"], "drawer": "drawer 3"},
    {"id": "gauze", "name": "gauze", "family": "gauze", "attributes": {},
     "aliases": ["gauze pad", "gauze dressing"], "drawer": "drawer 5"},
    {"id": "gloves", "name": "gloves", "family": "gloves", "attributes": {},
     "aliases": ["a pair of gloves"], "drawer": "drawer 6"},
]


def gold(action, item=None, family=None, attrs=None, missing=None, failure="none", strategy="none"):
    return Decision(action=action, item_family=item["family"] if item else family,
                    resolved_item_id=item["id"] if item else None,
                    attributes=copy.deepcopy(item["attributes"] if item else attrs or {}),
                    missing_attributes=list(missing or []), failure_detected=failure != "none",
                    failure_type=failure, recovery_strategy=strategy).model_dump()


def record(group, scenario, inventory, history, target, memory=None, timing=None, allowed=None, proposed=None):
    payload = make_payload(history, memory or empty_memory(), inventory, timing, allowed, proposed)
    return {"id": f"{group}-{scenario}", "group_id": group, "source": "synthetic_template",
            "scenario": scenario, "inventory": copy.deepcopy(inventory), "input": payload,
            "target": target, "acceptable_recovery_strategies": [target["recovery_strategy"]]
            if target["failure_detected"] else [],
            "evaluation_mask": {"item_slots": True}}


def turns(*texts):
    return [{"role": "user" if i % 2 == 0 else "robot", "text": text} for i, text in enumerate(texts)]


PHRASES = {
    "train": [
        ["I need the {item}.", "Please find the {item}.", "Where is the {item}?", "Can you locate the {item}?"],
        ["I need tape.", "Where is the tape?", "Please find some tape."],
        ["Blue.", "The blue one.", "Blue, please."],
        ["No, I said pen.", "I asked for a pen, not a pencil.", "You misunderstood: I need the pen."],
        ["Tell me a joke.", "What is the weather?", "Write a poem about the moon."],
    ],
    "validation": [
        ["Could you point me to the {item}?"], ["Which drawer contains tape?"], ["Make it blue."],
        ["Please correct that: pen was the item I requested."], ["Who won the football game?"],
    ],
    "test": [
        ["Help me track down the {item}, please."], ["I'm looking for adhesive tape."], ["I'll take blue."],
        ["A pencil isn't what I requested; I wanted a pen."], ["Plan my next holiday."],
    ],
}


def world_rows(split, world_index):
    offset = {"train": 0, "validation": 10000, "test": 20000}[split]
    rng = random.Random(offset + world_index)
    inventory = copy.deepcopy(DEMO_INVENTORY)
    for item in inventory:
        item["id"] = f"obj_{offset + world_index}_{rng.randrange(100000, 999999)}"
        item["drawer"] = f"drawer {rng.randrange(1, 10)}"
    rng.shuffle(inventory)
    by_name = {x["name"]: x for x in inventory}
    blue, white, pen, pencil = [by_name[k] for k in ["blue tape", "white tape", "pen", "pencil"]]
    group = f"{split}-world-{world_index:03d}"
    phr = PHRASES[split]
    rows = []
    def add(scenario, history, target, memory=None, timing=None, allowed=None, proposed=None):
        rows.append(record(group, scenario, inventory, history, target, memory, timing, allowed, proposed))
    for item in inventory:
        text = rng.choice(phr[0]).format(item=item["name"])
        add("direct_" + item["name"].replace(" ", "_"), turns(text), gold("locate", item))
    add("synonym", turns(rng.choice(phr[0]).format(item="ballpoint")), gold("locate", pen))
    add("ambiguity", turns(rng.choice(phr[1])), gold("clarify", family="tape", missing=["color"]))
    mem = empty_memory()
    mem.update(pending_item_family="tape", missing_attributes=["color"], last_robot_action="clarify")
    add("context_blue", turns(rng.choice(phr[1]), "Do you want blue tape or white tape?", rng.choice(phr[2])),
        gold("locate", blue), mem)
    add("context_white", turns(rng.choice(phr[1]), "Do you want blue tape or white tape?", "White, please."),
        gold("locate", white), mem)
    add("bare_color", turns(rng.choice(phr[2])), gold("clarify"))
    add("correction", turns("I need the pen.", f"The pencil is in {pencil['drawer']}.", rng.choice(phr[3])),
        gold("recover", pen, failure="comprehension_failure", strategy="self_correction"))
    add("context_failure", turns("I need tape.", "Blue tape or white tape?", "Blue.",
        "What item are you asking for?", "The blue tape we were just discussing."),
        gold("recover", blue, failure="context_failure", strategy="self_correction"))
    add("new_request", turns("I need tape.", "Do you want blue tape or white tape?", "Actually, find the pen instead."),
        gold("locate", pen), mem)
    add("out_of_domain", turns(rng.choice(phr[4])), gold("redirect"))
    add("interrupt_pending", turns("I need tape.", "Blue tape or white tape?", rng.choice(phr[4])), gold("redirect"), mem)
    add("not_found", turns(rng.choice(phr[0]).format(item="red tape")), gold("not_found", family="tape", attrs={"color": "red"}))
    add("unknown_item", turns(rng.choice(phr[0]).format(item="stapler")), gold("not_found", family="stapler"))
    add("cancel", turns("Please cancel my request."), gold("cancel"))
    add("speech_failure", turns("Where is the gauze?", "Open the drawer.", "Which drawer did you mean?"),
        gold("recover", by_name["gauze"], failure="speech_failure", strategy="specific_redirection"))
    # Timing evidence is explicit metadata AND user feedback, not guessed from wording alone.
    add("timing_failure", turns("Find the pen.", f"The pen is in {pen['drawer']}.", "Why did you take so long?"),
        gold("recover", pen, failure="timing_failure", strategy="transparency_cue"),
        timing={"last_response_latency_ms": 14500})
    wrong = "drawer 99"
    add("search_failure", turns("I need the pen.", f"The pen is in {wrong}.", "That drawer has no pen."),
        gold("recover", pen, failure="search_failure", strategy="specific_redirection"))
    add("unclear_repair", turns("I need something to write with.", "I could not understand.", "Something for writing."),
        gold("recover", failure="comprehension_failure", strategy="clarifying_prompt"))
    add("partial_repair", turns("I need tape, the [inaudible] one.", "I heard tape but lost the rest.", "Can you ask about the missing part?"),
        gold("recover", family="tape", missing=["color"], failure="comprehension_failure", strategy="partial_understanding_repair"))
    add("reset_repair", turns("Find the pen.", "The pencil is in drawer 3.", "You misunderstood. Let's start this request over."),
        gold("recover", failure="comprehension_failure", strategy="guided_reset"))
    add("confirm_repair", turns("I said pen.", "The pencil is in drawer 3.", "Please check that you understood pen correctly."),
        gold("recover", pen, failure="comprehension_failure", strategy="confidence_check"))
    add("disallowed_locate", turns("Find the blue tape."), gold("clarify", family="tape"),
        allowed=["clarify", "redirect", "cancel"])
    # Pre-send controls prevent the model from learning that every proposed answer is a failure.
    add("review_correct_pen", turns("I need the pen."), gold("locate", pen),
        timing={"last_response_latency_ms": 900.0}, allowed=["locate", "recover"],
        proposed=f"The {pen['name']} is listed in {pen['drawer']}.")
    add("review_wrong_item", turns("I need the pen."),
        gold("recover", pen, failure="comprehension_failure", strategy="self_correction"),
        timing={"last_response_latency_ms": 700.0}, allowed=["locate", "recover"],
        proposed=f"The {pencil['name']} is listed in {pencil['drawer']}.")
    add("review_wrong_drawer", turns("I need the pen."),
        gold("recover", pen, failure="search_failure", strategy="specific_redirection"),
        timing={"last_response_latency_ms": 1100.0}, allowed=["locate", "recover"],
        proposed="The pen is listed in drawer 99.")
    add("review_incomplete", turns("Where is the gauze?"),
        gold("recover", by_name["gauze"], failure="speech_failure", strategy="specific_redirection"),
        timing={"last_response_latency_ms": 800.0}, allowed=["locate", "recover"],
        proposed="Open the drawer.")
    return rows


def make_demo_splits():
    # Worlds (IDs and drawer assignments) and direct-request/clarification/correction wording
    # differ between splits. Other failure templates recur: this is still a synthetic smoke test.
    return {split: [row for i in range(n) for row in world_rows(split, i)]
            for split, n in [("train", 16), ("validation", 3), ("test", 3)]}


def audit_splits(splits):
    ids, fingerprints, seen_groups = set(), {}, {}
    required = {"id", "group_id", "source", "scenario", "inventory", "input", "target"}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty {split} split")
        for r in rows:
            if not required <= set(r):
                raise ValueError(f"Record needs fields: {sorted(required)}")
            if r["id"] in ids:
                raise ValueError(f"Duplicate ID: {r['id']}")
            ids.add(r["id"])
            group = str(r.get("participant_id") or r["group_id"])
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError(f"Group/participant {group} leaks across splits")
            seen_groups[group] = split
            inv = check_inventory(r["inventory"])
            p = r["input"]
            # Rebuild to check role labels, memory presence, and consistent candidate inventory.
            rebuilt = make_payload(p["conversation"], p["memory"], inv, p["timing"], p["allowed_actions"], p.get("proposed_robot_answer"))
            if p != rebuilt:
                raise ValueError(f"Input context does not match inventory/history: {r['id']}")
            validate_decision(Decision.model_validate(r["target"]), p, inv)
            acceptable = r.get("acceptable_recovery_strategies", [])
            if r["target"]["failure_detected"] and acceptable and r["target"]["recovery_strategy"] not in acceptable:
                raise ValueError("Chosen target strategy must be in the acceptable strategy set")
            fingerprint = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
            if fingerprint in fingerprints and fingerprints[fingerprint] != split:
                raise ValueError("Identical input duplicated across splits")
            fingerprints[fingerprint] = split
    return {split: {"rows": len(rows), "groups": len({str(r.get('participant_id') or r['group_id']) for r in rows}),
                    "scenarios": dict(Counter(r["scenario"] for r in rows))} for split, rows in splits.items()}


def load_grouped_jsonl(path, seed=42):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("No records in JSONL")
    if not all("target" in r and "group_id" in r for r in records):
        raise ValueError("Use canonical structured records. Legacy input/output strategy labels need manual annotation; see notebook.")
    if any("split" in r for r in records):
        if not all(r.get("split") in {"train", "validation", "test"} for r in records):
            raise ValueError("Supply train/validation/test for every row, or omit split on every row")
        splits = {s: [r for r in records if r["split"] == s] for s in ["train", "validation", "test"]}
    else:
        groups = sorted({str(r.get("participant_id") or r["group_id"]) for r in records})
        if len(groups) < 5:
            raise ValueError("Need at least 5 independent groups, or supply explicit reviewed splits")
        random.Random(seed).shuffle(groups)
        n = max(1, round(len(groups) * .15))
        assignment = {g: "test" if i < n else "validation" if i < 2*n else "train" for i, g in enumerate(groups)}
        splits = {s: [r for r in records if assignment[str(r.get("participant_id") or r["group_id"])] == s]
                  for s in ["train", "validation", "test"]}
    audit_splits(splits)
    return splits


def write_splits(splits, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


SUPPORTED_RECOVERIES = set(STRATEGIES) - {"none"}
FAMILY_ATTRIBUTES = {
    "blue_tape": ("tape", {"color": "blue"}),
    "white_tape": ("tape", {"color": "white"}),
    "orange_pencils": ("pencil", {"color": "orange"}),
    "yellow_pencils": ("pencil", {"color": "yellow"}),
    "red_pencils": ("pencil", {"color": "red"}),
    "blue_pencils": ("pencil", {"color": "blue"}),
    "blue_markers": ("marker", {"color": "blue"}),
    "orange_multimeter": ("multimeter", {"color": "orange"}),
    "line_draw_arterial_blood_sample_syringe": ("syringe", {"type": "arterial blood sample"}),
    "1_ml_tuberculin_syringe": ("syringe", {"volume": "1 mL", "type": "tuberculin"}),
    "6_ml_syringe": ("syringe", {"volume": "6 mL"}),
}
RECOVERY_PRIORITIES = {
    "speech_failure": ["specific_redirection", "self_correction", "clarifying_prompt",
                       "partial_understanding_repair", "confidence_check", "transparency_cue", "guided_reset"],
    "timing_failure": ["transparency_cue", "clarifying_prompt", "self_correction",
                       "confidence_check", "specific_redirection", "guided_reset", "partial_understanding_repair"],
    "comprehension_failure": ["clarifying_prompt", "partial_understanding_repair", "confidence_check",
                              "self_correction", "specific_redirection", "guided_reset", "transparency_cue"],
    "search_failure": ["self_correction", "specific_redirection", "guided_reset", "confidence_check",
                       "clarifying_prompt", "partial_understanding_repair", "transparency_cue"],
}


def normalize_inventory_item(raw):
    family, attrs = FAMILY_ATTRIBUTES.get(raw["item_id"], (raw["item_id"], {}))
    aliases = list(dict.fromkeys(raw.get("aliases", [])))
    return {"id": raw["item_id"], "name": raw["name"].strip(), "family": family,
            "attributes": copy.deepcopy(attrs), "aliases": aliases,
            "drawer": f"drawer {int(raw['drawer'])}"}


def _normal(text):
    import re, unicodedata
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^a-z0-9%]+", " ", text).strip()


def infer_source_item(history, inventory):
    # Exact cart-scoped matching on participant text only; the robot proposal must not define user intent.
    text = _normal(" ".join(t["text"] for t in history if t["role"] == "user"))
    hits = []
    for item in inventory:
        phrases = sorted(set([item["name"], *item.get("aliases", [])]), key=len, reverse=True)
        score = max((len(_normal(p)) for p in phrases if _normal(p) and
                     f" {_normal(p)} " in f" {text} "), default=0)
        if score:
            hits.append((score, item))
    hits.sort(key=lambda pair: pair[0], reverse=True)
    if not hits or (len(hits) > 1 and hits[0][0] == hits[1][0]):
        return None
    return hits[0][1]


def choose_primary_strategy(failure_type, preferences):
    supported = [x for x in preferences if x in SUPPORTED_RECOVERIES]
    for strategy in RECOVERY_PRIORITIES[failure_type]:
        if strategy in supported:
            return strategy, supported
    return None, supported


def load_real_statelm(data_dir, synthetic_train_worlds=2, synthetic_validation_worlds=1,
                      synthetic_challenge_worlds=2):
    data_dir = Path(data_dir)
    original = json.loads((data_dir / "original_cases.json").read_text())["cases"]
    raw_inventory = json.loads((data_dir / "inventory.json").read_text())["items"]
    fold = json.loads((data_dir / "folds.json").read_text())[0]
    inventory_by_setting = {}
    for setting in sorted({x["setting"] for x in raw_inventory}):
        inventory_by_setting[setting] = [normalize_inventory_item(x) for x in raw_inventory if x["setting"] == setting]
        check_inventory(inventory_by_setting[setting])
    assignment = {str(pid): split for split, key in [("train", "train_participants"),
                  ("validation", "validation_participants"), ("test", "test_participants")]
                  for pid in fold[key]}
    splits = {"train": [], "validation": [], "test": [], "challenge": []}
    excluded = Counter()
    for case in original:
        pid = str(case["participant_id"])
        if pid not in assignment:
            excluded["participant_not_in_fold"] += 1
            continue
        setting = case["input"]["cart"]
        inventory = inventory_by_setting[setting]
        history = [{"role": "user" if t["role"] == "participant" else "robot", "text": t["text"]}
                   for t in case["input"]["history"]]
        # A recovery prefix ends at the induced robot failure. Move every final contiguous robot turn
        # into an explicit pre-send proposal while keeping the latest participant request as history.
        proposed_turns = []
        while history and history[-1]["role"] == "robot":
            proposed_turns.append(history.pop()["text"])
        proposed = " ".join(reversed(proposed_turns)).strip() or None
        if not history or history[-1]["role"] != "user":
            excluded["no_user_decision_boundary"] += 1
            continue
        timing = {"last_response_latency_ms": None}
        wait = case["input"].get("observed_wait_seconds")
        if isinstance(wait, (int, float)):
            timing["last_response_latency_ms"] = round(float(wait) * 1000, 3)
        allowed = ["locate", "recover"] if proposed is not None else None
        payload = make_payload(history, empty_memory(), inventory, timing, allowed, proposed)
        if case["task"] == "retrieval":
            item = next((x for x in inventory if x["id"] == case["targets"]["item_id"]), None)
            if item is None:
                excluded["retrieval_item_missing"] += 1
                continue
            target = gold("locate", item)
            acceptable = []
            scenario = "real_success_retrieval"
        elif case["task"] == "recovery":
            failure_types = case["targets"].get("failure_types", [])
            if len(failure_types) != 1:
                excluded["multi_failure"] += 1
                continue
            failure = failure_types[0]
            strategy, acceptable = choose_primary_strategy(failure, case["targets"].get("preferences", []))
            if strategy is None:
                excluded["no_voice_recovery_preference"] += 1
                continue
            item = infer_source_item(history, inventory)
            target = gold("recover", item, failure=failure, strategy=strategy)
            scenario = "real_" + failure
        else:
            excluded["unknown_task"] += 1
            continue
        provenance = copy.deepcopy(case.get("provenance", {}))
        provenance.update({"converter_version": "qwen3_robot_slm_v1",
                           "raw_preferences": copy.deepcopy(case.get("targets", {}).get("preferences", [])),
                           "observed_robot_turn_reframed_as_proposal": proposed is not None,
                           "reframed_robot_turn_count": len(proposed_turns),
                           "item_match_method": "exact_user_alias" if item is not None else "unresolved"})
        record_out = {"id": case["case_id"], "group_id": "participant_" + pid,
                      "participant_id": pid, "trial_id": case["trial_id"],
                      "source": "real_statelm_v5_reviewed", "scenario": scenario,
                      "inventory": copy.deepcopy(inventory), "input": payload, "target": target,
                      "acceptable_recovery_strategies": acceptable,
                      "evaluation_mask": {"item_slots": case["task"] == "retrieval"},
                      "source_provenance": provenance}
        splits[assignment[pid]].append(record_out)
        if case["task"] == "retrieval":
            # Matched *derived* no-failure controls, not human-observed robot responses.
            correct_proposals = [
                f"The {item['name']} is listed in {item['drawer']}.",
                f"You can find the {item['name']} in {item['drawer']}.",
                f"Please open {item['drawer']} for the {item['name']}.",
            ]
            normal_review_latencies_ms = [0.0, 1500.0, 3200.0]
            for control_index, correct_proposal in enumerate(correct_proposals, 1):
                control_timing = {"last_response_latency_ms": normal_review_latencies_ms[control_index - 1]}
                control_input = make_payload(history, empty_memory(), inventory, control_timing,
                                             ["locate", "recover"], correct_proposal)
                control = {"id": case["case_id"] + f":derived_correct_review_{control_index}",
                           "group_id": "participant_" + pid, "participant_id": pid,
                           "trial_id": case["trial_id"], "source": "derived_inventory_review_control",
                           "scenario": "derived_correct_review", "inventory": copy.deepcopy(inventory),
                           "input": control_input, "target": target,
                           "acceptable_recovery_strategies": [],
                           "evaluation_mask": {"item_slots": True},
                           "source_provenance": {"derived_from_case_id": case["case_id"],
                               "derivation": "verified_inventory_template_v1", "template_index": control_index,
                               "derived_normal_latency_ms": normal_review_latencies_ms[control_index - 1],
                               "synthetic_text": True, "not_human_observation": True}}
                splits[assignment[pid]].append(control)
    # Synthetic worlds teach uncovered behaviors. Their dev/smoke rows are labelled separately.
    for i in range(synthetic_train_worlds):
        splits["train"].extend(world_rows("train", 500 + i))
    for i in range(synthetic_validation_worlds):
        splits["validation"].extend(world_rows("validation", 700 + i))
    for i in range(synthetic_challenge_worlds):
        splits["challenge"].extend(world_rows("test", 900 + i))
    report = {"real_input_cases": len(original), "observed_rows_included": {
              k: sum(r["source"] == "real_statelm_v5_reviewed" for r in rows)
              for k, rows in splits.items()}, "derived_review_controls": {
              k: sum(r["source"] == "derived_inventory_review_control" for r in rows)
              for k, rows in splits.items()}, "excluded": dict(excluded),
              "synthetic_train": sum(r["source"] == "synthetic_template" for r in splits["train"]),
              "synthetic_validation": sum(r["source"] == "synthetic_template" for r in splits["validation"]),
              "synthetic_challenge": len(splits["challenge"]),
              "note": "Derived controls and challenge rows are synthetic smoke/development evidence, not human outcomes."}
    audit_splits(splits)
    return splits, inventory_by_setting, report
