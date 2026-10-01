"""Structured decision and inventory contract shared by the FSM and SLM.

The final drawer answer is rendered from the live inventory. Neither the SLM
nor a cached FSM transition is allowed to store or invent a drawer value.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence


ACTION_ORDER = ["locate", "clarify", "recover", "redirect", "not_found", "cancel"]  # bundle order
ACTIONS = set(ACTION_ORDER)
MAX_HISTORY_TURNS = 8
FAILURE_TYPES = {
    "none",
    "comprehension_failure",
    "context_failure",
    "speech_failure",
    "timing_failure",
    "search_failure",
}
RECOVERY_STRATEGIES = {
    "none",
    "clarifying_prompt",
    "self_correction",
    "specific_redirection",
    "confidence_check",
    "guided_reset",
    "transparency_cue",
    "partial_understanding_repair",
}
DECISION_KEYS = (
    "action",
    "item_family",
    "resolved_item_id",
    "attributes",
    "missing_attributes",
    "failure_detected",
    "failure_type",
    "recovery_strategy",
)


def _location(item: Mapping[str, Any]) -> str | None:
    value = item.get("drawer", item.get("location"))
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def normalize_inventory(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(inventory, Sequence) or isinstance(inventory, (str, bytes)) or not inventory:
        raise ValueError("Inventory must be a nonempty sequence")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in inventory:
        item = copy.deepcopy(dict(raw))
        item_id = str(item.get("id", "")).strip()
        name = str(item.get("name", "")).strip()
        family = str(item.get("family", "")).strip()
        if not item_id or not name or not family:
            raise ValueError("Every inventory item needs id, name, and family")
        if item_id in seen:
            raise ValueError(f"Duplicate inventory ID: {item_id}")
        seen.add(item_id)
        aliases = item.get("aliases", [])
        attributes = item.get("attributes", {})
        if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
            raise ValueError("aliases must be a list of strings")
        if not isinstance(attributes, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in attributes.items()
        ):
            raise ValueError("attributes must map strings to strings")
        item["id"] = item_id
        item["name"] = name
        item["family"] = family
        item["aliases"] = list(aliases)
        item["attributes"] = dict(attributes)
        item["drawer"] = _location(item)
        result.append(item)
    return result


def decision_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError("Decision must be a mapping or JSON object")
    missing = [key for key in DECISION_KEYS if key not in value]
    extra = [key for key in value if key not in DECISION_KEYS]
    if missing or extra:
        raise ValueError(f"Decision keys disagree; missing={missing}, extra={extra}")
    result = {key: copy.deepcopy(value[key]) for key in DECISION_KEYS}
    if result["action"] not in ACTIONS:
        raise ValueError("Unknown action")
    if result["failure_type"] not in FAILURE_TYPES:
        raise ValueError("Unknown failure type")
    if result["recovery_strategy"] not in RECOVERY_STRATEGIES:
        raise ValueError("Unknown recovery strategy")
    if not isinstance(result["attributes"], dict) or not all(
        isinstance(key, str) and isinstance(val, str)
        for key, val in result["attributes"].items()
    ):
        raise ValueError("attributes must map strings to strings")
    if not isinstance(result["missing_attributes"], list) or not all(
        isinstance(x, str) for x in result["missing_attributes"]
    ):
        raise ValueError("missing_attributes must be a list of strings")
    if not isinstance(result["failure_detected"], bool):
        raise ValueError("failure_detected must be boolean")
    return result


def validate_decision(value: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    d = decision_dict(value)
    inventory = normalize_inventory(payload.get("inventory_candidates") or [])
    allowed = set(payload.get("allowed_actions") or ACTIONS)
    if d["action"] not in allowed:
        raise ValueError("Disallowed action")
    if d["failure_detected"] != (d["failure_type"] != "none"):
        raise ValueError("failure flag and type disagree")
    if d["failure_detected"] != (d["action"] == "recover"):
        raise ValueError("Only recover may represent a detected failure")
    if d["action"] == "recover" and d["recovery_strategy"] == "none":
        raise ValueError("Recovery needs a strategy")
    if d["action"] != "recover" and d["recovery_strategy"] != "none":
        raise ValueError("Non-recovery action cannot select a recovery strategy")
    if d["action"] == "locate" and d["resolved_item_id"] is None:
        raise ValueError("locate needs an item ID")
    if d["action"] not in {"locate", "recover"} and d["resolved_item_id"] is not None:
        raise ValueError("Only locate/recover may select an item")
    if d["resolved_item_id"] is not None and d["missing_attributes"]:
        raise ValueError("A resolved item cannot have missing attributes")
    if len(set(d["missing_attributes"])) != len(d["missing_attributes"]):
        raise ValueError("Duplicate missing attributes")
    if set(d["attributes"]) & set(d["missing_attributes"]):
        raise ValueError("Attribute cannot be both known and missing")

    by_id = {item["id"]: item for item in inventory}
    if d["resolved_item_id"] is not None:
        item = by_id.get(d["resolved_item_id"])
        if item is None:
            raise ValueError("Unsupported item ID")
        if d["item_family"] != item["family"] or d["attributes"] != item["attributes"]:
            raise ValueError("Item ID, family, and attributes disagree")
    elif d["action"] in {"clarify", "recover"} and d["item_family"] is not None:
        family_items = [item for item in inventory if item["family"] == d["item_family"]]
        if not family_items:
            raise ValueError("Unsupported clarification family")
        if not any(
            all(item["attributes"].get(key) == val for key, val in d["attributes"].items())
            for item in family_items
        ):
            raise ValueError("Unsupported known attributes")
        valid_keys = {key for item in family_items for key in item["attributes"]}
        if not set(d["missing_attributes"]) <= valid_keys:
            raise ValueError("Unsupported missing attribute")
    if d["action"] in {"redirect", "cancel"} and (
        d["item_family"] is not None or d["attributes"] or d["missing_attributes"]
    ):
        raise ValueError("redirect/cancel must not inject item context")
    return d


def empty_memory() -> dict[str, Any]:
    return {
        "pending_item_family": None,
        "known_attributes": {},
        "missing_attributes": [],
        "resolved_item_id": None,
        "last_robot_action": None,
        "recovery_count": 0,
    }


def make_payload(
    history: Sequence[Mapping[str, str]],
    memory: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    *,
    allowed_actions: Sequence[str] | None = None,
    proposed_robot_answer: str | None = None,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not history or history[-1].get("role") != "user":
        raise ValueError("History must end with the latest user utterance")
    normalized_inventory = normalize_inventory(inventory)
    allowed = list(allowed_actions or ACTION_ORDER)
    mode = "pre_send_review" if proposed_robot_answer is not None else "dialogue_decision"
    state = (
        "reviewing_proposed_answer"
        if mode == "pre_send_review"
        else "awaiting_clarification"
        if memory.get("missing_attributes")
        else "awaiting_request"
    )
    return {
        "decision_mode": mode,
        "state": state,
        "allowed_actions": allowed,
        "conversation": [dict(turn) for turn in history[-MAX_HISTORY_TURNS:]],
        "memory": copy.deepcopy(dict(memory)),
        "inventory_candidates": normalized_inventory,
        "timing": copy.deepcopy(dict(timing or {"last_response_latency_ms": None})),
        "proposed_robot_answer": proposed_robot_answer,
    }


def next_memory(memory: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    d = decision_dict(decision)
    result = copy.deepcopy(dict(memory))
    result.setdefault("recovery_count", 0)
    result["last_robot_action"] = d["action"]
    if d["action"] == "cancel":
        return empty_memory()
    if d["action"] == "redirect":
        return result
    result.update(
        pending_item_family=d["item_family"],
        known_attributes=dict(d["attributes"]),
        missing_attributes=list(d["missing_attributes"]),
        resolved_item_id=d["resolved_item_id"],
    )
    if d["action"] == "recover":
        result["recovery_count"] += 1
    if d["recovery_strategy"] == "guided_reset":
        result.update(
            pending_item_family=None,
            known_attributes={},
            missing_attributes=[],
            resolved_item_id=None,
        )
    return result


def _clarification_text(decision: Mapping[str, Any], inventory: list[dict[str, Any]]) -> str:
    family = decision["item_family"]
    attrs = decision["attributes"]
    matches = [
        item
        for item in inventory
        if item["family"] == family
        and all(item["attributes"].get(key) == value for key, value in attrs.items())
    ]
    if 2 <= len(matches) <= 5:
        return "Do you want " + " or ".join(item["name"] for item in matches) + "?"
    if len(matches) > 5:
        return "Which type do you need? Please give its color, size, or exact name."
    return "Which item do you mean? Please say its full name and any color or size."


def render_response(decision: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]) -> str:
    """Spoken reply.  Wording follows the deployed bundle's ``slm_runtime``.

    The drawer is always read from the live inventory for the bound item;
    neither the SLM nor a cached transition can store or invent a drawer.
    """

    d = decision_dict(decision)
    normalized = normalize_inventory(inventory)
    by_id = {item["id"]: item for item in normalized}
    item = by_id.get(d["resolved_item_id"])
    location = None
    if item is not None:
        if not item["drawer"]:
            raise ValueError(f"Inventory has no drawer for {item['id']}")
        location = f"The {item['name']} is listed in {item['drawer']}."
    if d["action"] == "locate":
        if location is None:
            raise ValueError("locate response has no grounded inventory item")
        return location
    if d["action"] == "clarify":
        return _clarification_text(d, normalized)
    if d["action"] == "redirect":
        return "I can help you locate items in the inventory. Which item do you need?"
    if d["action"] == "not_found":
        return "I could not match that request to the inventory. Please check the name or ask a staff member."
    if d["action"] == "cancel":
        return "Okay, I have cancelled this request."
    if d["recovery_strategy"] == "self_correction":
        return "Sorry, I misunderstood. " + (location or "Please repeat the full item name.")
    if d["recovery_strategy"] == "specific_redirection":
        return "Sorry, let me give the inventory location more clearly. " + (
            location or "Please repeat the full item name."
        )
    if d["recovery_strategy"] == "transparency_cue":
        prefix = (
            "Sorry for the delay."
            if d["failure_type"] == "timing_failure"
            else "I could not reliably interpret the earlier request."
        )
        return prefix + ((" " + location) if location else " Please repeat the full item name.")
    if d["recovery_strategy"] == "confidence_check":
        return (
            "I may have misunderstood. Please confirm that you requested the " + item["name"] + "."
            if item
            else "I may have misunderstood. Please confirm the full item name."
        )
    if d["recovery_strategy"] == "partial_understanding_repair":
        understood = ("the " + d["item_family"]) if d["item_family"] else "part of the request"
        return f"I understood {understood}, but I missed a detail. Please repeat the full item name."
    if d["recovery_strategy"] == "clarifying_prompt":
        return "Sorry, I need to clarify. Please repeat the full item name, including any color or size."
    if d["recovery_strategy"] == "guided_reset":
        return "Let's start this request again. Please say the full item name."
    raise ValueError("Unsupported recovery strategy")
