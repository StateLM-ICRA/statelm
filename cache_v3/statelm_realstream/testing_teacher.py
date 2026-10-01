"""Deterministic teacher used ONLY to test the controller without a GPU.

It is a test double for the code paths (routing, caching, validity,
persistence).  It is never used to produce results: every number in the
experiment comes from the trained SLM's cached decisions.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contract import normalize_inventory, validate_decision
from .features import extract_review_features, extract_text_features, normal_text


def _decision(**updates: Any) -> dict[str, Any]:
    value = {
        "action": None,
        "item_family": None,
        "resolved_item_id": None,
        "attributes": {},
        "missing_attributes": [],
        "failure_detected": False,
        "failure_type": "none",
        "recovery_strategy": "none",
    }
    value.update(updates)
    return value


class OracleTeacher:
    """A small reference policy for controller tests; never the SLM result."""

    def __init__(self, inventory: Sequence[Mapping[str, Any]]):
        self.inventory = normalize_inventory(inventory)
        self.calls = 0

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        text = str(payload["conversation"][-1]["text"])
        normalized = normal_text(text)
        inventory = normalize_inventory(payload.get("inventory_candidates") or self.inventory)
        features = extract_text_features(text, inventory)
        by_id = {item["id"]: item for item in inventory}
        memory = payload.get("memory") or {}
        proposal = payload.get("proposed_robot_answer")

        item_id = None
        if len(features.item_ids) == 1:
            item_id = features.item_ids[0]
        elif not features.item_ids and memory.get("pending_item_family"):
            attrs = dict(features.attributes)
            matches = [
                item
                for item in inventory
                if item["family"] == memory["pending_item_family"]
                and all(item["attributes"].get(key) == value for key, value in attrs.items())
            ]
            if len(matches) == 1:
                item_id = matches[0]["id"]

        if proposal is not None:
            review = extract_review_features(text, proposal, inventory, requested_item_id=item_id)
            item = by_id.get(item_id) if item_id else None
            fields = (
                dict(item_family=item["family"], resolved_item_id=item["id"],
                     attributes=dict(item["attributes"]))
                if item
                else {}
            )
            typed = review.proposal_typed or ""
            if item and "<item:other>" in typed:
                d = _decision(action="recover", failure_detected=True,
                              failure_type="comprehension_failure",
                              recovery_strategy="self_correction", **fields)
            elif item and "<drawer:mismatch>" in typed:
                d = _decision(action="recover", failure_detected=True,
                              failure_type="search_failure",
                              recovery_strategy="self_correction", **fields)
            elif "delayed" in normal_text(proposal):
                d = _decision(action="recover", failure_detected=True,
                              failure_type="timing_failure",
                              recovery_strategy="transparency_cue", **fields)
            elif "not understand" in normal_text(proposal) or "did not understand" in normal_text(proposal):
                d = _decision(action="recover", failure_detected=True,
                              failure_type="comprehension_failure",
                              recovery_strategy="clarifying_prompt", **fields)
            elif item and "<drawer:match>" in typed:
                d = _decision(action="locate", **fields)
            elif item:
                d = _decision(action="recover", failure_detected=True,
                              failure_type="speech_failure",
                              recovery_strategy="specific_redirection", **fields)
            else:
                d = _decision(action="recover", failure_detected=True,
                              failure_type="comprehension_failure",
                              recovery_strategy="clarifying_prompt")
            return validate_decision(d, payload)

        if any(cue in normalized for cue in ("never mind", "cancel", "stop the request")):
            return validate_decision(_decision(action="cancel"), payload)
        if features.correction:
            target = features.item_ids[-1] if features.item_ids else None
            if target in by_id:
                item = by_id[target]
                return validate_decision(
                    _decision(
                        action="recover",
                        item_family=item["family"],
                        resolved_item_id=target,
                        attributes=dict(item["attributes"]),
                        failure_detected=True,
                        failure_type="comprehension_failure",
                        recovery_strategy="self_correction",
                    ),
                    payload,
                )
        if any(
            phrase in normalized
            for phrase in ("joke", "weather", "what time", "your name", "who built", "music", "how are you")
        ):
            return validate_decision(_decision(action="redirect"), payload)
        if item_id in by_id:
            item = by_id[item_id]
            return validate_decision(
                _decision(
                    action="locate",
                    item_family=item["family"],
                    resolved_item_id=item["id"],
                    attributes=dict(item["attributes"]),
                ),
                payload,
            )
        family = features.families[-1] if len(features.families) == 1 else None
        if family is not None or features.ambiguous or len(features.item_ids) > 1:
            missing = []
            if family:
                family_items = [item for item in inventory if item["family"] == family]
                keys = {key for item in family_items for key in item["attributes"]}
                missing = sorted(
                    key for key in keys
                    if len({item["attributes"].get(key) for item in family_items}) > 1
                )
            return validate_decision(
                _decision(action="clarify", item_family=family, missing_attributes=missing),
                payload,
            )
        return validate_decision(_decision(action="not_found"), payload)
