"""State-aware semantic cache that turns verified SLM behaviour into FSM transitions.

Derived from the 21 Sept ``statelm_combined.router`` (paper-aligned router).
Changes relative to that version:

* Review mode (``pre_send_review``) embeds ``typed(user) || typed(proposal)``
  instead of the user utterance alone, so the robot's proposed line is part
  of the cached pattern (see ``features.extract_review_features``).
* Task validity additionally rejects, in review mode, a cached *approval*
  of a proposal that names another item or a wrong drawer, and a cached
  ``search_failure`` verdict on a proposal whose item and drawer are both
  correct.  Both facts are provable from the live inventory.
* A bound item must come from the user's own words (or dialogue memory);
  a decision that binds an item the request does not support is rejected
  (paper eq. 3, ``I(h) != bottom`` and ``i = I(h)``).
* ``seed_example`` installs success-labelled transitions as active patterns
  (the initial FSM G0 built from success data only).
* ``route`` returns the typed text and the winning similarity for logging.

Thresholds are fixed and validation-selected exactly as in the paper
(tau_z merge, tau_r routing, tau_c admission, minimum distinct evidence).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import json
import math
import random
import time
import uuid
from typing import Any, Callable, Mapping, Sequence

from .contract import decision_dict, normalize_inventory, render_response, validate_decision
from .features import (
    Embedder,
    HashingEmbedder,
    TextFeatures,
    cosine,
    extract_review_features,
    extract_text_features,
    is_correction,
    normal_text,
    tokens,
)


@dataclass(frozen=True)
class RouterConfig:
    """Validation-selected thresholds corresponding to the paper's tau values."""

    merge_similarity: float = 0.78  # tau_z
    routing_similarity: float = 0.82  # tau_r
    admission_threshold: float = 0.85  # tau_c
    minimum_observations: int = 2  # distinct evidence ids before admission
    audit_probability: float = 0.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        for name in ("merge_similarity", "routing_similarity", "admission_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if not 0.0 <= self.audit_probability <= 1.0:
            raise ValueError("audit_probability must be between 0 and 1")


@dataclass
class CachedPattern:
    """One executable cached pattern p=(state, embedding, action, successor)."""

    pattern_id: str
    state: str
    typed_example: str
    centroid: tuple[float, ...]
    template: dict[str, Any]
    next_state: str
    observations: int = 0
    successes: int = 0
    status: str = "candidate"
    served: int = 0
    audits: int = 0
    audit_failures: int = 0
    origin: str = "online"
    evidence_ids: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def reliability(self) -> float:
        """Beta(1,1)-smoothed score c=(k+1)/(n+2) from the paper (eq. 4)."""

        return (self.successes + 1.0) / (self.observations + 2.0)

    def observe(self, vector: Sequence[float], success: bool, evidence_id: str) -> None:
        previous = self.observations
        self.observations += 1
        self.successes += int(success)
        self.evidence_ids.add(str(evidence_id))
        if success:
            weighted = [
                self.centroid[index] * max(1, previous) + float(vector[index])
                for index in range(len(self.centroid))
            ]
            norm = math.sqrt(sum(value * value for value in weighted)) or 1.0
            self.centroid = tuple(value / norm for value in weighted)
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["centroid"] = list(self.centroid)
        value["evidence_ids"] = sorted(self.evidence_ids)
        value["reliability"] = self.reliability
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CachedPattern":
        fields = dict(value)
        fields.pop("reliability", None)
        fields["centroid"] = tuple(float(x) for x in fields["centroid"])
        fields["evidence_ids"] = set(str(x) for x in fields.get("evidence_ids", []))
        fields.setdefault("origin", "online")
        return cls(**fields)


@dataclass(frozen=True)
class RouteResult:
    decision: dict[str, Any]
    response: str
    source: str
    teacher_called: bool
    state: str
    next_state: str
    typed_text: str = ""
    similarity: float | None = None
    pattern_id: str | None = None
    audited: bool = False
    reason: str | None = None


def _latest_user_text(payload: Mapping[str, Any]) -> str:
    for turn in reversed(payload.get("conversation") or []):
        if turn.get("role") == "user":
            return str(turn.get("text", ""))
    return ""


def _state_signature(payload: Mapping[str, Any]) -> str:
    memory = payload.get("memory") or {}
    value = {
        "state": payload.get("state", "awaiting_request"),
        "mode": payload.get("decision_mode", "dialogue_decision"),
        "last_action": memory.get("last_robot_action"),
        "has_pending_family": bool(memory.get("pending_item_family")),
        "has_resolved_item": bool(memory.get("resolved_item_id")),
        "has_missing_attributes": bool(memory.get("missing_attributes")),
        "has_proposal": payload.get("proposed_robot_answer") is not None,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _next_state(decision: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    action = decision["action"]
    if action == "cancel":
        return "ended"
    if action == "clarify":
        return "awaiting_clarification"
    if action in {"locate", "recover"}:
        return "located"
    return "awaiting_request" if action == "not_found" else str(payload.get("state", "awaiting_request"))


class PaperAlignedRouter:
    """Route known task-valid patterns to an FSM and unfamiliar ones to an SLM.

    The teacher returns a structured decision or a JSON string.  The
    validator is called both when admitting and when reusing a transition:
    similarity alone never authorises a drawer response.
    """

    def __init__(
        self,
        teacher: Callable[[dict[str, Any]], Any],
        inventory: Sequence[Mapping[str, Any]],
        *,
        validator: Callable[[Any, Mapping[str, Any]], dict[str, Any]] = validate_decision,
        embedder: Embedder | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        self.teacher = teacher
        self.inventory = normalize_inventory(inventory)
        self.validator = validator
        self.embedder = embedder or HashingEmbedder()
        self.config = config or RouterConfig()
        self.patterns: list[CachedPattern] = []
        self.teacher_calls = 0
        self.fsm_calls = 0
        self.guard_repairs = 0
        self._rng = random.Random(self.config.random_seed)
        self._last_pattern_by_session: dict[str, str] = {}

    # Inventory helpers
    def _by_id(self) -> dict[str, dict[str, Any]]:
        return {item["id"]: item for item in self.inventory}

    def _payload_inventory(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        candidates = payload.get("inventory_candidates")
        if candidates:
            return normalize_inventory(candidates)
        return self.inventory

    def _features(self, payload: Mapping[str, Any]) -> TextFeatures:
        inventory = self._payload_inventory(payload)
        user_text = _latest_user_text(payload)
        base = extract_text_features(user_text, inventory)
        proposal = payload.get("proposed_robot_answer")
        if proposal is None:
            return base
        requested = self._resolve_item(base, payload, allow_memory=True, inventory=inventory)
        return extract_review_features(user_text, proposal, inventory, requested_item_id=requested)

    # Item resolution from the user's own words
    def _resolve_item(
        self,
        features: TextFeatures,
        payload: Mapping[str, Any],
        *,
        allow_memory: bool = True,
        inventory: Sequence[Mapping[str, Any]] | None = None,
    ) -> str | None:
        inventory = normalize_inventory(inventory) if inventory else self._payload_inventory(payload)
        by_id = {item["id"]: item for item in inventory}
        if features.correction:
            latest = _latest_user_text(payload)
            normalized_latest = " ".join(tokens(latest))
            for cue in ("i meant", "no i said", "i said"):
                cue_index = normalized_latest.rfind(cue)
                if cue_index < 0:
                    continue
                target_text = normalized_latest[cue_index + len(cue) :].strip()
                target = extract_text_features(target_text, inventory)
                if len(target.item_ids) == 1 and target.item_ids[0] in by_id:
                    return target.item_ids[0]
                target_attrs = dict(target.attributes)
                target_family = target.families[0] if len(target.families) == 1 else None
                if target_family:
                    target_matches = [
                        item
                        for item in inventory
                        if item["family"] == target_family
                        and all(
                            item["attributes"].get(key) == value
                            for key, value in target_attrs.items()
                        )
                    ]
                    if len(target_matches) == 1:
                        return target_matches[0]["id"]
                break

            rejection_only = any(
                cue in normalized_latest
                for cue in ("did not mean", "didn t mean", "not that one")
            )
            if len(features.item_ids) == 1 and not rejection_only:
                return features.item_ids[0] if features.item_ids[0] in by_id else None

            # A short rejection such as "No, I did not mean tape" may name only
            # the robot's wrong item.  Recover the intended item from the
            # preceding user request instead of treating the rejected family
            # as a new request.
            conversation = list(payload.get("conversation") or [])
            for turn in reversed(conversation[:-1]):
                if turn.get("role") != "user":
                    continue
                prior = extract_text_features(turn.get("text", ""), inventory)
                if len(prior.item_ids) == 1 and prior.item_ids[0] in by_id:
                    return prior.item_ids[0]
                prior_attrs = dict(prior.attributes)
                prior_family = prior.families[0] if len(prior.families) == 1 else None
                if prior_family:
                    matches = [
                        item
                        for item in inventory
                        if item["family"] == prior_family
                        and all(
                            item["attributes"].get(key) == value
                            for key, value in prior_attrs.items()
                        )
                    ]
                    if len(matches) == 1:
                        return matches[0]["id"]
                if prior.item_ids or prior.families or prior.attributes:
                    break
        if len(features.item_ids) == 1:
            return features.item_ids[0] if features.item_ids[0] in by_id else None
        if len(features.item_ids) > 1:
            return None
        attrs = dict(features.attributes)
        memory = payload.get("memory") or {}
        family = features.families[-1] if len(features.families) == 1 else None
        if family is None and attrs:
            family = memory.get("pending_item_family")
        if family:
            matches = [
                item
                for item in inventory
                if item["family"] == family
                and all(item["attributes"].get(key) == val for key, val in attrs.items())
            ]
            if len(matches) == 1:
                return matches[0]["id"]
        if allow_memory and not attrs and not features.families:
            words = set(tokens(_latest_user_text(payload)))
            if words & {"it", "one", "that", "again"}:
                remembered = memory.get("resolved_item_id")
                if remembered in by_id:
                    return str(remembered)
        return None

    def _resolve_family(self, features: TextFeatures, payload: Mapping[str, Any]) -> str | None:
        if len(features.families) == 1:
            return features.families[0]
        item_id = self._resolve_item(features, payload)
        if item_id:
            by_id = {item["id"]: item for item in self._payload_inventory(payload)}
            if item_id in by_id:
                return by_id[item_id]["family"]
        memory = payload.get("memory") or {}
        value = memory.get("pending_item_family")
        return str(value) if value else None

    # Templates: abstract a concrete decision into its binding sources
    def _compile_template(
        self,
        decision: Mapping[str, Any],
        features: TextFeatures,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        d = decision_dict(decision)
        memory = payload.get("memory") or {}
        item_source = "none"
        if d["resolved_item_id"] is not None:
            current = self._resolve_item(features, payload, allow_memory=False)
            if d["resolved_item_id"] == current:
                item_source = "current_resolution"
            elif d["resolved_item_id"] == memory.get("resolved_item_id"):
                item_source = "memory_item"
            else:
                return None
        family_source = "none"
        if d["resolved_item_id"] is not None:
            family_source = "resolved_item"
        elif d["item_family"] is not None:
            if d["item_family"] in features.families:
                family_source = "current_family"
            elif d["item_family"] == memory.get("pending_item_family"):
                family_source = "memory_family"
            else:
                return None
        attribute_sources: dict[str, str] = {}
        if d["resolved_item_id"] is None:
            current_attrs = dict(features.attributes)
            memory_attrs = memory.get("known_attributes") or {}
            for key, value in d["attributes"].items():
                if current_attrs.get(key) == value:
                    attribute_sources[key] = "current_attribute"
                elif memory_attrs.get(key) == value:
                    attribute_sources[key] = "memory_attribute"
                else:
                    return None
        return {
            "action": d["action"],
            "item_source": item_source,
            "family_source": family_source,
            "attribute_sources": attribute_sources,
            "missing_attributes": list(d["missing_attributes"]),
            "failure_detected": d["failure_detected"],
            "failure_type": d["failure_type"],
            "recovery_strategy": d["recovery_strategy"],
        }

    def _instantiate_template(
        self,
        template: Mapping[str, Any],
        features: TextFeatures,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        by_id = {item["id"]: item for item in self._payload_inventory(payload)}
        memory = payload.get("memory") or {}
        if template["item_source"] == "current_resolution":
            item_id = self._resolve_item(features, payload, allow_memory=False)
        elif template["item_source"] == "memory_item":
            item_id = memory.get("resolved_item_id")
        elif template["item_source"] == "none":
            item_id = None
        else:
            raise ValueError("Unknown item source")
        item = by_id.get(item_id) if item_id is not None else None
        if item_id is not None and item is None:
            raise ValueError("Bound item is absent from inventory")
        if template["item_source"] == "current_resolution" and item is None:
            raise ValueError("The request does not resolve to one inventory item")

        if template["family_source"] == "resolved_item":
            family = item["family"] if item else None
        elif template["family_source"] == "current_family":
            family = self._resolve_family(features, payload)
        elif template["family_source"] == "memory_family":
            family = memory.get("pending_item_family")
        elif template["family_source"] == "none":
            family = None
        else:
            raise ValueError("Unknown family source")

        if item is not None:
            attributes = dict(item["attributes"])
        else:
            attributes: dict[str, str] = {}
            current_attrs = dict(features.attributes)
            memory_attrs = memory.get("known_attributes") or {}
            for key, source in template.get("attribute_sources", {}).items():
                value = current_attrs.get(key) if source == "current_attribute" else memory_attrs.get(key)
                if value is None:
                    raise ValueError(f"Could not bind attribute {key}")
                attributes[key] = value
        return {
            "action": template["action"],
            "item_family": family,
            "resolved_item_id": item["id"] if item else None,
            "attributes": attributes,
            "missing_attributes": list(template.get("missing_attributes", [])),
            "failure_detected": bool(template.get("failure_detected")),
            "failure_type": template.get("failure_type", "none"),
            "recovery_strategy": template.get("recovery_strategy", "none"),
        }

    # Task validity (paper eq. 3 plus the review-mode checks)
    def _task_valid(
        self,
        decision: Mapping[str, Any],
        features: TextFeatures,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        validated = self.validator(decision, payload)
        action = validated["action"]
        by_id = {item["id"]: item for item in self._payload_inventory(payload)}
        requested_item = self._resolve_item(features, payload)
        looks_inventory = self._looks_like_inventory_request(payload)
        review = payload.get("proposed_robot_answer") is not None
        if action in {"locate", "recover"} and validated["resolved_item_id"] is not None:
            item = by_id.get(validated["resolved_item_id"])
            if item is None or not item.get("drawer"):
                raise ValueError("Item-drawer binding is not valid")
            if requested_item is None:
                raise ValueError("A bound item must come from the user's request or memory")
            if validated["resolved_item_id"] != requested_item:
                raise ValueError("Resolved item disagrees with the user's supported item")
        if action == "locate" and review:
            # Approving a proposal is provably wrong when the proposal names
            # another item or a drawer that is not the item's drawer.
            requested = by_id.get(requested_item) if requested_item else None
            if any(pid != requested_item for pid in features.proposal_items):
                raise ValueError("Cannot approve a proposal that names another item")
            if requested is not None and any(
                normal_text(drawer) != normal_text(requested["drawer"])
                for drawer in features.proposal_drawers
            ):
                raise ValueError("Cannot approve a proposal with a wrong drawer")
        if action == "recover" and review and validated["failure_type"] == "search_failure":
            requested = by_id.get(requested_item) if requested_item else None
            same_item = all(pid == requested_item for pid in features.proposal_items)
            drawers = list(features.proposal_drawers)
            if (
                requested is not None
                and same_item
                and drawers
                and all(normal_text(d) == normal_text(requested["drawer"]) for d in drawers)
            ):
                raise ValueError("A correct drawer cannot be a search failure")
        if action == "clarify":
            if requested_item is not None:
                raise ValueError("A uniquely resolved item should not be clarified")
            family = validated["item_family"]
            if family is not None:
                candidates = [item for item in self._payload_inventory(payload) if item["family"] == family]
                if len(candidates) < 2 and not validated["missing_attributes"]:
                    raise ValueError("Clarification does not have multiple candidates")
            elif not (features.ambiguous or len(features.item_ids) > 1 or not features.item_ids):
                raise ValueError("Generic clarification is unsupported by the current request")
        if action == "not_found":
            if features.item_ids or features.families:
                raise ValueError("A present inventory match cannot be reported missing")
            if not looks_inventory:
                raise ValueError("An unrelated request cannot be reported as a missing item")
            if validated["item_family"] is not None or validated["attributes"] or validated["missing_attributes"]:
                raise ValueError("not_found must not inject unsupported item fields")
        if action == "redirect" and (features.item_ids or features.families or looks_inventory):
            raise ValueError("An inventory request cannot be redirected as irrelevant")
        if action == "recover" and not (
            features.correction
            or review
            or (payload.get("memory") or {}).get("last_robot_action")
        ):
            raise ValueError("Recovery lacks evidence of an earlier/proposed answer")
        return validated

    @staticmethod
    def _looks_like_inventory_request(payload: Mapping[str, Any]) -> bool:
        ordered = tokens(_latest_user_text(payload))
        words = set(ordered)
        if any(left == "get" and right == "to" for left, right in zip(ordered, ordered[1:])):
            return False
        asks_if_present = any(
            ordered[index : index + 3] == ("is", "there", "any")
            for index in range(max(0, len(ordered) - 2))
        )
        return bool(
            asks_if_present
            or words
            & {
                "where",
                "find",
                "locate",
                "need",
                "have",
                "get",
                "grab",
                "drawer",
                "inventory",
                "looking",
                "show",
                "give",
                "retrieve",
                "bring",
                "fetch",
            }
        )

    def _guard_decision(
        self,
        features: TextFeatures,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Deterministic task guard applied when the checked SLM output is unsafe.

        The guard never guesses an item or drawer.  It derives only behaviour
        that is provable from the inventory and the dialogue state.
        """

        base = {
            "action": None,
            "item_family": None,
            "resolved_item_id": None,
            "attributes": {},
            "missing_attributes": [],
            "failure_detected": False,
            "failure_type": "none",
            "recovery_strategy": "none",
        }
        by_id = {item["id"]: item for item in self._payload_inventory(payload)}
        item_id = self._resolve_item(features, payload)
        review = payload.get("proposed_robot_answer") is not None
        if review:
            # Only locate/recover are allowed while reviewing a proposal.
            if item_id is not None:
                item = by_id[item_id]
                base.update(item_family=item["family"], resolved_item_id=item["id"],
                            attributes=dict(item["attributes"]))
                other_item = any(pid != item_id for pid in features.proposal_items)
                wrong_drawer = any(
                    normal_text(d) != normal_text(item["drawer"]) for d in features.proposal_drawers
                )
                if other_item:
                    base.update(action="recover", failure_detected=True,
                                failure_type="comprehension_failure",
                                recovery_strategy="self_correction")
                elif wrong_drawer:
                    base.update(action="recover", failure_detected=True,
                                failure_type="search_failure",
                                recovery_strategy="specific_redirection")
                elif features.proposal_drawers:
                    base.update(action="locate")
                else:
                    base.update(action="recover", failure_detected=True,
                                failure_type="comprehension_failure",
                                recovery_strategy="confidence_check")
            else:
                base.update(action="recover", failure_detected=True,
                            failure_type="comprehension_failure",
                            recovery_strategy="clarifying_prompt")
            return self._task_valid(base, features, payload)

        if item_id is not None:
            item = by_id[item_id]
            base.update(
                action="recover" if features.correction else "locate",
                item_family=item["family"],
                resolved_item_id=item["id"],
                attributes=dict(item["attributes"]),
            )
            if features.correction:
                base.update(
                    failure_detected=True,
                    failure_type="comprehension_failure",
                    recovery_strategy="self_correction",
                )
            return self._task_valid(base, features, payload)

        family = self._resolve_family(features, payload)
        if family is not None or features.ambiguous or len(features.item_ids) > 1:
            missing: list[str] = []
            if family is not None:
                family_items = [item for item in self._payload_inventory(payload) if item["family"] == family]
                attribute_keys = {key for item in family_items for key in item["attributes"]}
                missing = sorted(
                    key
                    for key in attribute_keys
                    if len({item["attributes"].get(key) for item in family_items}) > 1
                )
            base.update(action="clarify", item_family=family, missing_attributes=missing)
            return self._task_valid(base, features, payload)
        if self._looks_like_inventory_request(payload):
            base["action"] = "not_found"
            return self._task_valid(base, features, payload)
        latest = _latest_user_text(payload).casefold()
        if any(word in latest for word in ("cancel", "never mind", "stop the request")):
            base["action"] = "cancel"
        else:
            base["action"] = "redirect"
        return self._task_valid(base, features, payload)

    def _checked_teacher(
        self,
        payload: dict[str, Any],
        features: TextFeatures,
    ) -> tuple[dict[str, Any], str | None]:
        try:
            raw = self.teacher(payload)
        except Exception as exc:
            guarded = self._guard_decision(features, payload)
            if guarded is None:
                raise
            self.guard_repairs += 1
            return guarded, f"teacher_failure_guard:{type(exc).__name__}"
        try:
            return self._task_valid(raw, features, payload), None
        except Exception as exc:
            guarded = self._guard_decision(features, payload)
            if guarded is None:
                raise
            self.guard_repairs += 1
            return guarded, f"task_guard_repair:{type(exc).__name__}:{exc}"

    # Pattern learning
    @staticmethod
    def _template_key(template: Mapping[str, Any]) -> str:
        return json.dumps(template, sort_keys=True, separators=(",", ":"))

    def _maybe_promote(self, pattern: CachedPattern) -> None:
        if pattern.origin == "seed" and pattern.audit_failures == 0:
            pattern.status = "active"
            return
        ready = (
            pattern.observations >= self.config.minimum_observations
            and len(pattern.evidence_ids) >= self.config.minimum_observations
            and pattern.reliability >= self.config.admission_threshold
        )
        pattern.status = "active" if ready else "candidate"

    def _observe_teacher(
        self,
        payload: Mapping[str, Any],
        features: TextFeatures,
        vector: Sequence[float],
        decision: Mapping[str, Any],
        evidence_id: str,
        *,
        origin: str = "online",
    ) -> CachedPattern | None:
        template = self._compile_template(decision, features, payload)
        if template is None:
            return None
        instantiated = self._instantiate_template(template, features, payload)
        try:
            checked = self._task_valid(instantiated, features, payload)
            success = decision_dict(checked) == decision_dict(decision)
        except Exception:
            success = False
        state = _state_signature(payload)
        successor = _next_state(decision, payload)
        key = self._template_key(template)
        compatible = [
            (cosine(pattern.centroid, vector), pattern)
            for pattern in self.patterns
            if pattern.state == state
            and pattern.next_state == successor
            and self._template_key(pattern.template) == key
        ]
        compatible = [pair for pair in compatible if pair[0] >= self.config.merge_similarity]
        if compatible:
            _score, pattern = max(compatible, key=lambda pair: pair[0])
        else:
            pattern = CachedPattern(
                pattern_id="p_" + uuid.uuid4().hex[:16],
                state=state,
                typed_example=features.typed_text,
                centroid=tuple(float(x) for x in vector),
                template=copy.deepcopy(template),
                next_state=successor,
                origin=origin,
            )
            self.patterns.append(pattern)
        pattern.observe(vector, success, evidence_id)
        self._maybe_promote(pattern)
        return pattern

    def seed_example(
        self,
        payload: dict[str, Any],
        decision: Mapping[str, Any],
        evidence_id: str,
    ) -> CachedPattern | None:
        """Install a success-labelled transition as an active pattern (G0).

        The decision must pass the task-validity check for the payload; a
        seed that does not is refused (returns None).
        """

        features = self._features(payload)
        try:
            checked = self._task_valid(decision, features, payload)
        except Exception:
            return None
        vector = self.embedder.encode(features.typed_text)
        return self._observe_teacher(payload, features, vector, checked, evidence_id, origin="seed")

    def _active_match(
        self,
        payload: Mapping[str, Any],
        features: TextFeatures,
        vector: Sequence[float],
    ) -> tuple[CachedPattern, dict[str, Any], float] | None:
        state = _state_signature(payload)
        candidates = sorted(
            (
                (cosine(pattern.centroid, vector), pattern)
                for pattern in self.patterns
                if pattern.status == "active" and pattern.state == state
            ),
            key=lambda pair: (pair[0], pair[1].reliability),
            reverse=True,
        )
        for similarity, pattern in candidates:
            if similarity < self.config.routing_similarity:
                break
            try:
                decision = self._instantiate_template(pattern.template, features, payload)
                checked = self._task_valid(decision, features, payload)
            except Exception:
                continue
            return pattern, checked, similarity
        return None

    def best_similarity(self, payload: Mapping[str, Any]) -> tuple[float | None, str | None]:
        """Highest cosine to any active pattern in the same state (for logs)."""

        features = self._features(payload)
        vector = self.embedder.encode(features.typed_text)
        state = _state_signature(payload)
        best: tuple[float, str] | None = None
        for pattern in self.patterns:
            if pattern.status != "active" or pattern.state != state:
                continue
            score = cosine(pattern.centroid, vector)
            if best is None or score > best[0]:
                best = (score, pattern.pattern_id)
        return (best[0], best[1]) if best else (None, None)

    def record_feedback(self, pattern_id: str, success: bool, evidence_id: str) -> None:
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                pattern.observe(pattern.centroid, success, evidence_id)
                if not success:
                    pattern.audit_failures += 1
                self._maybe_promote(pattern)
                return
        raise KeyError(pattern_id)

    # Routing (paper eq. 1)
    def route(
        self,
        payload: dict[str, Any],
        *,
        evidence_id: str | None = None,
        session_id: str = "default",
        feedback: bool | None = None,
        learn: bool = True,
    ) -> RouteResult:
        evidence_id = evidence_id or f"online:{uuid.uuid4()}"
        features = self._features(payload)
        vector = self.embedder.encode(features.typed_text)
        state = _state_signature(payload)
        inventory = self._payload_inventory(payload)
        previous_pattern = self._last_pattern_by_session.get(session_id)
        if learn and previous_pattern is not None and feedback is not None:
            self.record_feedback(previous_pattern, feedback, f"feedback:{evidence_id}")
        elif learn and previous_pattern is not None and is_correction(_latest_user_text(payload)):
            self.record_feedback(previous_pattern, False, f"correction:{evidence_id}")

        match = self._active_match(payload, features, vector)
        if match is not None:
            pattern, cached, similarity = match
            audit = self._rng.random() < self.config.audit_probability
            if not audit:
                pattern.served += 1
                pattern.updated_at = time.time()
                self.fsm_calls += 1
                self._last_pattern_by_session[session_id] = pattern.pattern_id
                return RouteResult(
                    decision=cached,
                    response=render_response(cached, inventory),
                    source="fsm",
                    teacher_called=False,
                    state=state,
                    next_state=pattern.next_state,
                    typed_text=features.typed_text,
                    similarity=similarity,
                    pattern_id=pattern.pattern_id,
                )

            self.teacher_calls += 1
            teacher_decision, guard_reason = self._checked_teacher(payload, features)
            pattern.audits += 1
            if decision_dict(cached) == decision_dict(teacher_decision):
                if learn:
                    pattern.observe(vector, True, f"audit:{evidence_id}")
                    self._maybe_promote(pattern)
                pattern.served += 1
                self.fsm_calls += 1
                self._last_pattern_by_session[session_id] = pattern.pattern_id
                return RouteResult(
                    decision=cached,
                    response=render_response(cached, inventory),
                    source="fsm",
                    teacher_called=True,
                    state=state,
                    next_state=pattern.next_state,
                    typed_text=features.typed_text,
                    similarity=similarity,
                    pattern_id=pattern.pattern_id,
                    audited=True,
                )
            pattern.audit_failures += 1
            learned = None
            if learn:
                pattern.observe(vector, False, f"audit:{evidence_id}")
                self._maybe_promote(pattern)
                learned = self._observe_teacher(
                    payload, features, vector, teacher_decision, f"teacher:{evidence_id}"
                )
            self._last_pattern_by_session.pop(session_id, None)
            return RouteResult(
                decision=teacher_decision,
                response=render_response(teacher_decision, inventory),
                source="slm",
                teacher_called=True,
                state=state,
                next_state=_next_state(teacher_decision, payload),
                typed_text=features.typed_text,
                similarity=similarity,
                pattern_id=learned.pattern_id if learned else None,
                audited=True,
                reason="audit_disagreement" + (f";{guard_reason}" if guard_reason else ""),
            )

        self.teacher_calls += 1
        teacher_decision, guard_reason = self._checked_teacher(payload, features)
        learned = None
        if learn:
            learned = self._observe_teacher(payload, features, vector, teacher_decision, evidence_id)
        self._last_pattern_by_session.pop(session_id, None)
        best_similarity, _best_id = self.best_similarity(payload)
        return RouteResult(
            decision=teacher_decision,
            response=render_response(teacher_decision, inventory),
            source="slm",
            teacher_called=True,
            state=state,
            next_state=_next_state(teacher_decision, payload),
            typed_text=features.typed_text,
            similarity=best_similarity,
            pattern_id=learned.pattern_id if learned else None,
            reason=guard_reason,
        )

    # Persistence
    def summary(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.name,
            "config": asdict(self.config),
            "patterns": len(self.patterns),
            "active_patterns": sum(pattern.status == "active" for pattern in self.patterns),
            "seed_patterns": sum(pattern.origin == "seed" for pattern in self.patterns),
            "teacher_calls": self.teacher_calls,
            "fsm_calls": self.fsm_calls,
            "guard_repairs": self.guard_repairs,
            "fsm_share": self.fsm_calls / max(1, self.fsm_calls + self.teacher_calls),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "format": "statelm-realstream-cache-v1",
            "embedder": self.embedder.name,
            "config": asdict(self.config),
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "counters": {
                "teacher_calls": self.teacher_calls,
                "fsm_calls": self.fsm_calls,
                "guard_repairs": self.guard_repairs,
            },
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        teacher: Callable[[dict[str, Any]], Any],
        inventory: Sequence[Mapping[str, Any]],
        *,
        validator: Callable[[Any, Mapping[str, Any]], dict[str, Any]] = validate_decision,
        embedder: Embedder | None = None,
    ) -> "PaperAlignedRouter":
        if snapshot.get("format") != "statelm-realstream-cache-v1":
            raise ValueError("Unsupported snapshot format")
        router = cls(
            teacher,
            inventory,
            validator=validator,
            embedder=embedder,
            config=RouterConfig(**dict(snapshot["config"])),
        )
        if snapshot.get("embedder") != router.embedder.name:
            raise ValueError("Snapshot embedding backend does not match")
        router.patterns = [CachedPattern.from_dict(value) for value in snapshot.get("patterns", [])]
        counters = snapshot.get("counters") or {}
        router.teacher_calls = int(counters.get("teacher_calls", 0))
        router.fsm_calls = int(counters.get("fsm_calls", 0))
        router.guard_repairs = int(counters.get("guard_repairs", 0))
        return router
