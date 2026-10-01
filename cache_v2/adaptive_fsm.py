"""Guarded online FSM/cache for the structured Robot SLM runtime.

The router starts empty.  A request is served by the teacher SLM until a
state-conditioned transition has enough distinct, externally verified
exposures to become active.  Production evidence identifiers should represent
separate real interactions; repeated corpus replay is only a mechanics test.
Active transitions remain subject to the original
runtime validator and sampled teacher audits.

Only structured decisions are stored.  Raw dialogue text, rendered responses,
and drawer facts are never persisted in the transition entries.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


FORMAT_VERSION = 2
EXTRACTOR_VERSION = "robot-progressive-fsm-v1"
TOKEN_RE = re.compile(r"[a-z0-9]+")
DRAWER_RE = re.compile(r"\bdrawer\s+([a-z0-9]+)\b", re.IGNORECASE)
REASON_CODE_RE = re.compile(r"[a-z][a-z0-9_:-]{0,79}")
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normal_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(normalized.split())


def _tokens(text: Any) -> list[str]:
    return TOKEN_RE.findall(_normal_text(text))


def _decision_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise TypeError("A decision must be a mapping or expose model_dump()")
    missing = [key for key in DECISION_KEYS if key not in value]
    if missing:
        raise ValueError(f"Decision is missing keys: {missing}")
    return {key: copy.deepcopy(value[key]) for key in DECISION_KEYS}


def decisions_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(_decision_dict(left)) == _canonical_json(_decision_dict(right))
    except Exception:
        return False


def wilson_lower_bound(successes: int, failures: int, z: float = 1.96) -> float:
    """Two-sided 95% Wilson lower bound by default."""
    n = successes + failures
    if n <= 0:
        return 0.0
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - radius) / denominator)


@dataclass(frozen=True)
class Thresholds:
    min_teacher_agreements: int
    min_shadow_observations: int
    min_verified_successes: int
    min_distinct_evidence: int
    min_verified_wilson: float
    audit_probability: float


@dataclass(frozen=True)
class RouterPolicy:
    """Promotion and audit policy.

    ``research`` is intentionally conservative.  ``development`` is provided
    for smoke tests and demonstrations and must not be described as a safety
    guarantee.
    """

    routine: Thresholds
    risky: Thresholds
    promotion_mode: str = "verified_outcome"
    reject_any_verified_failure: bool = True
    reject_any_teacher_conflict: bool = True
    max_events_in_snapshot: int = 100_000
    max_traces_in_memory: int = 100_000
    pending_trace_ttl_seconds: float = 86_400.0
    timing_buckets_ms: tuple[float, ...] = (1000.0, 3000.0, 7000.0, 15000.0)

    @classmethod
    def research(cls) -> "RouterPolicy":
        return cls(
            routine=Thresholds(20, 20, 35, 35, 0.90, 0.05),
            risky=Thresholds(40, 40, 189, 189, 0.98, 0.10),
        )

    @classmethod
    def development(cls) -> "RouterPolicy":
        return cls(
            routine=Thresholds(2, 1, 2, 2, 0.34, 0.05),
            risky=Thresholds(3, 2, 3, 3, 0.43, 0.10),
        )

    @classmethod
    def calibrated(cls) -> "RouterPolicy":
        """Middle preset for corpus calibration before a frozen held-out test."""
        return cls(
            routine=Thresholds(5, 2, 5, 5, 0.55, 0.05),
            risky=Thresholds(10, 3, 10, 10, 0.72, 0.10),
        )

    @classmethod
    def imitation(cls) -> "RouterPolicy":
        """Open-world student policy learned only from consistent SLM outputs.

        External labels are not required for promotion. Distinct evidence IDs
        must correspond to separate traffic observations in production. Real
        task feedback remains useful as a rollback signal.
        """
        return cls(
            routine=Thresholds(8, 3, 0, 8, 0.0, 0.05),
            risky=Thresholds(15, 5, 0, 15, 0.0, 0.10),
            promotion_mode="imitate_teacher",
        )

    def thresholds_for(self, decision: Mapping[str, Any], decision_mode: str) -> Thresholds:
        risky = (
            decision_mode == "pre_send_review"
            or decision.get("action") in {"locate", "recover"}
            or bool(decision.get("failure_detected"))
        )
        return self.risky if risky else self.routine


@dataclass(frozen=True)
class TextFeatures:
    surface: str
    item_ids: tuple[str, ...]
    families: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    ambiguous: bool

    def last_attribute(self, key: str) -> str | None:
        for found_key, value in reversed(self.attributes):
            if found_key == key:
                return value
        return None


@dataclass(frozen=True)
class FeatureBundle:
    current: TextFeatures
    all_user: TextFeatures
    previous_robot: TextFeatures
    proposal: TextFeatures
    history_surfaces: tuple[str, ...]
    proposal_relation: str
    prior_robot_relation: str
    timing_bucket: str


@dataclass
class TransitionEntry:
    entry_id: str
    level: str
    key_hash: str
    decision_mode: str
    scenario: str
    decision: dict[str, Any] | None = None
    template: dict[str, Any] | None = None
    status: str = "candidate"
    teacher_agreements: int = 0
    teacher_disagreements: int = 0
    shadow_matches: int = 0
    shadow_mismatches: int = 0
    verified_successes: int = 0
    verified_failures: int = 0
    audit_matches: int = 0
    audit_mismatches: int = 0
    served: int = 0
    validator_failures: int = 0
    evidence_hashes: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    activated_at: float | None = None
    quarantine_reason: str | None = None
    generation: int = 0
    replaces_entry_id: str | None = None

    @property
    def shadow_observations(self) -> int:
        return self.shadow_matches + self.shadow_mismatches

    @property
    def verified_wilson(self) -> float:
        return wilson_lower_bound(self.verified_successes, self.verified_failures)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_hashes"] = sorted(self.evidence_hashes)
        value["verified_wilson"] = self.verified_wilson
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionEntry":
        fields = dict(value)
        fields.pop("verified_wilson", None)
        fields["evidence_hashes"] = set(fields.get("evidence_hashes", []))
        return cls(**fields)


@dataclass(frozen=True)
class RouteTrace:
    trace_id: str
    route: str
    primary: str
    teacher_called: bool
    audited: bool
    decision_mode: str
    scenario: str
    latency_ms: float
    exact_entry_id: str | None = None
    typed_entry_id: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _phrase_table(inventory: list[dict[str, Any]]) -> dict[tuple[str, ...], list[tuple[str, Any]]]:
    table: dict[tuple[str, ...], list[tuple[str, Any]]] = {}
    for item in inventory:
        for phrase in {item.get("name", ""), *item.get("aliases", [])}:
            key = tuple(_tokens(phrase))
            if key:
                table.setdefault(key, []).append(("item", item["id"]))
        family = tuple(_tokens(str(item.get("family", "")).replace("_", " ")))
        if family:
            table.setdefault(family, []).append(("family", item["family"]))
        for key, value in item.get("attributes", {}).items():
            phrase = tuple(_tokens(value))
            if phrase:
                table.setdefault(phrase, []).append(("attribute", (key, value)))
    return table


def extract_text_features(text: Any, inventory: list[dict[str, Any]]) -> TextFeatures:
    token_list = _tokens(text)
    table = _phrase_table(inventory)
    phrases = sorted(table, key=lambda phrase: (-len(phrase), phrase))
    priority = {"item": 3, "attribute": 2, "family": 1}
    surface: list[str] = []
    item_ids: list[str] = []
    families: list[str] = []
    attributes: list[tuple[str, str]] = []
    ambiguous = False
    index = 0
    while index < len(token_list):
        matches: list[tuple[int, int, tuple[str, Any]]] = []
        for phrase in phrases:
            width = len(phrase)
            if token_list[index : index + width] == list(phrase):
                matches.extend((width, priority[kind], (kind, value)) for kind, value in table[phrase])
        if not matches:
            surface.append(token_list[index])
            index += 1
            continue
        width = max(match[0] for match in matches)
        narrowed = [match for match in matches if match[0] == width]
        best_priority = max(match[1] for match in narrowed)
        descriptors = [match[2] for match in narrowed if match[1] == best_priority]
        kind = descriptors[0][0]
        values = {descriptor[1] for descriptor in descriptors}
        if len(values) != 1:
            ambiguous = True
            surface.append(f"<ambiguous:{kind}>")
        else:
            value = next(iter(values))
            if kind == "item":
                item_ids.append(value)
                surface.append("<item>")
            elif kind == "family":
                families.append(value)
                surface.append("<family>")
            else:
                key, attribute_value = value
                attributes.append((key, attribute_value))
                surface.append(f"<attr:{key}>")
        index += width
    if len(set(item_ids)) > 1:
        ambiguous = True
    if len(set(families)) > 1:
        ambiguous = True
    return TextFeatures(
        surface=" ".join(surface) or "<empty>",
        item_ids=tuple(item_ids),
        families=tuple(families),
        attributes=tuple(attributes),
        ambiguous=ambiguous,
    )


def _last_item(features: TextFeatures) -> str | None:
    return features.item_ids[-1] if features.item_ids else None


def _last_family(features: TextFeatures, by_id: Mapping[str, Mapping[str, Any]]) -> str | None:
    item_id = _last_item(features)
    if item_id in by_id:
        return str(by_id[item_id]["family"])
    return features.families[-1] if features.families else None


def _relation(requested: TextFeatures, answer: TextFeatures, inventory: list[dict[str, Any]], text: str) -> str:
    by_id = {item["id"]: item for item in inventory}
    requested_id = _last_item(requested)
    answer_id = _last_item(answer)
    if requested_id is None:
        return "unverifiable"
    if answer_id is None:
        return "missing_item"
    if requested_id != answer_id:
        return "wrong_item"
    drawer_match = DRAWER_RE.search(text or "")
    if not drawer_match:
        return "same_item_no_drawer"
    proposed_drawer = f"drawer {drawer_match.group(1).casefold()}"
    expected_drawer = _normal_text(by_id[requested_id].get("drawer", ""))
    return "correct" if proposed_drawer == expected_drawer else "wrong_drawer"


def _timing_bucket(timing: Mapping[str, Any], edges: Iterable[float]) -> str:
    value = timing.get("last_response_latency_ms")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "missing"
    number = float(value)
    previous = 0.0
    for edge in edges:
        if number <= edge:
            return f"{int(previous)}-{int(edge)}ms"
        previous = edge
    return f">{int(previous)}ms"


def extract_bundle(payload: Mapping[str, Any], policy: RouterPolicy) -> FeatureBundle:
    inventory = copy.deepcopy(payload.get("inventory_candidates") or [])
    if not inventory:
        raise ValueError("payload.inventory_candidates must be non-empty")
    conversation = list(payload.get("conversation") or [])
    latest_user = next(
        (str(turn.get("text", "")) for turn in reversed(conversation) if turn.get("role") == "user"),
        "",
    )
    prior_robot_text = next(
        (str(turn.get("text", "")) for turn in reversed(conversation[:-1]) if turn.get("role") == "robot"),
        "",
    )
    all_user_text = " \n ".join(
        str(turn.get("text", "")) for turn in conversation if turn.get("role") == "user"
    )
    current = extract_text_features(latest_user, inventory)
    all_user = extract_text_features(all_user_text, inventory)
    previous_robot = extract_text_features(prior_robot_text, inventory)
    proposal_text = str(payload.get("proposed_robot_answer") or "")
    proposal = extract_text_features(proposal_text, inventory)
    proposal_relation = (
        _relation(all_user, proposal, inventory, proposal_text)
        if payload.get("decision_mode") == "pre_send_review"
        else "none"
    )
    prior_robot_relation = (
        _relation(all_user, previous_robot, inventory, prior_robot_text)
        if prior_robot_text
        else "none"
    )
    history_surfaces = tuple(
        f"{turn.get('role')}:{extract_text_features(turn.get('text', ''), inventory).surface}"
        for turn in conversation[-8:]
    )
    return FeatureBundle(
        current=current,
        all_user=all_user,
        previous_robot=previous_robot,
        proposal=proposal,
        history_surfaces=history_surfaces,
        proposal_relation=proposal_relation,
        prior_robot_relation=prior_robot_relation,
        timing_bucket=_timing_bucket(payload.get("timing") or {}, policy.timing_buckets_ms),
    )


def _semantic_inventory(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "family": item.get("family"),
                "attributes": item.get("attributes") or {},
                "aliases": sorted(item.get("aliases") or []),
            }
            for item in inventory
        ],
        key=lambda item: str(item["id"]),
    )


def _location_inventory(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [{"id": item.get("id"), "drawer": item.get("drawer")} for item in inventory],
        key=lambda item: str(item["id"]),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_reason_code(reason: str | None) -> str:
    """Persist a bounded code, never arbitrary operator/user prose."""
    if reason is None:
        return "external_failure"
    candidate = str(reason).strip().casefold()
    if REASON_CODE_RE.fullmatch(candidate):
        return candidate
    return "external_failure:" + hashlib.sha256(str(reason).encode("utf-8")).hexdigest()[:12]


def _typed_state(payload: Mapping[str, Any], bundle: FeatureBundle) -> dict[str, Any]:
    memory = payload.get("memory") or {}
    inventory = payload.get("inventory_candidates") or []
    result = {
        "extractor_version": EXTRACTOR_VERSION,
        "decision_mode": payload.get("decision_mode", "dialogue_decision"),
        "state": payload.get("state"),
        "allowed_actions": sorted(payload.get("allowed_actions") or []),
        "memory": {
            "has_pending_family": bool(memory.get("pending_item_family")),
            "known_attribute_keys": sorted((memory.get("known_attributes") or {}).keys()),
            "missing_attributes": sorted(memory.get("missing_attributes") or []),
            "has_resolved_item": bool(memory.get("resolved_item_id")),
            "last_robot_action": memory.get("last_robot_action"),
            "recovery_count": min(int(memory.get("recovery_count") or 0), 2),
        },
        "history_surfaces": list(bundle.history_surfaces),
        "proposal_relation": bundle.proposal_relation,
        "prior_robot_relation": bundle.prior_robot_relation,
        "timing_bucket": bundle.timing_bucket,
        "semantic_inventory": _digest(_semantic_inventory(inventory)),
    }
    if payload.get("decision_mode") == "pre_send_review":
        result["location_inventory"] = _digest(_location_inventory(inventory))
    return result


def _strict_payload(payload: Mapping[str, Any], policy: RouterPolicy) -> dict[str, Any]:
    inventory = payload.get("inventory_candidates") or []
    result = copy.deepcopy(dict(payload))
    result["conversation"] = [
        {"role": turn.get("role"), "text": _normal_text(turn.get("text", ""))}
        for turn in result.get("conversation") or []
    ]
    result["proposed_robot_answer"] = (
        _normal_text(result.get("proposed_robot_answer"))
        if result.get("proposed_robot_answer") is not None
        else None
    )
    result["allowed_actions"] = sorted(result.get("allowed_actions") or [])
    result["timing"] = {
        "last_response_latency_bucket": _timing_bucket(
            result.get("timing") or {}, policy.timing_buckets_ms
        )
    }
    result["inventory_candidates"] = {
        "semantic": _digest(_semantic_inventory(inventory)),
        "locations": _digest(_location_inventory(inventory)),
    }
    return result


def scenario_for(decision: Mapping[str, Any], decision_mode: str) -> str:
    if decision.get("action") == "recover" or decision.get("failure_detected"):
        return f"failure:{decision.get('failure_type', 'unknown')}"
    if decision_mode == "pre_send_review":
        return "review:accepted"
    return f"success:{decision.get('action', 'unknown')}"


def _source_for_item(
    item_id: str,
    payload: Mapping[str, Any],
    bundle: FeatureBundle,
) -> str | None:
    if _last_item(bundle.current) == item_id:
        return "current_item"
    if _last_item(bundle.all_user) == item_id:
        return "context_item"
    if (payload.get("memory") or {}).get("resolved_item_id") == item_id:
        return "memory_item"
    return None


def _source_for_family(
    family: str,
    payload: Mapping[str, Any],
    bundle: FeatureBundle,
    by_id: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if _last_family(bundle.current, by_id) == family:
        return "current_family"
    if _last_family(bundle.all_user, by_id) == family:
        return "context_family"
    if (payload.get("memory") or {}).get("pending_item_family") == family:
        return "memory_family"
    return None


def compile_template(
    decision: Mapping[str, Any],
    payload: Mapping[str, Any],
    bundle: FeatureBundle,
) -> dict[str, Any] | None:
    """Compile only templates whose slots bind to current evidence.

    Any fixed item, family, or attribute value would risk replaying an old item
    into a new request, so such a transition remains exact-cache-only.
    """
    data = _decision_dict(decision)
    inventory = payload.get("inventory_candidates") or []
    by_id = {item["id"]: item for item in inventory}
    # Arbitrary proposal prose can reverse meaning while preserving the same
    # item/drawer mentions (for example, "do not use drawer A1").  Until a
    # trusted proposal grammar is available, review decisions may use only the
    # strict normalized-payload cache, never a delexicalized typed transition.
    if payload.get("decision_mode") == "pre_send_review":
        return None
    if bundle.current.ambiguous or bundle.all_user.ambiguous:
        return None

    item_id = data.get("resolved_item_id")
    if item_id is None:
        item_source = "none"
    else:
        item_source = _source_for_item(item_id, payload, bundle)
        if item_source is None:
            return None

    family = data.get("item_family")
    if item_id is not None:
        family_source = "resolved_item"
    elif family is None:
        family_source = "none"
    else:
        family_source = _source_for_family(family, payload, bundle, by_id)
        if family_source is None:
            return None

    attribute_sources: dict[str, str] = {}
    if item_id is None:
        current_attributes = dict(bundle.current.attributes)
        context_attributes = dict(bundle.all_user.attributes)
        memory_attributes = (payload.get("memory") or {}).get("known_attributes") or {}
        for key, value in (data.get("attributes") or {}).items():
            if current_attributes.get(key) == value:
                attribute_sources[key] = "current_attribute"
            elif context_attributes.get(key) == value:
                attribute_sources[key] = "context_attribute"
            elif memory_attributes.get(key) == value:
                attribute_sources[key] = "memory_attribute"
            else:
                return None

    return {
        "action": data["action"],
        "item_source": item_source,
        "family_source": family_source,
        "attribute_sources": attribute_sources,
        "missing_attributes": list(data.get("missing_attributes") or []),
        "failure_detected": bool(data.get("failure_detected")),
        "failure_type": data.get("failure_type", "none"),
        "recovery_strategy": data.get("recovery_strategy", "none"),
    }


def instantiate_template(
    template: Mapping[str, Any],
    payload: Mapping[str, Any],
    bundle: FeatureBundle,
) -> dict[str, Any]:
    inventory = payload.get("inventory_candidates") or []
    by_id = {item["id"]: item for item in inventory}
    memory = payload.get("memory") or {}

    item_source = template["item_source"]
    if item_source == "current_item":
        item_id = _last_item(bundle.current)
    elif item_source == "context_item":
        item_id = _last_item(bundle.all_user)
    elif item_source == "memory_item":
        item_id = memory.get("resolved_item_id")
    elif item_source == "none":
        item_id = None
    else:
        raise LookupError(f"Unknown item source: {item_source}")
    item = by_id.get(item_id) if item_id is not None else None
    if item_id is not None and item is None:
        raise LookupError("Bound item is absent from the live inventory")

    family_source = template["family_source"]
    if family_source == "resolved_item":
        family = item["family"] if item else None
    elif family_source == "current_family":
        family = _last_family(bundle.current, by_id)
    elif family_source == "context_family":
        family = _last_family(bundle.all_user, by_id)
    elif family_source == "memory_family":
        family = memory.get("pending_item_family")
    elif family_source == "none":
        family = None
    else:
        raise LookupError(f"Unknown family source: {family_source}")

    if item is not None:
        attributes = copy.deepcopy(item.get("attributes") or {})
    else:
        attributes = {}
        current_attributes = dict(bundle.current.attributes)
        context_attributes = dict(bundle.all_user.attributes)
        memory_attributes = memory.get("known_attributes") or {}
        for key, source in (template.get("attribute_sources") or {}).items():
            if source == "current_attribute":
                value = current_attributes.get(key)
            elif source == "context_attribute":
                value = context_attributes.get(key)
            elif source == "memory_attribute":
                value = memory_attributes.get(key)
            else:
                raise LookupError(f"Unknown attribute source: {source}")
            if value is None:
                raise LookupError(f"Could not bind attribute {key}")
            attributes[key] = value

    return {
        "action": template["action"],
        "item_family": family,
        "resolved_item_id": item["id"] if item else None,
        "attributes": attributes,
        "missing_attributes": list(template.get("missing_attributes") or []),
        "failure_detected": bool(template.get("failure_detected")),
        "failure_type": template.get("failure_type", "none"),
        "recovery_strategy": template.get("recovery_strategy", "none"),
    }


class ProgressiveFSMRouter:
    """SLM-first teacher/student router with guarded promotion and rollback."""

    def __init__(
        self,
        teacher: Callable[[dict[str, Any]], str],
        validate_raw: Callable[[str, dict[str, Any]], Any],
        *,
        namespace: str,
        policy: RouterPolicy | None = None,
        random_seed: int = 42,
        secret: bytes | None = None,
    ) -> None:
        self.teacher = teacher
        self.validate_raw = validate_raw
        self.namespace = str(namespace)
        self.policy = policy or RouterPolicy.research()
        self._secret = secret or os.urandom(32)
        self._rng = random.Random(random_seed)
        self._lock = threading.RLock()
        self.entries: dict[str, TransitionEntry] = {}
        self.key_index: dict[str, str] = {}
        self.traces: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.learning_enabled = True
        self.audit_enabled = True
        self.last_trace: RouteTrace | None = None

    def _hash(self, value: Any) -> str:
        message = (self.namespace + "\n" + _canonical_json(value)).encode("utf-8")
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def _evidence_hash(self, evidence_id: str) -> str:
        return self._hash({"evidence_id": str(evidence_id)})

    def _entry_for_key(self, level: str, key_hash: str) -> TransitionEntry | None:
        entry_id = self.key_index.get(f"{level}:{key_hash}")
        return self.entries.get(entry_id) if entry_id else None

    def _keys(self, payload: dict[str, Any], bundle: FeatureBundle) -> tuple[str, str]:
        exact = self._hash({"level": "exact", "payload": _strict_payload(payload, self.policy)})
        typed = self._hash({"level": "typed", "state": _typed_state(payload, bundle)})
        return exact, typed

    def _event(self, kind: str, **values: Any) -> None:
        event = {"time": time.time(), "event": kind, **copy.deepcopy(values)}
        self.events.append(event)
        excess = len(self.events) - self.policy.max_events_in_snapshot
        if excess > 0:
            del self.events[:excess]

    def _validate(self, decision: Mapping[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        raw = _canonical_json(_decision_dict(decision))
        return _decision_dict(self.validate_raw(raw, payload))

    def _candidate(
        self,
        *,
        level: str,
        key_hash: str,
        decision_mode: str,
        decision: dict[str, Any],
        template: dict[str, Any] | None,
    ) -> TransitionEntry:
        previous = self._entry_for_key(level, key_hash)
        generation = previous.generation + 1 if previous is not None else 0
        entry_id = "p_" + self._hash(
            {"level": level, "key": key_hash, "generation": generation}
        )[:20]
        entry = TransitionEntry(
            entry_id=entry_id,
            level=level,
            key_hash=key_hash,
            decision_mode=decision_mode,
            scenario=scenario_for(decision, decision_mode),
            decision=copy.deepcopy(decision) if level == "exact" else None,
            template=copy.deepcopy(template) if level == "typed" else None,
            generation=generation,
            replaces_entry_id=previous.entry_id if previous is not None else None,
        )
        self.entries[entry_id] = entry
        self.key_index[f"{level}:{key_hash}"] = entry_id
        self._event(
            "candidate_restarted" if previous is not None else "candidate_created",
            entry_id=entry_id,
            level=level,
            scenario=entry.scenario,
            generation=generation,
            replaces_entry_id=entry.replaces_entry_id,
        )
        return entry

    def _entry_prediction(
        self, entry: TransitionEntry, payload: dict[str, Any], bundle: FeatureBundle
    ) -> dict[str, Any]:
        if entry.level == "exact":
            if entry.decision is None:
                raise LookupError("Exact entry has no decision")
            decision = copy.deepcopy(entry.decision)
        elif entry.level == "typed":
            if entry.template is None:
                raise LookupError("Typed entry has no template")
            decision = instantiate_template(entry.template, payload, bundle)
        else:
            raise LookupError(f"Unknown entry level: {entry.level}")
        return self._validate(decision, payload)

    def _quarantine(self, entry: TransitionEntry, reason: str) -> None:
        entry.status = "quarantined"
        entry.quarantine_reason = reason
        entry.updated_at = time.time()
        self._event("quarantined", entry_id=entry.entry_id, reason=reason, scenario=entry.scenario)

    def _maybe_promote(self, entry: TransitionEntry, *, allow_activation: bool = True) -> None:
        if entry.status == "quarantined":
            return
        thresholds = self.policy.thresholds_for(
            entry.decision or {
                "action": entry.template.get("action") if entry.template else None,
                "failure_detected": entry.template.get("failure_detected") if entry.template else False,
            },
            entry.decision_mode,
        )
        if entry.teacher_agreements >= thresholds.min_teacher_agreements and entry.status == "candidate":
            entry.status = "shadow"
            self._event("shadow_started", entry_id=entry.entry_id, scenario=entry.scenario)
        common_ready = (
            entry.status in {"candidate", "shadow"}
            and entry.teacher_agreements >= thresholds.min_teacher_agreements
            and entry.shadow_observations >= thresholds.min_shadow_observations
            and entry.shadow_mismatches == 0
            and len(entry.evidence_hashes) >= thresholds.min_distinct_evidence
            and (not self.policy.reject_any_teacher_conflict or entry.teacher_disagreements == 0)
        )
        if self.policy.promotion_mode == "imitate_teacher":
            ready = (
                common_ready
                and (not self.policy.reject_any_verified_failure or entry.verified_failures == 0)
            )
        elif self.policy.promotion_mode == "verified_outcome":
            ready = (
                common_ready
                and entry.verified_successes >= thresholds.min_verified_successes
                and entry.verified_wilson >= thresholds.min_verified_wilson
                and (not self.policy.reject_any_verified_failure or entry.verified_failures == 0)
            )
        else:
            raise ValueError(f"Unknown promotion_mode: {self.policy.promotion_mode}")
        if ready and allow_activation:
            entry.status = "active"
            entry.activated_at = time.time()
            entry.updated_at = time.time()
            self._event("promoted", entry_id=entry.entry_id, level=entry.level, scenario=entry.scenario)

    def _observe_one(
        self,
        *,
        level: str,
        key_hash: str,
        decision: dict[str, Any],
        template: dict[str, Any] | None,
        payload: dict[str, Any],
        bundle: FeatureBundle,
        evidence_id: str,
    ) -> TransitionEntry | None:
        if level == "typed" and template is None:
            return None
        entry = self._entry_for_key(level, key_hash)
        if entry is None or entry.status == "quarantined":
            entry = self._candidate(
                level=level,
                key_hash=key_hash,
                decision_mode=str(payload.get("decision_mode", "dialogue_decision")),
                decision=decision,
                template=template,
            )
            entry.teacher_agreements = 1
        else:
            prior_status = entry.status
            try:
                predicted = self._entry_prediction(entry, payload, bundle)
            except Exception as exc:
                entry.validator_failures += 1
                self._quarantine(entry, f"binding_or_validator_failure:{type(exc).__name__}")
                return entry
            if decisions_equal(predicted, decision):
                entry.teacher_agreements += 1
                # Shadow evidence must be observed strictly after the entry
                # has entered shadow; candidate observations do not count.
                if prior_status == "shadow":
                    entry.shadow_matches += 1
            else:
                entry.teacher_disagreements += 1
                if prior_status == "shadow":
                    entry.shadow_mismatches += 1
                self._quarantine(entry, "teacher_conflict")
        entry.evidence_hashes.add(self._evidence_hash(evidence_id))
        entry.updated_at = time.time()
        # Observation may start the shadow stage, but activation waits for the
        # outcome feedback attached to this same trace.
        self._maybe_promote(
            entry,
            allow_activation=self.policy.promotion_mode == "imitate_teacher",
        )
        return entry

    def observe_teacher(
        self,
        payload: dict[str, Any],
        decision: Mapping[str, Any],
        *,
        evidence_id: str,
    ) -> tuple[TransitionEntry, TransitionEntry | None]:
        """Record a validated teacher decision without declaring it correct."""
        with self._lock:
            bundle = extract_bundle(payload, self.policy)
            validated = self._validate(decision, payload)
            exact_key, typed_key = self._keys(payload, bundle)
            template = compile_template(validated, payload, bundle)
            exact = self._observe_one(
                level="exact",
                key_hash=exact_key,
                decision=validated,
                template=None,
                payload=payload,
                bundle=bundle,
                evidence_id=evidence_id,
            )
            typed = self._observe_one(
                level="typed",
                key_hash=typed_key,
                decision=validated,
                template=template,
                payload=payload,
                bundle=bundle,
                evidence_id=evidence_id,
            )
            return exact, typed

    def _active_match(
        self, payload: dict[str, Any], bundle: FeatureBundle, exact_key: str, typed_key: str
    ) -> tuple[TransitionEntry | None, dict[str, Any] | None]:
        for level, key in (("exact", exact_key), ("typed", typed_key)):
            entry = self._entry_for_key(level, key)
            if entry is None or entry.status != "active":
                continue
            try:
                return entry, self._entry_prediction(entry, payload, bundle)
            except Exception as exc:
                entry.validator_failures += 1
                self._quarantine(entry, f"active_validator_failure:{type(exc).__name__}")
        return None, None

    def _equivalent_active_entries(
        self,
        payload: dict[str, Any],
        bundle: FeatureBundle,
        exact_key: str,
        typed_key: str,
        decision: Mapping[str, Any],
    ) -> list[TransitionEntry]:
        """Return every active rule that could reproduce this same decision.

        Feedback must invalidate exact and generalized siblings together;
        otherwise a rejected exact hit could immediately reappear through its
        still-active typed transition.
        """
        matches: list[TransitionEntry] = []
        for level, key in (("exact", exact_key), ("typed", typed_key)):
            entry = self._entry_for_key(level, key)
            if entry is None or entry.status != "active":
                continue
            try:
                predicted = self._entry_prediction(entry, payload, bundle)
            except Exception as exc:
                entry.validator_failures += 1
                self._quarantine(entry, f"active_validator_failure:{type(exc).__name__}")
                continue
            if decisions_equal(predicted, decision):
                matches.append(entry)
        return matches

    def predict(
        self,
        payload: dict[str, Any],
        *,
        evidence_id: str | None = None,
        allow_teacher: bool = True,
    ) -> tuple[str, RouteTrace]:
        started = time.perf_counter()
        evidence_id = evidence_id or f"online:{uuid.uuid4()}"
        with self._lock:
            bundle = extract_bundle(payload, self.policy)
            exact_key, typed_key = self._keys(payload, bundle)
            active, cached = self._active_match(payload, bundle, exact_key, typed_key)

            if active is not None and cached is not None:
                equivalent_entries = self._equivalent_active_entries(
                    payload, bundle, exact_key, typed_key, cached
                )
                thresholds = self.policy.thresholds_for(cached, str(payload.get("decision_mode")))
                audit = self.audit_enabled and self._rng.random() < thresholds.audit_probability
                if not audit:
                    active.served += 1
                    active.updated_at = time.time()
                    trace = RouteTrace(
                        trace_id=str(uuid.uuid4()),
                        route=f"fsm_{active.level}",
                        primary="fsm",
                        teacher_called=False,
                        audited=False,
                        decision_mode=str(payload.get("decision_mode")),
                        scenario=scenario_for(cached, str(payload.get("decision_mode"))),
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        exact_entry_id=active.entry_id if active.level == "exact" else None,
                        typed_entry_id=active.entry_id if active.level == "typed" else None,
                    )
                    self._remember_trace(
                        trace, [entry.entry_id for entry in equivalent_entries], evidence_id
                    )
                    self._event("routed", **trace.to_dict())
                    self.last_trace = trace
                    return _canonical_json(cached), trace

                if not allow_teacher:
                    raise LookupError("Teacher audit required but teacher calls are disabled")
                teacher_raw = self.teacher(payload)
                teacher_decision = _decision_dict(self.validate_raw(teacher_raw, payload))
                if decisions_equal(cached, teacher_decision):
                    active.audit_matches += 1
                    active.served += 1
                    route, primary, reason = f"fsm_{active.level}_audited", "fsm", None
                    returned = cached
                else:
                    for entry in equivalent_entries:
                        entry.audit_mismatches += 1
                        self._quarantine(entry, "audit_disagreement")
                    route, primary, reason = "slm_audit_fallback", "slm", "audit_disagreement"
                    returned = teacher_decision
                trace = RouteTrace(
                    trace_id=str(uuid.uuid4()),
                    route=route,
                    primary=primary,
                    teacher_called=True,
                    audited=True,
                    decision_mode=str(payload.get("decision_mode")),
                    scenario=scenario_for(returned, str(payload.get("decision_mode"))),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    exact_entry_id=active.entry_id if active.level == "exact" else None,
                    typed_entry_id=active.entry_id if active.level == "typed" else None,
                    reason=reason,
                )
                # Outcome feedback belongs to the component that supplied the
                # returned decision.  On audit disagreement the cached rule is
                # already quarantined and the teacher, not that rule, answered.
                feedback_entries = (
                    [entry.entry_id for entry in equivalent_entries] if primary == "fsm" else []
                )
                self._remember_trace(trace, feedback_entries, evidence_id)
                self._event("routed", **trace.to_dict())
                self.last_trace = trace
                return _canonical_json(returned), trace

            if not allow_teacher:
                raise LookupError("No active FSM transition and teacher calls are disabled")

            teacher_raw = self.teacher(payload)
            teacher_decision = _decision_dict(self.validate_raw(teacher_raw, payload))
            exact = typed = None
            if self.learning_enabled:
                exact, typed = self.observe_teacher(
                    payload, teacher_decision, evidence_id=evidence_id
                )
            scenario = scenario_for(teacher_decision, str(payload.get("decision_mode")))
            trace = RouteTrace(
                trace_id=str(uuid.uuid4()),
                route="slm",
                primary="slm",
                teacher_called=True,
                audited=False,
                decision_mode=str(payload.get("decision_mode")),
                scenario=scenario,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                exact_entry_id=exact.entry_id if exact else None,
                typed_entry_id=typed.entry_id if typed else None,
                reason="empty_or_untrusted_transition",
            )
            entry_ids = [entry.entry_id for entry in (exact, typed) if entry is not None]
            self._remember_trace(trace, entry_ids, evidence_id)
            self._event("routed", **trace.to_dict())
            self.last_trace = trace
            return _canonical_json(teacher_decision), trace

    def __call__(self, payload: dict[str, Any]) -> str:
        raw, _trace = self.predict(payload)
        return raw

    def _remember_trace(self, trace: RouteTrace, entry_ids: list[str], evidence_id: str) -> None:
        self._prune_traces(reserve=1)
        self.traces[trace.trace_id] = {
            "entry_ids": list(entry_ids),
            "evidence_hash": self._evidence_hash(evidence_id),
            "route": trace.route,
            "scenario": trace.scenario,
            "verified": False,
            "created_at": time.time(),
        }

    def _prune_traces(self, *, reserve: int = 0) -> None:
        """Bound causal trace memory while retaining a feedback window."""
        now = time.time()
        expired = [
            trace_id
            for trace_id, trace in self.traces.items()
            if not trace.get("verified", False)
            and now - float(trace.get("created_at", now)) > self.policy.pending_trace_ttl_seconds
        ]
        for trace_id in expired:
            self.traces.pop(trace_id, None)
            self._event("trace_expired_without_feedback", trace_id=trace_id)

        target = max(0, self.policy.max_traces_in_memory - reserve)
        while len(self.traces) > target:
            removable = next(
                (
                    trace_id
                    for trace_id, trace in self.traces.items()
                    if trace.get("verified", False)
                ),
                None,
            )
            if removable is None:
                removable = next(iter(self.traces))
                self._event("trace_evicted_without_feedback", trace_id=removable)
            self.traces.pop(removable, None)

    def discard_trace(self, trace_id: str, *, reason: str = "unlabeled") -> None:
        """Forget a trace without treating it as positive or negative evidence."""
        with self._lock:
            if self.traces.pop(trace_id, None) is None:
                raise KeyError(f"Unknown trace_id: {trace_id}")
            self._event("trace_discarded", trace_id=trace_id, reason=_safe_reason_code(reason))

    def feedback(
        self,
        trace_id: str,
        *,
        success: bool,
        reason: str | None = None,
    ) -> None:
        """Attach an external task outcome to the exact trace that caused it."""
        with self._lock:
            trace = self.traces.get(trace_id)
            if trace is None:
                raise KeyError(f"Unknown trace_id: {trace_id}")
            if trace["verified"]:
                raise ValueError("Feedback was already recorded for this trace")
            trace["verified"] = True
            reason_code = _safe_reason_code(reason)
            for entry_id in trace["entry_ids"]:
                entry = self.entries.get(entry_id)
                if entry is None:
                    continue
                entry.evidence_hashes.add(trace["evidence_hash"])
                if success:
                    entry.verified_successes += 1
                    self._maybe_promote(entry)
                else:
                    entry.verified_failures += 1
                    self._quarantine(entry, reason_code)
                entry.updated_at = time.time()
            self._event(
                "feedback",
                trace_id=trace_id,
                success=bool(success),
                reason=reason_code if not success else None,
                scenario=trace["scenario"],
            )

    def freeze(self, *, audits: bool = False) -> None:
        self.learning_enabled = False
        self.audit_enabled = bool(audits)

    def unfreeze(self, *, audits: bool = True) -> None:
        self.learning_enabled = True
        self.audit_enabled = bool(audits)

    def entry_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [entry.to_dict() for entry in sorted(self.entries.values(), key=lambda x: x.entry_id)]

    def routing_events(self) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("event") == "routed"]

    def summary(self) -> dict[str, Any]:
        entries = list(self.entries.values())
        routes: dict[str, int] = {}
        for event in self.routing_events():
            routes[event["route"]] = routes.get(event["route"], 0) + 1
        statuses: dict[str, int] = {}
        scenarios: dict[str, dict[str, int]] = {}
        for entry in entries:
            statuses[entry.status] = statuses.get(entry.status, 0) + 1
            bucket = scenarios.setdefault(entry.scenario, {})
            bucket[entry.status] = bucket.get(entry.status, 0) + 1
        return {
            "namespace": self.namespace,
            "promotion_mode": self.policy.promotion_mode,
            "entries": len(entries),
            "statuses": statuses,
            "routes": routes,
            "scenarios": scenarios,
            "teacher_calls": sum(bool(event.get("teacher_called")) for event in self.routing_events()),
            "primary_fsm": sum(event.get("primary") == "fsm" for event in self.routing_events()),
            "primary_slm": sum(event.get("primary") == "slm" for event in self.routing_events()),
        }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "format_version": FORMAT_VERSION,
                "extractor_version": EXTRACTOR_VERSION,
                "namespace": self.namespace,
                "policy": asdict(self.policy),
                "secret_hex": self._secret.hex(),
                "entries": [entry.to_dict() for entry in self.entries.values()],
                "key_index": dict(self.key_index),
                "events": copy.deepcopy(self.events),
                "pending_traces": {
                    trace_id: copy.deepcopy(trace)
                    for trace_id, trace in self.traces.items()
                    if not trace.get("verified", False)
                },
                "learning_enabled": self.learning_enabled,
                "audit_enabled": self.audit_enabled,
                "saved_at": time.time(),
                "privacy_note": "Keys are HMACs; raw dialogue text is not stored. Hashing is not anonymization.",
            }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, target)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        teacher: Callable[[dict[str, Any]], str],
        validate_raw: Callable[[str, dict[str, Any]], Any],
        expected_namespace: str | None = None,
    ) -> "ProgressiveFSMRouter":
        data = json.loads(Path(path).read_text())
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported FSM snapshot format")
        if data.get("extractor_version") != EXTRACTOR_VERSION:
            raise ValueError("FSM extractor version changed; rebuild or migrate the snapshot")
        namespace = str(data["namespace"])
        if expected_namespace is not None and namespace != expected_namespace:
            raise ValueError("FSM namespace does not match the selected teacher/runtime")
        raw_policy = dict(data["policy"])
        raw_policy["routine"] = Thresholds(**raw_policy["routine"])
        raw_policy["risky"] = Thresholds(**raw_policy["risky"])
        raw_policy["timing_buckets_ms"] = tuple(raw_policy["timing_buckets_ms"])
        result = cls(
            teacher,
            validate_raw,
            namespace=namespace,
            policy=RouterPolicy(**raw_policy),
            secret=bytes.fromhex(data["secret_hex"]),
        )
        result.entries = {
            entry.entry_id: entry
            for entry in (TransitionEntry.from_dict(value) for value in data.get("entries", []))
        }
        result.key_index = dict(data.get("key_index", {}))
        result.events = list(data.get("events", []))
        result.traces = dict(data.get("pending_traces", {}))
        result.learning_enabled = bool(data.get("learning_enabled", True))
        result.audit_enabled = bool(data.get("audit_enabled", True))
        return result


def make_namespace(
    *,
    model_id: str,
    model_revision: str,
    adapter_sha256: str,
    runtime_sha256: str,
    schema_sha256: str,
    policy_name: str,
) -> str:
    return _digest(
        {
            "model_id": model_id,
            "model_revision": model_revision,
            "adapter_sha256": adapter_sha256,
            "runtime_sha256": runtime_sha256,
            "schema_sha256": schema_sha256,
            "extractor_version": EXTRACTOR_VERSION,
            "policy_name": policy_name,
        }
    )
