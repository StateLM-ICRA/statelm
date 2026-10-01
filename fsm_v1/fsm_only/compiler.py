"""Compile labeled JSONL examples into a deterministic FSM/FST artifact."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .errors import TrainingDataError
from .model import CompiledFSM, FORMAT_NAME
from .normalize import canonical_symbol, normalize_phrase, slot_name, tokenize
from .similarity import METHOD as SIMILARITY_METHOD, fit_idf, slot_token


DEFAULT_LOOKUP_EVENTS = {
    "zero": "ZERO_MATCHES",
    "one_with_location": "ONE_WITH_LOCATION",
    "one_without_location": "ONE_WITHOUT_LOCATION",
    "multiple": "MULTIPLE_MATCHES",
}
DEFAULT_TEMPLATES = {
    "RETURN_LOCATION": "{name} is in {location}.",
    "ASK_WHICH_ONE": "I found more than one match: {choices}. Which one do you mean?",
    "OBJECT_MISSING": "I couldn't find '{query}' in the inventory.",
    "LOCATION_MISSING": "{name} is in the inventory, but its location is missing.",
    "UNSUPPORTED_REQUEST": "I can only help locate objects in the inventory.",
    "UNRECOGNIZED": "I couldn't match that request to a learned query pattern.",
}
REQUIRED_RESULT_ACTIONS = {
    "zero": "OBJECT_MISSING",
    "one_with_location": "RETURN_LOCATION",
    "one_without_location": "LOCATION_MISSING",
    "multiple": "ASK_WHICH_ONE",
}
DEFAULT_SIMILARITY = {
    "enabled": True,
    "minimum": 0.72,
    "minimum_margin": 0.015,
    "slot_margin": 0.03,
    "max_slot_tokens": 12,
    "inventory_minimum": 0.55,
    "inventory_margin": 0.04,
}


def _minimize_acyclic_transducer(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge states with exactly equivalent accepting suffix behavior.

    The learned query graph is an acyclic prefix tree. Bottom-up structural
    hashing therefore gives deterministic minimal suffix equivalence without
    weakening or reinterpreting any slot transition.
    """

    signature_to_state: dict[tuple[Any, ...], int] = {}
    old_to_minimized: dict[int, int] = {}
    minimized: list[dict[str, Any]] = []

    def intern(old_id: int) -> int:
        if old_id in old_to_minimized:
            return old_to_minimized[old_id]
        old = states[old_id]
        child_pairs = tuple(
            (symbol, intern(int(child)))
            for symbol, child in sorted(old.get("transitions", {}).items())
        )
        final = old.get("final")
        final_signature = None
        if final is not None:
            final_signature = (str(final["intent"]), int(final.get("support", 1)))
        signature = (final_signature, child_pairs)
        state_id = signature_to_state.get(signature)
        if state_id is None:
            state_id = len(minimized)
            signature_to_state[signature] = state_id
            value: dict[str, Any] = {"transitions": dict(child_pairs)}
            if final is not None:
                value["final"] = {"intent": final_signature[0], "support": final_signature[1]}
            minimized.append(value)
        old_to_minimized[old_id] = state_id
        return state_id

    root = intern(0)
    # Renumber reachable minimized states in deterministic root-first order so
    # the public automaton always starts at state zero.
    order: list[int] = []
    seen: set[int] = set()

    def visit(state_id: int) -> None:
        if state_id in seen:
            return
        seen.add(state_id)
        order.append(state_id)
        for _symbol, child in sorted(minimized[state_id]["transitions"].items()):
            visit(int(child))

    visit(root)
    remap = {old_id: new_id for new_id, old_id in enumerate(order)}
    result: list[dict[str, Any]] = []
    for old_id in order:
        old = minimized[old_id]
        value: dict[str, Any] = {
            "transitions": {
                symbol: remap[int(child)] for symbol, child in sorted(old["transitions"].items())
            }
        }
        if "final" in old:
            value["final"] = dict(old["final"])
        result.append(value)
    return result


def _error(source: str, message: str) -> TrainingDataError:
    return TrainingDataError(f"{source}: {message}")


def _kind(record: Mapping[str, Any]) -> str:
    value = record.get("type", record.get("kind", record.get("record_type", "")))
    return str(value).strip().casefold()


def _pattern_string(value: str) -> list[str]:
    pieces = re.findall(r"\{[^{}]+\}|<[^<>]+>|[^\W_]+(?:['\u2019-][^\W_]+)*", value, flags=re.UNICODE)
    return pieces


def _slot_label(label: Any) -> str | None:
    raw = str(label).strip()
    if not raw or raw.upper() == "O":
        return None
    raw = re.sub(r"^[BI](?:-|_)", "", raw, flags=re.IGNORECASE)
    raw = raw.casefold().replace(":", ".")
    if raw.startswith("attribute."):
        raw = raw[len("attribute.") :]
    if not re.fullmatch(r"[a-z][a-z0-9_.-]*", raw):
        raise TrainingDataError(f"invalid token label {label!r}")
    return raw


def _symbols_from_token_labels(record: Mapping[str, Any], source: str) -> tuple[list[str], dict[str, set[str]]]:
    raw_tokens = record.get("tokens")
    labels = record.get("token_labels")
    if not isinstance(raw_tokens, list) or not isinstance(labels, list) or len(raw_tokens) != len(labels):
        raise _error(source, "tokens and token_labels must be equal-length arrays")
    tokens = [normalize_phrase(str(value)) for value in raw_tokens]
    if any(not value or len(tokenize(value)) != 1 for value in tokens):
        raise _error(source, "each labeled token must normalize to exactly one token")
    symbols: list[str] = []
    observed: dict[str, set[str]] = {}
    index = 0
    while index < len(tokens):
        name = _slot_label(labels[index])
        if name is None:
            symbols.append(tokens[index])
            index += 1
            continue
        end = index + 1
        while end < len(tokens) and _slot_label(labels[end]) == name:
            end += 1
        symbols.append(f"{{{name}}}")
        observed.setdefault(name, set()).add(" ".join(tokens[index:end]))
        index = end
    return symbols, observed


def _symbols_from_text_slots(record: Mapping[str, Any], source: str) -> tuple[list[str], dict[str, set[str]]]:
    text = record.get("text", record.get("utterance"))
    slots = record.get("slots", {})
    if not isinstance(text, str) or not isinstance(slots, Mapping):
        raise _error(source, "query record needs text/utterance plus a slots object")
    tokens = list(tokenize(text))
    spans: list[tuple[int, int, str, str]] = []
    observed: dict[str, set[str]] = {}
    explicit = record.get("slot_spans")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise _error(source, "slot_spans must be an array")
        for raw in explicit:
            if not isinstance(raw, Mapping):
                raise _error(source, "each slot span must be an object")
            name = str(raw.get("slot", "")).casefold()
            start, end = raw.get("start"), raw.get("end")
            if not name or not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(tokens)):
                raise _error(source, f"invalid slot span {raw!r}")
            value = " ".join(tokens[start:end])
            spans.append((start, end, name, value))
            observed.setdefault(name, set()).add(value)
    else:
        occupied: set[int] = set()
        entries: list[tuple[str, tuple[str, ...]]] = []
        for raw_name, raw_value in slots.items():
            name = str(raw_name).casefold()
            if isinstance(raw_value, Mapping):
                raw_value = raw_value.get("value", raw_value.get("text"))
            value_tokens = tokenize(str(raw_value)) if raw_value is not None else ()
            if not name or not value_tokens:
                raise _error(source, f"slot {raw_name!r} has no textual value")
            entries.append((name, value_tokens))
        for name, value_tokens in sorted(entries, key=lambda item: (-len(item[1]), item[0])):
            starts = [
                i
                for i in range(len(tokens) - len(value_tokens) + 1)
                if tuple(tokens[i : i + len(value_tokens)]) == value_tokens
                and not any(j in occupied for j in range(i, i + len(value_tokens)))
            ]
            if not starts:
                raise _error(source, f"slot {name!r} value {' '.join(value_tokens)!r} is not an unclaimed span in the text")
            start = starts[0]
            end = start + len(value_tokens)
            occupied.update(range(start, end))
            value = " ".join(value_tokens)
            spans.append((start, end, name, value))
            observed.setdefault(name, set()).add(value)
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            raise _error(source, "slot spans overlap")
    symbols: list[str] = []
    cursor = 0
    for start, end, name, _value in spans:
        symbols.extend(tokens[cursor:start])
        symbols.append(f"{{{name}}}")
        cursor = end
    symbols.extend(tokens[cursor:])
    if not symbols:
        raise _error(source, "query normalizes to an empty pattern")
    return symbols, observed


def _query_symbols(record: Mapping[str, Any], source: str) -> tuple[list[str], dict[str, set[str]]]:
    if "token_labels" in record:
        return _symbols_from_token_labels(record, source)
    supplied = record.get("symbols", record.get("pattern"))
    if supplied is not None:
        if isinstance(supplied, str):
            symbols = _pattern_string(supplied)
        elif isinstance(supplied, list):
            symbols = [str(value) for value in supplied]
        else:
            raise _error(source, "symbols/pattern must be a string or array")
        if not symbols:
            raise _error(source, "query pattern is empty")
        return symbols, {}
    return _symbols_from_text_slots(record, source)


def compile_records(
    records: Iterable[Mapping[str, Any]],
    *,
    start_state: str | None = None,
    entity_slot: str | None = None,
    require_complete: bool = True,
    provenance: Sequence[Mapping[str, str]] = (),
) -> CompiledFSM:
    """Compile labeled records into an immutable deterministic artifact.

    Duplicate compatible examples add support. Contradictory labels fail instead
    of being silently resolved by a handwritten priority rule.
    """

    query_examples: list[tuple[list[str], str, str]] = []
    observed_slots: dict[str, set[str]] = {}
    transition_counts: dict[tuple[str, str, str, str], int] = Counter()
    transition_target: dict[tuple[str, str], tuple[str, str]] = {}
    config: dict[str, Any] = {}
    first_transition_state: str | None = None

    for index, raw_record in enumerate(records, 1):
        if not isinstance(raw_record, Mapping):
            raise _error(f"record {index}", "record must be a JSON object")
        record = dict(raw_record)
        source = str(record.pop("_source", f"record {index}"))
        kind = _kind(record)
        if kind in {"config", "meta"}:
            for key in ("start_state", "entity_slot", "lookup_events", "templates", "similarity"):
                if key in record:
                    if key in config and config[key] != record[key]:
                        raise _error(source, f"conflicting config value for {key}")
                    config[key] = record[key]
            continue
        if kind in {"query", "query_pattern", "utterance"}:
            intent = str(record.get("intent", "")).strip()
            if not intent:
                raise _error(source, "query record requires intent")
            symbols, observed = _query_symbols(record, source)
            query_examples.append((symbols, intent, source))
            for name, values in observed.items():
                observed_slots.setdefault(name, set()).update(values)
            continue
        if kind in {"transition", "controller", "controller_transition"}:
            state = str(record.get("state", "")).strip()
            event = str(record.get("event", "")).strip()
            action = str(record.get("action", "")).strip()
            next_state = str(record.get("next_state", "")).strip()
            if not all((state, event, action, next_state)):
                raise _error(source, "transition requires state, event, action, and next_state")
            if first_transition_state is None:
                first_transition_state = state
            key = (state, event)
            target = (action, next_state)
            previous = transition_target.get(key)
            if previous is not None and previous != target:
                raise _error(source, f"contradictory transition for {state!r} + {event!r}: {previous!r} vs {target!r}")
            transition_target[key] = target
            transition_counts[(state, event, action, next_state)] += 1
            continue
        raise _error(source, f"unknown record type {kind!r}")

    if not query_examples:
        raise TrainingDataError("training data contains no query examples")
    if not transition_target:
        raise TrainingDataError("training data contains no controller transitions")
    chosen_start = str(start_state or config.get("start_state") or first_transition_state or "").strip()
    chosen_entity = str(entity_slot or config.get("entity_slot") or "object").strip().casefold()
    if not chosen_start or not chosen_entity:
        raise TrainingDataError("start_state and entity_slot must be non-empty")

    lookup_events = dict(DEFAULT_LOOKUP_EVENTS)
    if "lookup_events" in config:
        if not isinstance(config["lookup_events"], Mapping):
            raise TrainingDataError("config.lookup_events must be an object")
        lookup_events.update({str(key): str(value) for key, value in config["lookup_events"].items()})
    if set(lookup_events) != set(DEFAULT_LOOKUP_EVENTS) or any(not value for value in lookup_events.values()):
        raise TrainingDataError(f"lookup_events must define exactly {sorted(DEFAULT_LOOKUP_EVENTS)}")
    templates = dict(DEFAULT_TEMPLATES)
    if "templates" in config:
        if not isinstance(config["templates"], Mapping):
            raise TrainingDataError("config.templates must be an object")
        templates.update({str(key): str(value) for key, value in config["templates"].items()})
    similarity = dict(DEFAULT_SIMILARITY)
    if "similarity" in config:
        if not isinstance(config["similarity"], Mapping):
            raise TrainingDataError("config.similarity must be an object")
        similarity.update(dict(config["similarity"]))
    if not isinstance(similarity["enabled"], bool):
        raise TrainingDataError("config.similarity.enabled must be boolean")
    for key in ("minimum", "minimum_margin", "slot_margin", "inventory_minimum", "inventory_margin"):
        value = similarity[key]
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise TrainingDataError(f"config.similarity.{key} must be between zero and one")
        similarity[key] = float(value)
    if not isinstance(similarity["max_slot_tokens"], int) or similarity["max_slot_tokens"] < 1:
        raise TrainingDataError("config.similarity.max_slot_tokens must be a positive integer")

    states: list[dict[str, Any]] = [{"transitions": {}}]
    pattern_targets: dict[tuple[str, ...], str] = {}
    pattern_counts: Counter[tuple[str, ...]] = Counter()
    for raw_symbols, intent, source in query_examples:
        try:
            symbols = tuple(canonical_symbol(value) for value in raw_symbols)
        except ValueError as exc:
            raise _error(source, str(exc)) from exc
        existing = pattern_targets.get(symbols)
        if existing is not None and existing != intent:
            raise _error(source, f"same query pattern has conflicting intents {existing!r} and {intent!r}")
        pattern_targets[symbols] = intent
        pattern_counts[symbols] += 1
    for symbols in sorted(pattern_targets):
        state_id = 0
        for symbol in symbols:
            transitions = states[state_id]["transitions"]
            if symbol not in transitions:
                transitions[symbol] = len(states)
                states.append({"transitions": {}})
            state_id = transitions[symbol]
        states[state_id]["final"] = {"intent": pattern_targets[symbols], "support": pattern_counts[symbols]}
    state_count_before = len(states)
    states = _minimize_acyclic_transducer(states)
    state_count_after = len(states)

    semantic_prototypes: list[dict[str, Any]] = []
    entity_symbol = f"S:{chosen_entity}"
    for symbols in sorted(pattern_targets):
        # The similarity fallback currently learns one entity span. Exact FSM
        # traversal continues to support patterns containing repeated slots.
        if symbols.count(entity_symbol) != 1:
            continue
        semantic_tokens = [
            symbol[2:] if symbol.startswith("L:") else slot_token(symbol[2:])
            for symbol in symbols
        ]
        semantic_prototypes.append({
            "tokens": semantic_tokens,
            "intent": pattern_targets[symbols],
            "support": pattern_counts[symbols],
        })
    similarity_artifact = {
        "enabled": similarity["enabled"],
        "method": SIMILARITY_METHOD,
        "minimum": similarity["minimum"],
        "minimum_margin": similarity["minimum_margin"],
        "slot_margin": similarity["slot_margin"],
        "max_slot_tokens": similarity["max_slot_tokens"],
        "inventory_minimum": similarity["inventory_minimum"],
        "inventory_margin": similarity["inventory_margin"],
        "idf": fit_idf(prototype["tokens"] for prototype in semantic_prototypes),
        "prototypes": semantic_prototypes,
        "training_scope": "query patterns from compiler input only",
    }

    controller: dict[str, dict[str, dict[str, Any]]] = {}
    for (state, event), (action, next_state) in sorted(transition_target.items()):
        controller.setdefault(state, {})[event] = {
            "action": action,
            "next_state": next_state,
            "support": transition_counts[(state, event, action, next_state)],
        }

    if require_complete:
        intents = sorted(set(pattern_targets.values()))
        lookup_states: set[str] = set()
        for intent in intents:
            transition = controller.get(chosen_start, {}).get(intent)
            if transition is None:
                raise TrainingDataError(f"start state {chosen_start!r} has no learned transition for query intent {intent!r}")
            if transition["action"] != "LOOKUP":
                raise TrainingDataError(f"query intent {intent!r} must learn action 'LOOKUP', got {transition['action']!r}")
            lookup_states.add(transition["next_state"])
        checked: set[str] = set()
        while lookup_states - checked:
            state = sorted(lookup_states - checked)[0]
            checked.add(state)
            for outcome, event in lookup_events.items():
                transition = controller.get(state, {}).get(event)
                expected = REQUIRED_RESULT_ACTIONS[outcome]
                if transition is None:
                    raise TrainingDataError(f"state {state!r} lacks learned lookup transition for {event!r}")
                if transition["action"] != expected:
                    raise TrainingDataError(
                        f"acceptance invariant failed: {state!r} + {event!r} learned {transition['action']!r}, expected {expected!r}"
                    )
                if outcome == "multiple":
                    lookup_states.add(transition["next_state"])

    artifact = {
        "format": FORMAT_NAME,
        "architecture": "similarity-augmented-fsm",
        "entity_slot": chosen_entity,
        "lookup_events": lookup_events,
        "templates": templates,
        "similarity": similarity_artifact,
        "query_automaton": {
            "start_state": 0,
            "states": states,
            "pattern_count": len(pattern_targets),
            "example_count": sum(pattern_counts.values()),
            "minimization": "exact_suffix_equivalence",
            "state_count_before_minimization": state_count_before,
            "state_count_after_minimization": state_count_after,
            "observed_slot_values": {key: sorted(values) for key, values in sorted(observed_slots.items())},
        },
        "controller": {
            "start_state": chosen_start,
            "transitions": controller,
            "example_count": sum(transition_counts.values()),
        },
        "provenance": list(provenance),
    }
    return CompiledFSM(artifact)


def compile_jsonl(
    inputs: str | Path | Sequence[str | Path],
    output: str | Path | None = None,
    *,
    start_state: str | None = None,
    entity_slot: str | None = None,
    require_complete: bool = True,
    indent: int = 2,
) -> CompiledFSM:
    """Read labeled JSONL files, compile them, and optionally write JSON."""

    paths = [Path(inputs)] if isinstance(inputs, (str, Path)) else [Path(value) for value in inputs]
    records: list[Mapping[str, Any]] = []
    provenance: list[Mapping[str, str]] = []
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise TrainingDataError(f"cannot read {path!s}: {exc}") from exc
        provenance.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, Mapping):
                raise TrainingDataError(f"{path}:{line_number}: JSONL record must be an object")
            record = dict(record)
            record["_source"] = f"{path}:{line_number}"
            records.append(record)
    model = compile_records(
        records,
        start_state=start_state,
        entity_slot=entity_slot,
        require_complete=require_complete,
        provenance=provenance,
    )
    if output is not None:
        model.dump(output, indent=indent)
    return model
