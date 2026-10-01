"""Compiled query transducer and learned controller runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import ControllerError, ModelFormatError
from .normalize import tokenize
from .similarity import METHOD as SIMILARITY_METHOD, cosine, embed, slot_token


FORMAT_NAME = "fsm-only-v1"


@dataclass(frozen=True)
class QueryMatch:
    """One accepting path through the learned query automaton."""

    intent: str
    slots: Mapping[str, str]
    support: int
    literal_tokens: int
    slot_transitions: int
    match_method: str = "exact"
    similarity: float = 1.0


@dataclass(frozen=True)
class ControllerStep:
    """One learned controller transition."""

    state: str
    event: str
    action: str
    next_state: str
    support: int


class CompiledFSM:
    """Immutable facade over a compiler-produced JSON artifact."""

    def __init__(self, artifact: Mapping[str, Any]):
        self.artifact = dict(artifact)
        if self.artifact.get("format") != FORMAT_NAME:
            raise ModelFormatError(f"expected model format {FORMAT_NAME!r}")
        query = self.artifact.get("query_automaton")
        controller = self.artifact.get("controller")
        if not isinstance(query, Mapping) or not isinstance(query.get("states"), list):
            raise ModelFormatError("query_automaton.states must be a list")
        if not isinstance(controller, Mapping) or not isinstance(controller.get("transitions"), Mapping):
            raise ModelFormatError("controller.transitions must be an object")
        self._states = query["states"]
        self._start_query_state = int(query.get("start_state", 0))
        self.start_state = str(controller.get("start_state", ""))
        self.entity_slot = str(self.artifact.get("entity_slot", "object")).casefold()
        self.lookup_events = dict(self.artifact.get("lookup_events", {}))
        self.templates = dict(self.artifact.get("templates", {}))
        raw_similarity = self.artifact.get("similarity", {})
        self.similarity = dict(raw_similarity) if isinstance(raw_similarity, Mapping) else {}
        self._similarity_idf = {
            str(key): float(value)
            for key, value in self.similarity.get("idf", {}).items()
        } if isinstance(self.similarity.get("idf", {}), Mapping) else {}
        self._similarity_prototypes: list[tuple[str, int, dict[str, float]]] = []
        prototypes = self.similarity.get("prototypes", [])
        if prototypes and self.similarity.get("method") != SIMILARITY_METHOD:
            raise ModelFormatError(f"unsupported similarity method {self.similarity.get('method')!r}")
        if not isinstance(prototypes, list):
            raise ModelFormatError("similarity.prototypes must be a list")
        for prototype in prototypes:
            if not isinstance(prototype, Mapping) or not isinstance(prototype.get("tokens"), list):
                raise ModelFormatError("each similarity prototype needs a tokens array")
            tokens = tuple(str(value) for value in prototype["tokens"])
            self._similarity_prototypes.append((
                str(prototype.get("intent", "")),
                int(prototype.get("support", 1)),
                embed(tokens, self._similarity_idf),
            ))
        if not self.start_state:
            raise ModelFormatError("controller.start_state is required")

    @classmethod
    def load(cls, path: str | Path) -> "CompiledFSM":
        """Load a compiled FSM artifact from JSON."""

        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelFormatError(f"cannot load compiled model {path!s}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ModelFormatError("compiled model root must be a JSON object")
        return cls(value)

    def dump(self, path: str | Path, *, indent: int = 2) -> None:
        """Write this artifact deterministically."""

        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.artifact, handle, ensure_ascii=False, indent=indent, sort_keys=True)
            handle.write("\n")

    def controller_step(self, state: str, event: str) -> ControllerStep:
        """Follow the learned transition for ``(state, event)``."""

        raw = self.artifact["controller"]["transitions"].get(state, {}).get(event)
        if not isinstance(raw, Mapping):
            raise ControllerError(f"no learned controller transition for state={state!r}, event={event!r}")
        return ControllerStep(
            state=state,
            event=event,
            action=str(raw["action"]),
            next_state=str(raw["next_state"]),
            support=int(raw.get("support", 1)),
        )

    def has_controller_step(self, state: str, event: str) -> bool:
        """Return whether the learned table contains ``(state, event)``."""

        return event in self.artifact["controller"]["transitions"].get(state, {})

    def match_all(self, text: str, *, limit: int = 256) -> tuple[QueryMatch, ...]:
        """Return all exact accepting interpretations of a query.

        Literal transitions consume one normalized token. Slot transitions consume
        one or more tokens and copy their exact normalized phrase to the output.
        The traversal is bounded and contains no statistical or fuzzy matching.
        """

        tokens = tokenize(text)
        if not tokens or len(tokens) > 128:
            return ()
        found: list[QueryMatch] = []

        def walk(
            state_id: int,
            position: int,
            captures: dict[str, str],
            literal_count: int,
            slot_count: int,
        ) -> None:
            if len(found) >= limit:
                return
            try:
                state = self._states[state_id]
            except (IndexError, TypeError) as exc:
                raise ModelFormatError(f"invalid query state id {state_id}") from exc
            final = state.get("final")
            if position == len(tokens) and isinstance(final, Mapping):
                found.append(
                    QueryMatch(
                        intent=str(final["intent"]),
                        slots=dict(captures),
                        support=int(final.get("support", 1)),
                        literal_tokens=literal_count,
                        slot_transitions=slot_count,
                        match_method="exact",
                        similarity=1.0,
                    )
                )
            transitions = state.get("transitions", {})
            if not isinstance(transitions, Mapping) or position >= len(tokens):
                return
            for symbol, next_state in sorted(transitions.items(), key=lambda pair: (not pair[0].startswith("L:"), pair[0])):
                if symbol.startswith("L:"):
                    if tokens[position] == symbol[2:]:
                        walk(int(next_state), position + 1, captures, literal_count + 1, slot_count)
                    continue
                if not symbol.startswith("S:"):
                    raise ModelFormatError(f"unknown automaton symbol {symbol!r}")
                name = symbol[2:]
                for end in range(position + 1, len(tokens) + 1):
                    value = " ".join(tokens[position:end])
                    previous = captures.get(name)
                    if previous is not None and previous != value:
                        continue
                    updated = dict(captures)
                    updated[name] = value
                    walk(int(next_state), end, updated, literal_count, slot_count + 1)

        walk(self._start_query_state, 0, {}, 0, 0)
        unique: dict[tuple[str, tuple[tuple[str, str], ...]], QueryMatch] = {}
        for match in found:
            key = (match.intent, tuple(sorted(match.slots.items())))
            existing = unique.get(key)
            score = (match.literal_tokens, -match.slot_transitions, match.support)
            if existing is None or score > (existing.literal_tokens, -existing.slot_transitions, existing.support):
                unique[key] = match
        return tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    -item.literal_tokens,
                    item.slot_transitions,
                    -item.support,
                    item.intent,
                    tuple(sorted(item.slots.items())),
                ),
            )
        )

    def match(self, text: str) -> QueryMatch | None:
        """Return an exact interpretation or a cosine-similarity fallback."""

        matches = self.match_all(text)
        if matches:
            return matches[0]
        return self.match_similar(text)

    def match_similar(self, text: str) -> QueryMatch | None:
        """Infer one object span by cosine similarity to learned query patterns."""

        if not self.similarity.get("enabled") or not self._similarity_prototypes:
            return None
        tokens = tokenize(text)
        if len(tokens) < 2 or len(tokens) > 128:
            return None
        maximum = min(int(self.similarity.get("max_slot_tokens", 12)), len(tokens) - 1)
        marker = slot_token(self.entity_slot)
        candidates: dict[tuple[str, str], QueryMatch] = {}
        for start in range(len(tokens)):
            for end in range(start + 1, min(len(tokens), start + maximum) + 1):
                if start == 0 and end == len(tokens):
                    continue
                template = (*tokens[:start], marker, *tokens[end:])
                vector = embed(template, self._similarity_idf)
                captured = " ".join(tokens[start:end])
                for intent, support, prototype_vector in self._similarity_prototypes:
                    score = cosine(vector, prototype_vector)
                    key = (intent, captured)
                    previous = candidates.get(key)
                    if previous is None or score > previous.similarity:
                        candidates[key] = QueryMatch(
                            intent=intent,
                            slots={self.entity_slot: captured},
                            support=support,
                            literal_tokens=len(tokens) - (end - start),
                            slot_transitions=1,
                            match_method="cosine_similarity",
                            similarity=score,
                        )
        ranked = sorted(
            candidates.values(),
            key=lambda match: (
                -match.similarity,
                -match.literal_tokens,
                -match.support,
                len(match.slots[self.entity_slot].split()),
                match.intent,
                match.slots[self.entity_slot],
            ),
        )
        if not ranked:
            return None
        top_score = ranked[0].similarity
        slot_margin = float(self.similarity.get("slot_margin", 0.0))
        near_best = [item for item in ranked if top_score - item.similarity <= slot_margin]
        best = sorted(
            near_best,
            key=lambda match: (
                -len(match.slots[self.entity_slot].split()),
                -match.similarity,
                -match.support,
                match.intent,
                match.slots[self.entity_slot],
            ),
        )[0]
        if best.similarity < float(self.similarity.get("minimum", 1.0)):
            return None
        # The margin is measured against a genuinely different intent. Nearby
        # spans for the same learned intent are resolved deterministically.
        competitor = next((item for item in ranked[1:] if item.intent != best.intent), None)
        if competitor is not None and best.similarity - competitor.similarity < float(
            self.similarity.get("minimum_margin", 0.0)
        ):
            return None
        return best

    @property
    def controller_states(self) -> tuple[str, ...]:
        return tuple(sorted(self.artifact["controller"]["transitions"]))
