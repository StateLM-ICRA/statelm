"""Session API for deterministic inventory-location dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import ControllerError
from .inventory import Inventory, InventoryItem
from .model import CompiledFSM, ControllerStep, QueryMatch


@dataclass(frozen=True)
class TurnResult:
    """Structured result returned by :meth:`InventorySession.handle`."""

    text: str
    action: str
    state: str
    event: str | None = None
    intent: str | None = None
    slots: Mapping[str, str] = field(default_factory=dict)
    matches: tuple[str, ...] = ()
    query_match_method: str | None = None
    query_similarity: float | None = None
    inventory_match_method: str | None = None
    inventory_similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "action": self.action,
            "state": self.state,
            "event": self.event,
            "intent": self.intent,
            "slots": dict(self.slots),
            "matches": list(self.matches),
            "query_match_method": self.query_match_method,
            "query_similarity": self.query_similarity,
            "inventory_match_method": self.inventory_match_method,
            "inventory_similarity": self.inventory_similarity,
        }


class InventorySession:
    """A single deterministic dialogue session.

    The compiled artifact chooses actions. The runtime only performs exact
    inventory operations and formats the selected action's response.
    """

    def __init__(
        self,
        model: CompiledFSM,
        inventory: Inventory,
        *,
        templates: Mapping[str, str] | None = None,
    ):
        self.model = model
        self.inventory = inventory
        self.templates = dict(model.templates)
        if templates:
            self.templates.update({str(key): str(value) for key, value in templates.items()})
        self.state = model.start_state
        self._pending_ids: tuple[str, ...] = ()

    @property
    def awaiting_clarification(self) -> bool:
        return bool(self._pending_ids)

    def reset(self) -> None:
        """Clear dialogue memory and return to the learned start state."""

        self.state = self.model.start_state
        self._pending_ids = ()

    def reply(self, text: str) -> str:
        """Convenience wrapper returning only user-facing text."""

        return self.handle(text).text

    def handle(self, text: str) -> TurnResult:
        """Process one user turn through the learned FST and controller."""

        if self._pending_ids:
            return self._handle_clarification(text)
        query = self.model.match(text)
        if query is None:
            if self.model.has_controller_step(self.state, "UNRECOGNIZED_QUERY"):
                step = self._step("UNRECOGNIZED_QUERY")
                return self._execute(step, (), query_text=text)
            return TurnResult(
                text=self.templates["UNRECOGNIZED"],
                action="UNRECOGNIZED",
                state=self.state,
            )
        first = self._step(query.intent)
        if first.action != "LOOKUP":
            return self._execute(first, (), query_text=text, query=query)
        lookup = self.inventory.lookup_detailed(
            query.slots,
            entity_slot=self.model.entity_slot,
            minimum_similarity=float(self.model.similarity.get("inventory_minimum", 0.55)),
            similarity_margin=float(self.model.similarity.get("inventory_margin", 0.04)),
        )
        candidates = lookup.items
        event = self._lookup_event(candidates)
        result_step = self._step(event)
        return self._execute(
            result_step,
            candidates,
            query_text=text,
            query=query,
            event=event,
            inventory_match_method=lookup.method,
            inventory_similarity=lookup.similarity,
        )

    def _handle_clarification(self, text: str) -> TurnResult:
        candidates = self.inventory.narrow(self._pending_ids, text)
        event = self._lookup_event(candidates)
        step = self._step(event)
        return self._execute(step, candidates, query_text=text, event=event)

    def _step(self, event: str) -> ControllerStep:
        step = self.model.controller_step(self.state, event)
        self.state = step.next_state
        return step

    def _lookup_event(self, candidates: tuple[InventoryItem, ...]) -> str:
        if not candidates:
            return self.model.lookup_events["zero"]
        if len(candidates) > 1:
            return self.model.lookup_events["multiple"]
        if candidates[0].location is None:
            return self.model.lookup_events["one_without_location"]
        return self.model.lookup_events["one_with_location"]

    def _execute(
        self,
        step: ControllerStep,
        candidates: tuple[InventoryItem, ...],
        *,
        query_text: str,
        query: QueryMatch | None = None,
        event: str | None = None,
        inventory_match_method: str | None = None,
        inventory_similarity: float | None = None,
    ) -> TurnResult:
        action = step.action
        names = tuple(item.name for item in candidates)
        if action == "RETURN_LOCATION":
            if len(candidates) != 1 or candidates[0].location is None:
                raise ControllerError("RETURN_LOCATION requires exactly one item with a location")
            item = candidates[0]
            rendered = self.templates[action].format(name=item.name, location=item.location)
            self._pending_ids = ()
        elif action == "ASK_WHICH_ONE":
            if len(candidates) < 2:
                raise ControllerError("ASK_WHICH_ONE requires at least two candidates")
            rendered = self.templates[action].format(choices=", ".join(self._choice_labels(candidates)))
            self._pending_ids = tuple(item.id for item in candidates)
        elif action == "OBJECT_MISSING":
            missing_query = query.slots.get(self.model.entity_slot, query_text) if query else query_text
            rendered = self.templates[action].format(query=missing_query)
            self._pending_ids = ()
        elif action == "LOCATION_MISSING":
            if len(candidates) != 1 or candidates[0].location is not None:
                raise ControllerError("LOCATION_MISSING requires exactly one item without a location")
            rendered = self.templates[action].format(name=candidates[0].name)
            self._pending_ids = ()
        elif action in {"UNRECOGNIZED", "UNSUPPORTED_REQUEST"}:
            rendered = self.templates[action]
            self._pending_ids = ()
        else:
            raise ControllerError(f"runtime does not implement learned action {action!r}")
        return TurnResult(
            text=rendered,
            action=action,
            state=self.state,
            event=event or step.event,
            intent=query.intent if query else None,
            slots=dict(query.slots) if query else {},
            matches=names,
            query_match_method=query.match_method if query else None,
            query_similarity=query.similarity if query else None,
            inventory_match_method=inventory_match_method,
            inventory_similarity=inventory_similarity,
        )

    @staticmethod
    def _choice_labels(candidates: tuple[InventoryItem, ...]) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        for item in candidates:
            key = item.name.casefold()
            counts[key] = counts.get(key, 0) + 1
        labels: list[str] = []
        for item in candidates:
            if counts[item.name.casefold()] == 1:
                labels.append(item.name)
            else:
                qualifier = item.setting or item.id
                labels.append(f"{item.name} ({qualifier})")
        return tuple(labels)
