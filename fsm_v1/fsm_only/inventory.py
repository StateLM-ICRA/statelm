"""Exact, data-driven inventory lookup."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import InventoryFormatError
from .normalize import normalize_phrase, phrase_is_present, tokenize
from .similarity import cosine, embed, fit_idf


@dataclass(frozen=True)
class InventoryLookup:
    """Inventory candidates plus the method and score used to retrieve them."""

    items: tuple["InventoryItem", ...]
    method: str
    similarity: float | None = None


@dataclass(frozen=True)
class InventoryItem:
    id: str
    name: str
    aliases: tuple[str, ...]
    family: str | None
    attributes: Mapping[str, str]
    location: str | None
    setting: str | None = None

    @property
    def selectors(self) -> tuple[str, ...]:
        values = {normalize_phrase(self.id), normalize_phrase(self.name)}
        values.update(normalize_phrase(value) for value in self.aliases)
        if self.family:
            values.add(normalize_phrase(self.family))
        return tuple(sorted(value for value in values if value))

    @property
    def semantic_selectors(self) -> tuple[str, ...]:
        values = {normalize_phrase(self.name)}
        values.update(normalize_phrase(value) for value in self.aliases)
        if self.family:
            values.add(normalize_phrase(self.family))
        return tuple(sorted(value for value in values if value))


class Inventory:
    """A validated immutable inventory snapshot."""

    def __init__(self, items: Iterable[InventoryItem]):
        self.items = tuple(items)
        if not self.items:
            raise InventoryFormatError("inventory is empty")
        by_id: dict[str, InventoryItem] = {}
        for item in self.items:
            key = normalize_phrase(item.id)
            if key in by_id:
                raise InventoryFormatError(f"duplicate inventory id {item.id!r}")
            by_id[key] = item
        self._by_id = by_id
        semantic_rows: list[tuple[str, tuple[str, ...]]] = []
        for item in self.items:
            for selector in item.semantic_selectors:
                semantic_rows.append((item.id, tokenize(selector)))
        self._semantic_idf = fit_idf(tokens for _item_id, tokens in semantic_rows)
        self._semantic_rows = tuple(
            (item_id, embed(tokens, self._semantic_idf))
            for item_id, tokens in semantic_rows
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        location_field: str = "location",
        setting: str | None = None,
    ) -> "Inventory":
        items: list[InventoryItem] = []
        requested_setting = normalize_phrase(setting or "")
        for index, raw in enumerate(records, 1):
            if not isinstance(raw, Mapping):
                raise InventoryFormatError(f"inventory record {index} must be an object")
            item_id = str(raw.get("id", "")).strip()
            name = str(raw.get("name", "")).strip()
            item_setting_raw = raw.get("setting")
            item_setting = str(item_setting_raw).strip() if item_setting_raw is not None else None
            if requested_setting and normalize_phrase(item_setting or "") != requested_setting:
                continue
            aliases_raw = raw.get("aliases", [])
            attributes_raw = raw.get("attributes", {})
            if not item_id or not name:
                raise InventoryFormatError(f"inventory record {index} requires non-empty id and name")
            if not isinstance(aliases_raw, list) or not all(isinstance(value, str) for value in aliases_raw):
                raise InventoryFormatError(f"inventory record {index} aliases must be an array of strings")
            if not isinstance(attributes_raw, Mapping):
                raise InventoryFormatError(f"inventory record {index} attributes must be an object")
            attributes: dict[str, str] = {}
            for key, value in attributes_raw.items():
                normalized_key = str(key).strip().casefold()
                if not normalized_key or value is None or not normalize_phrase(str(value)):
                    raise InventoryFormatError(f"inventory record {index} has an invalid attribute")
                attributes[normalized_key] = str(value).strip()
            location = raw.get(location_field)
            if location_field == "location" and "location" not in raw and "drawer" in raw:
                location = raw.get("drawer")
            if location is not None:
                location = str(location).strip() or None
            family_value = raw.get("family")
            family = str(family_value).strip() if family_value is not None else None
            items.append(
                InventoryItem(
                    id=item_id,
                    name=name,
                    aliases=tuple(value.strip() for value in aliases_raw if value.strip()),
                    family=family or None,
                    attributes=attributes,
                    location=location,
                    setting=item_setting or None,
                )
            )
        return cls(items)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        location_field: str = "location",
        setting: str | None = None,
    ) -> "Inventory":
        """Load either a JSON array/object or JSONL inventory file."""

        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise InventoryFormatError(f"cannot read inventory {path!s}: {exc}") from exc
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                records = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise InventoryFormatError(f"invalid inventory JSON/JSONL: {exc}") from exc
        else:
            if isinstance(parsed, Mapping):
                parsed = parsed.get("items")
            if not isinstance(parsed, list):
                raise InventoryFormatError("inventory JSON must be an array or an object with an items array")
            records = parsed
        return cls.from_records(records, location_field=location_field, setting=setting)

    def by_ids(self, ids: Iterable[str]) -> tuple[InventoryItem, ...]:
        """Resolve normalized IDs while preserving inventory order."""

        wanted = {normalize_phrase(value) for value in ids}
        return tuple(item for item in self.items if normalize_phrase(item.id) in wanted)

    def lookup(self, slots: Mapping[str, str], *, entity_slot: str = "object") -> tuple[InventoryItem, ...]:
        """Return items using exact matching followed by cosine similarity."""

        return self.lookup_detailed(slots, entity_slot=entity_slot).items

    def lookup_detailed(
        self,
        slots: Mapping[str, str],
        *,
        entity_slot: str = "object",
        minimum_similarity: float = 0.55,
        similarity_margin: float = 0.04,
    ) -> InventoryLookup:
        """Filter records and report whether exact or cosine retrieval was used."""

        candidates = list(self.items)
        entity_value = normalize_phrase(slots.get(entity_slot, ""))
        if entity_value:
            candidates = [item for item in candidates if entity_value in item.selectors]
            method = "exact"
            score: float | None = 1.0
            if not candidates:
                query_vector = embed(tokenize(entity_value), self._semantic_idf)
                item_scores: dict[str, float] = {}
                for item_id, selector_vector in self._semantic_rows:
                    item_scores[item_id] = max(
                        item_scores.get(item_id, 0.0),
                        cosine(query_vector, selector_vector),
                    )
                score = max(item_scores.values(), default=0.0)
                if score >= minimum_similarity:
                    selected_ids = {
                        item_id
                        for item_id, item_score in item_scores.items()
                        if item_score >= minimum_similarity and score - item_score <= similarity_margin
                    }
                    candidates = [item for item in self.items if item.id in selected_ids]
                    method = "cosine_similarity"
                else:
                    method = "none"
                    candidates = []
        else:
            method = "exact"
            score = 1.0
        for raw_key, raw_value in slots.items():
            key = str(raw_key).casefold()
            if key == entity_slot:
                continue
            value = normalize_phrase(raw_value)
            if key == "family":
                candidates = [item for item in candidates if normalize_phrase(item.family or "") == value]
                continue
            for prefix in ("attribute.", "attr."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            candidates = [item for item in candidates if normalize_phrase(item.attributes.get(key, "")) == value]
        return InventoryLookup(tuple(candidates), method, score)

    def narrow(self, candidate_ids: Iterable[str], clarification: str) -> tuple[InventoryItem, ...]:
        """Narrow candidates from an exact name, alias, family, or attribute phrase."""

        candidates = self.by_ids(candidate_ids)
        tokens = tokenize(clarification)
        if not tokens:
            return ()
        scored: list[tuple[int, InventoryItem]] = []
        full = " ".join(tokens)
        for item in candidates:
            score = 0
            identity_values = {normalize_phrase(item.id), normalize_phrase(item.name)}
            identity_values.update(normalize_phrase(value) for value in item.aliases)
            for value in identity_values:
                phrase = tokenize(value)
                if value == full:
                    score = max(score, 1000 + len(phrase))
                elif phrase_is_present(tokens, phrase):
                    score = max(score, 100 + len(phrase))
            if item.family and phrase_is_present(tokens, tokenize(item.family)):
                score += 10
            if item.setting and phrase_is_present(tokens, tokenize(item.setting)):
                score += 30
            for value in item.attributes.values():
                phrase = tokenize(value)
                if phrase_is_present(tokens, phrase):
                    score += 20 + len(phrase)
            if score:
                scored.append((score, item))
        if not scored:
            return ()
        best = max(score for score, _item in scored)
        selected = {normalize_phrase(item.id) for score, item in scored if score == best}
        return tuple(item for item in candidates if normalize_phrase(item.id) in selected)
