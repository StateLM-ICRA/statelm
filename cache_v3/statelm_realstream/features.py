"""Typed-slot extraction and embedding backends for the semantic FSM.

This module extends the 21 Sept ``statelm_combined.features`` in two ways:

1. Spoken number words in drawer mentions are normalised ("drawer two"
   becomes "drawer 2") so that a drawer mention is recognised as a slot.
2. ``extract_review_features`` types the robot's *proposed* answer relative
   to the user's request.  Item mentions in the proposal become
   ``<item:same>`` or ``<item:other>`` and drawer mentions become
   ``<drawer:match>`` or ``<drawer:mismatch>``, judged against the live
   inventory.  The review-mode embedding text is therefore
   ``typed(user) || typed(proposal)``; the failure cue carried by the
   robot's line is part of the pattern instead of being discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata
from typing import Any, Mapping, Protocol, Sequence

from .contract import normalize_inventory


TOKEN_RE = re.compile(r"[a-z0-9]+")
CORRECTION_CUES = (
    "did not mean",
    "didn't mean",
    "i meant",
    "no i said",
    "no, i said",
    "i said",
    "that is wrong",
    "that's wrong",
    "wrong item",
    "not that one",
)
NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    # ASR homophones observed in the transcripts ("drawer too"); "to" and
    # "for" are deliberately left alone ("which drawer for the tape").
    "won": "1", "too": "2", "tree": "3", "fore": "4",
}
REVIEW_SEPARATOR = "||"


def normal_text(text: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def tokens(text: Any) -> tuple[str, ...]:
    raw = TOKEN_RE.findall(normal_text(text))
    out: list[str] = []
    for index, token in enumerate(raw):
        # "drawer two" -> "drawer 2": only after the word drawer, so that
        # "one" in "the blue one" keeps its ordinary meaning.
        if token in NUMBER_WORDS and index > 0 and raw[index - 1] in {"drawer", "door", "draw"}:
            out.append(NUMBER_WORDS[token])
        else:
            out.append(token)
    return tuple(out)


def is_correction(text: Any) -> bool:
    value = normal_text(text)
    return any(cue in value for cue in CORRECTION_CUES)


def _number_variants(phrase: tuple[str, ...]) -> set[tuple[str, ...]]:
    """Return conservative singular/plural variants for an inventory phrase."""

    if not phrase:
        return set()
    variants = {phrase}
    last = phrase[-1]
    if last.endswith("ies") and len(last) > 3:
        variants.add((*phrase[:-1], last[:-3] + "y"))
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 1:
        variants.add((*phrase[:-1], last[:-1]))
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        variants.add((*phrase[:-1], last[:-1] + "ies"))
    else:
        variants.add((*phrase[:-1], last + "s"))
    return variants


@dataclass(frozen=True)
class TextFeatures:
    typed_text: str
    item_ids: tuple[str, ...]
    families: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    ambiguous: bool
    correction: bool
    # Review mode only: what the robot's proposed line says, relative to the
    # request.  ``proposal_items`` are inventory ids mentioned in the
    # proposal, ``proposal_drawers`` the drawer strings mentioned.
    proposal_typed: str | None = None
    proposal_items: tuple[str, ...] = ()
    proposal_drawers: tuple[str, ...] = ()
    user_typed: str | None = None

    @property
    def last_item_id(self) -> str | None:
        return self.item_ids[-1] if self.item_ids else None


def _phrase_table(inventory: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], list[tuple[str, Any]]]:
    table: dict[tuple[str, ...], list[tuple[str, Any]]] = {}
    for item in normalize_inventory(inventory):
        for phrase in {item["name"], *item.get("aliases", [])}:
            for variant in _number_variants(tokens(phrase)):
                if variant:
                    entry = ("item", item["id"])
                    if entry not in table.setdefault(variant, []):
                        table[variant].append(entry)
        family = tokens(str(item["family"]).replace("_", " "))
        for variant in _number_variants(family):
            entry = ("family", item["family"])
            if entry not in table.setdefault(variant, []):
                table[variant].append(entry)
        for key, value in item["attributes"].items():
            phrase = tokens(value)
            if phrase:
                entry = ("attribute", (key, value))
                if entry not in table.setdefault(phrase, []):
                    table[phrase].append(entry)
        drawer = tokens(item.get("drawer") or "")
        if drawer:
            entry = ("drawer", item["drawer"])
            if entry not in table.setdefault(drawer, []):
                table[drawer].append(entry)
    return table


def _scan(
    token_list: list[str],
    table: Mapping[tuple[str, ...], list[tuple[str, Any]]],
) -> list[tuple[str, Any, int]]:
    """Longest-match scan.  Returns (kind, value(s), width) per position.

    Plain tokens are returned as ("word", token, 1).  A slot match returns
    ("item"|"family"|"attribute"|"drawer", [values], width).
    """

    phrases = sorted(table, key=lambda phrase: (-len(phrase), phrase))
    priority = {"item": 4, "drawer": 3, "attribute": 2, "family": 1}
    out: list[tuple[str, Any, int]] = []
    index = 0
    while index < len(token_list):
        matches: list[tuple[int, int, str, Any]] = []
        for phrase in phrases:
            width = len(phrase)
            if tuple(token_list[index : index + width]) == phrase:
                matches.extend((width, priority[kind], kind, value) for kind, value in table[phrase])
        if not matches:
            out.append(("word", token_list[index], 1))
            index += 1
            continue
        width = max(match[0] for match in matches)
        narrowed = [match for match in matches if match[0] == width]
        best_priority = max(match[1] for match in narrowed)
        narrowed = [match for match in narrowed if match[1] == best_priority]
        kind = narrowed[0][2]
        values: list[Any] = []
        for _width, _priority, _kind, value in narrowed:
            if value not in values:
                values.append(value)
        out.append((kind, values, width))
        index += width
    return out


def extract_text_features(text: Any, inventory: Sequence[Mapping[str, Any]]) -> TextFeatures:
    normalized_inventory = normalize_inventory(inventory)
    token_list = list(tokens(text))
    table = _phrase_table(normalized_inventory)
    family_sizes: dict[str, int] = {}
    for item in normalized_inventory:
        family = str(item["family"])
        family_sizes[family] = family_sizes.get(family, 0) + 1
    item_map = {item["id"]: item for item in normalized_inventory}
    surface: list[str] = []
    item_ids: list[str] = []
    families: list[str] = []
    attributes: list[tuple[str, str]] = []
    ambiguous = False
    for kind, value, _width in _scan(token_list, table):
        if kind == "word":
            surface.append(value)
        elif kind == "item":
            item_ids.extend(str(v) for v in value)
            if len(value) == 1:
                surface.append("<item>")
            else:
                item_families = {item_map[v]["family"] for v in value if v in item_map}
                families.extend(sorted(item_families))
                surface.append("<family>" if len(item_families) == 1 else "<ambiguous-item>")
                ambiguous = True
        elif kind == "family":
            families.extend(str(v) for v in value)
            surface.append("<family>")
            ambiguous |= len(set(value)) > 1 or any(
                family_sizes.get(str(v), 0) > 1 for v in value
            )
        elif kind == "attribute":
            attributes.extend(value)
            keys = {v[0] for v in value}
            surface.append(f"<attribute:{next(iter(keys))}>" if len(keys) == 1 else "<attribute>")
            ambiguous |= len(set(value)) > 1
        else:
            surface.append("<drawer>")
            ambiguous |= len(set(value)) > 1
    unique_items = tuple(dict.fromkeys(item_ids))
    unique_families = tuple(dict.fromkeys(families))
    unique_attributes = tuple(dict.fromkeys(attributes))
    correction = is_correction(text)
    if len(unique_items) > 1 and not correction:
        ambiguous = True
    typed = " ".join(surface) or "<empty>"
    return TextFeatures(
        typed_text=typed,
        item_ids=unique_items,
        families=unique_families,
        attributes=unique_attributes,
        ambiguous=ambiguous,
        correction=correction,
        user_typed=typed,
    )


def extract_review_features(
    user_text: Any,
    proposal_text: Any,
    inventory: Sequence[Mapping[str, Any]],
    requested_item_id: str | None = None,
) -> TextFeatures:
    """Type the user's request and the robot's proposed answer together.

    ``requested_item_id`` is the item the request resolves to (or None).
    When it is None and the request itself names exactly one inventory
    item, that item is used.
    """

    base = extract_text_features(user_text, inventory)
    normalized_inventory = normalize_inventory(inventory)
    item_map = {item["id"]: item for item in normalized_inventory}
    if requested_item_id is None and len(base.item_ids) == 1:
        requested_item_id = base.item_ids[0]
    requested = item_map.get(requested_item_id) if requested_item_id else None
    requested_drawer = normal_text(requested["drawer"]) if requested and requested.get("drawer") else None
    table = _phrase_table(normalized_inventory)
    surface: list[str] = []
    proposal_items: list[str] = []
    proposal_drawers: list[str] = []
    scanned = _scan(list(tokens(proposal_text)), table)
    # "drawer 99": a drawer number that no inventory item uses is still a
    # drawer mention, and it can never match the requested item's drawer.
    merged: list[tuple[str, Any, int]] = []
    index = 0
    while index < len(scanned):
        kind, value, width = scanned[index]
        nxt = scanned[index + 1] if index + 1 < len(scanned) else None
        if (
            kind == "word"
            and value == "drawer"
            and nxt is not None
            and nxt[0] == "word"
            and str(nxt[1]).isdigit()
        ):
            merged.append(("drawer", [f"drawer {nxt[1]}"], 2))
            index += 2
            continue
        merged.append((kind, value, width))
        index += 1
    for kind, value, _width in merged:
        if kind == "word":
            surface.append(value)
        elif kind == "item":
            proposal_items.extend(str(v) for v in value)
            if requested is None:
                surface.append("<item>")
            elif len(value) == 1 and value[0] == requested["id"]:
                surface.append("<item:same>")
            else:
                surface.append("<item:other>")
        elif kind == "family":
            if requested is not None and requested["family"] in {str(v) for v in value}:
                surface.append("<family:same>")
            else:
                surface.append("<family>")
        elif kind == "attribute":
            keys = {v[0] for v in value}
            surface.append(f"<attribute:{next(iter(keys))}>" if len(keys) == 1 else "<attribute>")
        else:
            proposal_drawers.extend(normal_text(v) for v in value)
            if requested_drawer is None:
                surface.append("<drawer>")
            elif len(value) == 1 and normal_text(value[0]) == requested_drawer:
                surface.append("<drawer:match>")
            else:
                surface.append("<drawer:mismatch>")
    proposal_typed = " ".join(surface) or "<empty>"
    # In review mode the object under review is the proposal.  The request
    # enters through what it resolves to (one item, a family, or nothing);
    # its phrasing does not change the review decision and would only
    # fragment the patterns.
    if requested is not None:
        marker = "<request:item>"
    elif base.families or base.ambiguous:
        marker = "<request:family>"
    else:
        marker = "<request:none>"
    if base.correction:
        marker += " <request:correction>"
    return TextFeatures(
        typed_text=f"{marker} {REVIEW_SEPARATOR} {proposal_typed}",
        user_typed=base.typed_text,
        item_ids=base.item_ids,
        families=base.families,
        attributes=base.attributes,
        ambiguous=base.ambiguous,
        correction=base.correction,
        proposal_typed=proposal_typed,
        proposal_items=tuple(dict.fromkeys(proposal_items)),
        proposal_drawers=tuple(dict.fromkeys(proposal_drawers)),
    )


class Embedder(Protocol):
    name: str

    def encode(self, text: str) -> tuple[float, ...]: ...


class HashingEmbedder:
    """Dependency-free word/character n-gram embedding for reproducible tests."""

    name = "hash-word-char-ngram-v1"

    def __init__(self, dimensions: int = 2048):
        if dimensions < 128:
            raise ValueError("dimensions must be at least 128")
        self.dimensions = dimensions

    @staticmethod
    def _features(text: str) -> list[str]:
        words = list(TOKEN_RE.findall(normal_text(text).replace("<", " <").replace(">", "> ")))
        values = [f"w:{word}" for word in words]
        values.extend(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
        compact = " " + " ".join(words) + " "
        for width in (3, 4, 5):
            values.extend(f"c{width}:{compact[i:i+width]}" for i in range(len(compact) - width + 1))
        return values or ["<empty>"]

    def encode(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            index = number % self.dimensions
            sign = -1.0 if (number >> 63) else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return tuple(value / norm for value in vector)


class SentenceTransformerEmbedder:
    """Neural sentence embedding (Colab).  Vectors are L2-normalised."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use the neural embedder") from exc
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self._cache: dict[str, tuple[float, ...]] = {}

    def encode(self, text: str) -> tuple[float, ...]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        value = self.model.encode(text, normalize_embeddings=True)
        result = tuple(float(x) for x in value)
        self._cache[text] = result
        return result


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions disagree")
    return float(sum(a * b for a, b in zip(left, right)))


def mean_unit(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("Cannot average no embeddings")
    size = len(vectors[0])
    if any(len(vector) != size for vector in vectors):
        raise ValueError("Embedding dimensions disagree")
    summed = [sum(vector[i] for vector in vectors) for i in range(size)]
    norm = math.sqrt(sum(value * value for value in summed)) or 1.0
    return tuple(value / norm for value in summed)
