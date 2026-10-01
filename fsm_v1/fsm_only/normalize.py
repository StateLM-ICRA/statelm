"""Shared, deterministic text normalization."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)
_SLOT_RE = re.compile(r"^(?:\{([^{}]+)\}|<([^<>]+)>)$")


def tokenize(text: str) -> tuple[str, ...]:
    """Normalize *text* into a stable tuple of Unicode word tokens."""

    normalized = unicodedata.normalize("NFKC", str(text)).replace("\u2019", "'")
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(normalized))


def normalize_phrase(text: str) -> str:
    """Return the canonical space-separated form used by exact lookup."""

    return " ".join(tokenize(text))


def phrase_is_present(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    """Whether *phrase* occurs contiguously in *tokens*."""

    if not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    return any(tuple(tokens[index : index + width]) == tuple(phrase) for index in range(len(tokens) - width + 1))


def slot_name(symbol: str) -> str | None:
    """Extract a slot name from ``{slot}`` or ``<slot>``."""

    match = _SLOT_RE.match(str(symbol).strip())
    if not match:
        return None
    name = (match.group(1) or match.group(2) or "").strip().casefold()
    if not name or not re.fullmatch(r"[a-z][a-z0-9_.-]*", name):
        return None
    return name


def canonical_symbol(symbol: str) -> str:
    """Encode a training symbol as a literal or capture transition key."""

    capture = slot_name(symbol)
    if capture is not None:
        return f"S:{capture}"
    tokens = tokenize(symbol)
    if len(tokens) != 1:
        raise ValueError(f"literal pattern symbols must contain one token: {symbol!r}")
    return f"L:{tokens[0]}"


def flatten_tokens(parts: Iterable[str]) -> tuple[str, ...]:
    """Tokenize and concatenate string parts."""

    return tuple(token for part in parts for token in tokenize(part))
