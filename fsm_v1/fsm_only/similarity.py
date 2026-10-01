"""Deterministic sparse embeddings and cosine similarity.

The embedding vocabulary and inverse-document-frequency weights are learned
from the supplied training strings.  No external model, network access, or
randomness is involved.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Mapping, Sequence


METHOD = "tfidf_word_char_ngram_cosine_v1"


def _features(tokens: Sequence[str]) -> Counter[str]:
    """Return word and character n-gram features for a token sequence."""

    normalized = tuple(str(token).casefold() for token in tokens if str(token))
    counts: Counter[str] = Counter()
    for token in normalized:
        counts[f"w:{token}"] += 2
    for left, right in zip(normalized, normalized[1:]):
        counts[f"b:{left}\u241f{right}"] += 2
    surface = "^" + " ".join(normalized) + "$"
    for width in (3, 4, 5):
        for index in range(max(0, len(surface) - width + 1)):
            counts[f"c{width}:{surface[index:index + width]}"] += 1
    return counts


def fit_idf(documents: Iterable[Sequence[str]]) -> dict[str, float]:
    """Learn smoothed IDF weights from training documents."""

    materialized = [tuple(document) for document in documents]
    if not materialized:
        return {}
    document_frequency: Counter[str] = Counter()
    for document in materialized:
        document_frequency.update(_features(document).keys())
    count = len(materialized)
    return {
        feature: math.log((1.0 + count) / (1.0 + frequency)) + 1.0
        for feature, frequency in sorted(document_frequency.items())
    }


def embed(tokens: Sequence[str], idf: Mapping[str, float]) -> dict[str, float]:
    """Create a unit-length sparse TF-IDF embedding."""

    weighted: dict[str, float] = {}
    for feature, frequency in _features(tokens).items():
        if feature not in idf:
            continue
        weighted[feature] = (1.0 + math.log(float(frequency))) * float(idf[feature])
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0.0:
        return {}
    return {feature: value / norm for feature, value in weighted.items()}


def cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Cosine similarity for embeddings already normalized to unit length."""

    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(feature, 0.0) for feature, value in left.items())


def slot_token(name: str) -> str:
    """Return the stable token used to represent a learned slot."""

    return f"<slot:{name.casefold()}>"
