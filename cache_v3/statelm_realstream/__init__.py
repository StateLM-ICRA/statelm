"""StateLM real-stream package: FSM_0 seeding, online semantic caching, replay."""

from .contract import (
    DECISION_KEYS,
    decision_dict,
    empty_memory,
    make_payload,
    next_memory,
    render_response,
    validate_decision,
)
from .features import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    extract_review_features,
    extract_text_features,
)
from .router import CachedPattern, PaperAlignedRouter, RouteResult, RouterConfig

__all__ = [
    "DECISION_KEYS",
    "CachedPattern",
    "HashingEmbedder",
    "PaperAlignedRouter",
    "RouteResult",
    "RouterConfig",
    "SentenceTransformerEmbedder",
    "decision_dict",
    "empty_memory",
    "extract_review_features",
    "extract_text_features",
    "make_payload",
    "next_memory",
    "render_response",
    "validate_decision",
]
