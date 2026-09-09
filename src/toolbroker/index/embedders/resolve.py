"""Choosing an embedder when the caller did not name one.

Order: whatever fastembed can load, falling back to the hashing embedder with a
warning. Never an import error, never a silent network call the user did not
ask for beyond the one fastembed makes on its own.
"""

from __future__ import annotations

from typing import Any

from ...observability import get_logger
from ...protocols import Embedder
from ...registry import GROUP_EMBEDDERS, create
from .hashing import HashingEmbedder

logger = get_logger("embedders")


def get_embedder(name: str, /, **kwargs: Any) -> Embedder:
    """Instantiate the embedder registered under ``name``."""
    embedder: Embedder = create(GROUP_EMBEDDERS, name, **kwargs)
    return embedder


def default_embedder(*, prefer_fastembed: bool = True) -> Embedder:
    """Return the best embedder available without configuration."""
    if prefer_fastembed:
        try:
            from .fastembed import FastEmbedEmbedder

            return FastEmbedEmbedder()
        except Exception as exc:
            logger.warning(
                "falling back to HashingEmbedder (lexical only); "
                "install 'toolbroker[fastembed]' for semantic matching",
                extra={"reason": str(exc)},
            )
    return HashingEmbedder()
