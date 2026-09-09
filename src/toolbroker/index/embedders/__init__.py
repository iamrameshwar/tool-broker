"""Embedders, plus the resolution order used when none is specified."""

from .base import CachedEmbedder
from .hashing import HashingEmbedder
from .resolve import default_embedder, get_embedder

__all__ = ["CachedEmbedder", "HashingEmbedder", "default_embedder", "get_embedder"]
