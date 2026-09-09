"""Embedder utilities: caching and batching."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path

from ..._vectors import l2_normalize
from ...determinism import cache_dir, content_hash
from ...observability import get_logger
from ...protocols import Embedder

logger = get_logger("embedders")


class CachedEmbedder:
    """Wraps an embedder with an in-process and optional on-disk cache.

    Two reasons this exists. Re-indexing an unchanged catalogue should not pay
    for embeddings twice, and deterministic mode needs the same text to produce
    the same vector across runs even when the underlying model is not pinned.
    """

    def __init__(
        self,
        inner: Embedder,
        *,
        persist: bool = False,
        directory: Path | None = None,
        max_entries: int = 50_000,
    ) -> None:
        """Wrap ``inner``.

        Args:
            inner: The embedder to delegate to on a cache miss.
            persist: Also cache to disk, keyed by embedder id and text hash.
            directory: Override the cache directory.
            max_entries: In-memory cache ceiling before eviction.
        """
        self._inner = inner
        self._persist = persist
        self._max_entries = max_entries
        self._memory: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._dir = directory
        self.hits = 0
        self.misses = 0

    @property
    def dim(self) -> int:
        """Dimensionality of the wrapped embedder."""
        return self._inner.dim

    @property
    def id(self) -> str:
        """Identifier of the wrapped embedder."""
        return self._inner.id

    @property
    def inner(self) -> Embedder:
        """The wrapped embedder."""
        return self._inner

    def _path(self) -> Path:
        base = self._dir or cache_dir()
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{content_hash(self._inner.id)}.jsonl"

    def _load_disk(self) -> None:
        if not self._persist:
            return
        path = self._path()
        if not path.exists():
            return
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    self._memory.setdefault(entry["k"], entry["v"])
        except (OSError, json.JSONDecodeError, KeyError):
            logger.warning("embedding cache unreadable; ignoring", extra={"path": str(path)})

    def _append_disk(self, pairs: Sequence[tuple[str, list[float]]]) -> None:
        if not self._persist or not pairs:
            return
        try:
            with self._path().open("a", encoding="utf-8") as handle:
                for key, vector in pairs:
                    handle.write(json.dumps({"k": key, "v": vector}) + "\n")
        except OSError:
            logger.warning("could not write embedding cache; continuing without it")

    def _key(self, text: str) -> str:
        return content_hash(self._inner.id, text)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts``, calling the inner embedder only for cache misses."""
        with self._lock:
            if self._persist and not self._memory:
                self._load_disk()

        keys = [self._key(text) for text in texts]
        results: list[list[float] | None] = [None] * len(texts)
        pending: list[int] = []

        with self._lock:
            for position, key in enumerate(keys):
                cached = self._memory.get(key)
                if cached is None:
                    pending.append(position)
                else:
                    results[position] = cached

        self.hits += len(texts) - len(pending)
        self.misses += len(pending)

        if pending:
            computed = self._inner.embed([texts[position] for position in pending])
            fresh: list[tuple[str, list[float]]] = []
            with self._lock:
                for position, vector in zip(pending, computed, strict=True):
                    results[position] = vector
                    self._memory[keys[position]] = vector
                    fresh.append((keys[position], vector))
                if len(self._memory) > self._max_entries:
                    self._evict()
            self._append_disk(fresh)

        return [vector for vector in results if vector is not None]

    def _evict(self) -> None:
        """Drop the oldest half of the cache. Called with the lock held."""
        keep = self._max_entries // 2
        for key in list(self._memory)[: len(self._memory) - keep]:
            del self._memory[key]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query, bypassing the document cache namespace."""
        return self._inner.embed_query(text)

    def clear(self) -> None:
        """Empty the in-memory cache."""
        with self._lock:
            self._memory.clear()

    def __repr__(self) -> str:
        """Show what is wrapped and the hit ratio."""
        return f"CachedEmbedder({self._inner!r}, hits={self.hits}, misses={self.misses})"


def batched(items: Sequence[str], size: int) -> list[Sequence[str]]:
    """Split ``items`` into chunks of at most ``size``."""
    if size <= 0:
        return [items]
    return [items[start : start + size] for start in range(0, len(items), size)]


def normalize_all(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """L2-normalize every vector."""
    return [l2_normalize(vector) for vector in vectors]
