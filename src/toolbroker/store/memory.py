"""In-process vector store.

Brute-force cosine over a normalized matrix. For the catalogue sizes this
library targets — hundreds to low thousands of tools — an approximate index
would add a dependency and a build step to save microseconds. Reach for
``toolbroker-qdrant`` and friends when the catalogue outgrows this.

Filters are applied *before* scoring so ``k`` means what the caller expects:
asking for 5 billing tools returns 5, not whatever survives filtering the
global top 5.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from .._vectors import Matrix, Vector
from ..determinism import stable_sort
from ..errors import DimensionMismatchError
from ..types import Filters, Hit, ToolRecord


class InMemoryStore:
    """Thread-safe brute-force vector store."""

    def __init__(self, dim: int) -> None:
        """Create an empty store for vectors of width ``dim``."""
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._matrix = Matrix(dim)
        self._records: list[ToolRecord] = []
        self._index_by_id: dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def dim(self) -> int:
        """Vector width this store was built for."""
        return self._dim

    def __len__(self) -> int:
        """Number of stored records."""
        with self._lock:
            return len(self._records)

    def upsert(self, records: Sequence[ToolRecord]) -> None:
        """Insert or replace records, keyed by tool id."""
        with self._lock:
            for record in records:
                if len(record.vector) != self._dim:
                    raise DimensionMismatchError(self._dim, len(record.vector))
                existing = self._index_by_id.get(record.id)
                if existing is None:
                    position = self._matrix.append(record.vector)
                    self._records.append(record)
                    self._index_by_id[record.id] = position
                else:
                    self._matrix.replace(existing, record.vector)
                    self._records[existing] = record

    def delete(self, tool_ids: Sequence[str]) -> int:
        """Remove records by tool id.

        Rebuilds the matrix rather than tombstoning: deletes are rare here, and
        a compact matrix keeps search simple and predictable.
        """
        with self._lock:
            targets = set(tool_ids)
            keep = [record for record in self._records if record.id not in targets]
            removed = len(self._records) - len(keep)
            if removed:
                self._rebuild(keep)
            return removed

    def _rebuild(self, records: Sequence[ToolRecord]) -> None:
        """Rewrite internal state from ``records``. Called with the lock held."""
        self._matrix.clear()
        self._records = []
        self._index_by_id = {}
        for record in records:
            position = self._matrix.append(record.vector)
            self._records.append(record)
            self._index_by_id[record.id] = position

    def get(self, tool_id: str) -> ToolRecord | None:
        """Return one record by tool id."""
        with self._lock:
            position = self._index_by_id.get(tool_id)
            return None if position is None else self._records[position]

    def all_records(self) -> Sequence[ToolRecord]:
        """Return a snapshot of every record."""
        with self._lock:
            return tuple(self._records)

    def search(self, vector: Vector, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return the ``k`` highest-scoring records that satisfy ``filters``."""
        if k <= 0:
            return []
        if len(vector) != self._dim:
            raise DimensionMismatchError(self._dim, len(vector))

        with self._lock:
            if not self._records:
                return []
            if filters is None or filters.is_empty():
                candidates = list(range(len(self._records)))
            else:
                candidates = [
                    position
                    for position, record in enumerate(self._records)
                    if filters.matches(record.tool)
                ]
            if not candidates:
                return []
            scores = self._matrix.similarities(vector, candidates)
            hits = [
                Hit(
                    tool=self._records[position].tool,
                    score=float(score),
                    components={"vector": float(score)},
                )
                for position, score in zip(candidates, scores, strict=True)
            ]

        return stable_sort(hits)[:k]

    def clear(self) -> None:
        """Drop everything."""
        with self._lock:
            self._matrix.clear()
            self._records = []
            self._index_by_id = {}

    def __repr__(self) -> str:
        """Show size and dimensionality."""
        return f"InMemoryStore(dim={self._dim}, size={len(self)})"
