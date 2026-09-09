"""Vector math with an optional numpy fast path.

The in-memory store must work on a machine with nothing installed but pydantic,
so every operation has a pure-Python implementation. When numpy is present the
same functions dispatch to it, which matters once a catalogue passes a few
thousand tools.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Vector = Sequence[float]

try:  # pragma: no cover - trivial import guard
    import numpy as _np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    HAS_NUMPY = False


def l2_normalize(vector: Vector) -> list[float]:
    """Return ``vector`` scaled to unit length, or unchanged if it is all zeros."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def dot(left: Vector, right: Vector) -> float:
    """Return the dot product of two equal-length vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Return cosine similarity, tolerating non-normalized inputs."""
    numerator = dot(left, right)
    denominator = math.sqrt(dot(left, left)) * math.sqrt(dot(right, right))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


class Matrix:
    """A stack of unit-length row vectors supporting batched similarity.

    Rows are assumed pre-normalized so similarity reduces to a dot product. The
    numpy path keeps a contiguous 2-D array; the fallback keeps a list of lists.
    """

    __slots__ = ("_dim", "_rows")

    def __init__(self, dim: int) -> None:
        """Create an empty matrix of row width ``dim``."""
        self._dim = dim
        self._rows: Any = _np.zeros((0, dim), dtype="float32") if HAS_NUMPY else []

    @property
    def dim(self) -> int:
        """Row width."""
        return self._dim

    def __len__(self) -> int:
        """Number of rows."""
        return int(self._rows.shape[0]) if HAS_NUMPY else len(self._rows)

    def append(self, vector: Vector) -> int:
        """Append a normalized copy of ``vector`` and return its row index."""
        if len(vector) != self._dim:
            from .errors import DimensionMismatchError

            raise DimensionMismatchError(self._dim, len(vector))
        normalized = l2_normalize(vector)
        index = len(self)
        if HAS_NUMPY:
            row: Any = _np.asarray(normalized, dtype="float32").reshape(1, self._dim)
            self._rows = _np.concatenate([self._rows, row], axis=0)
        else:
            self._rows.append(normalized)
        return index

    def replace(self, index: int, vector: Vector) -> None:
        """Overwrite the row at ``index`` with a normalized copy of ``vector``."""
        normalized = l2_normalize(vector)
        if HAS_NUMPY:
            self._rows[index] = _np.asarray(normalized, dtype="float32")
        else:
            self._rows[index] = normalized

    def similarities(self, query: Vector, candidates: Sequence[int]) -> list[float]:
        """Return similarity between ``query`` and each row in ``candidates``."""
        if not candidates:
            return []
        normalized = l2_normalize(query)
        if HAS_NUMPY:
            subset = self._rows[_np.asarray(candidates, dtype="int64")]
            vector: Any = _np.asarray(normalized, dtype="float32")
            return [float(value) for value in subset @ vector]
        return [dot(self._rows[index], normalized) for index in candidates]

    def clear(self) -> None:
        """Drop every row."""
        self._rows = _np.zeros((0, self._dim), dtype="float32") if HAS_NUMPY else []
