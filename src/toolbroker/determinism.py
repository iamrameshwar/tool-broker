"""Deterministic mode.

Retrieval that returns different tools on different runs is untestable. This
module pins the two sources of nondeterminism we control — tie-break ordering
and embedding drift — so a test suite can assert on exact tool lists.

Ties are broken by tool id, never by insertion order or dict iteration, which
makes results stable across Python versions and across store backends.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Hit

ENV_CACHE_DIR = "TOOLBROKER_CACHE_DIR"
DEFAULT_CACHE_DIR = "~/.cache/toolbroker"


def cache_dir() -> Path:
    """Return the embedding cache directory, honouring ``TOOLBROKER_CACHE_DIR``."""
    raw = os.environ.get(ENV_CACHE_DIR, DEFAULT_CACHE_DIR)
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_sort(hits: Sequence[Hit]) -> list[Hit]:
    """Sort hits by descending score, breaking ties by tool id.

    Deterministic across runs and backends, which floating-point ranking alone
    is not.
    """
    return sorted(hits, key=lambda hit: (-hit.score, hit.id))


def content_hash(*parts: object) -> str:
    """Return a stable short hash of ``parts``, for cache keys."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
