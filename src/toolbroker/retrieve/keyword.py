"""BM25 lexical retrieval.

Vector search misses exact identifiers: a user who types ``stripe_refund``
wants that tool, not the five things semantically near it. BM25 catches those,
which is why the hybrid retriever fuses both.

The index is built lazily from the store's records and invalidated by record
count and content hash, so a re-index is picked up without an explicit call.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from collections.abc import Sequence

from ..determinism import content_hash, stable_sort
from ..index.enrich import tokenize
from ..protocols import Store
from ..types import Filters, Hit, ToolRecord

K1 = 1.5
B = 0.75


class KeywordRetriever:
    """Okapi BM25 over the same index text the embedder saw."""

    def __init__(self, store: Store, *, k1: float = K1, b: float = B) -> None:
        """Wire the store this retriever reads documents from."""
        self._store = store
        self._k1 = k1
        self._b = b
        self._lock = threading.Lock()
        self._signature: str | None = None
        self._records: Sequence[ToolRecord] = ()
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._average_length: float = 0.0
        self._document_frequencies: Counter[str] = Counter()

    def _ensure_index(self) -> None:
        """Rebuild the lexical index if the store changed underneath us."""
        records = self._store.all_records()
        signature = content_hash(len(records), tuple(record.id for record in records))
        with self._lock:
            if signature == self._signature:
                return
            self._records = records
            self._term_frequencies = [Counter(tokenize(record.text)) for record in records]
            self._lengths = [sum(counter.values()) for counter in self._term_frequencies]
            self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
            self._document_frequencies = Counter()
            for counter in self._term_frequencies:
                self._document_frequencies.update(counter.keys())
            self._signature = signature

    def _idf(self, term: str) -> float:
        total = len(self._records)
        frequency = self._document_frequencies.get(term, 0)
        if total == 0 or frequency == 0:
            return 0.0
        return math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return up to ``k`` lexically matching tools."""
        if k <= 0:
            return []
        self._ensure_index()
        terms = tokenize(query)
        if not terms or not self._records:
            return []

        hits: list[Hit] = []
        for position, record in enumerate(self._records):
            if filters is not None and not filters.matches(record.tool):
                continue
            frequencies = self._term_frequencies[position]
            length = self._lengths[position] or 1
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / (self._average_length or 1.0)
                )
                score += self._idf(term) * frequency * (self._k1 + 1) / denominator
            if score > 0.0:
                hits.append(Hit(tool=record.tool, score=score, components={"bm25": score}))

        return stable_sort(hits)[:k]

    def __repr__(self) -> str:
        """Show the wiring."""
        return f"KeywordRetriever(store={self._store!r})"
