"""Dense vector retrieval. The default."""

from __future__ import annotations

from ..hooks import Event, HookManager
from ..observability import span
from ..protocols import Embedder, Store
from ..types import Filters, Hit


class SemanticRetriever:
    """Embeds the query and asks the store for nearest neighbours."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        *,
        hooks: HookManager | None = None,
        min_score: float = 0.0,
    ) -> None:
        """Wire the store and embedder.

        Args:
            store: Where vectors live.
            embedder: Must be the same embedder the index was built with.
            hooks: Optional hook manager for query rewriting.
            min_score: Drop hits below this similarity. Zero keeps everything.
        """
        self._store = store
        self._embedder = embedder
        self._hooks = hooks or HookManager()
        self._min_score = min_score

    @property
    def store(self) -> Store:
        """The store being queried."""
        return self._store

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return up to ``k`` semantically similar tools."""
        with span("toolbroker.retrieve.semantic", k=k):
            rewritten = self._hooks.transform(Event.TRANSFORM_QUERY, query)
            vector = self._embedder.embed_query(rewritten)
            hits = self._store.search(vector, k, filters)
            if self._min_score > 0.0:
                hits = [hit for hit in hits if hit.score >= self._min_score]
            return hits

    def __repr__(self) -> str:
        """Show the wiring."""
        return f"SemanticRetriever(store={self._store!r}, embedder={self._embedder!r})"
