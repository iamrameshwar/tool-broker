"""Composing a retriever with rerankers.

The shape is deliberate: retrieve wide and cheap, then narrow with progressively
more expensive stages. Each stage's contribution stays visible in the hit's
score components, so a surprising ranking can be attributed to a specific stage.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..hooks import Event, HookManager
from ..observability import span
from ..protocols import Reranker, Retriever
from ..types import Filters, Hit


class RetrievalPipeline:
    """A retriever followed by zero or more rerankers."""

    def __init__(
        self,
        retriever: Retriever,
        rerankers: Sequence[Reranker] = (),
        *,
        overfetch: float = 4.0,
        max_candidates: int = 100,
        hooks: HookManager | None = None,
    ) -> None:
        """Configure the pipeline.

        Args:
            retriever: The first stage.
            rerankers: Applied in order to the candidate set.
            overfetch: Multiple of ``k`` to retrieve before reranking. Without
                it, a reranker can only reorder what the retriever already
                ranked highly, which defeats the point.
            max_candidates: Ceiling on the overfetched set.
            hooks: Optional hook manager.
        """
        self._retriever = retriever
        self._rerankers = list(rerankers)
        self._overfetch = max(1.0, overfetch)
        self._max_candidates = max_candidates
        self._hooks = hooks or HookManager()

    @property
    def retriever(self) -> Retriever:
        """The first-stage retriever."""
        return self._retriever

    @property
    def rerankers(self) -> Sequence[Reranker]:
        """The reranking stages."""
        return tuple(self._rerankers)

    def with_reranker(self, reranker: Reranker) -> RetrievalPipeline:
        """Return a copy with ``reranker`` appended."""
        return RetrievalPipeline(
            self._retriever,
            [*self._rerankers, reranker],
            overfetch=self._overfetch,
            max_candidates=self._max_candidates,
            hooks=self._hooks,
        )

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """Run the pipeline and return up to ``k`` hits."""
        if k <= 0:
            return []
        with span("toolbroker.retrieve.pipeline", k=k, stages=len(self._rerankers)):
            self._hooks.emit(Event.BEFORE_RETRIEVAL, query=query, k=k, filters=filters)
            fetch = min(self._max_candidates, max(k, int(k * self._overfetch)))
            hits = self._retriever.retrieve(query, fetch, filters)

            for reranker in self._rerankers:
                if not hits:
                    break
                # Keep the wide set until the final stage: narrowing to k
                # between rerankers throws away candidates later stages might
                # have promoted.
                hits = list(reranker.rerank(query, hits, fetch))

            hits = self._hooks.transform(Event.TRANSFORM_HITS, hits, query=query)
            result = list(hits)[:k]
            self._hooks.emit(Event.AFTER_RETRIEVAL, query=query, hits=result)
            return result

    def __repr__(self) -> str:
        """Show the stages."""
        stages = " -> ".join(
            [type(self._retriever).__name__, *(type(r).__name__ for r in self._rerankers)]
        )
        return f"RetrievalPipeline({stages})"
