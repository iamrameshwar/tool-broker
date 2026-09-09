"""Rerankers: cheap signal first, expensive signal on the shortlist."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from ..determinism import stable_sort
from ..errors import RetrievalError
from ..observability import get_logger
from ..types import Hit
from ..usage import UsageTracker

logger = get_logger("rerank")

LLMScorer = Callable[[str, Sequence[str]], Sequence[float]]


class UsageBooster:
    """Nudges ranking toward tools that actually get called.

    Retrieval sees a description; usage sees reality. The boost is small and
    hard-capped on purpose: at ``weight``, a hit can gain at most that fraction
    of its own score, which guarantees two tools more than ``weight`` apart in
    relative score cannot swap places however lopsided their usage.

    That bound is the whole safety argument. Without it, usage boosting is a
    feedback loop — popular tools rank higher, get called more, rank higher
    still — that entrenches whatever was popular first and starves every tool
    added afterwards.

    Counts come from a :class:`~toolbroker.usage.UsageTracker`, which decays them
    over time so the loop also forgets.
    """

    def __init__(
        self,
        tracker: UsageTracker | None = None,
        *,
        weight: float = 0.1,
    ) -> None:
        """Configure the boost.

        Args:
            tracker: Where counts come from. A fresh one is created if omitted.
            weight: Maximum fraction of a hit's own score the boost may add.
                Keep it small; the point is to break ties between plausible
                candidates, not to override relevance.

        Raises:
            ValueError: If ``weight`` is negative.
        """
        if weight < 0:
            raise ValueError("weight must not be negative")
        self._weight = weight
        self._tracker = tracker if tracker is not None else UsageTracker()

    @property
    def tracker(self) -> UsageTracker:
        """The underlying usage tracker."""
        return self._tracker

    @property
    def weight(self) -> float:
        """Maximum fraction of a hit's score the boost may add."""
        return self._weight

    def record(self, tool_id: str, count: float = 1.0) -> None:
        """Record ``count`` uses of ``tool_id``."""
        self._tracker.record(tool_id, count)

    def snapshot(self) -> dict[str, float]:
        """Return current decayed counts."""
        return self._tracker.counts()

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        """Apply usage boosts and re-sort."""
        del query
        counts = self._tracker.counts()
        if not counts or self._weight == 0:
            return list(hits)[:k]

        # Logarithmic, and normalised against the most-used tool in the
        # catalogue: the difference between 1 and 10 uses should matter far more
        # than the difference between 1000 and 1010.
        peak = math.log1p(max(counts.values()))
        if peak <= 0:
            return list(hits)[:k]

        boosted: list[Hit] = []
        for hit in hits:
            factor = math.log1p(counts.get(hit.id, 0.0)) / peak
            boost = self._weight * factor * abs(hit.score)
            boosted.append(
                Hit(
                    tool=hit.tool,
                    score=hit.score + boost,
                    components={**hit.components, "usage_boost": boost},
                )
            )
        return stable_sort(boosted)[:k]

    def __repr__(self) -> str:
        """Show the weight and how many tools have counts."""
        return f"UsageBooster(weight={self._weight}, tracked={len(self._tracker)})"


class LLMReranker:
    """Scores a shortlist with a caller-supplied model.

    ToolBroker does not ship an LLM client, pick a provider, or read an API key.
    You pass a callable taking ``(query, tool_texts)`` and returning one score
    per text; that keeps the core provider-neutral and your credentials out of
    this library entirely.

    On failure it logs and returns the input order. A reranker outage should
    degrade ranking quality, not break the agent.
    """

    def __init__(
        self,
        scorer: LLMScorer,
        *,
        candidates: int = 20,
        blend: float = 1.0,
        strict: bool = False,
    ) -> None:
        """Configure reranking.

        Args:
            scorer: Callable returning one score per candidate.
            candidates: How many candidates to score. Cost scales with this.
            blend: 1.0 uses the reranker's score alone; 0.5 averages with the
                retriever's.
            strict: Raise on scorer failure instead of degrading gracefully.
        """
        self._scorer = scorer
        self._candidates = candidates
        self._blend = min(max(blend, 0.0), 1.0)
        self._strict = strict

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        """Rescore the top candidates and return the best ``k``."""
        if not hits:
            return []
        shortlist = list(hits)[: self._candidates]
        tail = list(hits)[self._candidates :]
        texts = [self._describe(hit) for hit in shortlist]

        try:
            scores = list(self._scorer(query, texts))
            if len(scores) != len(shortlist):
                raise RetrievalError(
                    f"reranker returned {len(scores)} scores for {len(shortlist)} candidates"
                )
        except Exception as exc:
            if self._strict:
                raise RetrievalError(f"reranker failed: {exc}") from exc
            logger.warning("reranker failed; keeping retrieval order", extra={"error": str(exc)})
            return list(hits)[:k]

        rescored = [
            Hit(
                tool=hit.tool,
                score=self._blend * float(score) + (1.0 - self._blend) * hit.score,
                components={**hit.components, "rerank": float(score)},
            )
            for hit, score in zip(shortlist, scores, strict=True)
        ]
        return (
            stable_sort(rescored)[:k]
            if len(rescored) >= k
            else stable_sort(rescored) + tail[: k - len(rescored)]
        )

    @staticmethod
    def _describe(hit: Hit) -> str:
        """Render a candidate for the scorer."""
        return f"{hit.tool.id}: {hit.tool.description}".strip()
