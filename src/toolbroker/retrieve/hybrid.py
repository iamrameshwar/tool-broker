"""Fusing dense and lexical retrieval.

Two fusion modes:

* **RRF** (default) combines rank positions. It needs no score calibration,
  which matters because BM25 scores are unbounded and cosine scores are not.
  Comparing them numerically is meaningless; comparing their ranks is not.
* **Weighted** normalizes each retriever's scores to ``[0, 1]`` and blends
  them. Available for callers who have calibrated their own scores and want
  explicit control.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ..determinism import stable_sort
from ..protocols import Retriever
from ..types import Filters, Hit

FusionMode = Literal["rrf", "weighted"]
RRF_K = 60.0


class HybridRetriever:
    """Runs several retrievers and fuses their rankings."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Sequence[float] | None = None,
        mode: FusionMode = "rrf",
        rrf_k: float = RRF_K,
        overfetch: float = 3.0,
        names: Sequence[str] | None = None,
    ) -> None:
        """Configure fusion.

        Args:
            retrievers: The retrievers to fuse. Order defines component names.
            weights: Per-retriever weights; defaults to equal.
            mode: ``"rrf"`` or ``"weighted"``.
            rrf_k: RRF damping constant. Higher flattens rank influence.
            overfetch: Fetch this multiple of ``k`` from each retriever, so a
                tool ranked well by only one of them can still surface.
            names: Component labels for score explanations.
        """
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._retrievers = list(retrievers)
        self._weights = list(weights) if weights else [1.0] * len(self._retrievers)
        if len(self._weights) != len(self._retrievers):
            raise ValueError("weights must have the same length as retrievers")
        self._mode = mode
        self._rrf_k = rrf_k
        self._overfetch = max(1.0, overfetch)
        self._names = (
            list(names)
            if names
            else [
                type(retriever).__name__.replace("Retriever", "").lower()
                for retriever in self._retrievers
            ]
        )

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return up to ``k`` fused results."""
        if k <= 0:
            return []
        fetch = max(k, int(k * self._overfetch))
        runs = [retriever.retrieve(query, fetch, filters) for retriever in self._retrievers]
        fused = self._fuse_rrf(runs) if self._mode == "rrf" else self._fuse_weighted(runs)
        return stable_sort(fused)[:k]

    def _fuse_rrf(self, runs: Sequence[Sequence[Hit]]) -> list[Hit]:
        scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        tools: dict[str, Hit] = {}
        for run_index, run in enumerate(runs):
            name = self._names[run_index]
            weight = self._weights[run_index]
            for rank, hit in enumerate(run, start=1):
                contribution = weight / (self._rrf_k + rank)
                scores[hit.id] = scores.get(hit.id, 0.0) + contribution
                components.setdefault(hit.id, {})[f"{name}_rrf"] = contribution
                components[hit.id][f"{name}_rank"] = float(rank)
                tools.setdefault(hit.id, hit)
        return [
            Hit(tool=tools[tool_id].tool, score=score, components=components[tool_id])
            for tool_id, score in scores.items()
        ]

    def _fuse_weighted(self, runs: Sequence[Sequence[Hit]]) -> list[Hit]:
        scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        tools: dict[str, Hit] = {}
        for run_index, run in enumerate(runs):
            name = self._names[run_index]
            weight = self._weights[run_index]
            if not run:
                continue
            top = max(hit.score for hit in run)
            bottom = min(hit.score for hit in run)
            spread = top - bottom
            for hit in run:
                normalized = 1.0 if spread == 0.0 else (hit.score - bottom) / spread
                contribution = weight * normalized
                scores[hit.id] = scores.get(hit.id, 0.0) + contribution
                components.setdefault(hit.id, {})[name] = normalized
                tools.setdefault(hit.id, hit)
        total = sum(self._weights) or 1.0
        return [
            Hit(
                tool=tools[tool_id].tool,
                score=score / total,
                components=components[tool_id],
            )
            for tool_id, score in scores.items()
        ]

    def __repr__(self) -> str:
        """Show the fusion mode and members."""
        return f"HybridRetriever(mode={self._mode!r}, retrievers={self._names})"
