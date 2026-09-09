"""Measuring whether the retrieval actually helps.

The central claim of this library is that selecting 5 of 500 tools beats
sending all 500. That claim is only worth making if it is measurable, so the
benchmark is part of the library rather than a script in a blog post — anyone
can run it against their own catalogue and check.

Metrics:

* **recall@k** — did the correct tool make the cut? The one that matters: if
  the right tool is not in the context, the agent cannot pick it.
* **precision@k** — how much of the selection was relevant.
* **MRR** — how highly the first correct tool ranked.
* **context saving** — how many tool schemas were kept out of the prompt.

A case with no expected tools is a *negative* case: the right answer is to
return nothing. Those matter as much as the positives, because a retriever that
always returns its five best guesses looks perfect on positives alone.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .catalog import ToolBroker


class BenchmarkCase(BaseModel):
    """One labelled query."""

    model_config = ConfigDict(extra="forbid")

    query: str
    expected: list[str] = Field(default_factory=list)
    agent: str | None = None
    scopes: list[str] = Field(default_factory=list)
    note: str = ""
    #: What kind of query this is — ``paraphrase``, ``goal``, ``jargon``,
    #: ``distractor``, and so on. An aggregate score hides which *kinds* of
    #: query a retriever fails on, and that is the actionable part.
    category: str = "uncategorised"

    @property
    def is_negative(self) -> bool:
        """Whether the correct answer is an empty selection."""
        return not self.expected


class CaseResult(BaseModel):
    """What happened for one case."""

    model_config = ConfigDict(extra="forbid")

    query: str
    expected: list[str]
    selected: list[str]
    hit: bool
    reciprocal_rank: float
    latency_ms: float
    category: str = "uncategorised"

    @property
    def recall(self) -> float:
        """Fraction of expected tools that were selected."""
        if not self.expected:
            return 1.0 if not self.selected else 0.0
        found = len(set(self.expected) & set(self.selected))
        return found / len(self.expected)

    @property
    def precision(self) -> float:
        """Fraction of selected tools that were expected."""
        if not self.selected:
            return 1.0 if not self.expected else 0.0
        return len(set(self.expected) & set(self.selected)) / len(self.selected)


class CategoryResult(BaseModel):
    """Scores for one category of query."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cases: list[CaseResult]

    @property
    def hit_rate(self) -> float:
        """Fraction of cases in this category that succeeded."""
        return _mean([1.0 if case.hit else 0.0 for case in self.cases])

    @property
    def recall(self) -> float:
        """Mean recall within this category."""
        return _mean([case.recall for case in self.cases])

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank within this category."""
        return _mean([case.reciprocal_rank for case in self.cases])

    @property
    def interval(self) -> tuple[float, float]:
        """95% confidence interval on this category's hit rate."""
        return wilson_interval(sum(1 for case in self.cases if case.hit), len(self.cases))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        low, high = self.interval
        return {
            "cases": len(self.cases),
            "hit_rate": round(self.hit_rate, 4),
            "recall": round(self.recall, 4),
            "mrr": round(self.mrr, 4),
            "ci95": [round(low, 4), round(high, 4)],
        }


class BenchmarkResult(BaseModel):
    """Aggregate outcome across every case."""

    model_config = ConfigDict(extra="forbid")

    k: int
    catalogue_size: int
    cases: list[CaseResult]

    @property
    def recall_at_k(self) -> float:
        """Mean recall across cases."""
        return _mean([case.recall for case in self.cases])

    @property
    def precision_at_k(self) -> float:
        """Mean precision across cases."""
        return _mean([case.precision for case in self.cases])

    @property
    def hit_rate(self) -> float:
        """Fraction of cases where at least one expected tool was selected."""
        return _mean([1.0 if case.hit else 0.0 for case in self.cases])

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank of the first correct tool."""
        return _mean([case.reciprocal_rank for case in self.cases])

    @property
    def p50_latency_ms(self) -> float:
        """Median selection latency."""
        return _percentile([case.latency_ms for case in self.cases], 50)

    @property
    def p95_latency_ms(self) -> float:
        """95th-percentile selection latency."""
        return _percentile([case.latency_ms for case in self.cases], 95)

    @property
    def context_reduction(self) -> float:
        """Fraction of the catalogue kept out of the prompt."""
        if not self.catalogue_size:
            return 0.0
        mean_selected = _mean([float(len(case.selected)) for case in self.cases])
        return 1.0 - (mean_selected / self.catalogue_size)

    @property
    def hit_interval(self) -> tuple[float, float]:
        """95% confidence interval on the hit rate."""
        return wilson_interval(sum(1 for case in self.cases if case.hit), len(self.cases))

    def categories(self) -> dict[str, CategoryResult]:
        """Break the results down by query category.

        An aggregate hides the actionable part. "recall 0.79" says nothing;
        "0.95 on paraphrase, 0.48 on jargon" tells you what to fix.
        """
        grouped: dict[str, list[CaseResult]] = {}
        for case in self.cases:
            grouped.setdefault(case.category, []).append(case)
        return {
            name: CategoryResult(name=name, cases=cases) for name, cases in sorted(grouped.items())
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary plus per-case detail."""
        return {
            "k": self.k,
            "catalogue_size": self.catalogue_size,
            "cases": len(self.cases),
            "recall_at_k": round(self.recall_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "context_reduction": round(self.context_reduction, 4),
            "hit_rate_ci95": [round(bound, 4) for bound in self.hit_interval],
            "by_category": {name: group.to_dict() for name, group in self.categories().items()},
            "failures": [
                {"query": case.query, "expected": case.expected, "selected": case.selected}
                for case in self.cases
                if not case.hit
            ],
        }

    def summary(self) -> str:
        """Render a readable report, including the failures."""
        data = self.to_dict()
        low, high = self.hit_interval
        lines = [
            f"catalogue: {self.catalogue_size} tools, k={self.k}, cases={len(self.cases)}",
            f"recall@{self.k}:    {data['recall_at_k']:.3f}",
            f"precision@{self.k}: {data['precision_at_k']:.3f}",
            f"hit rate:      {data['hit_rate']:.3f}  (95% CI {low:.3f}-{high:.3f})",
            f"MRR:           {data['mrr']:.3f}",
            f"latency:       p50 {data['p50_latency_ms']:.2f}ms / "
            f"p95 {data['p95_latency_ms']:.2f}ms",
            f"context saved: {data['context_reduction'] * 100:.1f}% of tool schemas",
        ]

        groups = self.categories()
        if len(groups) > 1:
            lines.append("")
            lines.append(f"{'category':<14}{'n':>5}{'hit':>8}{'recall':>9}{'MRR':>8}   95% CI")
            lines.append("-" * 62)
            for name, group in sorted(groups.items(), key=lambda item: item[1].hit_rate):
                bottom, top = group.interval
                lines.append(
                    f"{name:<14}{len(group.cases):>5}{group.hit_rate:>8.3f}"
                    f"{group.recall:>9.3f}{group.mrr:>8.3f}   {bottom:.3f}-{top:.3f}"
                )
        failures = data["failures"]
        if failures:
            lines.append("")
            lines.append(f"failures ({len(failures)}):")
            # Printing failures by default is deliberate. A benchmark that only
            # reports its aggregate is a marketing number.
            for failure in failures[:20]:
                lines.append(
                    f"  {failure['query']!r}\n"
                    f"    expected: {failure['expected']}\n"
                    f"    got:      {failure['selected']}"
                )
            if len(failures) > 20:
                lines.append(f"  ... and {len(failures) - 20} more")
        return "\n".join(lines)


def run_benchmark(
    broker: ToolBroker,
    cases: Sequence[BenchmarkCase],
    *,
    k: int = 5,
    agent: str | None = None,
) -> BenchmarkResult:
    """Run every case against ``broker`` and aggregate the results."""
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        selection = broker.select(
            case.query,
            k=k,
            agent=case.agent or agent,
            scopes=case.scopes,
        )
        latency = (time.perf_counter() - started) * 1000
        selected = list(selection.tool_ids)
        expected = set(case.expected)

        rank = next(
            (index for index, tool_id in enumerate(selected, start=1) if tool_id in expected),
            0,
        )
        # A negative case succeeds only by returning nothing. Crediting one that
        # returned five wrong tools was scoring the failure as a perfect answer.
        hit = (not selected) if case.is_negative else rank > 0
        if case.is_negative:
            reciprocal_rank = 1.0 if hit else 0.0
        elif rank:
            reciprocal_rank = 1.0 / rank
        else:
            reciprocal_rank = 0.0

        results.append(
            CaseResult(
                query=case.query,
                expected=case.expected,
                selected=selected,
                hit=hit,
                reciprocal_rank=reciprocal_rank,
                latency_ms=latency,
                category=case.category,
            )
        )

    return BenchmarkResult(k=k, catalogue_size=len(broker), cases=results)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a 95% Wilson score interval for a proportion.

    Wilson rather than the textbook normal approximation because benchmark
    sample sizes are small and proportions land near 0 or 1, where the normal
    approximation produces intervals that run past 0% or 100%.

    This exists so nobody reads a 2-point difference between two runs as an
    improvement. If the intervals overlap, the benchmark cannot tell them apart.
    """
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean, zero for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(percentile / 100 * (len(ordered) - 1)))
    return ordered[index]
