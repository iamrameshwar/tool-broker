"""Choosing a score floor without labelled data.

By default a retriever always returns its ``k`` best guesses, so a query no
tool can serve still produces confident suggestions and the model uses one. The
fix is a :class:`~toolbroker.policy.rules.MinScore` floor, but the useful value
depends on the embedder's score distribution and the catalogue, so no default
can be shipped.

This module derives one from the catalogue itself. The method is simple and
makes no claim to be clever:

1. Run a set of queries that are certainly irrelevant to any business tool
   catalogue — geography, cooking, poetry.
2. Record the top score each one gets. That is what *noise* looks like against
   this catalogue with this embedder.
3. Put the floor above most of it.

The honest part is what this reveals rather than what it recommends. Noise
scores and genuine scores **overlap**, so no floor separates them cleanly. The
report shows the overlap so the choice is an informed trade-off rather than a
number someone copied from a docs page.

    from toolbroker.calibrate import calibrate_floor

    report = calibrate_floor(broker)
    print(report.summary())
    broker.set_policy(PolicyEngine(selection_rules=[MinScore(report.recommended())]))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .catalog import ToolBroker

#: Queries chosen to be irrelevant to any plausible business tool catalogue.
#: Deliberately mundane: the point is to measure what an embedder scores when
#: nothing matches, not to find adversarial inputs.
DEFAULT_NOISE: tuple[str, ...] = (
    "who won the 1974 world cup",
    "what is the capital of Mongolia",
    "recommend a restaurant in Lisbon",
    "how do I make sourdough starter",
    "name three novels by Ursula K Le Guin",
    "what is the speed of light in a vacuum",
    "explain quantum entanglement to a child",
    "what is the tallest mountain in Africa",
    "who painted the Mona Lisa",
    "teach me to play the ukulele",
    "translate this poem into old norse",
    "what year did the Berlin wall come down",
    "summarise the plot of Moby Dick",
    "convert forty celsius to fahrenheit",
    "how many bones are in the human foot",
    "what is the airspeed velocity of an unladen swallow",
    "compose a limerick about the sea",
    "what time zone is Reykjavik in",
    "tell me about the migration of arctic terns",
    "what is a good name for a golden retriever",
    "describe the rules of cricket",
    "when is the next lunar eclipse",
    "what is the difference between a crow and a raven",
    "how long does it take to boil an egg",
    "who wrote the Epic of Gilgamesh",
)

DEFAULT_PERCENTILES: tuple[int, ...] = (50, 75, 90, 95, 100)


class FloorSuggestion(BaseModel):
    """One candidate floor and what it would cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    percentile: int
    floor: float
    #: Share of noise queries where the floor removes every candidate — the
    #: outcome you want for a query no tool can serve.
    noise_rejected: float
    #: Share of sample queries left with nothing at all. The severe failure.
    samples_emptied: float | None = None
    #: Share of sample queries that lose at least one of their top ``k``.
    #: Always larger than ``samples_emptied``, and the number people forget:
    #: a floor set from top-1 scores looks free while quietly deleting the
    #: third-ranked tool that was the right answer.
    samples_thinned: float | None = None


class CalibrationReport(BaseModel):
    """What the catalogue's score distribution looks like."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    noise_scores: tuple[float, ...]
    sample_scores: tuple[float, ...] = ()
    #: Weakest of the top ``k`` scores for each sample query.
    sample_tail_scores: tuple[float, ...] = ()
    suggestions: tuple[FloorSuggestion, ...] = ()
    embedder: str = ""
    catalogue_size: int = 0
    k: int = 5
    #: Highest noise score, i.e. the best a certainly-irrelevant query managed.
    noise_ceiling: float = 0.0
    #: Lowest sample score, when samples were supplied.
    sample_floor: float | None = None

    @property
    def overlaps(self) -> bool:
        """Whether noise ever outscores a genuine query.

        When this is true — and it usually is — no floor separates the two, and
        the choice is which error you would rather make.
        """
        return self.sample_floor is not None and self.noise_ceiling > self.sample_floor

    def recommended(self, *, percentile: int = 90) -> float:
        """Return the suggested floor at ``percentile`` of the noise scores."""
        for suggestion in self.suggestions:
            if suggestion.percentile == percentile:
                return suggestion.floor
        return _percentile(self.noise_scores, percentile)

    def summary(self) -> str:
        """Render the distributions and the trade-off in readable form."""
        lines = [
            f"catalogue: {self.catalogue_size} tools    embedder: {self.embedder}",
            f"noise queries: {len(self.noise_scores)}"
            + (f"    sample queries: {len(self.sample_scores)}" if self.sample_scores else ""),
            "",
            "top-1 score when nothing should match:",
            f"  min {min(self.noise_scores):.3f}   median "
            f"{_percentile(self.noise_scores, 50):.3f}   max {self.noise_ceiling:.3f}",
        ]
        if self.sample_scores:
            lines.append("")
            lines.append("top-1 score for your sample queries:")
            lines.append(
                f"  min {self.sample_floor:.3f}   median "
                f"{_percentile(self.sample_scores, 50):.3f}   "
                f"max {max(self.sample_scores):.3f}"
            )

        lines.append("")
        header = f"{'floor':>7}{'blocks noise':>14}"
        if self.sample_scores:
            header += f"{'samples emptied':>17}{'samples thinned':>17}"
        lines.append(header)
        lines.append("-" * len(header))
        for suggestion in self.suggestions:
            row = f"{suggestion.floor:>7.3f}{suggestion.noise_rejected:>13.0%}"
            if suggestion.samples_emptied is not None:
                row += f"{suggestion.samples_emptied:>17.0%}"
                row += f"{suggestion.samples_thinned or 0.0:>17.0%}"
            lines.append(row)
        if self.sample_scores:
            lines.append("")
            lines.append(
                f'"emptied" means the query got nothing back; "thinned" means it lost '
                f"at least one of its top {self.k}."
            )
            lines.append(
                "Thinned is the number that gets forgotten: a floor chosen from best "
                "scores alone looks free while deleting lower-ranked tools that were "
                "sometimes the right answer."
            )

        lines.append("")
        if self.overlaps:
            lines.append(
                f"Noise reached {self.noise_ceiling:.3f} and your weakest genuine query "
                f"scored {self.sample_floor:.3f}."
            )
            lines.append(
                "The distributions overlap, so no floor separates them cleanly. Pick "
                "based on which error costs you more:"
            )
            lines.append(
                "  a low floor answers more queries and invents more tools for "
                "questions no tool can serve;"
            )
            lines.append("  a high floor says 'no tool fits' more often, correctly and not.")
        elif self.sample_scores:
            lines.append(
                "The distributions do not overlap on this sample, so a floor between "
                f"{self.noise_ceiling:.3f} and {self.sample_floor:.3f} separates them. "
                "Verify on more queries before trusting that."
            )
        else:
            lines.append(
                "Pass sample queries you expect to succeed to see the overlap; without "
                "them this only says what noise scores, not what it would cost you."
            )
        return "\n".join(lines)


def calibrate_floor(
    broker: ToolBroker,
    *,
    noise: Sequence[str] | None = None,
    samples: Sequence[str] = (),
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
    k: int = 5,
) -> CalibrationReport:
    """Measure what noise scores against ``broker`` and suggest a floor.

    Args:
        broker: An indexed catalogue.
        noise: Queries that certainly match nothing. Defaults to
            :data:`DEFAULT_NOISE`. Replace these if your catalogue happens to
            cover cooking or astronomy.
        samples: Queries you expect to succeed. Optional, but without them the
            report can only say what noise scores — not what a floor would cost.
        percentiles: Which noise percentiles to suggest floors at.
        k: How many tools a selection returns. The floor is judged against the
            whole top ``k``, not only the best hit, because that is what it
            filters.

    Returns:
        A :class:`CalibrationReport`.
    """
    queries = list(noise) if noise is not None else list(DEFAULT_NOISE)
    noise_scores = tuple(_scores(broker, query, k)[0] for query in queries)

    sample_pairs = [_scores(broker, query, k) for query in samples]
    sample_scores = tuple(best for best, _ in sample_pairs)
    sample_tail_scores = tuple(weakest for _, weakest in sample_pairs)

    if not noise_scores:
        raise ValueError("calibration needs at least one noise query")

    suggestions: list[FloorSuggestion] = []
    for percentile in percentiles:
        floor = _percentile(noise_scores, percentile)
        suggestions.append(
            FloorSuggestion(
                percentile=percentile,
                floor=floor,
                noise_rejected=_fraction_below(noise_scores, floor),
                samples_emptied=(_fraction_below(sample_scores, floor) if sample_scores else None),
                samples_thinned=(
                    _fraction_below(sample_tail_scores, floor) if sample_tail_scores else None
                ),
            )
        )

    return CalibrationReport(
        noise_scores=noise_scores,
        sample_scores=sample_scores,
        sample_tail_scores=sample_tail_scores,
        suggestions=tuple(suggestions),
        embedder=broker.embedder.id,
        catalogue_size=len(broker),
        k=k,
        noise_ceiling=max(noise_scores),
        sample_floor=min(sample_scores) if sample_scores else None,
    )


def _scores(broker: ToolBroker, query: str, k: int) -> tuple[float, float]:
    """Return ``(best, weakest)`` score across the top ``k`` hits for ``query``.

    Retrieval is used directly rather than :meth:`ToolBroker.select`, because a
    policy that already caps or filters would distort what is being measured.
    """
    hits = broker.pipeline.retrieve(query, k)
    if not hits:
        return (0.0, 0.0)
    scores = [float(hit.score) for hit in hits]
    return (max(scores), min(scores))


def _percentile(values: Sequence[float], percentile: int) -> float:
    """Nearest-rank percentile of ``values``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(percentile / 100 * (len(ordered) - 1))))
    return ordered[index]


def _fraction_below(values: Sequence[float], floor: float) -> float:
    """Share of ``values`` strictly below ``floor``."""
    if not values:
        return 0.0
    return sum(1 for value in values if value < floor) / len(values)
