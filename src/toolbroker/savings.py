"""What retrieval is actually saving you, on your own traffic.

The claim this library rests on is that sending five tools beats sending five
hundred, and the benchmark measures that on a synthetic catalogue. Nobody
approving a budget cares about a synthetic catalogue. They care what happened
last month on their traffic, and the only component that can answer is the one
that sees both numbers on every request: how many tools exist, and how many
were sent.

    tally = SavingsTally(price_per_million=0.15)
    broker.on_selection(tally.record)          # or a hook
    ...
    print(tally.summary())

Two things this is careful about, because a savings number nobody trusts is
worse than no number.

**The counterfactual is stated, not assumed.** "Saved" means *measured against
sending the whole catalogue on every turn*, which is what the alternative
actually is for an agent with one tool list. It is not a comparison against
some cleverer baseline, and the report says so.

**Token counts are estimates unless you supply real ones.** The default is the
usual four-characters-per-token approximation over the rendered JSON. Pass
``actual_tokens`` from your provider's usage field and the tally uses it. The
report always says which it is, because a cost figure that silently mixes the
two is not auditable.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .types import Selection, Tool

#: Characters per token, the usual English approximation. Only used when the
#: caller does not supply a real count.
CHARS_PER_TOKEN = 4


def estimate_tokens(tools: Sequence[Tool] | Sequence[Any]) -> int:
    """Approximate the prompt cost of rendering ``tools`` into a tool block.

    Deliberately crude and deliberately consistent: the value of this number is
    the *ratio* between two conditions measured the same way, not its absolute
    accuracy. A real tokeniser would make the ratio no more truthful.
    """
    payload = [
        tool.model_dump(mode="json") if hasattr(tool, "model_dump") else tool for tool in tools
    ]
    return len(json.dumps(payload, default=str)) // CHARS_PER_TOKEN


class SavingsReport(BaseModel):
    """What was sent, against what would have been sent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selections: int = 0
    tools_sent: int = 0
    tools_in_catalogue: int = 0
    tokens_sent: int = 0
    tokens_if_full: int = 0
    #: Whether every recorded figure came from a provider rather than the
    #: estimator. Mixed sources count as estimated, because the weaker number
    #: is what the total is worth.
    measured: bool = False
    price_per_million: float | None = None

    @property
    def tokens_saved(self) -> int:
        """Tokens not sent, versus the whole catalogue every turn."""
        return max(0, self.tokens_if_full - self.tokens_sent)

    @property
    def ratio(self) -> float:
        """How many times larger the full catalogue would have been."""
        return self.tokens_if_full / self.tokens_sent if self.tokens_sent else 0.0

    @property
    def cost_saved(self) -> float | None:
        """Money not spent, when a price was configured."""
        if self.price_per_million is None:
            return None
        return self.tokens_saved / 1_000_000 * self.price_per_million

    def summary(self) -> str:
        """Render the figure and the caveat that makes it honest."""
        if not self.selections:
            return "No selections recorded yet."

        basis = "measured" if self.measured else "estimated"
        lines = [
            f"{self.selections:,} selections",
            f"  tools sent      {self.tools_sent:,} of a possible {self.tools_in_catalogue:,}",
            f"  tokens sent     {self.tokens_sent:,} ({basis})",
            f"  full catalogue  {self.tokens_if_full:,} ({basis})",
            f"  not sent        {self.tokens_saved:,}  ({self.ratio:.0f}x smaller)",
        ]
        saved = self.cost_saved
        if saved is not None:
            lines.append(f"  cost avoided    {saved:,.2f}")
        lines.append("")
        lines.append(
            'Counterfactual: "full catalogue" is every tool sent on every turn,'
            "\nwhich is what an agent with one tool list actually does."
        )
        if not self.measured:
            lines.append(
                f"Token counts are estimated at {CHARS_PER_TOKEN} characters per token."
                "\nPass actual_tokens from your provider's usage to make them measured."
            )
        return "\n".join(lines)


class SavingsTally:
    """Accumulates what each selection sent, against what it could have.

    Thread-safe, because a broker is usually shared across request handlers and
    a torn counter would make the whole number unusable.
    """

    def __init__(self, *, price_per_million: float | None = None) -> None:
        """Configure the tally.

        Args:
            price_per_million: Input-token price for your model, so the report
                can carry a currency figure. Left out, only tokens are shown.
        """
        self._lock = threading.Lock()
        self._price = price_per_million
        self._selections = 0
        self._tools_sent = 0
        self._catalogue_total = 0
        self._tokens_sent = 0
        self._tokens_full = 0
        self._all_measured = True

    def record(
        self,
        selection: Selection,
        catalogue: Sequence[Tool],
        *,
        actual_tokens: int | None = None,
    ) -> None:
        """Record one selection against the catalogue it was drawn from.

        Args:
            selection: What went to the model.
            catalogue: Every tool that existed at the time — the counterfactual.
            actual_tokens: Real prompt tokens for the tool block, if your
                provider reported them. Without it the estimate is used and the
                report is marked estimated.
        """
        sent = estimate_tokens(selection.tools) if actual_tokens is None else actual_tokens
        # The full-catalogue figure is always estimated: nobody sent it, so no
        # provider ever counted it. Scale it from the same estimator so the
        # ratio compares like with like.
        full = estimate_tokens(catalogue)
        if actual_tokens is not None:
            estimated_sent = estimate_tokens(selection.tools)
            if estimated_sent:
                full = int(full * (actual_tokens / estimated_sent))

        with self._lock:
            self._selections += 1
            self._tools_sent += len(selection.hits)
            self._catalogue_total += len(catalogue)
            self._tokens_sent += sent
            self._tokens_full += full
            if actual_tokens is None:
                self._all_measured = False

    def report(self) -> SavingsReport:
        """Return the tally so far."""
        with self._lock:
            return SavingsReport(
                selections=self._selections,
                tools_sent=self._tools_sent,
                tools_in_catalogue=self._catalogue_total,
                tokens_sent=self._tokens_sent,
                tokens_if_full=self._tokens_full,
                measured=self._all_measured and self._selections > 0,
                price_per_million=self._price,
            )

    def summary(self) -> str:
        """Render the current report."""
        return self.report().summary()

    def reset(self) -> None:
        """Clear the tally, for a fresh reporting period."""
        with self._lock:
            self._selections = 0
            self._tools_sent = 0
            self._catalogue_total = 0
            self._tokens_sent = 0
            self._tokens_full = 0
            self._all_measured = True
