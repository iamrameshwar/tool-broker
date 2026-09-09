"""Selecting tools for turn five, not just for turn five's sentence.

``select("check stock there")`` cannot succeed. *There* was named three turns
ago, and a single user message is often the least informative thing in the
conversation — pronouns, ellipsis, and follow-ups like "now cancel it" carry
almost no retrievable signal on their own.

The fix is not subtle: put some recent history in the query. What takes care is
*how much*. Concatenating a whole transcript should drift every query toward the
average of the conversation, so the turn that actually asks for something stops
dominating — that is the reasoning behind bounding it, and it is reasoning
rather than a result: the corpus in ``bench/multiturn.py`` is too shallow to
show the dilution, as the table below says.

So three bounds:

* **Only the last few turns.** ``max_turns`` of history, most recent first.
  Older turns are usually resolved already.
* **Only user turns, by default.** Assistant text is generated, often long, and
  describes what was already done rather than what is wanted next.
* **The current turn is weighted.** It is repeated ``weight`` times so history
  informs the query without outvoting it. This is the same device enrichment
  uses for tool names, for the same reason.

What the benchmark actually settles, and what it does not
---------------------------------------------------------

Including history at all is worth a lot, and the gain grows with the catalogue —
more distractors means more to disambiguate against:

===========  ==========  ============
tools        last only   with history
===========  ==========  ============
50           0.533       0.667
200          0.467       0.633
1000         0.400       0.633
===========  ==========  ============

The two knobs are **not** settled by it, and saying otherwise would be the
mistake this project already withdrew a claim for. Every conversation in
``bench/multiturn.py`` has exactly one prior user turn, so ``max_turns`` above 1
is untestable there — the rows come out identical by construction, not by
finding. It stays at 2 because real conversations are longer than the corpus,
which is a judgement, not a measurement. And ``weight`` moves recall by one or
two queries out of thirty, below the resolution of a set that size.

    context = Conversation(max_turns=2)
    context.add_user("what is in the Berlin warehouse")
    context.add_assistant("Berlin holds 412 SKUs.")
    context.add_user("check stock there")

    broker.select(context.query(), k=5)

``Conversation`` is a convenience, not a requirement: anything that produces a
string works, and :func:`contextual_query` is the same logic over plain tuples
for callers who already hold their own history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

#: Turns of history folded into a query. Not settled by the benchmark, whose
#: conversations are one turn deep; 2 hedges toward real conversations being
#: longer than that. See the module docstring.
DEFAULT_MAX_TURNS = 2

#: How many times the current turn is repeated relative to history. Moves recall
#: by one or two queries out of thirty — below the resolution of that set, so
#: this is a default rather than a finding.
DEFAULT_WEIGHT = 2

#: Longest history turn worth including. Past this it is a document, not
#: context, and it swamps whatever the user just asked for.
MAX_TURN_LENGTH = 400

USER = "user"
ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Turn:
    """One message in a conversation."""

    role: str
    text: str


@dataclass
class Conversation:
    """Accumulates turns and renders a retrieval query from them."""

    max_turns: int = DEFAULT_MAX_TURNS
    weight: int = DEFAULT_WEIGHT
    include_assistant: bool = False
    turns: list[Turn] = field(default_factory=list)

    def add_user(self, text: str) -> Conversation:
        """Record a user turn and return self, so calls chain."""
        self.turns.append(Turn(USER, text))
        return self

    def add_assistant(self, text: str) -> Conversation:
        """Record an assistant turn."""
        self.turns.append(Turn(ASSISTANT, text))
        return self

    def add(self, role: str, text: str) -> Conversation:
        """Record a turn with an arbitrary role."""
        self.turns.append(Turn(role, text))
        return self

    @property
    def latest(self) -> str:
        """The most recent user turn, which is what is actually being asked."""
        for turn in reversed(self.turns):
            if turn.role == USER:
                return turn.text
        return ""

    def query(self) -> str:
        """Render the retrieval query for the current turn."""
        return contextual_query(
            self.turns,
            max_turns=self.max_turns,
            weight=self.weight,
            include_assistant=self.include_assistant,
        )

    def __len__(self) -> int:
        """Number of recorded turns."""
        return len(self.turns)


def contextual_query(
    turns: Sequence[Turn] | Sequence[tuple[str, str]],
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    weight: int = DEFAULT_WEIGHT,
    include_assistant: bool = False,
) -> str:
    """Fold recent history into the current turn.

    Args:
        turns: The conversation, oldest first. Either :class:`Turn` objects or
            ``(role, text)`` pairs.
        max_turns: How many *previous* turns to include.
        weight: How many times to repeat the current turn, so history informs
            the query without outvoting it. ``1`` disables the weighting.
        include_assistant: Whether assistant turns count as history.

    Returns:
        A query string. Empty input gives an empty string rather than raising —
        a caller threading history through should not have to special-case the
        first turn.
    """
    normalised = [
        turn if isinstance(turn, Turn) else Turn(turn[0], turn[1])
        for turn in turns
        if (turn.text if isinstance(turn, Turn) else turn[1]).strip()
    ]
    if not normalised:
        return ""

    current = normalised[-1]
    eligible = [
        turn
        for turn in normalised[:-1]
        if include_assistant or turn.role == USER
        if len(turn.text) <= MAX_TURN_LENGTH
    ]
    # Not `eligible[-max_turns:]`: at zero that slice is `[-0:]`, which is the
    # whole list rather than none of it, so max_turns=0 would silently include
    # everything — the exact opposite of what it asks for.
    history = eligible[-max_turns:] if max_turns > 0 else []

    # History first, current turn last and repeated. Order matters less than
    # the repetition does, but putting the live question last matches how the
    # turn reads and keeps the string legible in a trace.
    parts = [turn.text.strip() for turn in history]
    parts.extend([current.text.strip()] * max(1, weight))
    return "\n".join(parts)
