"""Learning how your users actually ask for a tool.

The benchmark's most stubborn failure is not a ranking problem. ``bounce the
pods`` never reaches ``restart_service`` at *any* depth, because nothing about
"Restart a running service." is near that sentence in embedding space. No
reranker, no fusion and no larger ``k`` recovers it; on the reference catalogue
that class of miss is 8.3% of queries, larger than everything a retrieval tweak
could win.

The catalogue owner could fix it by writing better descriptions. Most never
will. But the broker sees something they cannot: the query a user actually
typed, and — once the framework reports it — the tool that actually got called.
That pair *is* the missing alias, and it costs nothing to collect.

    aliases = AliasLearner()
    broker.hooks.register(Event.TRANSFORM_INDEX_TEXT, aliases.enrich)

    selection = broker.select("bounce the pods", k=5)
    # ... the framework runs the loop and something gets called ...
    aliases.record("bounce the pods", "infrastructure/restart_service")

Aliases take effect at the next re-index, because they change what is embedded.

The trap, and the three bounds against it
-----------------------------------------

The naive version is actively harmful. Learn from a *wrong* selection and you
teach the index to make the same mistake faster, forever. So:

* **Only confirmed calls are learned.** :meth:`record` is given the tool the
  framework actually invoked, not the tool that ranked first. A wrong guess the
  user ignored teaches nothing.
* **A single occurrence changes nothing.** An alias must be seen
  ``min_count`` times before it enters the index, so one accidental call cannot
  rewrite retrieval.
* **The text is capped.** At most ``max_aliases`` phrases per tool, chosen by
  weight. A tool cannot win by accumulating an ever-growing blob of text — the
  same bound, for the same reason, as the cap on usage boosting.

Counts decay on a half-life, so vocabulary that falls out of use fades instead
of anchoring the index to how people spoke a year ago.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .observability import get_logger

if TYPE_CHECKING:
    from .types import Tool

logger = get_logger("aliases")

#: Thirty days. Long enough that seasonal vocabulary survives a quiet month,
#: short enough that a renamed internal system stops haunting the index.
DEFAULT_HALF_LIFE = 30 * 24 * 3600.0

#: How many times a phrase must be seen before it is indexed.
DEFAULT_MIN_COUNT = 3.0

#: How many phrases a single tool may contribute.
DEFAULT_MAX_ALIASES = 5

#: Longest phrase worth keeping. Past this it is a sentence, not an alias, and
#: embedding it dilutes the tool's own description.
MAX_PHRASE_LENGTH = 120

#: Tolerance on the ``min_count`` comparison. Decay is applied on read, so the
#: microseconds between recording the third observation and asking about it
#: leave the weight at 0.999999999999 rather than 3.0 — and a threshold of 3
#: would then never fire. Worse, whether it fired would depend on clock
#: granularity, so the bug would present as flakiness rather than as a rule
#: that plainly does not work.
_THRESHOLD_TOLERANCE = 1e-6


def normalise(query: str) -> str:
    """Reduce a query to the form two spellings of it should share."""
    return " ".join(query.strip().lower().split())


class AliasLearner:
    """Records query-to-tool pairs and feeds the confident ones into the index."""

    def __init__(
        self,
        *,
        half_life: float = DEFAULT_HALF_LIFE,
        min_count: float = DEFAULT_MIN_COUNT,
        max_aliases: int = DEFAULT_MAX_ALIASES,
        counts: Mapping[str, Mapping[str, float]] | None = None,
        clock: Any = time.time,
    ) -> None:
        """Configure the learner.

        Args:
            half_life: Seconds after which an observation counts for half as
                much. ``math.inf`` disables decay.
            min_count: Decayed weight a phrase needs before it is indexed. The
                bound that stops one stray call rewriting retrieval.
            max_aliases: Most phrases any single tool may contribute.
            counts: Initial ``{tool_id: {phrase: weight}}``, treated as if
                recorded now.
            clock: Current time in seconds. Injectable so decay is testable
                without sleeping.

        Raises:
            ValueError: If ``half_life`` is not positive.
        """
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self._half_life = half_life
        self._min_count = min_count
        self._max_aliases = max(0, max_aliases)
        self._clock = clock
        self._lock = threading.Lock()
        now = float(clock())
        self._counts: dict[str, dict[str, tuple[float, float]]] = {
            tool_id: {phrase: (weight, now) for phrase, weight in phrases.items()}
            for tool_id, phrases in (counts or {}).items()
        }

    def record(self, query: str, tool_id: str, weight: float = 1.0) -> None:
        """Record that ``query`` led to ``tool_id`` actually being called.

        Give this the tool the framework *invoked*, never the one that ranked
        first — learning from an unconfirmed guess is how the index teaches
        itself to be wrong faster.
        """
        phrase = normalise(query)
        if not phrase or len(phrase) > MAX_PHRASE_LENGTH:
            return
        now = float(self._clock())
        with self._lock:
            phrases = self._counts.setdefault(tool_id, {})
            current, stamp = phrases.get(phrase, (0.0, now))
            phrases[phrase] = (self._decay(current, stamp, now) + weight, now)

    def record_many(self, query: str, tool_ids: Iterable[str]) -> None:
        """Record one query against several confirmed calls."""
        for tool_id in tool_ids:
            self.record(query, tool_id)

    def aliases_for(self, tool_id: str) -> tuple[str, ...]:
        """Return the phrases confident enough to index, strongest first."""
        now = float(self._clock())
        with self._lock:
            phrases = self._counts.get(tool_id, {})
            scored = [
                (self._decay(weight, stamp, now), phrase)
                for phrase, (weight, stamp) in phrases.items()
            ]
        floor = self._min_count - _THRESHOLD_TOLERANCE
        qualifying = [(score, phrase) for score, phrase in scored if score >= floor]
        qualifying.sort(key=lambda pair: (-pair[0], pair[1]))
        return tuple(phrase for _, phrase in qualifying[: self._max_aliases])

    def weight_of(self, tool_id: str, query: str) -> float:
        """Return the current decayed weight of one phrase."""
        now = float(self._clock())
        with self._lock:
            entry = self._counts.get(tool_id, {}).get(normalise(query))
        return self._decay(entry[0], entry[1], now) if entry else 0.0

    def enrich(self, text: str, tool: Tool | None = None, **_: Any) -> str:
        """Hook handler: append learned aliases to a tool's index text.

        Register on :attr:`~toolbroker.hooks.Event.TRANSFORM_INDEX_TEXT`. The
        aliases are appended rather than woven in, so the tool's own
        description keeps its weight and the addition is visibly bounded.
        """
        if tool is None:
            return text
        aliases = self.aliases_for(tool.id)
        if not aliases:
            return text
        return text + "\n" + "\n".join(aliases)

    def known_tools(self) -> tuple[str, ...]:
        """Tools with at least one recorded phrase."""
        with self._lock:
            return tuple(sorted(self._counts))

    def forget(self, tool_id: str, query: str | None = None) -> bool:
        """Drop one phrase, or every phrase for a tool.

        The escape hatch that makes this safe to switch on: a bad alias can be
        removed without discarding everything else that was learned.
        """
        with self._lock:
            if query is None:
                return self._counts.pop(tool_id, None) is not None
            phrases = self._counts.get(tool_id)
            if not phrases:
                return False
            return phrases.pop(normalise(query), None) is not None

    def _decay(self, weight: float, stamp: float, now: float) -> float:
        """Apply decay lazily, so there is no sweep to schedule and forget."""
        if math.isinf(self._half_life):
            return weight
        elapsed = max(0.0, now - stamp)
        return weight * math.pow(0.5, elapsed / self._half_life)

    # -- persistence ------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return decayed weights, suitable for saving."""
        now = float(self._clock())
        with self._lock:
            return {
                tool_id: {
                    phrase: self._decay(weight, stamp, now)
                    for phrase, (weight, stamp) in phrases.items()
                }
                for tool_id, phrases in self._counts.items()
            }

    def save(self, path: str | Path) -> Path:
        """Persist learned aliases.

        Timestamps are written too, so a process down for a month comes back
        with month-old weights rather than fresh ones.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "half_life": self._half_life,
                "saved_at": float(self._clock()),
                "counts": {
                    tool_id: {phrase: list(entry) for phrase, entry in phrases.items()}
                    for tool_id, phrases in self._counts.items()
                },
            }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> AliasLearner:
        """Restore a learner, tolerating a missing or corrupt file."""
        source = Path(path)
        learner = cls(**kwargs)
        if not source.exists():
            return learner
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            raw = payload["counts"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("could not read aliases from %s: %s", source, exc)
            return learner
        learner._counts = {
            str(tool_id): {
                str(phrase): (float(entry[0]), float(entry[1])) for phrase, entry in phrases.items()
            }
            for tool_id, phrases in raw.items()
        }
        return learner
