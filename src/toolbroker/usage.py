"""Recording which tools actually get called.

Retrieval sees a description. Usage sees reality: the tool whose description
reads best is not always the one that works, and a catalogue's users find that
out long before its author does.

ToolBroker never executes a tool, so it cannot observe this itself — the caller
reports it:

    selection = broker.select("refund the customer", k=5)
    # ... your framework runs the loop, the model calls one tool ...
    broker.record_use("billing/issue_refund")

Two properties this has to have or it does more harm than good.

**Old popularity has to fade.** Counts decay exponentially with a half-life, so
a tool that was heavily used last quarter and abandoned since stops dominating.
Decay is applied lazily at read time, so there is no cron job to forget.

**The boost has to be bounded.** A popular-but-wrong tool must not displace an
unused-but-right one. The boost is capped at ``weight`` of a hit's own score,
which gives a hard guarantee: two tools more than ``weight`` apart in relative
score cannot swap places, no matter how lopsided their usage. Without that,
usage boosting is a feedback loop that entrenches whatever was popular first
and starves every tool added afterwards.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .observability import get_logger

logger = get_logger("usage")

#: Thirty days. Long enough that a genuinely popular tool stays boosted, short
#: enough that a deprecated one stops mattering within a quarter.
DEFAULT_HALF_LIFE = 30 * 24 * 60 * 60.0


class UsageTracker:
    """A decaying count of how often each tool has been used.

    Thread-safe: a background refresher, a request handler, and a persistence
    hook can all touch it at once.
    """

    def __init__(
        self,
        *,
        half_life: float = DEFAULT_HALF_LIFE,
        counts: Mapping[str, float] | None = None,
        clock: Any = time.time,
    ) -> None:
        """Create a tracker.

        Args:
            half_life: Seconds after which a recorded use counts for half as
                much. Set to ``math.inf`` to disable decay.
            counts: Initial counts, treated as if recorded now.
            clock: Returns the current time in seconds. Injectable so decay is
                testable without sleeping.

        Raises:
            ValueError: If ``half_life`` is not positive.
        """
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self._half_life = half_life
        self._clock = clock
        self._lock = threading.Lock()
        now = clock()
        # value and the moment it was last decayed to, per tool. Storing the
        # timestamp lets decay be applied on read in O(1) rather than needing a
        # scheduled sweep that someone will forget to run.
        self._entries: dict[str, tuple[float, float]] = {
            tool_id: (float(value), now) for tool_id, value in (counts or {}).items()
        }

    @property
    def half_life(self) -> float:
        """Seconds after which a use counts for half as much."""
        return self._half_life

    def _decayed(self, value: float, since: float, now: float) -> float:
        """Return ``value`` decayed from ``since`` to ``now``."""
        if math.isinf(self._half_life):
            return value
        elapsed = max(0.0, now - since)
        return value * math.pow(0.5, elapsed / self._half_life)

    def record(self, tool_id: str, count: float = 1.0) -> None:
        """Record ``count`` uses of ``tool_id``."""
        if count <= 0:
            return
        now = self._clock()
        with self._lock:
            value, since = self._entries.get(tool_id, (0.0, now))
            self._entries[tool_id] = (self._decayed(value, since, now) + count, now)

    def record_many(self, tool_ids: Iterable[str]) -> None:
        """Record one use of each id."""
        for tool_id in tool_ids:
            self.record(tool_id)

    def counts(self) -> dict[str, float]:
        """Return current counts, decayed to now.

        Entries that have decayed to effectively nothing are dropped, so a
        long-lived tracker does not accumulate one row per tool ever seen.
        """
        now = self._clock()
        with self._lock:
            live = {
                tool_id: self._decayed(value, since, now)
                for tool_id, (value, since) in self._entries.items()
            }
            self._entries = {
                tool_id: (value, now) for tool_id, value in live.items() if value >= 1e-6
            }
            return {tool_id: value for tool_id, value in live.items() if value >= 1e-6}

    def count(self, tool_id: str) -> float:
        """Decayed count for one tool."""
        return self.counts().get(tool_id, 0.0)

    def clear(self) -> None:
        """Forget everything."""
        with self._lock:
            self._entries = {}

    def __len__(self) -> int:
        """Number of tools with a non-negligible count."""
        return len(self.counts())

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write counts to ``path`` as JSON.

        Timestamps are stored alongside the values so decay survives a restart —
        a process that was down for a month should come back with month-old
        counts, not fresh ones.
        """
        now = self._clock()
        with self._lock:
            payload = {
                "half_life": self._half_life,
                "saved_at": now,
                "entries": {
                    tool_id: [value, since] for tool_id, (value, since) in self._entries.items()
                },
            }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash mid-write must not leave a truncated file
        # that fails to load on the next boot.
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        half_life: float | None = None,
        clock: Any = time.time,
        missing_ok: bool = True,
    ) -> UsageTracker:
        """Read counts from ``path``.

        Args:
            path: File written by :meth:`save`.
            half_life: Override the stored half-life.
            clock: Time source.
            missing_ok: Return an empty tracker if the file is absent or
                unreadable, rather than raising. On by default: a corrupt usage
                file should degrade ranking, not stop a service from starting.
        """
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            entries = payload["entries"]
            stored_half_life = float(payload.get("half_life", DEFAULT_HALF_LIFE))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if not missing_ok:
                raise
            if source.exists():
                logger.warning(
                    "usage file unreadable; starting from empty",
                    extra={"path": str(source), "error": str(exc)},
                )
            return cls(half_life=half_life or DEFAULT_HALF_LIFE, clock=clock)

        tracker = cls(half_life=half_life or stored_half_life, clock=clock)
        restored: dict[str, tuple[float, float]] = {}
        for tool_id, pair in entries.items():
            try:
                value, since = float(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            restored[str(tool_id)] = (value, since)
        tracker._entries = restored
        return tracker

    def __repr__(self) -> str:
        """Show the size and half-life."""
        days = self._half_life / 86400
        return f"UsageTracker(tools={len(self._entries)}, half_life={days:.1f}d)"
