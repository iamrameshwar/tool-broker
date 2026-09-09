"""Keeping a long-running catalogue current.

A process that indexes once at startup serves a stale catalogue forever. MCP
servers gain and lose tools; an OpenAPI spec gets redeployed. Nothing notices.

:class:`PeriodicRefresher` runs :meth:`ToolBroker.refresh` on an interval. That
is deliberately the dumbest mechanism that works: polling needs no protocol
support, survives a server that never sends notifications, and costs almost
nothing because :meth:`~toolbroker.index.indexer.Indexer.sync` only embeds what
actually changed.

Three properties it has to have, because the alternative is worse than not
refreshing at all:

* **It never dies.** A failing refresh is logged and retried on the next tick.
  A background thread that exits on the first exception leaves a process
  serving stale tools with no indication anything is wrong.
* **It never deletes on failure.** That guarantee lives in ``sync``: a source
  that could not be reached keeps its tools.
* **It stops cleanly.** ``stop()`` interrupts the wait rather than waiting out
  the interval, so shutdown is not gated on the poll period.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING

from .observability import get_logger

if TYPE_CHECKING:
    from .catalog import ToolBroker
    from .index.indexer import RefreshReport

logger = get_logger("refresh")

#: Called after each refresh, successful or not.
Listener = Callable[["RefreshReport"], None]

DEFAULT_INTERVAL = 300.0


class PeriodicRefresher:
    """Refreshes a catalogue on a fixed interval, in a daemon thread."""

    def __init__(
        self,
        broker: ToolBroker,
        *,
        interval: float = DEFAULT_INTERVAL,
        prune: bool = True,
        on_refresh: Listener | None = None,
        refresh_on_start: bool = False,
        name: str = "toolbroker-refresh",
    ) -> None:
        """Configure the refresher.

        Args:
            broker: The catalogue to keep current.
            interval: Seconds between refreshes.
            prune: Whether a refresh removes tools that are genuinely gone.
            on_refresh: Called with each report. Use it to log, emit metrics, or
                alert on ``report.failed_sources``.
            refresh_on_start: Refresh immediately on :meth:`start` rather than
                waiting out the first interval.
            name: Thread name, for debuggers and stack dumps.

        Raises:
            ValueError: If ``interval`` is not positive.
        """
        if interval <= 0:
            raise ValueError("interval must be positive")
        self._broker = broker
        self._interval = interval
        self._prune = prune
        self._listener = on_refresh
        self._refresh_on_start = refresh_on_start
        self._name = name

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.refreshes = 0
        self.failures = 0

    # -- lifecycle --------------------------------------------------------

    @property
    def running(self) -> bool:
        """Whether the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> PeriodicRefresher:
        """Start refreshing in the background. Returns self for chaining."""
        if self.running:
            return self
        self._stop.clear()
        # Daemon: a stalled refresh must never keep the process from exiting.
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()
        return self

    def stop(self, *, timeout: float | None = 5.0) -> None:
        """Stop refreshing and wait briefly for the thread to finish."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def __enter__(self) -> PeriodicRefresher:
        """Start on entry, so ``with PeriodicRefresher(...)`` works."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop on exit."""
        self.stop()

    # -- work -------------------------------------------------------------

    def refresh_once(self) -> RefreshReport | None:
        """Run a single refresh, swallowing and logging any failure.

        Returns ``None`` if the refresh raised. Callers who want the exception
        should use :meth:`ToolBroker.refresh` directly.
        """
        try:
            report = self._broker.refresh(prune=self._prune)
        except Exception:
            self.failures += 1
            logger.exception("refresh failed; will retry on the next interval")
            return None

        self.refreshes += 1
        if report.changed or report.failed_sources:
            logger.info("catalogue refreshed", extra={"summary": report.summary()})
        if self._listener is not None:
            try:
                self._listener(report)
            except Exception:
                # A broken listener must not take down refreshing.
                logger.exception("refresh listener failed")
        return report

    def _loop(self) -> None:
        """Refresh until stopped, waiting interruptibly between passes."""
        if self._refresh_on_start:
            self.refresh_once()
        # Event.wait rather than sleep: stop() interrupts it immediately, so
        # shutdown does not have to wait out the poll period.
        while not self._stop.wait(self._interval):
            self.refresh_once()

    def __repr__(self) -> str:
        """Show the interval and whether it is running."""
        return (
            f"PeriodicRefresher(interval={self._interval}s, "
            f"running={self.running}, refreshes={self.refreshes})"
        )
