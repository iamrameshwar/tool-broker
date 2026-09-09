"""The background refresher: it must never die, never block shutdown."""

from __future__ import annotations

import threading
import time

import pytest

from toolbroker import PeriodicRefresher, Tool, ToolBroker
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.sources.base import BaseSource


class MutableSource(BaseSource):
    def __init__(self, source_id: str = "srv", tools=()):
        super().__init__(source_id)
        self.tools = list(tools)
        self.error: Exception | None = None

    def _discover(self):
        if self.error is not None:
            raise self.error
        return list(self.tools)


def tool(name: str) -> Tool:
    return Tool(name=name, namespace="srv", description=f"Does {name}.")


@pytest.fixture
def source():
    return MutableSource(tools=[tool("alpha")])


@pytest.fixture
def broker(source):
    catalogue = ToolBroker(embedder=HashingEmbedder(dim=64), cache_embeddings=False)
    catalogue.add_source(source)
    catalogue.refresh()
    return catalogue


def wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll until ``predicate`` holds. Beats sleeping a fixed duration."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# -- configuration --------------------------------------------------------


def test_interval_must_be_positive(broker):
    with pytest.raises(ValueError, match="interval must be positive"):
        PeriodicRefresher(broker, interval=0)


def test_not_running_before_start(broker):
    assert PeriodicRefresher(broker).running is False


def test_repr_shows_state(broker):
    assert "running=False" in repr(PeriodicRefresher(broker))


# -- manual refresh -------------------------------------------------------


def test_refresh_once_returns_a_report(broker):
    report = PeriodicRefresher(broker).refresh_once()
    assert report is not None
    assert report.unchanged == 1


def test_refresh_once_picks_up_new_tools(broker, source):
    source.tools.append(tool("beta"))
    report = PeriodicRefresher(broker).refresh_once()
    assert report.added == ("srv/beta",)


def test_refresh_once_survives_a_source_failure(broker, source):
    # A source that raises is isolated by refresh(), so this still returns.
    source.error = RuntimeError("down")
    refresher = PeriodicRefresher(broker)
    report = refresher.refresh_once()
    assert report is not None
    assert report.failed_sources


def test_refresh_once_swallows_a_catastrophic_failure(broker, monkeypatch):
    # A background loop that dies on the first exception leaves the process
    # serving stale tools with nothing indicating a problem.
    def boom(**kwargs):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(broker, "refresh", boom)
    refresher = PeriodicRefresher(broker)
    assert refresher.refresh_once() is None
    assert refresher.failures == 1


def test_listener_receives_the_report(broker):
    seen = []
    PeriodicRefresher(broker, on_refresh=seen.append).refresh_once()
    assert len(seen) == 1


def test_a_broken_listener_does_not_break_refreshing(broker):
    def broken(report):
        raise RuntimeError("listener is buggy")

    refresher = PeriodicRefresher(broker, on_refresh=broken)
    assert refresher.refresh_once() is not None
    assert refresher.refreshes == 1


# -- background loop ------------------------------------------------------


def test_start_refreshes_immediately_when_asked(broker):
    refresher = PeriodicRefresher(broker, interval=60, refresh_on_start=True)
    try:
        refresher.start()
        assert wait_for(lambda: refresher.refreshes >= 1)
    finally:
        refresher.stop()


def test_background_loop_picks_up_changes(broker, source):
    refresher = PeriodicRefresher(broker, interval=0.02)
    try:
        refresher.start()
        source.tools.append(tool("beta"))
        assert wait_for(lambda: broker.get("srv/beta") is not None)
    finally:
        refresher.stop()


def test_loop_keeps_running_after_a_failure(broker, source):
    refresher = PeriodicRefresher(broker, interval=0.02)
    try:
        refresher.start()
        source.error = RuntimeError("down")
        assert wait_for(lambda: refresher.refreshes >= 2)
        source.error = None
        source.tools.append(tool("beta"))
        assert wait_for(lambda: broker.get("srv/beta") is not None)
    finally:
        refresher.stop()


def test_stop_does_not_wait_out_the_interval(broker):
    # Shutdown must not be gated on the poll period.
    refresher = PeriodicRefresher(broker, interval=30)
    refresher.start()
    started = time.monotonic()
    refresher.stop()
    assert time.monotonic() - started < 2.0
    assert refresher.running is False


def test_start_is_idempotent(broker):
    refresher = PeriodicRefresher(broker, interval=30)
    try:
        refresher.start()
        thread_count = threading.active_count()
        refresher.start()
        assert threading.active_count() == thread_count
    finally:
        refresher.stop()


def test_stop_is_safe_when_never_started(broker):
    PeriodicRefresher(broker).stop()


def test_context_manager_starts_and_stops(broker):
    with PeriodicRefresher(broker, interval=0.02, refresh_on_start=True) as refresher:
        assert wait_for(lambda: refresher.refreshes >= 1)
        assert refresher.running
    assert refresher.running is False


def test_the_thread_is_a_daemon(broker):
    # A stalled refresh must never keep the process from exiting.
    refresher = PeriodicRefresher(broker, interval=30)
    try:
        refresher.start()
        assert refresher._thread.daemon is True
    finally:
        refresher.stop()


def test_selection_works_while_refreshing(broker, source):
    # Reads must stay correct while the background thread rewrites the store.
    # The loop yields: a tight CPU-bound loop starves the refresher of the GIL,
    # which would make this a test of Python's scheduler rather than of us.
    refresher = PeriodicRefresher(broker, interval=0.005, refresh_on_start=True)
    results = []
    try:
        refresher.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and refresher.refreshes < 3:
            results.append(broker.select("does alpha", k=1).tool_ids)
            time.sleep(0.001)
        assert refresher.refreshes >= 3, "the refresher did not run"
        assert results, "no selections ran"
        assert all(ids == ("srv/alpha",) for ids in results)
    finally:
        refresher.stop()


def test_concurrent_selection_never_sees_a_torn_catalogue(broker, source):
    source.tools = [tool(f"tool_{i:02d}") for i in range(20)]
    broker.refresh()
    refresher = PeriodicRefresher(broker, interval=0.005)
    errors: list[Exception] = []

    def reader():
        try:
            for _ in range(200):
                assert len(broker.select("does tool", k=5)) <= 5
        except Exception as exc:  # pragma: no cover - only on a real bug
            errors.append(exc)

    try:
        refresher.start()
        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert not errors
    finally:
        refresher.stop()
