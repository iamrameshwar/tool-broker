"""Usage tracking: decay, persistence, and the bound that stops a feedback loop."""

from __future__ import annotations

import json
import math

import pytest

from toolbroker import Hit, Tool, ToolBroker, UsageTracker
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.retrieve.rerank import UsageBooster
from toolbroker.usage import DEFAULT_HALF_LIFE

DAY = 86400.0


class FakeClock:
    """A clock the test advances by hand, so decay needs no sleeping."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# -- recording ------------------------------------------------------------


def test_counts_start_empty():
    assert UsageTracker().counts() == {}


def test_recording_accumulates():
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("a/b")
    tracker.record("a/b", 2)
    assert tracker.count("a/b") == 3.0


def test_record_many():
    tracker = UsageTracker(clock=FakeClock())
    tracker.record_many(["a/b", "a/c", "a/b"])
    assert tracker.count("a/b") == 2.0


def test_non_positive_counts_are_ignored():
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("a/b", 0)
    tracker.record("a/b", -5)
    assert tracker.counts() == {}


def test_rejects_non_positive_half_life():
    with pytest.raises(ValueError, match="half_life must be positive"):
        UsageTracker(half_life=0)


def test_initial_counts_are_accepted():
    assert UsageTracker(counts={"a/b": 5.0}, clock=FakeClock()).count("a/b") == 5.0


def test_clear():
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("a/b")
    tracker.clear()
    assert len(tracker) == 0


# -- decay ----------------------------------------------------------------


def test_one_half_life_halves_the_count():
    clock = FakeClock()
    tracker = UsageTracker(half_life=30 * DAY, clock=clock)
    tracker.record("a/b", 10)
    clock.advance(30 * DAY)
    assert tracker.count("a/b") == pytest.approx(5.0)


def test_two_half_lives_quarter_it():
    clock = FakeClock()
    tracker = UsageTracker(half_life=DAY, clock=clock)
    tracker.record("a/b", 8)
    clock.advance(2 * DAY)
    assert tracker.count("a/b") == pytest.approx(2.0)


def test_recording_after_decay_adds_to_the_decayed_value():
    clock = FakeClock()
    tracker = UsageTracker(half_life=DAY, clock=clock)
    tracker.record("a/b", 4)
    clock.advance(DAY)
    tracker.record("a/b", 1)
    assert tracker.count("a/b") == pytest.approx(3.0)


def test_recent_use_outweighs_old_use():
    # The point of decay: a tool abandoned last quarter should stop dominating.
    clock = FakeClock()
    tracker = UsageTracker(half_life=30 * DAY, clock=clock)
    tracker.record("old/tool", 100)
    clock.advance(365 * DAY)
    tracker.record("new/tool", 5)
    counts = tracker.counts()
    assert counts["new/tool"] > counts["old/tool"]


def test_infinite_half_life_disables_decay():
    clock = FakeClock()
    tracker = UsageTracker(half_life=math.inf, clock=clock)
    tracker.record("a/b", 3)
    clock.advance(1000 * DAY)
    assert tracker.count("a/b") == 3.0


def test_fully_decayed_entries_are_dropped():
    # A long-lived tracker must not keep one row per tool ever seen.
    clock = FakeClock()
    tracker = UsageTracker(half_life=DAY, clock=clock)
    tracker.record("a/b")
    clock.advance(100 * DAY)
    assert tracker.counts() == {}
    assert len(tracker) == 0


def test_default_half_life_is_thirty_days():
    assert pytest.approx(30 * DAY) == DEFAULT_HALF_LIFE


# -- persistence ----------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    clock = FakeClock()
    tracker = UsageTracker(half_life=DAY, clock=clock)
    tracker.record("a/b", 4)
    path = tmp_path / "usage.json"
    tracker.save(path)

    restored = UsageTracker.load(path, clock=clock)
    assert restored.count("a/b") == pytest.approx(4.0)
    assert restored.half_life == DAY


def test_decay_survives_a_restart(tmp_path):
    # A process down for a month should come back with month-old counts, not
    # fresh ones.
    clock = FakeClock()
    tracker = UsageTracker(half_life=DAY, clock=clock)
    tracker.record("a/b", 8)
    path = tmp_path / "usage.json"
    tracker.save(path)

    clock.advance(3 * DAY)
    restored = UsageTracker.load(path, clock=clock)
    assert restored.count("a/b") == pytest.approx(1.0)


def test_save_creates_parent_directories(tmp_path):
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("a/b")
    tracker.save(tmp_path / "nested" / "deep" / "usage.json")
    assert (tmp_path / "nested" / "deep" / "usage.json").exists()


def test_save_leaves_no_temp_file(tmp_path):
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("a/b")
    tracker.save(tmp_path / "usage.json")
    assert [p.name for p in tmp_path.iterdir()] == ["usage.json"]


def test_missing_file_gives_an_empty_tracker(tmp_path):
    assert len(UsageTracker.load(tmp_path / "absent.json")) == 0


def test_corrupt_file_degrades_instead_of_crashing(tmp_path):
    # A corrupt usage file should cost ranking quality, not stop a service
    # from starting.
    path = tmp_path / "usage.json"
    path.write_text("{not json")
    assert len(UsageTracker.load(path)) == 0


def test_corrupt_file_can_be_made_fatal(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        UsageTracker.load(path, missing_ok=False)


def test_malformed_entries_are_skipped(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({"entries": {"good": [1.0, 0.0], "bad": "nonsense"}}))
    restored = UsageTracker.load(path, clock=FakeClock(now=0.0), half_life=math.inf)
    assert set(restored.counts()) == {"good"}


def test_half_life_can_be_overridden_on_load(tmp_path):
    tracker = UsageTracker(half_life=DAY, clock=FakeClock())
    tracker.save(tmp_path / "usage.json")
    assert UsageTracker.load(tmp_path / "usage.json", half_life=99.0).half_life == 99.0


# -- the bound ------------------------------------------------------------


def hit(name: str, score: float) -> Hit:
    return Hit(tool=Tool(name=name, namespace="t"), score=score)


def test_boosting_promotes_a_used_tool_over_a_close_rival():
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("t/b", 100)
    booster = UsageBooster(tracker, weight=0.5)
    order = [h.id for h in booster.rerank("q", [hit("a", 0.50), hit("b", 0.49)], 2)]
    assert order[0] == "t/b"


def test_boost_never_exceeds_the_weight():
    # The safety guarantee: two tools more than `weight` apart in relative
    # score cannot swap, however lopsided their usage.
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("t/popular", 1_000_000)
    booster = UsageBooster(tracker, weight=0.1)
    hits = [hit("relevant", 1.0), hit("popular", 0.85)]
    order = [h.id for h in booster.rerank("q", hits, 2)]
    assert order[0] == "t/relevant"


def test_the_boost_is_recorded_in_the_components():
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("t/a", 10)
    result = UsageBooster(tracker, weight=0.2).rerank("q", [hit("a", 0.5)], 1)
    assert result[0].components["usage_boost"] > 0


def test_no_usage_means_no_change():
    hits = [hit("a", 0.5), hit("b", 0.4)]
    assert UsageBooster(UsageTracker(clock=FakeClock())).rerank("q", hits, 2) == hits


def test_zero_weight_disables_boosting():
    tracker = UsageTracker(clock=FakeClock())
    tracker.record("t/b", 100)
    hits = [hit("a", 0.5), hit("b", 0.4)]
    assert UsageBooster(tracker, weight=0.0).rerank("q", hits, 2) == hits


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        UsageBooster(weight=-0.1)


def test_booster_creates_its_own_tracker_if_not_given():
    booster = UsageBooster()
    booster.record("t/a")
    # approx, not equality: this uses the real clock, so the microseconds
    # between recording and reading decay the count very slightly.
    assert booster.snapshot()["t/a"] == pytest.approx(1.0)


# -- catalogue integration ------------------------------------------------


def tools() -> list[Tool]:
    return [
        Tool(name="search_orders", namespace="shop", description="Find customer orders"),
        Tool(name="search_invoices", namespace="shop", description="Find customer invoices"),
    ]


def test_usage_boosting_is_off_by_default():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index(tools())
    assert broker.usage is None
    # Instrumenting call sites must be safe before deciding to turn it on.
    broker.record_use("shop/search_orders")


def test_record_use_reaches_the_tracker():
    tracker = UsageTracker(clock=FakeClock())
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False, usage=tracker)
    broker.index(tools())
    broker.record_use("shop/search_orders", 3)
    assert tracker.count("shop/search_orders") == 3.0


def test_record_uses_records_each():
    tracker = UsageTracker(clock=FakeClock())
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False, usage=tracker)
    broker.index(tools())
    broker.record_uses(["shop/search_orders", "shop/search_invoices"])
    assert len(tracker) == 2


def test_usage_changes_the_ranking():
    tracker = UsageTracker(clock=FakeClock())
    broker = ToolBroker(
        embedder=HashingEmbedder(dim=128),
        cache_embeddings=False,
        usage=tracker,
        usage_weight=0.9,
    )
    broker.index(tools())
    query = "find something for a customer"
    before = broker.select(query, k=2).tool_ids

    tracker.record(before[1], 500)
    after = broker.select(query, k=2).tool_ids
    assert after[0] == before[1]


def test_the_boost_shows_up_in_the_explanation():
    tracker = UsageTracker(clock=FakeClock())
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False, usage=tracker)
    broker.index(tools())
    broker.record_use("shop/search_orders", 10)
    selection = broker.select("find customer orders", k=2)
    assert any("usage_boost" in h.components for h in selection.hits)


def test_swapping_the_retriever_keeps_boosting_attached():
    # Assigning a pipeline directly used to drop the booster silently, and the
    # symptom -- boosting quietly stops working -- is invisible.
    from toolbroker.retrieve import KeywordRetriever, RetrievalPipeline

    tracker = UsageTracker(clock=FakeClock())
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False, usage=tracker)
    broker.index(tools())
    broker.set_retriever(RetrievalPipeline(KeywordRetriever(broker.store)))

    broker.record_use("shop/search_invoices", 100)
    selection = broker.select("find customer", k=2)
    assert any("usage_boost" in hit.components for hit in selection.hits)


def test_set_retriever_accepts_a_bare_retriever():
    from toolbroker.retrieve import KeywordRetriever

    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index(tools())
    broker.set_retriever(KeywordRetriever(broker.store))
    assert len(broker.select("orders", k=1)) == 1


def test_set_retriever_does_not_double_attach():
    from toolbroker.retrieve import KeywordRetriever, RetrievalPipeline

    tracker = UsageTracker(clock=FakeClock())
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False, usage=tracker)
    broker.index(tools())
    pipeline = RetrievalPipeline(KeywordRetriever(broker.store))
    broker.set_retriever(pipeline)
    broker.set_retriever(broker.pipeline)
    assert len(broker.pipeline.rerankers) == 1
