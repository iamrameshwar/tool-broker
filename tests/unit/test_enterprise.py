"""Savings accounting, learned aliases, and degradation under failure."""

from __future__ import annotations

import math

import pytest

from toolbroker import (
    DenyTools,
    Event,
    PolicyEngine,
    RiskTier,
    Tool,
    ToolBroker,
)
from toolbroker.aliases import MAX_PHRASE_LENGTH, AliasLearner, normalise
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.retrieve.keyword import KeywordRetriever
from toolbroker.retrieve.resilient import ResilientRetriever
from toolbroker.savings import SavingsTally, estimate_tokens
from toolbroker.types import Filters, Hit

CATALOGUE = [
    Tool(name="issue_refund", namespace="billing", description="Refund a customer payment."),
    Tool(name="track_shipment", namespace="shipping", description="Locate a parcel in transit."),
    Tool(
        name="restart_service",
        namespace="infra",
        description="Restart a running service.",
        risk=RiskTier.HIGH,
    ),
]


class Clock:
    """Injectable time, so decay is testable without sleeping."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BrokenRetriever:
    """Stands in for a vector store that is down."""

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        self.calls += 1
        raise ConnectionError("qdrant unreachable")


class StubRetriever:
    """Returns a fixed set, so a fallback's contents are predictable."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = tools

    def retrieve(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        return [Hit(tool=tool, score=0.5) for tool in self.tools][:k]


def _broker(**kwargs) -> ToolBroker:
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False, **kwargs)
    broker.index(CATALOGUE)
    return broker


# --- savings ---------------------------------------------------------------


def test_an_empty_tally_says_so():
    assert "No selections recorded" in SavingsTally().summary()


def test_a_selection_is_counted_against_the_whole_catalogue():
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=1), broker.tools())

    report = tally.report()
    assert report.selections == 1
    assert report.tools_sent == 1
    assert report.tools_in_catalogue == 3
    assert report.tokens_saved > 0


def test_the_ratio_reflects_how_much_was_not_sent():
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=1), broker.tools())
    assert tally.report().ratio > 1.0


def test_sending_everything_saves_nothing():
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=3), broker.tools())
    assert tally.report().tokens_saved == 0


def test_estimates_are_labelled_as_estimates():
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=1), broker.tools())
    assert not tally.report().measured
    assert "estimated" in tally.summary()


def test_supplying_real_token_counts_marks_the_report_measured():
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=1), broker.tools(), actual_tokens=120)
    report = tally.report()
    assert report.measured
    assert report.tokens_sent == 120


def test_one_estimate_makes_the_whole_report_estimated():
    """Silently mixing measured and estimated figures is not auditable."""
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=1), broker.tools(), actual_tokens=120)
    tally.record(broker.select("refund", k=1), broker.tools())
    assert not tally.report().measured


def test_a_price_produces_a_cost_figure():
    broker = _broker()
    tally = SavingsTally(price_per_million=3.0)
    for _ in range(50):
        tally.record(broker.select("refund", k=1), broker.tools())
    assert tally.report().cost_saved is not None
    assert "cost avoided" in tally.summary()


def test_no_price_means_no_cost_figure():
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=1), broker.tools())
    assert tally.report().cost_saved is None
    assert "cost avoided" not in tally.summary()


def test_the_summary_states_its_counterfactual():
    """A savings number without its baseline is marketing, not measurement."""
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=1), broker.tools())
    assert "Counterfactual" in tally.summary()


def test_reset_clears_the_period():
    broker = _broker()
    tally = SavingsTally()
    tally.record(broker.select("refund", k=1), broker.tools())
    tally.reset()
    assert tally.report().selections == 0


def test_estimating_an_empty_catalogue_is_zero_ish():
    assert estimate_tokens([]) == 0


# --- learned aliases -------------------------------------------------------


def test_normalise_collapses_spelling_differences():
    assert normalise("  Bounce   The PODS ") == "bounce the pods"


def test_one_observation_does_not_reach_the_index():
    """The bound that stops a stray call rewriting retrieval."""
    learner = AliasLearner(min_count=3)
    learner.record("bounce the pods", "infra/restart_service")
    assert learner.aliases_for("infra/restart_service") == ()


def test_enough_observations_promote_an_alias():
    learner = AliasLearner(min_count=3)
    for _ in range(3):
        learner.record("bounce the pods", "infra/restart_service")
    assert learner.aliases_for("infra/restart_service") == ("bounce the pods",)


def test_aliases_are_capped_per_tool():
    """A tool cannot win by accumulating an ever-growing blob of text."""
    learner = AliasLearner(min_count=1, max_aliases=2)
    for phrase in ("one", "two", "three", "four"):
        learner.record(phrase, "infra/restart_service")
    assert len(learner.aliases_for("infra/restart_service")) == 2


def test_the_strongest_aliases_win_the_cap():
    learner = AliasLearner(min_count=1, max_aliases=1)
    learner.record("rare", "infra/restart_service")
    for _ in range(5):
        learner.record("common", "infra/restart_service")
    assert learner.aliases_for("infra/restart_service") == ("common",)


def test_aliases_decay():
    clock = Clock()
    learner = AliasLearner(min_count=1, half_life=100.0, clock=clock)
    for _ in range(2):
        learner.record("bounce the pods", "infra/restart_service")
    assert learner.aliases_for("infra/restart_service")

    clock.advance(1000.0)
    assert learner.aliases_for("infra/restart_service") == ()


def test_decay_can_be_disabled():
    clock = Clock()
    learner = AliasLearner(min_count=1, half_life=math.inf, clock=clock)
    learner.record("bounce the pods", "infra/restart_service")
    clock.advance(10**9)
    assert learner.aliases_for("infra/restart_service")


def test_an_empty_query_is_ignored():
    learner = AliasLearner(min_count=1)
    learner.record("   ", "infra/restart_service")
    assert learner.aliases_for("infra/restart_service") == ()


def test_an_essay_is_not_an_alias():
    learner = AliasLearner(min_count=1)
    learner.record("x" * (MAX_PHRASE_LENGTH + 1), "infra/restart_service")
    assert learner.aliases_for("infra/restart_service") == ()


def test_a_bad_alias_can_be_forgotten():
    learner = AliasLearner(min_count=1)
    learner.record("wrong", "infra/restart_service")
    assert learner.forget("infra/restart_service", "wrong")
    assert learner.aliases_for("infra/restart_service") == ()


def test_every_alias_for_a_tool_can_be_dropped():
    learner = AliasLearner(min_count=1)
    learner.record("a", "infra/restart_service")
    learner.record("b", "infra/restart_service")
    assert learner.forget("infra/restart_service")
    assert learner.aliases_for("infra/restart_service") == ()


def test_a_negative_half_life_is_rejected():
    with pytest.raises(ValueError, match="half_life"):
        AliasLearner(half_life=0)


def test_aliases_persist(tmp_path):
    path = tmp_path / "aliases.json"
    learner = AliasLearner(min_count=1)
    learner.record("bounce the pods", "infra/restart_service")
    learner.save(path)
    assert AliasLearner.load(path, min_count=1).aliases_for("infra/restart_service")


def test_a_missing_alias_file_is_not_an_error(tmp_path):
    assert AliasLearner.load(tmp_path / "nope.json").known_tools() == ()


def test_a_corrupt_alias_file_does_not_stop_startup(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text("{not json", encoding="utf-8")
    assert AliasLearner.load(path).known_tools() == ()


def test_the_enrich_hook_appends_aliases():
    learner = AliasLearner(min_count=1)
    learner.record("bounce the pods", "infra/restart_service")
    text = learner.enrich("Restart a running service.", tool=CATALOGUE[2])
    assert "bounce the pods" in text
    assert text.startswith("Restart a running service.")


def test_the_enrich_hook_leaves_unknown_tools_alone():
    learner = AliasLearner(min_count=1)
    assert learner.enrich("Refund a customer payment.", tool=CATALOGUE[0]) == (
        "Refund a customer payment."
    )


def test_a_learned_alias_makes_a_tool_retrievable():
    """The whole point: a phrase that missed entirely now lands."""
    learner = AliasLearner(min_count=1)

    plain = _broker()
    before = plain.select("bounce the pods", k=1).tool_ids

    taught = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    taught.hooks.register(Event.TRANSFORM_INDEX_TEXT, learner.enrich)
    for _ in range(3):
        learner.record("bounce the pods", "infra/restart_service")
    taught.index(CATALOGUE)
    after = taught.select("bounce the pods", k=1).tool_ids

    assert "infra/restart_service" not in before
    assert "infra/restart_service" in after


# --- degradation -----------------------------------------------------------


def test_by_default_a_broken_store_still_raises():
    """Silence is not a safe default; the operator has to choose."""
    retriever = ResilientRetriever(BrokenRetriever())
    with pytest.raises(ConnectionError):
        retriever.retrieve("anything", 5)


def test_empty_mode_returns_nothing_deliberately():
    retriever = ResilientRetriever(BrokenRetriever(), on_error="empty")
    assert retriever.retrieve("anything", 5) == []
    assert retriever.degraded


def test_fallback_mode_uses_the_second_retriever():
    retriever = ResilientRetriever(
        BrokenRetriever(), fallback=StubRetriever(CATALOGUE), on_error="fallback"
    )
    hits = retriever.retrieve("anything", 2)
    assert len(hits) == 2
    assert retriever.degraded


def test_a_healthy_primary_is_not_degraded():
    retriever = ResilientRetriever(
        StubRetriever(CATALOGUE), fallback=StubRetriever([]), on_error="fallback"
    )
    retriever.retrieve("anything", 1)
    assert not retriever.degraded


def test_recovery_clears_the_degraded_flag():
    primary = StubRetriever(CATALOGUE)
    retriever = ResilientRetriever(primary, on_error="empty")
    retriever._primary = BrokenRetriever()
    retriever.retrieve("anything", 1)
    assert retriever.degraded

    retriever._primary = primary
    retriever.retrieve("anything", 1)
    assert not retriever.degraded


def test_fallback_mode_requires_a_fallback():
    with pytest.raises(ValueError, match="needs a fallback"):
        ResilientRetriever(BrokenRetriever(), on_error="fallback")


def test_when_both_fail_the_original_error_surfaces():
    """The primary's failure names what actually broke."""
    retriever = ResilientRetriever(
        BrokenRetriever(), fallback=BrokenRetriever(), on_error="fallback"
    )
    with pytest.raises(ConnectionError, match="qdrant"):
        retriever.retrieve("anything", 5)


def test_the_degraded_path_is_still_filtered_by_policy():
    """The security property. A fallback must never widen access."""
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index(CATALOGUE)
    broker.set_policy(PolicyEngine([DenyTools(["infra/*"])]))
    broker.set_retriever(
        ResilientRetriever(
            BrokenRetriever(),
            fallback=StubRetriever(CATALOGUE),
            on_error="fallback",
        )
    )

    selection = broker.select("restart the service", k=5)
    assert selection.tool_ids
    assert "infra/restart_service" not in selection.tool_ids


def test_the_empty_path_is_also_still_policed():
    broker = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    broker.index(CATALOGUE)
    broker.set_policy(PolicyEngine([DenyTools(["infra/*"])]))
    broker.set_retriever(ResilientRetriever(BrokenRetriever(), on_error="empty"))
    assert broker.select("restart the service", k=5).tool_ids == ()


def test_a_lexical_fallback_needs_no_external_service():
    """The realistic configuration: BM25 over records already in memory."""
    broker = _broker()
    retriever = ResilientRetriever(
        BrokenRetriever(),
        fallback=KeywordRetriever(broker.store),
        on_error="fallback",
    )
    assert retriever.retrieve("refund a customer payment", 1)
