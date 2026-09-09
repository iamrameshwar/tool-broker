from __future__ import annotations

import pytest

from toolbroker import Filters, Hit, Tool
from toolbroker.index.indexer import Indexer
from toolbroker.retrieve import (
    HybridRetriever,
    KeywordRetriever,
    LLMReranker,
    RetrievalPipeline,
    SemanticRetriever,
    UsageBooster,
)


def build(embedder, store, tools):
    Indexer(embedder, store).index(tools)
    return SemanticRetriever(store, embedder)


def test_semantic_retrieval_returns_relevant_tools(embedder, store, sample_tools):
    retriever = build(embedder, store, sample_tools)
    hits = retriever.retrieve("give the customer their money back for an order", k=3)
    assert "test/issue_refund" in [hit.id for hit in hits]


def test_semantic_retrieval_respects_k(embedder, store, sample_tools):
    retriever = build(embedder, store, sample_tools)
    assert len(retriever.retrieve("customer", k=2)) == 2


def test_semantic_retrieval_applies_filters(embedder, store, sample_tools):
    retriever = build(embedder, store, sample_tools)
    hits = retriever.retrieve("customer", k=5, filters=Filters(tags_any=frozenset({"warehouse"})))
    assert [hit.id for hit in hits] == ["test/check_inventory"]


def test_min_score_drops_weak_hits(embedder, store, sample_tools):
    Indexer(embedder, store).index(sample_tools)
    strict = SemanticRetriever(store, embedder, min_score=0.99)
    assert strict.retrieve("something entirely unrelated to any of these", k=5) == []


def test_keyword_retrieval_finds_exact_identifiers(embedder, store, sample_tools):
    Indexer(embedder, store).index(sample_tools)
    hits = KeywordRetriever(store).retrieve("check_inventory", k=1)
    assert hits[0].id == "test/check_inventory"


def test_keyword_retrieval_returns_nothing_for_unknown_words(embedder, store, sample_tools):
    Indexer(embedder, store).index(sample_tools)
    assert KeywordRetriever(store).retrieve("zzzzqqq", k=5) == []


def test_keyword_index_follows_a_reindex(embedder, store, sample_tools):
    Indexer(embedder, store).index(sample_tools[:1])
    retriever = KeywordRetriever(store)
    assert retriever.retrieve("inventory", k=5) == []

    Indexer(embedder, store).index(sample_tools, replace=True)
    assert retriever.retrieve("inventory", k=5)


def test_hybrid_combines_both_signals(embedder, store, sample_tools):
    Indexer(embedder, store).index(sample_tools)
    hybrid = HybridRetriever(
        [SemanticRetriever(store, embedder), KeywordRetriever(store)],
        names=["semantic", "keyword"],
    )
    hits = hybrid.retrieve("refund", k=3)
    assert hits
    assert any("rrf" in key for key in hits[0].components)


def test_hybrid_weighted_mode_normalises_scores(embedder, store, sample_tools):
    Indexer(embedder, store).index(sample_tools)
    hybrid = HybridRetriever(
        [SemanticRetriever(store, embedder), KeywordRetriever(store)],
        mode="weighted",
        names=["semantic", "keyword"],
    )
    hits = hybrid.retrieve("refund the customer", k=3)
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


def test_usage_booster_promotes_used_tools():
    hits = [
        Hit(tool=Tool(name="a"), score=0.50),
        Hit(tool=Tool(name="b"), score=0.49),
    ]
    booster = UsageBooster(weight=0.5)
    booster.record("default/b", 100)
    assert next(hit.id for hit in booster.rerank("q", hits, 2)) == "default/b"


def test_usage_booster_without_history_changes_nothing():
    hits = [Hit(tool=Tool(name="a"), score=0.5)]
    assert UsageBooster().rerank("q", hits, 1) == hits


def test_usage_booster_decay_is_now_time_based():
    # decay_all() is gone: counts decay on a half-life, applied lazily on read,
    # so there is no sweep for anyone to forget to schedule.
    from toolbroker import UsageTracker

    clock = [1_000_000.0]
    tracker = UsageTracker(half_life=100.0, clock=lambda: clock[0])
    booster = UsageBooster(tracker)
    booster.record("x", 10)
    clock[0] += 100.0
    assert booster.snapshot()["x"] == pytest.approx(5.0)


def test_llm_reranker_reorders_by_score():
    hits = [Hit(tool=Tool(name="a"), score=0.9), Hit(tool=Tool(name="b"), score=0.1)]
    reranker = LLMReranker(lambda query, texts: [0.0, 1.0])
    assert next(hit.id for hit in reranker.rerank("q", hits, 2)) == "default/b"


def test_llm_reranker_degrades_gracefully_on_failure():
    hits = [Hit(tool=Tool(name="a"), score=0.9), Hit(tool=Tool(name="b"), score=0.1)]

    def broken(query, texts):
        raise RuntimeError("provider is down")

    assert LLMReranker(broken).rerank("q", hits, 2) == hits


def test_llm_reranker_can_be_strict():
    import pytest

    from toolbroker.errors import RetrievalError

    def broken(query, texts):
        raise RuntimeError("provider is down")

    with pytest.raises(RetrievalError):
        LLMReranker(broken, strict=True).rerank("q", [Hit(tool=Tool(name="a"), score=1.0)], 1)


def test_pipeline_overfetches_so_rerankers_have_room(embedder, store, sample_tools):
    Indexer(embedder, store).index(sample_tools)
    seen: list[int] = []

    class Spy:
        def rerank(self, query, hits, k):
            seen.append(len(hits))
            return list(hits)

    pipeline = RetrievalPipeline(SemanticRetriever(store, embedder), [Spy()], overfetch=4.0)
    pipeline.retrieve("customer", k=1)
    assert seen[0] > 1


def test_pipeline_returns_exactly_k(embedder, store, sample_tools):
    Indexer(embedder, store).index(sample_tools)
    pipeline = RetrievalPipeline(SemanticRetriever(store, embedder))
    assert len(pipeline.retrieve("customer", k=2)) == 2


def test_with_reranker_returns_a_new_pipeline(embedder, store):
    pipeline = RetrievalPipeline(SemanticRetriever(store, embedder))
    extended = pipeline.with_reranker(UsageBooster())
    assert len(pipeline.rerankers) == 0
    assert len(extended.rerankers) == 1
