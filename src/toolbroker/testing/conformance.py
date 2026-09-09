"""Executable contracts for every extension point with a registry group."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from ..errors import AdapterError
from ..index.embedders.hashing import HashingEmbedder
from ..types import Filters, RiskTier, Tool
from .factories import make_records, make_tools


class StoreConformanceSuite:
    """Behaviour every :class:`~toolbroker.protocols.Store` must exhibit.

    Subclass and provide a ``store`` fixture returning an empty store of
    dimension :attr:`DIM`.
    """

    DIM = 128

    @pytest.fixture
    def store(self) -> Any:  # pragma: no cover - subclasses override
        """Return an empty store of width :attr:`DIM`.

        It must be empty *and isolated from other tests*. Several backends hand
        out a process-wide shared client, so two stores naming the same
        collection are the same collection; give each test a unique collection
        name, or a temporary directory, rather than relying on construction to
        start clean.
        """
        raise NotImplementedError("provide a 'store' fixture returning an empty store")

    @pytest.fixture
    def populated(self, store: Any) -> Any:
        """A store holding 20 varied tools."""
        store.upsert(make_records(make_tools(20), dim=self.DIM))
        return store

    @pytest.fixture
    def query_vector(self) -> list[float]:
        """A query vector in the same space as the fixtures."""
        return HashingEmbedder(dim=self.DIM).embed_query("tool number 3 topic 3")

    def test_reports_its_dimension(self, store: Any) -> None:
        """``dim`` matches what the store was constructed with."""
        assert store.dim == self.DIM

    def test_starts_empty(self, store: Any) -> None:
        """A fresh store holds nothing."""
        assert len(store) == 0

    def test_upsert_then_len(self, populated: Any) -> None:
        """Every upserted record is counted once."""
        assert len(populated) == 20

    def test_upsert_is_idempotent_by_id(self, store: Any) -> None:
        """Re-upserting the same ids replaces rather than duplicates."""
        records = make_records(make_tools(5), dim=self.DIM)
        store.upsert(records)
        store.upsert(records)
        assert len(store) == 5

    def test_get_returns_the_record(self, populated: Any) -> None:
        """A stored record is retrievable by tool id."""
        assert populated.get("test/tool_003").tool.name == "tool_003"

    def test_get_missing_returns_none(self, populated: Any) -> None:
        """An unknown id yields ``None`` rather than raising."""
        assert populated.get("test/does_not_exist") is None

    def test_search_returns_at_most_k(self, populated: Any, query_vector: Any) -> None:
        """``k`` is an upper bound."""
        assert len(populated.search(query_vector, k=5)) <= 5

    def test_search_orders_by_descending_score(self, populated: Any, query_vector: Any) -> None:
        """Hits come back best first."""
        scores = [hit.score for hit in populated.search(query_vector, k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_search_is_deterministic(self, populated: Any, query_vector: Any) -> None:
        """The same query returns the same ids in the same order."""
        first = [hit.id for hit in populated.search(query_vector, k=10)]
        second = [hit.id for hit in populated.search(query_vector, k=10)]
        assert first == second

    def test_search_on_empty_store(self, store: Any, query_vector: Any) -> None:
        """An empty store returns no hits rather than raising."""
        assert store.search(query_vector, k=5) == []

    def test_zero_k_returns_nothing(self, populated: Any, query_vector: Any) -> None:
        """``k=0`` is valid and returns nothing."""
        assert populated.search(query_vector, k=0) == []

    def test_namespace_filter(self, populated: Any, query_vector: Any) -> None:
        """Namespace filtering excludes everything else."""
        hits = populated.search(
            query_vector, k=20, filters=Filters(namespaces=frozenset({"other"}))
        )
        assert hits == []

    def test_tag_filter(self, populated: Any, query_vector: Any) -> None:
        """``tags_any`` keeps only tools carrying one of the tags."""
        hits = populated.search(query_vector, k=20, filters=Filters(tags_any=frozenset({"even"})))
        assert hits
        assert all("even" in hit.tool.tags for hit in hits)

    def test_risk_filter(self, populated: Any, query_vector: Any) -> None:
        """``max_risk`` excludes higher tiers."""
        hits = populated.search(query_vector, k=20, filters=Filters(max_risk=RiskTier.LOW))
        assert all(hit.tool.risk is RiskTier.LOW for hit in hits)

    @pytest.mark.parametrize(
        "filters",
        [
            Filters(namespaces=frozenset()),
            Filters(tags_any=frozenset()),
            Filters(tool_ids=frozenset()),
        ],
        ids=["namespaces", "tags_any", "tool_ids"],
    )
    def test_empty_allow_set_matches_nothing(
        self, populated: Any, query_vector: Any, filters: Filters
    ) -> None:
        """An empty allow-set excludes everything, matching ``Filters.matches``.

        The dangerous failure here is silent widening: a backend that drops an
        empty clause returns the whole catalogue for a filter the caller wrote
        to restrict it.
        """
        assert populated.search(query_vector, k=5, filters=filters) == []

    def test_filters_apply_before_k(self, populated: Any, query_vector: Any) -> None:
        """Filtering happens before truncation, so ``k`` stays meaningful."""
        hits = populated.search(query_vector, k=3, filters=Filters(max_risk=RiskTier.LOW))
        assert len(hits) == 3

    def test_delete_removes_records(self, populated: Any) -> None:
        """Deleting reports the count and the record is gone."""
        assert populated.delete(["test/tool_000"]) == 1
        assert populated.get("test/tool_000") is None
        assert len(populated) == 19

    def test_delete_unknown_id_is_a_noop(self, populated: Any) -> None:
        """Deleting something absent removes nothing and does not raise."""
        assert populated.delete(["test/absent"]) == 0

    def test_search_works_after_delete(self, populated: Any, query_vector: Any) -> None:
        """Deletion leaves the index queryable and consistent."""
        populated.delete(["test/tool_000", "test/tool_001"])
        hits = populated.search(query_vector, k=20)
        assert "test/tool_000" not in [hit.id for hit in hits]
        assert len(hits) == 18

    def test_all_records_returns_everything(self, populated: Any) -> None:
        """``all_records`` exposes the full contents for lexical retrievers."""
        assert len(populated.all_records()) == 20

    def test_clear_empties_the_store(self, populated: Any) -> None:
        """``clear`` removes everything."""
        populated.clear()
        assert len(populated) == 0

    def test_records_round_trip_their_tool(self, populated: Any) -> None:
        """Tool metadata survives storage unchanged."""
        record = populated.get("test/tool_005")
        assert record.tool.tags
        assert record.text


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, used by the embedder suite's tolerance checks."""
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    magnitude = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if magnitude == 0.0 else numerator / magnitude


def _rendered_name(entry: Any) -> str:
    """Pull the tool name out of whatever shape an adapter emitted."""
    name = getattr(entry, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(entry, dict):
        if isinstance(entry.get("name"), str):
            return str(entry["name"])
        function = entry.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return str(function["name"])
    raise AssertionError(f"cannot determine a tool name from {entry!r}")


class AdapterConformanceSuite:
    """Behaviour every :class:`~toolbroker.protocols.Adapter` must exhibit.

    Subclass and provide an ``adapter`` fixture.
    """

    @pytest.fixture
    def adapter(self) -> Any:  # pragma: no cover - subclasses override
        """Return the adapter under test."""
        raise NotImplementedError("provide an 'adapter' fixture")

    @staticmethod
    def _example_handler(**kwargs: Any) -> str:
        """Stand-in implementation for the fixture tools."""
        return f"called with {sorted(kwargs)}"

    @pytest.fixture
    def tools(self) -> list[Tool]:
        """A small, varied tool set.

        Every tool carries a local callable. Adapters that only render schemas
        ignore it; adapters that bind an implementation need one, and without it
        this suite could only ever test half of them.
        """
        handler = type(self)._example_handler
        return [
            Tool(
                name="search",
                namespace="docs",
                description="Search the docs",
                metadata={"callable": handler},
            ),
            Tool(
                name="create_issue",
                namespace="github",
                description="Open an issue",
                input_schema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
                metadata={"callable": handler},
            ),
        ]

    def test_has_an_id(self, adapter: Any) -> None:
        """Adapters identify their target framework."""
        assert isinstance(adapter.id, str)
        assert adapter.id

    def test_renders_one_entry_per_tool(self, adapter: Any, tools: list[Tool]) -> None:
        """Nothing is dropped during rendering."""
        assert len(adapter.render(tools)) == len(tools)

    def test_empty_input_renders_empty(self, adapter: Any) -> None:
        """An empty selection renders to an empty result, not an error."""
        assert len(adapter.render([])) == 0

    def test_rendering_is_repeatable(self, adapter: Any, tools: list[Tool]) -> None:
        """Rendering twice produces the same tools, in the same order.

        Compared by name rather than by object: adapters that bind an
        implementation embed a fresh closure each time, so object equality would
        be a contract no such adapter could ever satisfy.
        """
        first = [_rendered_name(entry) for entry in adapter.render(tools)]
        second = [_rendered_name(entry) for entry in adapter.render(tools)]
        assert first == second
        assert len(first) == len(tools)

    def test_tool_without_schema_renders(self, adapter: Any) -> None:
        """A tool with no parameters is still renderable."""
        ping = Tool(name="ping", metadata={"callable": type(self)._example_handler})
        assert len(adapter.render([ping])) == 1

    def test_colliding_names_raise_or_are_disambiguated(
        self, adapter: Any, tools: list[Tool]
    ) -> None:
        """Ambiguous names must not silently drop a tool.

        Adapters resolve this differently — some flatten the namespace in, some
        raise and ask the caller to opt into qualified names. Either is fine;
        quietly emitting one tool where two were requested is not.
        """
        handler = type(self)._example_handler
        colliding = [
            Tool(name="b__c", namespace="a", metadata={"callable": handler}),
            Tool(name="c", namespace="a__b", metadata={"callable": handler}),
            Tool(name="c", namespace="a", metadata={"callable": handler}),
        ]
        try:
            rendered = adapter.render(colliding)
        except AdapterError:
            return
        assert len(rendered) == len(colliding)


class EmbedderConformanceSuite:
    """Behaviour every :class:`~toolbroker.protocols.Embedder` must exhibit.

    Subclass and provide an ``embedder`` fixture. The suite is written to pass
    for a purely lexical embedder as well as a semantic one, so the similarity
    checks use texts that overlap on words rather than only on meaning — a
    contract that only a neural model could satisfy would exclude the offline
    default.
    """

    #: How far apart two embeddings of the same text may be. Local models are
    #: bit-identical; a hosted API is not always, so subclasses can relax this.
    SIMILARITY_TOLERANCE = 0.999

    @pytest.fixture
    def embedder(self) -> Any:  # pragma: no cover - subclasses override
        """Return the embedder under test."""
        raise NotImplementedError("provide an 'embedder' fixture")

    def test_dim_is_a_positive_int(self, embedder: Any) -> None:
        """``dim`` describes the output width."""
        assert isinstance(embedder.dim, int)
        assert embedder.dim > 0

    def test_id_identifies_the_model(self, embedder: Any) -> None:
        """``id`` is a stable, non-empty cache key."""
        assert isinstance(embedder.id, str)
        assert embedder.id
        assert embedder.id == embedder.id

    def test_empty_batch_returns_empty(self, embedder: Any) -> None:
        """Embedding nothing costs nothing and returns nothing."""
        assert embedder.embed([]) == []

    def test_one_vector_per_text_in_order(self, embedder: Any) -> None:
        """Results line up positionally with the inputs."""
        texts = ["refund a customer order", "restart the web server", "list open invoices"]
        vectors = embedder.embed(texts)
        assert len(vectors) == len(texts)

    def test_every_vector_has_the_declared_width(self, embedder: Any) -> None:
        """``dim`` is a promise, not a hint."""
        for vector in embedder.embed(["alpha", "beta"]):
            assert len(vector) == embedder.dim

    def test_query_vector_has_the_declared_width(self, embedder: Any) -> None:
        """Query and document vectors share a space, so they share a width."""
        assert len(embedder.embed_query("refund a customer")) == embedder.dim

    def test_vectors_are_finite_floats(self, embedder: Any) -> None:
        """No NaN or infinity: either would poison every ranking downstream."""
        for value in embedder.embed(["refund a customer order"])[0]:
            assert isinstance(value, float)
            assert math.isfinite(value)

    def test_embedding_is_deterministic(self, embedder: Any) -> None:
        """The same text embeds the same way twice.

        Without this a re-index silently reranks the catalogue, and no test that
        asserts on a tool list can be trusted.
        """
        first = embedder.embed(["refund a customer order"])[0]
        second = embedder.embed(["refund a customer order"])[0]
        assert _cosine(first, second) >= self.SIMILARITY_TOLERANCE

    def test_batching_does_not_change_results(self, embedder: Any) -> None:
        """A text embeds the same alone as it does in a batch."""
        alone = embedder.embed(["restart the web server"])[0]
        batched = embedder.embed(["refund a customer order", "restart the web server"])[1]
        assert _cosine(alone, batched) >= self.SIMILARITY_TOLERANCE

    def test_related_text_beats_unrelated_text(self, embedder: Any) -> None:
        """The whole point: nearness has to mean something."""
        query = embedder.embed_query("refund a customer order")
        near, far = embedder.embed(
            ["issue a refund for a customer order", "compile the kernel from source"]
        )
        assert _cosine(query, near) > _cosine(query, far)

    def test_empty_string_does_not_crash(self, embedder: Any) -> None:
        """A tool with no description still has to be indexable."""
        assert len(embedder.embed([""])[0]) == embedder.dim

    def test_unicode_is_handled(self, embedder: Any) -> None:
        """Tool descriptions are not always ASCII."""
        assert len(embedder.embed(["débiter le compte client 日本語 🎯"])[0]) == embedder.dim

    def test_long_text_is_handled(self, embedder: Any) -> None:
        """Enriched index text for a large schema gets long; truncate, do not fail."""
        assert len(embedder.embed(["refund " * 2000])[0]) == embedder.dim

    def test_works_with_the_catalogue(self, embedder: Any) -> None:
        """The real integration: index and select using this embedder."""
        from ..catalog import ToolBroker
        from ..types import Tool

        broker = ToolBroker(embedder=embedder, cache_embeddings=False)
        broker.index(
            [
                Tool(name="issue_refund", namespace="billing", description="Refund a payment"),
                Tool(name="restart_service", namespace="ops", description="Restart a service"),
            ]
        )
        assert broker.select("refund a payment", k=1).tool_ids == ("billing/issue_refund",)


class RetrieverConformanceSuite:
    """Behaviour every :class:`~toolbroker.protocols.Retriever` must exhibit.

    Subclass and provide a ``retriever`` fixture, plus the ``tools`` those
    results should be drawn from if the default set does not suit.

        class TestGraphRetriever(RetrieverConformanceSuite):
            @pytest.fixture
            def retriever(self, store):
                return GraphRetriever(store)

    Ranking quality is deliberately **not** tested. Which tools a retriever
    returns is its whole reason for existing and is not something a shared
    contract can adjudicate — ``docs/stability.md`` is explicit that retrieval
    results are not an API. What is tested is the shape of the answer, because
    every stage downstream depends on it: ``k`` is honoured, filters narrow
    rather than widen, and scores are comparable within one result.
    """

    @pytest.fixture
    def retriever(self) -> Any:  # pragma: no cover - subclasses override
        """Return the retriever under test, wired to ``store``."""
        raise NotImplementedError("provide a 'retriever' fixture")

    @pytest.fixture
    def tools(self) -> list[Tool]:
        """A small, varied set with distinguishable text."""
        return [
            Tool(name="issue_refund", namespace="billing", description="Refund a payment."),
            Tool(name="list_payments", namespace="billing", description="List payments."),
            Tool(
                name="restart_service",
                namespace="ops",
                description="Restart a running service.",
                risk=RiskTier.HIGH,
                tags=frozenset({"destructive"}),
            ),
        ]

    def test_returns_hits(self, retriever: Any) -> None:
        """A query against a populated store returns something."""
        assert retriever.retrieve("refund a payment", 3)

    def test_respects_k(self, retriever: Any) -> None:
        """Never more than asked for. Everything downstream sizes on this."""
        for k in (1, 2, 3):
            assert len(retriever.retrieve("payment", k)) <= k

    def test_zero_k_returns_nothing(self, retriever: Any) -> None:
        """``k=0`` is a legitimate degenerate case, not an error."""
        assert retriever.retrieve("payment", 0) == []

    def test_hits_carry_a_tool_and_a_score(self, retriever: Any) -> None:
        """The two fields every downstream stage reads."""
        for hit in retriever.retrieve("refund a payment", 3):
            assert isinstance(hit.tool, Tool)
            assert isinstance(hit.score, float | int)

    def test_scores_are_ordered(self, retriever: Any) -> None:
        """Descending, so a caller can truncate without re-sorting."""
        scores = [hit.score for hit in retriever.retrieve("payment", 3)]
        assert scores == sorted(scores, reverse=True)

    def test_ids_are_unique(self, retriever: Any) -> None:
        """One result per tool. A duplicate silently costs a caller a slot."""
        ids = [hit.id for hit in retriever.retrieve("payment", 3)]
        assert len(ids) == len(set(ids))

    def test_an_empty_query_does_not_raise(self, retriever: Any) -> None:
        """Real callers pass empty strings. Return nothing, or return anything."""
        retriever.retrieve("", 3)

    def test_filters_narrow_the_result(self, retriever: Any) -> None:
        """A namespace filter cannot return a tool outside it."""
        hits = retriever.retrieve("payment", 5, Filters(namespaces=frozenset({"billing"})))
        assert all(hit.tool.namespace == "billing" for hit in hits)

    def test_an_empty_allow_set_matches_nothing(self, retriever: Any) -> None:
        """The dangerous direction.

        A backend that *drops* an empty clause returns the whole catalogue for
        a filter written to restrict it. Held to the same rule as stores.
        """
        assert retriever.retrieve("payment", 5, Filters(tool_ids=frozenset())) == []

    def test_exclusions_are_honoured(self, retriever: Any) -> None:
        """An excluded tool must not come back under any query."""
        hits = retriever.retrieve(
            "refund a payment", 5, Filters(exclude_tool_ids=frozenset({"billing/issue_refund"}))
        )
        assert all(hit.id != "billing/issue_refund" for hit in hits)

    def test_repeated_queries_agree(self, retriever: Any) -> None:
        """Same query, same store, same tools — in the same order."""
        first = [hit.id for hit in retriever.retrieve("refund a payment", 3)]
        second = [hit.id for hit in retriever.retrieve("refund a payment", 3)]
        assert first == second


class RerankerConformanceSuite:
    """Behaviour every :class:`~toolbroker.protocols.Reranker` must exhibit.

    Subclass and provide a ``reranker`` fixture.

    A reranker's job is to *reorder*, and the contract exists because the two
    ways of getting that wrong are both silent: inventing a hit the retriever
    never produced, and dropping one without being asked to.
    """

    @pytest.fixture
    def reranker(self) -> Any:  # pragma: no cover - subclasses override
        """Return the reranker under test."""
        raise NotImplementedError("provide a 'reranker' fixture")

    @pytest.fixture
    def hits(self) -> list[Any]:
        """Candidates in the shape a retriever hands over."""
        from ..types import Hit

        return [
            Hit(tool=tool, score=1.0 - index * 0.1)
            for index, tool in enumerate(make_tools(5, namespace="bench"))
        ]

    def test_returns_at_most_k(self, reranker: Any, hits: list[Any]) -> None:
        """The cut is the reranker's to make, but it may not exceed ``k``."""
        for k in (1, 3, 5):
            assert len(reranker.rerank("a query", hits, k)) <= k

    def test_invents_nothing(self, reranker: Any, hits: list[Any]) -> None:
        """Every returned hit was in the input. The silent-corruption case."""
        known = {hit.id for hit in hits}
        assert {hit.id for hit in reranker.rerank("a query", hits, 5)}.issubset(known)

    def test_keeps_everything_when_k_allows(self, reranker: Any, hits: list[Any]) -> None:
        """Reordering is not filtering. Dropping without being asked loses tools."""
        assert len(reranker.rerank("a query", hits, len(hits))) == len(hits)

    def test_no_duplicates(self, reranker: Any, hits: list[Any]) -> None:
        """A duplicated hit costs the caller a slot for nothing."""
        ids = [hit.id for hit in reranker.rerank("a query", hits, 5)]
        assert len(ids) == len(set(ids))

    def test_empty_input_is_empty_output(self, reranker: Any) -> None:
        """A retriever that found nothing must not become a crash here."""
        assert reranker.rerank("a query", [], 5) == []

    def test_zero_k_returns_nothing(self, reranker: Any, hits: list[Any]) -> None:
        """Consistent with retrievers."""
        assert reranker.rerank("a query", hits, 0) == []

    def test_the_input_is_not_mutated(self, reranker: Any, hits: list[Any]) -> None:
        """Pipelines reuse the candidate list; in-place reordering corrupts it."""
        before = [hit.id for hit in hits]
        reranker.rerank("a query", hits, 5)
        assert [hit.id for hit in hits] == before

    def test_repeated_calls_agree(self, reranker: Any, hits: list[Any]) -> None:
        """Same query, same candidates, same order."""
        first = [hit.id for hit in reranker.rerank("a query", hits, 5)]
        second = [hit.id for hit in reranker.rerank("a query", hits, 5)]
        assert first == second


class TracerConformanceSuite:
    """Behaviour every :class:`~toolbroker.protocols.Tracer` must exhibit.

    Subclass and provide a ``tracer`` fixture.

    The bar is deliberately low and one rule is absolute: **a tracer must not
    change what the application does**. Observability is never worth an outage,
    so a tracer that raises, swallows, or alters an exception is worse than no
    tracer at all.
    """

    @pytest.fixture
    def tracer(self) -> Any:  # pragma: no cover - subclasses override
        """Return the tracer under test."""
        raise NotImplementedError("provide a 'tracer' fixture")

    def test_span_is_a_context_manager(self, tracer: Any) -> None:
        """The whole protocol."""
        with tracer.span("toolbroker.test"):
            pass

    def test_attributes_are_accepted(self, tracer: Any) -> None:
        """The core passes ints, strings and None as span attributes."""
        with tracer.span("toolbroker.test", k=5, agent="support", missing=None):
            pass

    def test_spans_nest(self, tracer: Any) -> None:
        """Retrieval opens a span inside selection's."""
        with tracer.span("toolbroker.outer"), tracer.span("toolbroker.inner"):
            pass

    def test_an_exception_propagates_unchanged(self, tracer: Any) -> None:
        """A tracer that swallows an error hides a production failure."""
        sentinel = RuntimeError("boom")
        with pytest.raises(RuntimeError) as caught:  # noqa: SIM117
            with tracer.span("toolbroker.test"):
                raise sentinel
        assert caught.value is sentinel

    def test_reusable(self, tracer: Any) -> None:
        """One tracer serves every request for the life of the process."""
        for index in range(3):
            with tracer.span(f"toolbroker.test.{index}"):
                pass
