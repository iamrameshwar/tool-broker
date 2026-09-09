"""Behaviour specific to the Postgres backend, beyond the shared contract."""

from __future__ import annotations

import pytest
from toolbroker_pgvector import PgVectorStore
from toolbroker_pgvector.store import build_where

from toolbroker import Filters, RiskTier, Tool, ToolBroker
from toolbroker.errors import DimensionMismatchError, StoreError
from toolbroker.index.embedders import HashingEmbedder
from toolbroker.testing import make_records, make_tools
from toolbroker.types import ToolRecord

from .conftest import requires_postgres

DIM = 128

pytestmark = requires_postgres


@pytest.fixture
def store(dsn, table):
    created = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    yield created
    created.drop()
    created.close()


@pytest.fixture
def populated(store):
    store.upsert(make_records(make_tools(20), dim=DIM))
    return store


def record(name: str, vector, **kwargs) -> ToolRecord:
    return ToolRecord(
        tool=Tool(name=name, namespace="test", **kwargs), text=name, vector=tuple(vector)
    )


# -- construction ---------------------------------------------------------


def test_rejects_non_positive_dim(dsn, table):
    with pytest.raises(ValueError, match="positive"):
        PgVectorStore(dim=0, dsn=dsn, table=table)


def test_rejects_unknown_metric(dsn, table):
    with pytest.raises(ValueError, match="metric must be"):
        PgVectorStore(dim=DIM, dsn=dsn, table=table, metric="hamming")


def test_requires_something_to_connect_with():
    with pytest.raises(StoreError, match="dsn, connection, or pool"):
        PgVectorStore(dim=DIM)


def test_table_is_created_and_reopened(dsn, table):
    first = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    first.upsert(make_records(make_tools(3), dim=DIM))
    first.close()

    second = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    try:
        assert len(second) == 3
    finally:
        second.drop()
        second.close()


def test_dimension_change_on_an_existing_table_is_refused(dsn, table):
    # Querying a table built by a different embedder returns confident
    # nonsense, which is worse than failing at construction.
    first = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    try:
        with pytest.raises(StoreError, match="different embedder"):
            PgVectorStore(dim=DIM * 2, dsn=dsn, table=table)
    finally:
        first.drop()
        first.close()


def test_recreate_wipes_the_table(dsn, table):
    first = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    first.upsert(make_records(make_tools(3), dim=DIM))
    first.close()

    second = PgVectorStore(dim=DIM, dsn=dsn, table=table, recreate=True)
    try:
        assert len(second) == 0
    finally:
        second.drop()
        second.close()


def test_wrong_vector_width_is_rejected(store):
    with pytest.raises(DimensionMismatchError):
        store.upsert([record("a", [1.0, 0.0])])
    with pytest.raises(DimensionMismatchError):
        store.search([1.0, 0.0], k=1)


def test_an_existing_connection_can_be_shared(dsn, table):
    import psycopg
    from pgvector.psycopg import register_vector

    connection = psycopg.connect(dsn, autocommit=True)
    register_vector(connection)
    store = PgVectorStore(dim=DIM, connection=connection, table=table)
    try:
        store.upsert(make_records(make_tools(2), dim=DIM))
        assert len(store) == 2
    finally:
        store.drop()
        connection.close()


# -- where translation ----------------------------------------------------


def sql_text(filters) -> str:
    clause, _ = build_where(filters)
    return clause.as_string(None) if hasattr(clause, "as_string") else str(clause)


def test_empty_filters_produce_no_clause():
    assert sql_text(None) == ""
    assert sql_text(Filters()) == ""


def test_namespace_becomes_an_any_clause():
    assert "namespace = ANY" in sql_text(Filters(namespaces=frozenset({"a"})))


def test_tags_any_uses_the_overlap_operator():
    assert "tags &&" in sql_text(Filters(tags_any=frozenset({"a"})))


def test_tags_all_uses_the_contains_operator():
    assert "tags @>" in sql_text(Filters(tags_all=frozenset({"a", "b"})))


def test_risk_becomes_an_ordinal_range():
    clause, params = build_where(Filters(max_risk=RiskTier.MEDIUM))
    assert "risk_level <= " in clause.as_string(None)
    assert params == [1]


@pytest.mark.parametrize(
    "filters",
    [
        Filters(namespaces=frozenset()),
        Filters(tags_any=frozenset()),
        Filters(tool_ids=frozenset()),
    ],
)
def test_empty_allow_sets_match_nothing(filters):
    # Dropping the clause would silently widen the query to the whole
    # catalogue, which is the dangerous direction for a guard rail.
    assert sql_text(filters) == "WHERE false"


def test_filters_run_before_k(populated):
    # The contract: ask for 3 low-risk tools, get 3 -- not whatever survives
    # filtering the global top 3.
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3")
    hits = populated.search(query, k=3, filters=Filters(max_risk=RiskTier.LOW))
    assert len(hits) == 3
    assert all(hit.tool.risk is RiskTier.LOW for hit in hits)


def test_combined_filters_intersect(populated):
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3")
    hits = populated.search(
        query, k=20, filters=Filters(tags_any=frozenset({"even"}), max_risk=RiskTier.MEDIUM)
    )
    assert all("even" in hit.tool.tags for hit in hits)
    assert all(hit.tool.risk <= RiskTier.MEDIUM for hit in hits)


def test_exclusion_keeps_untagged_tools(store):
    store.upsert(
        [
            record("tagged", [1.0] + [0.0] * (DIM - 1), tags=frozenset({"internal"})),
            record("untagged", [0.9, 0.1] + [0.0] * (DIM - 2)),
        ]
    )
    hits = store.search(
        [1.0] + [0.0] * (DIM - 1), k=5, filters=Filters(exclude_tags=frozenset({"internal"}))
    )
    assert [hit.tool.name for hit in hits] == ["untagged"]


# -- scoring and round-tripping -------------------------------------------


def test_identical_vector_scores_near_one(store):
    vector = [1.0] + [0.0] * (DIM - 1)
    store.upsert([record("a", vector)])
    assert store.search(vector, k=1)[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_ordered_best_first(populated):
    query = HashingEmbedder(dim=DIM).embed_query("tool number 3 topic 3")
    scores = [hit.score for hit in populated.search(query, k=10)]
    assert scores == sorted(scores, reverse=True)


def test_tool_survives_the_round_trip(store):
    tool = Tool(
        name="issue_refund",
        namespace="billing",
        description="Refund a payment",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
        tags=frozenset({"write", "money"}),
        risk=RiskTier.HIGH,
        required_scopes=frozenset({"payments:write"}),
    )
    store.upsert([ToolRecord(tool=tool, text="refund", vector=tuple([0.1] * DIM))])
    assert store.get("billing/issue_refund").tool == tool


def test_vectors_survive_the_round_trip(store):
    vector = tuple(float(i) / DIM for i in range(DIM))
    store.upsert([record("a", vector)])
    stored = store.get("test/a").vector
    assert len(stored) == DIM
    assert stored[1] == pytest.approx(vector[1], abs=1e-6)


def test_upsert_replaces_rather_than_duplicating(store):
    store.upsert([record("a", [1.0] + [0.0] * (DIM - 1), description="first")])
    store.upsert([record("a", [1.0] + [0.0] * (DIM - 1), description="second")])
    assert len(store) == 1
    assert store.get("test/a").tool.description == "second"


def test_ties_break_by_tool_id(store):
    vector = [1.0] + [0.0] * (DIM - 1)
    store.upsert([record("zebra", vector), record("alpha", vector), record("mango", vector)])
    assert [h.tool.name for h in store.search(vector, k=3)] == ["alpha", "mango", "zebra"]


def test_clear_keeps_the_table(populated):
    populated.clear()
    assert len(populated) == 0
    populated.upsert(make_records(make_tools(2), dim=DIM))
    assert len(populated) == 2


# -- integration ----------------------------------------------------------


def test_drop_in_replacement_for_the_default_store(dsn, table):
    store = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    try:
        broker = ToolBroker(embedder=HashingEmbedder(dim=DIM), store=store, cache_embeddings=False)
        broker.index(
            [
                Tool(name="issue_refund", namespace="billing", description="Refund a payment"),
                Tool(name="check_inventory", namespace="shop", description="Stock for a SKU"),
            ]
        )
        assert broker.select("refund a payment", k=1).tool_ids == ("billing/issue_refund",)
    finally:
        store.drop()
        store.close()


def test_incremental_refresh_works_against_postgres(dsn, table):
    from toolbroker.sources.base import BaseSource

    class Mutable(BaseSource):
        def __init__(self):
            super().__init__("mut")
            self.tools = [Tool(name="alpha", namespace="mut", description="Does alpha.")]

        def _discover(self):
            return list(self.tools)

    source = Mutable()
    store = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    try:
        broker = ToolBroker(embedder=HashingEmbedder(dim=DIM), store=store, cache_embeddings=False)
        broker.add_source(source)
        broker.refresh()
        source.tools.append(Tool(name="beta", namespace="mut", description="Does beta."))
        report = broker.refresh()
        assert report.added == ("mut/beta",)
        assert report.unchanged == 1
    finally:
        store.drop()
        store.close()


def test_resolves_through_the_plugin_registry():
    from toolbroker.registry import GROUP_STORES, available, resolve

    assert "pgvector" in available(GROUP_STORES)
    assert resolve(GROUP_STORES, "pgvector") is PgVectorStore


def test_usable_as_a_context_manager(dsn, table):
    with PgVectorStore(dim=DIM, dsn=dsn, table=table) as store:
        store.upsert(make_records(make_tools(2), dim=DIM))
        assert len(store) == 2
    store.drop()


def test_close_leaves_an_injected_connection_alone(dsn, table):
    # The store did not open it and has no business deciding when it ends.
    import psycopg
    from pgvector.psycopg import register_vector

    connection = psycopg.connect(dsn, autocommit=True)
    register_vector(connection)
    store = PgVectorStore(dim=DIM, connection=connection, table=table)
    store.close()
    assert not connection.closed
    store.drop()
    connection.close()


def test_the_suite_leaves_no_tables_behind(dsn, table):
    # Each test uses a unique table and drops it; this pins that habit.
    import psycopg

    store = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    store.drop()
    store.close()
    with psycopg.connect(dsn) as connection:
        cursor = connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name=%s",
            (table,),
        )
        assert cursor.fetchone()[0] == 0


def test_bootstraps_the_extension_on_a_fresh_database(dsn, table):
    # register_vector() looks the type up and fails if the extension is absent,
    # so the extension has to be created first. Doing it the other way round
    # works on any database where someone already ran CREATE EXTENSION by hand,
    # and breaks on exactly the case that matters -- a fresh one.
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as connection:
        cursor = connection.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
        assert cursor.fetchone()[0] == 1, "fixture database should already have it by now"

    store = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    try:
        store.upsert(make_records(make_tools(1), dim=DIM))
        assert len(store) == 1
    finally:
        store.drop()
        store.close()


def test_drop_is_public_and_idempotent(dsn, table):
    store = PgVectorStore(dim=DIM, dsn=dsn, table=table)
    store.drop()
    store.drop()
    store.close()
