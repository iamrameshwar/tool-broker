"""A :class:`~toolbroker.protocols.Store` backed by PostgreSQL and pgvector.

The reason to choose this over a dedicated vector database is that you already
run Postgres. A tool catalogue is small — hundreds to low thousands of rows —
and rarely worth a second piece of infrastructure with its own backups,
upgrades, and on-call rota.

Filters compile to ``WHERE`` clauses so narrowing happens **before** ranking,
which is what keeps ``k`` meaning what the caller asked for. Tags live in a
``text[]`` with a GIN index; risk is an ordinal so a ceiling is a range scan
rather than an enumeration of every permitted tier.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from toolbroker._vectors import Vector
from toolbroker.determinism import stable_sort
from toolbroker.errors import DimensionMismatchError, StoreError
from toolbroker.observability import get_logger
from toolbroker.types import Filters, Hit, RiskTier, Tool, ToolRecord

logger = get_logger("stores.pgvector")

_RISK_ORDER = [RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL]

# `<=>` is cosine distance; similarity is 1 minus it. `<#>` is negative inner
# product and `<->` is L2, both of which are already "smaller is better", so
# they are negated to give the higher-is-better score ToolBroker ranks on.
_OPERATORS = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
_HNSW_OPS = {"cosine": "vector_cosine_ops", "l2": "vector_l2_ops", "ip": "vector_ip_ops"}


class PgVectorStore:
    """Stores tool records in a Postgres table with a pgvector column."""

    def __init__(
        self,
        dim: int,
        *,
        dsn: str | None = None,
        connection: Any | None = None,
        pool: Any | None = None,
        table: str = "toolbroker_tools",
        schema: str = "public",
        metric: str = "cosine",
        create_table: bool = True,
        recreate: bool = False,
        **connect_kwargs: Any,
    ) -> None:
        """Connect and ensure the table exists.

        Args:
            dim: Vector width. Must match the embedder.
            dsn: Connection string. Ignored when ``connection`` or ``pool`` is
                given.
            connection: An existing ``psycopg.Connection`` to reuse.
            pool: A ``psycopg_pool.ConnectionPool``, so the catalogue shares
                your application's pool instead of opening its own.
            table: Table name.
            schema: Schema name.
            metric: ``cosine`` (default), ``l2``, or ``ip``.
            create_table: Create the table and indexes if absent. Turn off when
                migrations are managed elsewhere.
            recreate: Drop and rebuild the table. Destroys data.
            **connect_kwargs: Passed to ``psycopg.connect``.

        Raises:
            StoreError: If nothing to connect with was supplied, or an existing
                table was built for a different vector width.
        """
        if dim <= 0:
            raise ValueError("dim must be positive")
        if metric not in _OPERATORS:
            raise ValueError(f"metric must be one of {sorted(_OPERATORS)}")
        if connection is None and pool is None and not dsn:
            raise StoreError("PgVectorStore needs one of dsn, connection, or pool")

        self._dim = dim
        self._metric = metric
        self._table = sql.Identifier(schema, table)
        self._table_name = f"{schema}.{table}"
        self._pool = pool
        self._external = connection
        self._dsn = dsn
        self._connect_kwargs = connect_kwargs
        self._own: psycopg.Connection[Any] | None = None
        self._lock = threading.RLock()

        if recreate:
            self.drop()
        if create_table:
            self._create()
        self._verify_dimension()

    # -- connection -------------------------------------------------------

    def _connection(self) -> Any:
        """Return a usable connection, opening one lazily if we own it."""
        if self._external is not None:
            return self._external
        with self._lock:
            if self._own is None or self._own.closed:
                self._own = psycopg.connect(
                    self._dsn or "", autocommit=True, **self._connect_kwargs
                )
                _prepare(self._own)
            return self._own

    class _CursorContext:
        """Yields a cursor from either a pool or a long-lived connection."""

        def __init__(self, store: PgVectorStore) -> None:
            self._store = store
            self._pooled: Any | None = None
            self._cursor: Any | None = None

        def __enter__(self) -> Any:
            if self._store._pool is not None:
                self._pooled = self._store._pool.connection()
                connection = self._pooled.__enter__()
                _prepare(connection)
            else:
                connection = self._store._connection()
            self._cursor = connection.cursor()
            return self._cursor

        def __exit__(self, *exc_info: Any) -> None:
            if self._cursor is not None:
                self._cursor.close()
            if self._pooled is not None:
                self._pooled.__exit__(*exc_info)

    def _cursor(self) -> _CursorContext:
        """Open a cursor, from the pool if one was supplied."""
        return PgVectorStore._CursorContext(self)

    def close(self) -> None:
        """Close the connection this store opened, if it opened one.

        A connection or pool you passed in is left alone: this store did not
        open it and has no business deciding when it ends.
        """
        with self._lock:
            if self._own is not None and not self._own.closed:
                self._own.close()
            self._own = None

    def __enter__(self) -> PgVectorStore:
        """Return self, so the store can be used as a context manager."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close any connection this store opened."""
        self.close()

    # -- schema -----------------------------------------------------------

    def _create(self) -> None:
        """Create the table and its indexes."""
        with self._cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        tool_id     text PRIMARY KEY,
                        namespace   text NOT NULL,
                        tags        text[] NOT NULL DEFAULT '{{}}',
                        risk_level  smallint NOT NULL DEFAULT 0,
                        doc         text NOT NULL DEFAULT '',
                        tool        jsonb NOT NULL,
                        embedding   vector({dim}) NOT NULL
                    )
                    """
                ).format(table=self._table, dim=sql.Literal(self._dim))
            )
            base = self._table_name.replace(".", "_")
            # HNSW rather than IVFFlat: no training step, and IVFFlat performs
            # silently badly until there are enough rows to build it properly —
            # which a tool catalogue may never reach.
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw (embedding {ops})"
                ).format(
                    name=sql.Identifier(f"{base}_embedding_idx"),
                    table=self._table,
                    ops=sql.SQL(_HNSW_OPS[self._metric]),
                )
            )
            for column, method in (("tags", "gin"), ("namespace", "btree")):
                cursor.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {name} ON {table} USING {method} ({column})"
                    ).format(
                        name=sql.Identifier(f"{base}_{column}_idx"),
                        table=self._table,
                        method=sql.SQL(method),
                        column=sql.Identifier(column),
                    )
                )

    def drop(self) -> None:
        """Drop the table and its indexes. Destroys data.

        The counterpart to ``create_table``: a caller that provisioned a table
        should be able to remove it without reaching for SQL.
        """
        with self._cursor() as cursor:
            cursor.execute(sql.SQL("DROP TABLE IF EXISTS {table}").format(table=self._table))

    def _verify_dimension(self) -> None:
        """Refuse a table built for a different embedder.

        Querying it would return confident nonsense, which is worse than
        failing at construction.
        """
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname || '.' || c.relname = %s AND a.attname = 'embedding'
                """,
                (self._table_name,),
            )
            row = cursor.fetchone()
        if row and isinstance(row[0], int) and row[0] > 0 and row[0] != self._dim:
            raise StoreError(
                f"table {self._table_name!r} holds {row[0]}-dimensional vectors but this "
                f"store expects {self._dim}. It was indexed with a different embedder; "
                "use a different table or pass recreate=True."
            )

    # -- introspection ----------------------------------------------------

    @property
    def dim(self) -> int:
        """Vector width this store was built for."""
        return self._dim

    @property
    def table(self) -> str:
        """Fully qualified table name."""
        return self._table_name

    def __len__(self) -> int:
        """Number of stored records."""
        with self._cursor() as cursor:
            cursor.execute(sql.SQL("SELECT count(*) FROM {table}").format(table=self._table))
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def __repr__(self) -> str:
        """Show the table and dimensionality."""
        return f"PgVectorStore(table={self._table_name!r}, dim={self._dim})"

    # -- writes -----------------------------------------------------------

    def upsert(self, records: Sequence[ToolRecord]) -> None:
        """Insert or replace records, keyed by tool id."""
        if not records:
            return
        rows = []
        for record in records:
            if len(record.vector) != self._dim:
                raise DimensionMismatchError(self._dim, len(record.vector))
            tool = record.tool
            rows.append(
                (
                    tool.id,
                    tool.namespace,
                    sorted(tool.tags),
                    _RISK_ORDER.index(tool.risk),
                    record.text,
                    json.dumps(tool.model_dump(mode="json")),
                    _literal(record.vector),
                )
            )

        statement = sql.SQL(
            """
            INSERT INTO {table} (tool_id, namespace, tags, risk_level, doc, tool, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (tool_id) DO UPDATE SET
                namespace  = EXCLUDED.namespace,
                tags       = EXCLUDED.tags,
                risk_level = EXCLUDED.risk_level,
                doc        = EXCLUDED.doc,
                tool       = EXCLUDED.tool,
                embedding  = EXCLUDED.embedding
            """
        ).format(table=self._table)
        with self._cursor() as cursor:
            cursor.executemany(statement, rows)

    def delete(self, tool_ids: Sequence[str]) -> int:
        """Remove records by tool id, returning how many existed."""
        if not tool_ids:
            return 0
        with self._cursor() as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {table} WHERE tool_id = ANY(%s)").format(table=self._table),
                (list(tool_ids),),
            )
            return int(cursor.rowcount or 0)

    def clear(self) -> None:
        """Remove every record, keeping the table and its indexes."""
        with self._cursor() as cursor:
            cursor.execute(sql.SQL("TRUNCATE {table}").format(table=self._table))

    # -- reads ------------------------------------------------------------

    def get(self, tool_id: str) -> ToolRecord | None:
        """Return one record by tool id."""
        with self._cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT tool, doc, embedding::text FROM {table} WHERE tool_id = %s").format(
                    table=self._table
                ),
                (tool_id,),
            )
            row = cursor.fetchone()
        return None if row is None else _to_record(row)

    def all_records(self) -> Sequence[ToolRecord]:
        """Return every record, ordered by tool id."""
        with self._cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT tool, doc, embedding::text FROM {table} ORDER BY tool_id").format(
                    table=self._table
                )
            )
            rows = cursor.fetchall()
        return tuple(_to_record(row) for row in rows)

    def search(self, vector: Vector, k: int, filters: Filters | None = None) -> list[Hit]:
        """Return the ``k`` highest-scoring records that satisfy ``filters``."""
        if k <= 0:
            return []
        if len(vector) != self._dim:
            raise DimensionMismatchError(self._dim, len(vector))

        where, params = build_where(filters)
        operator = _OPERATORS[self._metric]
        score = (
            sql.SQL("1 - (embedding <=> %s::vector)")
            if self._metric == "cosine"
            else sql.SQL("-(embedding {op} %s::vector)").format(op=sql.SQL(operator))
        )
        statement = sql.SQL(
            "SELECT tool, {score} AS score FROM {table} {where} "
            "ORDER BY embedding {op} %s::vector LIMIT %s"
        ).format(
            score=score,
            table=self._table,
            where=where,
            op=sql.SQL(operator),
        )

        literal = _literal(vector)
        with self._cursor() as cursor:
            cursor.execute(statement, (literal, *params, literal, k))
            rows = cursor.fetchall()

        hits = [
            Hit(tool=_to_tool(row[0]), score=float(row[1]), components={"vector": float(row[1])})
            for row in rows
        ]
        return stable_sort(hits)[:k]


def _prepare(connection: Any) -> None:
    """Make a connection ready to speak vector.

    The extension has to exist *before* the type adapter is registered:
    ``register_vector`` looks the type up and fails outright if it is absent.
    Doing it the other way round works on any database where someone already
    ran ``CREATE EXTENSION`` by hand, and breaks on exactly the case that
    matters — a fresh one.
    """
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if not connection.autocommit:
        connection.commit()
    register_vector(connection)


def _literal(vector: Vector) -> str:
    """Render a vector in the text form pgvector accepts."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def _to_tool(payload: Any) -> Tool:
    """Rebuild a tool from the stored jsonb."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise StoreError("stored row is missing its tool payload")
    return Tool.model_validate(data)


def _to_record(row: Sequence[Any]) -> ToolRecord:
    """Rebuild a full record from a selected row."""
    tool, doc, embedding = row[0], row[1], row[2]
    values = tuple(float(part) for part in str(embedding).strip("[]").split(",") if part)
    return ToolRecord(tool=_to_tool(tool), text=str(doc or ""), vector=values)


def build_where(filters: Filters | None) -> tuple[sql.Composable, list[Any]]:
    """Translate ToolBroker filters into a ``WHERE`` clause and its parameters.

    Returns an empty clause when nothing is constrained, so Postgres plans the
    query without a redundant predicate.
    """
    if filters is None or filters.is_empty():
        return sql.SQL(""), []

    clauses: list[sql.Composable] = []
    params: list[Any] = []

    # An empty allow-set matches nothing, the same way Filters.matches does.
    # Dropping the clause would silently widen the query to everything, which
    # is the dangerous direction for something used as a guard rail.
    if (
        (filters.namespaces is not None and not filters.namespaces)
        or (filters.tool_ids is not None and not filters.tool_ids)
        or (filters.tags_any is not None and not filters.tags_any)
    ):
        return sql.SQL("WHERE false"), []

    if filters.namespaces is not None:
        clauses.append(sql.SQL("namespace = ANY(%s)"))
        params.append(sorted(filters.namespaces))

    if filters.tool_ids is not None:
        clauses.append(sql.SQL("tool_id = ANY(%s)"))
        params.append(sorted(filters.tool_ids))

    if filters.exclude_tool_ids:
        clauses.append(sql.SQL("NOT (tool_id = ANY(%s))"))
        params.append(sorted(filters.exclude_tool_ids))

    if filters.tags_any is not None:
        clauses.append(sql.SQL("tags && %s"))
        params.append(sorted(filters.tags_any))

    if filters.tags_all is not None:
        clauses.append(sql.SQL("tags @> %s"))
        params.append(sorted(filters.tags_all))

    if filters.exclude_tags:
        clauses.append(sql.SQL("NOT (tags && %s)"))
        params.append(sorted(filters.exclude_tags))

    if filters.max_risk is not None:
        clauses.append(sql.SQL("risk_level <= %s"))
        params.append(_RISK_ORDER.index(filters.max_risk))

    if not clauses:
        return sql.SQL(""), []
    return sql.SQL("WHERE ") + sql.SQL(" AND ").join(clauses), params
