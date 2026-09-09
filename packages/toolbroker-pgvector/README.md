# toolbroker-pgvector

PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) store for
[ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
pip install toolbroker-pgvector
```

```python
from toolbroker import ToolBroker
from toolbroker_pgvector import PgVectorStore

broker = ToolBroker(
    store=PgVectorStore(
        dim=384,
        dsn="postgresql://user:pass@localhost/mydb",
    )
)
```

Or declaratively:

```yaml
store:
  name: pgvector
  options:
    dsn: "postgresql://user:pass@localhost/mydb"
    table: agent_tools
```

The reason to pick this one is that you already run Postgres. A tool catalogue is small —
hundreds to low thousands of rows — and not worth a second piece of infrastructure with
its own backups, upgrades, and on-call rota.

## Filters run in SQL

Namespace, tags, risk ceiling, and id filters become `WHERE` clauses, so narrowing
happens **before** ranking and `k` keeps meaning what the caller expects. Tags use a
`text[]` column with a GIN index; risk is stored as an ordinal so a ceiling is a range
scan rather than an enumeration.

## Indexing

An HNSW index is created on first use. It is good for the catalogue sizes this library
targets and needs no training step, unlike IVFFlat, which silently performs badly until
you have enough rows to build it properly.

## Connections

Pass a `dsn`, or hand in your own `connection` or `psycopg_pool` so the catalogue shares
your application's pool rather than opening its own.

## Conformance

Passes `toolbroker.testing.StoreConformanceSuite` unmodified:

```bash
docker run -d -e POSTGRES_PASSWORD=toolbroker -e POSTGRES_DB=toolbroker \
  -p 55432:5432 pgvector/pgvector:pg16
TOOLBROKER_PG_DSN=postgresql://postgres:toolbroker@localhost:55432/toolbroker \
  uv run pytest packages/toolbroker-pgvector
```

Without `TOOLBROKER_PG_DSN` the suite skips rather than failing, so a contributor without
Docker is not blocked.
