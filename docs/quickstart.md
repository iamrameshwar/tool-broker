# Quickstart

## Install

```bash
pip install toolbroker
```

The core depends on pydantic and nothing else. It works offline, with no API key.

For real semantic matching — recommended as soon as you are past the first five
minutes:

```bash
pip install 'toolbroker[fastembed]'   # local ONNX embeddings, still no API key
```

!!! note "Why two embedders"
    The default `HashingEmbedder` is a lexical model wearing a vector interface: good
    at *the query words appear in the tool*, bad at *"refund" matching "chargeback"*.
    It is the default because it needs no download and no network, so a first-time user
    is never blocked. `fastembed` downloads a model on first use and is substantially
    more accurate — see [Benchmarking](benchmarking.md) for the gap.

## Twenty lines

```python
from toolbroker import ToolBroker


def search_orders(customer_email: str, limit: int = 10) -> list[str]:
    """Find recent orders placed by a customer."""


def issue_refund(order_id: str, amount_cents: int, reason: str) -> dict:
    """Return money to a customer for a specific order."""


def check_inventory(sku: str) -> int:
    """Return how many units of a SKU are currently in stock."""


broker = ToolBroker()
broker.add_functions([search_orders, issue_refund, check_inventory])
broker.index()

selection = broker.select("customer is angry and wants their money back", k=2)
print(selection.tool_ids)
```

Docstrings become descriptions, type hints become JSON Schema, and the `Args:` section
becomes per-parameter documentation — all of which feed retrieval.

## Understanding a selection

```python
print(selection.explain())
```

```text
query: "customer is angry and wants their money back"
considered 4 tools, requested 2, selected 2

selected:
  1. python/issue_refund  score=0.6660  [vector=0.6660]
  2. python/search_orders  score=0.5193  [vector=0.5193]

excluded:
  - python/check_inventory (retrieval): ranked below the top 2

timings: retrieval=3.61ms, policy=0.06ms
```

When retrieval surprises you, this is the first thing to look at, and usually the whole
diagnosis.

## Handing tools to a model

=== "OpenAI"

    ```python
    tools = broker.openai_tools("book me a flight", k=3)
    response = client.chat.completions.create(model=..., tools=tools, messages=...)
    ```

=== "Anthropic"

    ```python
    tools = broker.anthropic_tools("book me a flight", k=3)
    response = client.messages.create(model=..., tools=tools, messages=...)
    ```

=== "LangGraph"

    ```python
    rendered, selection = broker.select_for("langgraph", "book me a flight", k=3)
    graph = create_react_agent(model, rendered)
    ```

Provider APIs reject `/` in tool names, so `billing/issue_refund` is rendered as
`billing__issue_refund`. Map it back with `adapter.resolve_name(...)`.

## Adding MCP servers

```python
broker.add_mcp_server("github", command="npx", args=["-y", "@modelcontextprotocol/server-github"])
broker.add_mcp_server("internal", url="https://tools.internal/mcp")
broker.index()
```

Requires `pip install 'toolbroker[mcp]'`.

## Adding a REST API

Any OpenAPI 3.x spec becomes a tool catalogue — one operation per tool:

```python
from toolbroker.sources import OpenAPISource

broker.add_source(OpenAPISource("openapi.yaml"))  # or a parsed dict
broker.index()
```

Path, query, and header parameters plus the JSON request body are merged into one flat
schema, because that is what function-calling APIs accept. The mapping back — which
field belongs in the path, which in the body — is preserved in `metadata["http"]`, so a
caller has everything it needs to issue the request. ToolBroker does not issue it.

Two things come free from a well-written spec: **risk is inferred from the HTTP verb**
(`GET` is low, `POST`/`PATCH` medium, `DELETE` high), and **`security` requirements
become `required_scopes`**, so an API that already documents its permissions gets policy
enforcement without any extra configuration.
