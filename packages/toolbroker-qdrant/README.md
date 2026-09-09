# toolbroker-qdrant

[Qdrant](https://qdrant.tech) vector store for [ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
pip install toolbroker-qdrant
```

```python
from toolbroker import ToolBroker
from toolbroker_qdrant import QdrantStore

broker = ToolBroker(store=QdrantStore(dim=384, url="http://localhost:6333"))
```

Or declaratively:

```yaml
store:
  name: qdrant
  options: {url: "http://localhost:6333", collection: "agent_tools"}
```

`location=":memory:"` (the default) runs entirely in-process with no server, which is
what the test suite uses.

Filters are translated into Qdrant's own filter language and applied **server-side
before scoring**, so asking for 5 low-risk tools returns 5 — not whatever survives
filtering the global top 5.

## Conformance

This package passes `toolbroker.testing.StoreConformanceSuite` unmodified:

```bash
uv run pytest packages/toolbroker-qdrant
```
