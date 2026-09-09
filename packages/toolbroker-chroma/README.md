# toolbroker-chroma

[Chroma](https://www.trychroma.com) vector store for
[ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
pip install toolbroker-chroma
```

```python
from toolbroker import ToolBroker
from toolbroker_chroma import ChromaStore

broker = ToolBroker(store=ChromaStore(dim=384, path="./chroma"))  # on disk
broker = ToolBroker(store=ChromaStore(dim=384))  # ephemeral
broker = ToolBroker(store=ChromaStore(dim=384, host="localhost", port=8000))
```

Or declaratively:

```yaml
store:
  name: chroma
  options: {path: "./chroma", collection: "agent-tools"}
```

## Tag encoding

Chroma metadata holds only scalars, so a tool's tags cannot be stored as a list. Each
tag becomes its own boolean key (`tag_<name>: true`). That is not just a workaround —
it is what lets every ToolBroker filter, including tag exclusion, run **server-side
before scoring**, so `k` keeps meaning what the caller expects.

## Conformance

This package passes `toolbroker.testing.StoreConformanceSuite` unmodified:

```bash
uv run pytest packages/toolbroker-chroma
```
