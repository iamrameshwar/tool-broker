# toolbroker-openai-embed

OpenAI and Azure OpenAI embeddings for
[ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
pip install toolbroker-openai-embed
export OPENAI_API_KEY=sk-...
```

```python
from toolbroker import ToolBroker
from toolbroker_openai_embed import OpenAIEmbedder

broker = ToolBroker(embedder=OpenAIEmbedder(model="text-embedding-3-small"))
```

Azure:

```python
from toolbroker_openai_embed import AzureOpenAIEmbedder

broker = ToolBroker(
    embedder=AzureOpenAIEmbedder(
        deployment="my-embedding-deployment",
        azure_endpoint="https://my-resource.openai.azure.com",
    )
)
```

Or declaratively:

```yaml
embedder:
  name: openai
  options: {model: text-embedding-3-small, dimensions: 512}
```

## Shrinking the vectors

`text-embedding-3-*` supports Matryoshka truncation, so you can trade a little accuracy
for a much smaller index:

```python
OpenAIEmbedder(model="text-embedding-3-large", dimensions=256)
```

`dimensions` is part of the embedder's `id`, so a cache built at one width is never
reused at another.

## Cost

Every re-index costs money. Wrap it:

```python
from toolbroker.index.embedders import CachedEmbedder

embedder = CachedEmbedder(OpenAIEmbedder(), persist=True)
```

Unchanged tools are then embedded once, ever, rather than once per re-index.
