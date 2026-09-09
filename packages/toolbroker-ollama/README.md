# toolbroker-ollama

[Ollama](https://ollama.com) embeddings for
[ToolBroker](https://github.com/iamrameshwar/tool-broker).

```bash
ollama pull nomic-embed-text
pip install toolbroker-ollama
```

```python
from toolbroker import ToolBroker
from toolbroker_ollama import OllamaEmbedder

broker = ToolBroker(embedder=OllamaEmbedder(model="nomic-embed-text"))
```

Or declaratively:

```yaml
embedder:
  name: ollama
  options: {model: nomic-embed-text, host: "http://localhost:11434"}
```

Local models, no API key, and no data leaving the machine — which is often the reason a
team cannot use a hosted embedder for an internal tool catalogue in the first place.

## No HTTP dependency

Ollama's embed endpoint is a single JSON POST, so this package uses the standard library
and adds no dependency beyond ToolBroker itself. If you want connection pooling, retries,
or your own auth, inject a transport:

```python
OllamaEmbedder(model="nomic-embed-text", transport=lambda url, payload, timeout: my_post(...))
```

## Dimension discovery

The vector width depends on the model, so it is discovered by embedding one probe string
at construction. Pass `dim=` to skip that round trip when you already know it.
