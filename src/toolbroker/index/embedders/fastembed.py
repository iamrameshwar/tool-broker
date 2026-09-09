"""fastembed-backed embedder.

The recommended default once a user is past the first five minutes: real
semantic matching, local inference, no API key. Costs a one-time model
download, which is why it is not the zero-config path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...errors import EmbeddingError

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    """Local ONNX embeddings via the ``fastembed`` package."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        batch_size: int = 64,
        **model_kwargs: Any,
    ) -> None:
        """Load ``model_name``, downloading it on first use."""
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise EmbeddingError(
                "FastEmbedEmbedder requires the 'fastembed' package. "
                "Install with: pip install 'toolbroker[fastembed]'"
            ) from exc

        self._model_name = model_name
        self._batch_size = batch_size
        try:
            self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, **model_kwargs)
        except Exception as exc:
            raise EmbeddingError(f"could not load fastembed model {model_name!r}: {exc}") from exc
        self._dim = self._probe_dim()

    def _probe_dim(self) -> int:
        """Determine output width, preferring metadata over a live call."""
        try:
            for description in type(self._model).list_supported_models():
                if description.get("model") == self._model_name:
                    dim = description.get("dim")
                    if isinstance(dim, int):
                        return dim
        except Exception:  # pragma: no cover - metadata shape varies by version
            pass
        return len(next(iter(self._model.embed(["dimension probe"]))))

    @property
    def dim(self) -> int:
        """Vector width of the loaded model."""
        return self._dim

    @property
    def id(self) -> str:
        """Cache key identifying the model."""
        return f"fastembed:{self._model_name}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        if not texts:
            return []
        try:
            return [
                [float(value) for value in vector]
                for vector in self._model.embed(list(texts), batch_size=self._batch_size)
            ]
        except Exception as exc:
            raise EmbeddingError(f"fastembed failed to embed {len(texts)} texts: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query, using the model's query prefix when it has one."""
        query_embed = getattr(self._model, "query_embed", None)
        if callable(query_embed):
            try:
                return [float(value) for value in next(iter(query_embed(text)))]
            except Exception:  # pragma: no cover - fall back to symmetric path
                pass
        return self.embed([text])[0]

    def __repr__(self) -> str:
        """Show the model name."""
        return f"FastEmbedEmbedder({self._model_name!r}, dim={self._dim})"
