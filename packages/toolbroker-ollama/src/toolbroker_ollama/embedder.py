"""An :class:`~toolbroker.protocols.Embedder` backed by a local Ollama server.

Local models with no API key and no data leaving the machine, which is often
the actual reason a team cannot put an internal tool catalogue through a hosted
embedder.

Ollama's embed endpoint is one JSON POST, so this uses the standard library and
adds no HTTP dependency. Anyone who wants pooling, retries, or custom auth
injects a :data:`Transport` instead.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

from toolbroker.errors import EmbeddingError

#: Called as ``transport(url, payload, timeout)`` and returns the decoded JSON.
Transport: TypeAlias = Callable[[str, dict[str, Any], float], dict[str, Any]]

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"
PROBE_TEXT = "dimension probe"


def urllib_transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST ``payload`` as JSON and decode the response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise EmbeddingError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError(
            f"cannot reach Ollama at {url}: {exc.reason}. "
            "Is it running? Start it with `ollama serve`."
        ) from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise EmbeddingError(f"Ollama returned a non-JSON response: {exc}") from exc
    if not isinstance(decoded, dict):
        raise EmbeddingError(f"Ollama returned {type(decoded).__name__}, expected an object")
    return decoded


class OllamaEmbedder:
    """Embeds text with a model served by Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str = DEFAULT_HOST,
        dim: int | None = None,
        batch_size: int = 32,
        timeout: float = 120.0,
        query_prefix: str = "",
        document_prefix: str = "",
        transport: Transport | None = None,
    ) -> None:
        """Connect to Ollama and determine the model's output width.

        Args:
            model: Ollama model tag, e.g. ``nomic-embed-text`` or
                ``qwen3-embedding:4b``. It must already be pulled.
            host: Base URL of the Ollama server.
            dim: Skip dimension discovery by declaring the width up front.
            batch_size: Texts per request. Ollama loads the whole batch into
                memory, so very large batches trade throughput for RAM.
            timeout: Per-request timeout in seconds. Generous by default: a
                cold model has to load before it can embed anything.
            query_prefix: Prepended to queries. Asymmetric models such as the
                E5 family need ``"query: "`` here and ``"passage: "`` below;
                omitting them costs real accuracy on those models.
            document_prefix: Prepended to documents.
            transport: Replace the stdlib HTTP call, for pooling or auth.

        Raises:
            EmbeddingError: If the server is unreachable or the model is absent.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model = model
        self._host = host.rstrip("/")
        self._batch_size = batch_size
        self._timeout = timeout
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._transport = transport or urllib_transport
        self._dim = dim if dim is not None else self._probe_dim()

    # -- introspection ----------------------------------------------------

    @property
    def dim(self) -> int:
        """Vector width of the loaded model."""
        return self._dim

    @property
    def id(self) -> str:
        """Cache key identifying the model and any prefixes.

        The prefixes are part of the key: the same model with a different query
        prefix produces different vectors, and reusing a cached embedding across
        that change would silently corrupt the index.
        """
        suffix = ""
        if self._query_prefix or self._document_prefix:
            suffix = f"+{self._query_prefix}|{self._document_prefix}"
        return f"ollama:{self._model}{suffix}"

    @property
    def model(self) -> str:
        """The Ollama model tag."""
        return self._model

    def __repr__(self) -> str:
        """Show the model and width."""
        return f"OllamaEmbedder({self._model!r}, dim={self._dim})"

    # -- embedding --------------------------------------------------------

    def _probe_dim(self) -> int:
        """Determine the output width by embedding one short string."""
        vectors = self._request([PROBE_TEXT])
        if not vectors or not vectors[0]:
            raise EmbeddingError(
                f"Ollama model {self._model!r} returned no embedding. "
                f"Pull it first: `ollama pull {self._model}`"
            )
        return len(vectors[0])

    def _request(self, texts: Sequence[str]) -> list[list[float]]:
        """Send one batch and return its vectors."""
        payload = {"model": self._model, "input": list(texts)}
        decoded = self._transport(f"{self._host}/api/embed", payload, self._timeout)

        if "error" in decoded:
            raise EmbeddingError(f"Ollama error for model {self._model!r}: {decoded['error']}")

        embeddings = decoded.get("embeddings")
        if embeddings is None:
            # The pre-0.3 endpoint returned a single "embedding" instead.
            single = decoded.get("embedding")
            embeddings = [single] if single is not None else None
        if not isinstance(embeddings, list):
            raise EmbeddingError(
                f"Ollama response had no 'embeddings' array (keys: {sorted(decoded)})"
            )
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        return [[float(value) for value in vector] for vector in embeddings]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        if not texts:
            return []
        prepared = [f"{self._document_prefix}{text}" for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self._batch_size):
            vectors.extend(self._request(prepared[start : start + self._batch_size]))
        self._check_widths(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query, applying the query prefix."""
        vectors = self._request([f"{self._query_prefix}{text}"])
        self._check_widths(vectors)
        return vectors[0]

    def _check_widths(self, vectors: Sequence[Sequence[float]]) -> None:
        """Fail loudly if the server changed models underneath us.

        Ollama will happily serve a different model under a re-pulled tag. A
        silent width change corrupts the index instead of erroring, and the
        symptom is unexplainable ranking rather than a stack trace.
        """
        for vector in vectors:
            if len(vector) != self._dim:
                raise EmbeddingError(
                    f"Ollama model {self._model!r} returned a {len(vector)}-dimensional "
                    f"vector but this embedder was built for {self._dim}. The model "
                    "behind that tag has changed; re-index against the new one."
                )
