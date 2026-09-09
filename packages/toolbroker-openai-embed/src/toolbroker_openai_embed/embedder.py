"""An :class:`~toolbroker.protocols.Embedder` backed by the OpenAI API.

Unlike the local embedders, every call here costs money and adds latency, which
shapes two decisions:

* **Dimension is looked up, not probed.** Known models have known widths, so
  construction makes no billable request. Only an unrecognised model falls back
  to a probe.
* **Batches are large by default.** The API accepts many inputs per request and
  the round trip dominates, so a bigger batch is strictly better until the token
  ceiling.

Wrap this in :class:`~toolbroker.index.embedders.CachedEmbedder` with
``persist=True`` unless you enjoy paying to re-embed unchanged tools.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from toolbroker.errors import EmbeddingError

DEFAULT_MODEL = "text-embedding-3-small"

# Published widths, so construction costs nothing. An unknown model is probed.
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Only the v3 models support Matryoshka truncation via `dimensions`.
SUPPORTS_DIMENSIONS = ("text-embedding-3-",)

PROBE_TEXT = "dimension probe"


def _require_openai() -> Any:
    """Import the OpenAI SDK with an actionable message when it is missing."""
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - depends on install
        raise EmbeddingError(
            "OpenAIEmbedder requires the openai package. "
            "Install with: pip install toolbroker-openai-embed"
        ) from exc
    return openai


class OpenAIEmbedder:
    """Embeds text with an OpenAI embedding model."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        dimensions: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        batch_size: int = 256,
        timeout: float | None = None,
        max_retries: int = 3,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Create a client and determine the output width.

        Args:
            model: Embedding model name.
            dimensions: Truncate to this width. Supported by the
                ``text-embedding-3-*`` models; a smaller index for a little
                accuracy.
            api_key: Overrides ``OPENAI_API_KEY``.
            base_url: Point at a compatible gateway or proxy.
            organization: OpenAI organization id.
            batch_size: Inputs per request.
            timeout: Per-request timeout in seconds.
            max_retries: Retries the SDK performs on transient failures.
            client: An already-configured ``OpenAI`` client, so a host
                application can share one connection pool and its own retry
                policy.
            **client_kwargs: Passed through to the client constructor.

        Raises:
            EmbeddingError: If ``dimensions`` is set on a model that does not
                support it, or the width cannot be determined.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if dimensions is not None:
            if dimensions <= 0:
                raise ValueError("dimensions must be positive")
            if not model.startswith(SUPPORTS_DIMENSIONS):
                raise EmbeddingError(
                    f"model {model!r} does not support the 'dimensions' parameter; "
                    f"only {', '.join(SUPPORTS_DIMENSIONS)}* models do"
                )

        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = (
            client
            if client is not None
            else self._build_client(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                timeout=timeout,
                max_retries=max_retries,
                **client_kwargs,
            )
        )
        self._dim = dimensions or self._lookup_dim()

    def _build_client(self, **kwargs: Any) -> Any:
        """Construct the SDK client, dropping unset optional arguments."""
        openai = _require_openai()
        return openai.OpenAI(**{k: v for k, v in kwargs.items() if v is not None})

    def _lookup_dim(self) -> int:
        """Return the model's width, probing only if it is unrecognised.

        A lookup keeps construction free; probing an unknown model costs one
        small billable request, which is better than guessing wrong.
        """
        known = MODEL_DIMENSIONS.get(self._model)
        if known is not None:
            return known
        vectors = self._request([PROBE_TEXT])
        if not vectors or not vectors[0]:
            raise EmbeddingError(f"model {self._model!r} returned no embedding")
        return len(vectors[0])

    # -- introspection ----------------------------------------------------

    @property
    def dim(self) -> int:
        """Vector width."""
        return self._dim

    @property
    def id(self) -> str:
        """Cache key identifying the model and requested width.

        Width is part of the key: a cache built at 1536 must never be reused at
        256, or the store would hold vectors of two different shapes.
        """
        return f"openai:{self._model}" + (f"@{self._dimensions}" if self._dimensions else "")

    @property
    def model(self) -> str:
        """The embedding model name."""
        return self._model

    @property
    def client(self) -> Any:
        """The underlying OpenAI client."""
        return self._client

    def __repr__(self) -> str:
        """Show the model and width."""
        return f"{type(self).__name__}({self._model!r}, dim={self._dim})"

    # -- embedding --------------------------------------------------------

    def _request(self, texts: Sequence[str]) -> list[list[float]]:
        """Send one batch and return its vectors, in input order."""
        payload: dict[str, Any] = {"model": self._model, "input": list(texts)}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions

        try:
            response = self._client.embeddings.create(**payload)
        except Exception as exc:
            raise EmbeddingError(
                f"OpenAI embeddings request failed for model {self._model!r}: {exc}"
            ) from exc

        data = list(getattr(response, "data", []) or [])
        if len(data) != len(texts):
            raise EmbeddingError(f"OpenAI returned {len(data)} embeddings for {len(texts)} inputs")
        # The API documents that results may arrive out of order, so they are
        # sorted by index rather than trusted positionally.
        ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
        return [[float(value) for value in item.embedding] for item in ordered]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        if not texts:
            return []
        # The API rejects an empty string; a tool with no description is
        # perfectly legal, so substitute a single space rather than failing.
        prepared = [text if text else " " for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self._batch_size):
            vectors.extend(self._request(prepared[start : start + self._batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Symmetric with :meth:`embed` for these models."""
        return self.embed([text])[0]


class AzureOpenAIEmbedder(OpenAIEmbedder):
    """Embeds text with an Azure OpenAI deployment.

    Azure addresses a *deployment*, not a model, and the deployment name is
    chosen by whoever provisioned it — so the width cannot be looked up and is
    probed unless declared.
    """

    def __init__(
        self,
        deployment: str,
        *,
        azure_endpoint: str | None = None,
        api_version: str = "2024-02-01",
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an Azure client for ``deployment``.

        Args:
            deployment: The Azure deployment name.
            azure_endpoint: Resource endpoint; falls back to
                ``AZURE_OPENAI_ENDPOINT``.
            api_version: Azure API version.
            api_key: Falls back to ``AZURE_OPENAI_API_KEY``.
            azure_ad_token: Entra ID token, as an alternative to a key.
            dimensions: Declare the width to skip the probe request.
            **kwargs: Forwarded to :class:`OpenAIEmbedder`.
        """
        self._azure = {
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
            "api_key": api_key,
            "azure_ad_token": azure_ad_token,
            "azure_deployment": deployment,
        }
        # Azure deployment names are arbitrary, so the dimensions guard in the
        # base class cannot tell whether truncation is supported. Bypass it and
        # let the service decide.
        self._deployment = deployment
        super().__init__(
            model=deployment,
            dimensions=None,
            **kwargs,
        )
        if dimensions is not None:
            self._dimensions = dimensions
            self._dim = dimensions

    def _build_client(self, **kwargs: Any) -> Any:
        """Construct an ``AzureOpenAI`` client."""
        openai = _require_openai()
        merged = {
            **{k: v for k, v in kwargs.items() if v is not None},
            **{k: v for k, v in self._azure.items() if v is not None},
        }
        return openai.AzureOpenAI(**merged)

    @property
    def id(self) -> str:
        """Cache key identifying the deployment and requested width."""
        return f"azure-openai:{self._deployment}" + (
            f"@{self._dimensions}" if self._dimensions else ""
        )
