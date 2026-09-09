"""Ollama embedder: protocol behaviour, wire format, and error handling."""

from __future__ import annotations

import pytest
from toolbroker_ollama import OllamaEmbedder

from toolbroker.errors import EmbeddingError

DIM = 8


class FakeTransport:
    """Deterministic stand-in for the HTTP call.

    Lets the wire format and error paths be tested without a server, which is
    what makes this package's CI job runnable anywhere.
    """

    def __init__(self, dim: int = DIM, response=None, error: Exception | None = None):
        self.dim = dim
        self.calls: list[tuple[str, dict, float]] = []
        self._response = response
        self._error = error

    def __call__(self, url: str, payload: dict, timeout: float) -> dict:
        self.calls.append((url, payload, timeout))
        if self._error is not None:
            raise self._error
        if self._response is not None:
            return self._response
        texts = payload["input"]
        return {
            "embeddings": [
                [float((len(text) + index) % 7) for index in range(self.dim)] for text in texts
            ]
        }

    @property
    def inputs(self) -> list[list[str]]:
        return [call[1]["input"] for call in self.calls]


def make(**kwargs) -> tuple[OllamaEmbedder, FakeTransport]:
    transport = kwargs.pop("transport", None) or FakeTransport()
    return OllamaEmbedder(model="fake-model", transport=transport, **kwargs), transport


# -- construction ---------------------------------------------------------


def test_dimension_is_probed_at_construction():
    embedder, transport = make()
    assert embedder.dim == DIM
    assert transport.inputs == [["dimension probe"]]


def test_declared_dimension_skips_the_probe():
    transport = FakeTransport()
    embedder = OllamaEmbedder(model="fake-model", dim=99, transport=transport)
    assert embedder.dim == 99
    assert transport.calls == []


def test_id_includes_the_model():
    embedder, _ = make()
    assert embedder.id == "ollama:fake-model"


def test_id_includes_prefixes_because_they_change_the_vectors():
    # Reusing a cached embedding across a prefix change would silently corrupt
    # the index.
    embedder, _ = make(query_prefix="query: ", document_prefix="passage: ")
    assert embedder.id != "ollama:fake-model"
    assert "query: " in embedder.id


def test_host_trailing_slash_is_normalised():
    transport = FakeTransport()
    OllamaEmbedder(model="m", host="http://localhost:11434/", transport=transport)
    assert transport.calls[0][0] == "http://localhost:11434/api/embed"


def test_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        make(batch_size=0)


# -- request shaping ------------------------------------------------------


def test_model_and_inputs_are_sent():
    embedder, transport = make()
    embedder.embed(["a", "b"])
    url, payload, _ = transport.calls[-1]
    assert url.endswith("/api/embed")
    assert payload == {"model": "fake-model", "input": ["a", "b"]}


def test_batches_are_split():
    embedder, transport = make(batch_size=2)
    embedder.embed(["a", "b", "c", "d", "e"])
    assert transport.inputs[1:] == [["a", "b"], ["c", "d"], ["e"]]


def test_batching_preserves_order():
    embedder, _ = make(batch_size=2)
    texts = ["x", "yy", "zzz", "wwww"]
    batched = embedder.embed(texts)
    individually = [embedder.embed([text])[0] for text in texts]
    assert batched == individually


def test_empty_batch_makes_no_request():
    embedder, transport = make()
    before = len(transport.calls)
    assert embedder.embed([]) == []
    assert len(transport.calls) == before


def test_document_prefix_is_applied():
    embedder, transport = make(document_prefix="passage: ")
    embedder.embed(["hello"])
    assert transport.inputs[-1] == ["passage: hello"]


def test_query_prefix_is_applied():
    embedder, transport = make(query_prefix="query: ")
    embedder.embed_query("hello")
    assert transport.inputs[-1] == ["query: hello"]


def test_timeout_is_passed_through():
    embedder, transport = make(timeout=5.0)
    embedder.embed(["a"])
    assert transport.calls[-1][2] == 5.0


# -- response handling ----------------------------------------------------


def test_legacy_single_embedding_response_is_accepted():
    transport = FakeTransport(response={"embedding": [1.0, 2.0, 3.0]})
    embedder = OllamaEmbedder(model="old", transport=transport)
    assert embedder.dim == 3


def test_server_error_field_is_surfaced():
    transport = FakeTransport(response={"error": 'model "nope" not found'})
    with pytest.raises(EmbeddingError, match="not found"):
        OllamaEmbedder(model="nope", transport=transport)


def test_missing_embeddings_key_lists_what_came_back():
    transport = FakeTransport(response={"unexpected": 1})
    with pytest.raises(EmbeddingError, match="unexpected"):
        OllamaEmbedder(model="m", transport=transport)


def test_count_mismatch_is_rejected():
    transport = FakeTransport()
    embedder = OllamaEmbedder(model="fake-model", dim=DIM, transport=transport)
    transport._response = {"embeddings": [[0.0] * DIM]}
    with pytest.raises(EmbeddingError, match="1 embeddings for 2 inputs"):
        embedder.embed(["a", "b"])


def test_empty_embedding_names_the_pull_command():
    transport = FakeTransport(response={"embeddings": [[]]})
    with pytest.raises(EmbeddingError, match="ollama pull"):
        OllamaEmbedder(model="missing", transport=transport)


def test_width_change_is_caught_rather_than_corrupting_the_index():
    # Ollama will serve a different model under a re-pulled tag. Silently
    # accepting a new width produces unexplainable ranking, not a stack trace.
    transport = FakeTransport()
    embedder = OllamaEmbedder(model="fake-model", dim=DIM, transport=transport)
    transport.dim = DIM * 2
    with pytest.raises(EmbeddingError, match="the new one"):
        embedder.embed(["a"])


def test_unreachable_server_error_propagates():
    import urllib.error

    transport = FakeTransport(error=urllib.error.URLError("connection refused"))
    with pytest.raises(urllib.error.URLError):
        OllamaEmbedder(model="m", transport=transport)


def test_real_transport_says_how_to_start_the_server():
    # The default transport is where the "is it running?" hint lives; an
    # injected fake bypasses it, so it needs its own test.
    from toolbroker_ollama.embedder import urllib_transport

    with pytest.raises(EmbeddingError, match="ollama serve"):
        urllib_transport("http://127.0.0.1:1/api/embed", {"model": "m", "input": ["x"]}, 1.0)


def test_registry_resolution():
    from toolbroker.registry import GROUP_EMBEDDERS, available, resolve

    assert "ollama" in available(GROUP_EMBEDDERS)
    assert resolve(GROUP_EMBEDDERS, "ollama") is OllamaEmbedder
