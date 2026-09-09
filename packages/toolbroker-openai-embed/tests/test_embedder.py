"""OpenAI embedder: request shaping, ordering, and cost-shaped decisions.

No API key is used. A fake client stands in for the SDK, which is enough to
pin the wire format and every error path — and is the only way to test a paid
API in CI without a bill.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from toolbroker_openai_embed import AzureOpenAIEmbedder, OpenAIEmbedder

from toolbroker.errors import EmbeddingError

DIM = 4


@dataclass
class FakeItem:
    index: int
    embedding: list[float]


@dataclass
class FakeResponse:
    data: list[FakeItem]


class FakeEmbeddings:
    def __init__(self, owner: FakeClient):
        self._owner = owner

    def create(self, **payload):
        self._owner.calls.append(payload)
        if self._owner.error is not None:
            raise self._owner.error
        if self._owner.response is not None:
            return self._owner.response
        width = payload.get("dimensions") or self._owner.dim
        items = [
            FakeItem(index=i, embedding=[float(len(str(text)) + j) for j in range(width)])
            for i, text in enumerate(payload["input"])
        ]
        if self._owner.shuffle:
            items = list(reversed(items))
        return FakeResponse(data=items)


class FakeClient:
    """Stands in for ``openai.OpenAI``."""

    def __init__(self, dim: int = DIM, response=None, error=None, shuffle: bool = False):
        self.dim = dim
        self.response = response
        self.error = error
        self.shuffle = shuffle
        self.calls: list[dict] = []
        self.embeddings = FakeEmbeddings(self)

    @property
    def inputs(self) -> list[list[str]]:
        return [call["input"] for call in self.calls]


def make(**kwargs) -> tuple[OpenAIEmbedder, FakeClient]:
    client = kwargs.pop("client", None) or FakeClient()
    return OpenAIEmbedder(client=client, **kwargs), client


# -- construction ---------------------------------------------------------


def test_known_model_width_is_looked_up_without_a_request():
    # Construction must not cost money.
    embedder, client = make(model="text-embedding-3-small")
    assert embedder.dim == 1536
    assert client.calls == []


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("text-embedding-3-small", 1536),
        ("text-embedding-3-large", 3072),
        ("text-embedding-ada-002", 1536),
    ],
)
def test_published_widths(model, expected):
    embedder, _ = make(model=model)
    assert embedder.dim == expected


def test_unknown_model_is_probed():
    embedder, client = make(model="some-new-model")
    assert embedder.dim == DIM
    assert client.inputs == [["dimension probe"]]


def test_dimensions_truncate_and_skip_the_probe():
    embedder, client = make(model="text-embedding-3-large", dimensions=256)
    assert embedder.dim == 256
    assert client.calls == []


def test_dimensions_on_an_unsupported_model_is_rejected():
    with pytest.raises(EmbeddingError, match="does not support the 'dimensions'"):
        make(model="text-embedding-ada-002", dimensions=256)


def test_dimensions_must_be_positive():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        make(model="text-embedding-3-small", dimensions=0)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        make(batch_size=0)


def test_id_includes_the_model():
    embedder, _ = make(model="text-embedding-3-small")
    assert embedder.id == "openai:text-embedding-3-small"


def test_id_includes_the_width_so_caches_cannot_be_mixed():
    # A cache built at 1536 must never be reused at 256.
    embedder, _ = make(model="text-embedding-3-small", dimensions=256)
    assert embedder.id == "openai:text-embedding-3-small@256"


# -- request shaping ------------------------------------------------------


def test_model_and_inputs_are_sent():
    embedder, client = make(model="text-embedding-3-small")
    embedder.embed(["a", "b"])
    assert client.calls[-1]["model"] == "text-embedding-3-small"
    assert client.calls[-1]["input"] == ["a", "b"]


def test_dimensions_are_sent_when_set():
    embedder, client = make(model="text-embedding-3-small", dimensions=64)
    embedder.embed(["a"])
    assert client.calls[-1]["dimensions"] == 64


def test_dimensions_are_omitted_when_unset():
    embedder, client = make(model="text-embedding-3-small")
    embedder.embed(["a"])
    assert "dimensions" not in client.calls[-1]


def test_batches_are_split():
    embedder, client = make(model="text-embedding-3-small", batch_size=2)
    embedder.embed(["a", "b", "c", "d", "e"])
    assert client.inputs == [["a", "b"], ["c", "d"], ["e"]]


def test_empty_batch_makes_no_request():
    embedder, client = make(model="text-embedding-3-small")
    assert embedder.embed([]) == []
    assert client.calls == []


def test_empty_strings_are_substituted():
    # The API rejects an empty string, but a tool with no description is legal.
    embedder, client = make(model="text-embedding-3-small")
    embedder.embed(["", "x"])
    assert client.calls[-1]["input"] == [" ", "x"]


# -- response handling ----------------------------------------------------


def test_results_are_reordered_by_index():
    # The API documents that results may arrive out of order.
    embedder, _ = make(model="unknown", client=FakeClient(shuffle=True))
    vectors = embedder.embed(["a", "bbbb"])
    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 4.0


def test_count_mismatch_is_rejected():
    client = FakeClient(response=FakeResponse(data=[FakeItem(0, [0.0] * DIM)]))
    embedder = OpenAIEmbedder(model="text-embedding-3-small", client=client)
    with pytest.raises(EmbeddingError, match="1 embeddings for 2 inputs"):
        embedder.embed(["a", "b"])


def test_api_errors_name_the_model():
    client = FakeClient(error=RuntimeError("rate limited"))
    with pytest.raises(EmbeddingError, match="rate limited"):
        OpenAIEmbedder(model="unknown", client=client)


def test_query_and_document_embedding_agree():
    embedder, _ = make(model="text-embedding-3-small")
    assert embedder.embed_query("hello") == embedder.embed(["hello"])[0]


# -- caching integration --------------------------------------------------


def test_cached_embedder_avoids_repeat_billing():
    from toolbroker.index.embedders import CachedEmbedder

    embedder, client = make(model="text-embedding-3-small")
    cached = CachedEmbedder(embedder)
    cached.embed(["a", "b"])
    cached.embed(["a", "b"])
    assert len(client.calls) == 1


# -- azure ----------------------------------------------------------------


def test_azure_uses_the_deployment_as_the_id():
    client = FakeClient()
    embedder = AzureOpenAIEmbedder(deployment="my-deploy", client=client)
    assert embedder.id == "azure-openai:my-deploy"


def test_azure_probes_because_deployment_names_are_arbitrary():
    client = FakeClient()
    embedder = AzureOpenAIEmbedder(deployment="my-deploy", client=client)
    assert embedder.dim == DIM
    assert client.inputs == [["dimension probe"]]


def test_azure_declared_dimensions_skip_the_probe_and_reach_the_id():
    client = FakeClient()
    embedder = AzureOpenAIEmbedder(deployment="d", client=client, dimensions=128)
    assert embedder.dim == 128
    assert embedder.id == "azure-openai:d@128"


def test_azure_accepts_dimensions_the_base_class_would_reject():
    # The base class guards on model name; an Azure deployment name says
    # nothing about which model is behind it, so the service decides.
    embedder = AzureOpenAIEmbedder(deployment="ada-legacy", client=FakeClient(), dimensions=64)
    assert embedder.dim == 64


# -- registry -------------------------------------------------------------


def test_registry_resolution():
    from toolbroker.registry import GROUP_EMBEDDERS, available, resolve

    assert "openai" in available(GROUP_EMBEDDERS)
    assert "azure-openai" in available(GROUP_EMBEDDERS)
    assert resolve(GROUP_EMBEDDERS, "openai") is OpenAIEmbedder
