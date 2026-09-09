from __future__ import annotations

import subprocess
import sys

import pytest

from toolbroker.index.embedders import CachedEmbedder, HashingEmbedder


def test_dimension_is_respected():
    embedder = HashingEmbedder(dim=64)
    assert embedder.dim == 64
    assert len(embedder.embed(["hello world"])[0]) == 64


def test_vectors_are_unit_length():
    vector = HashingEmbedder(dim=64).embed(["send an email to the customer"])[0]
    assert sum(component**2 for component in vector) == pytest.approx(1.0, abs=1e-6)


def test_identical_text_gives_identical_vectors():
    embedder = HashingEmbedder(dim=64)
    assert embedder.embed(["refund"]) == embedder.embed(["refund"])


def test_similar_text_scores_higher_than_unrelated():
    embedder = HashingEmbedder(dim=256)
    query = embedder.embed_query("refund a customer order")
    close = embedder.embed(["issue a refund for a customer order"])[0]
    far = embedder.embed(["compile the kernel from source"])[0]

    def dot(left, right):
        return sum(a * b for a, b in zip(left, right, strict=True))

    assert dot(query, close) > dot(query, far)


def test_hashing_is_stable_across_processes():
    # PYTHONHASHSEED randomisation must not reach the vectors, or a restarted
    # process would silently rank tools differently than the one that indexed.
    script = (
        "from toolbroker.index.embedders import HashingEmbedder;"
        "print(HashingEmbedder(dim=32).embed(['refund the order'])[0][:4])"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1


def test_empty_text_does_not_crash():
    assert len(HashingEmbedder(dim=32).embed([""])[0]) == 32


def test_rejects_tiny_dimensions():
    with pytest.raises(ValueError, match="at least 16"):
        HashingEmbedder(dim=4)


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self._inner = HashingEmbedder(dim=32)

    @property
    def dim(self) -> int:
        return 32

    @property
    def id(self) -> str:
        return "counting"

    def embed(self, texts):
        self.calls += 1
        return self._inner.embed(texts)

    def embed_query(self, text):
        return self._inner.embed_query(text)


def test_cache_avoids_recomputing():
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner)
    cached.embed(["a", "b"])
    cached.embed(["a", "b"])
    assert inner.calls == 1
    assert cached.hits == 2
    assert cached.misses == 2


def test_cache_only_computes_the_missing_entries():
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner)
    first = cached.embed(["a", "b"])
    second = cached.embed(["b", "c", "a"])
    assert inner.calls == 2
    assert second[0] == first[1]
    assert second[2] == first[0]


def test_cache_persists_to_disk(tmp_path):
    inner = CountingEmbedder()
    first = CachedEmbedder(inner, persist=True, directory=tmp_path)
    first.embed(["persisted text"])

    second = CachedEmbedder(CountingEmbedder(), persist=True, directory=tmp_path)
    second.embed(["persisted text"])
    assert second.hits == 1
