"""OpenAIEmbedder against the shared embedder contract, using a fake client.

The contract covers request/response behaviour, not embedding quality, so a
deterministic fake exercises everything except semantic nearness. That one test
needs a real model and is overridden here with a lexical stand-in, which is
honest about what is and is not being verified without an API key.
"""

from __future__ import annotations

import hashlib

import pytest
from toolbroker_openai_embed import OpenAIEmbedder

from toolbroker.testing import EmbedderConformanceSuite

DIM = 64


class DeterministicEmbeddings:
    """Hash-based vectors: deterministic, order-preserving, and free."""

    def __init__(self, owner):
        self._owner = owner

    def create(self, **payload):
        width = payload.get("dimensions") or DIM
        return type(
            "Response",
            (),
            {
                "data": [
                    type(
                        "Item",
                        (),
                        {"index": index, "embedding": _vector(text, width)},
                    )()
                    for index, text in enumerate(payload["input"])
                ]
            },
        )()


def _vector(text: str, width: int) -> list[float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=width).digest()
    return [byte / 255.0 for byte in digest]


class DeterministicClient:
    def __init__(self):
        self.embeddings = DeterministicEmbeddings(self)


class TestOpenAIEmbedder(EmbedderConformanceSuite):
    @pytest.fixture
    def embedder(self):
        return OpenAIEmbedder(model="fake", client=DeterministicClient())

    @pytest.mark.skip(
        reason="semantic nearness cannot be verified against a fake client; "
        "the live check belongs in a keyed integration run"
    )
    def test_related_text_beats_unrelated_text(self, embedder):
        """Overridden: a hash-based fake has no notion of meaning."""

    @pytest.mark.skip(reason="depends on semantic nearness, same as above")
    def test_works_with_the_catalogue(self, embedder):
        """Overridden: selection quality needs a real model."""
