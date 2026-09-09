"""A deterministic, dependency-free embedder.

This is what makes "works offline in 20 lines" true. fastembed downloads a
model on first use, which is not offline and not instant, so it cannot be the
thing a first-time user hits by default.

The method is signed feature hashing over word unigrams, bigrams, and
character 4-grams, with sublinear term weighting. It is a lexical model
wearing a vector interface: good at "the query words appear in the tool", bad
at "refund" matching "chargeback". That trade is the right default because it
is honest about what it does, requires no network, and gives a floor the
benchmark can measure real embedders against.

Use a real embedding model in production:

    pip install 'toolbroker[fastembed]'

Hashing uses blake2b rather than :func:`hash`, whose seed is randomised per
process; identical text must produce identical vectors across runs.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence

from ..._vectors import l2_normalize
from ..enrich import tokenize


class HashingEmbedder:
    """Signed feature hashing over word and character n-grams."""

    def __init__(
        self,
        dim: int = 512,
        *,
        word_ngrams: int = 2,
        char_ngram: int = 4,
        use_char_ngrams: bool = True,
    ) -> None:
        """Configure the feature space.

        Args:
            dim: Vector width. Larger reduces hash collisions.
            word_ngrams: Highest word n-gram order to include.
            char_ngram: Character n-gram width, for partial-word robustness.
            use_char_ngrams: Include character n-grams at all.
        """
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self._dim = dim
        self._word_ngrams = max(1, word_ngrams)
        self._char_ngram = max(2, char_ngram)
        self._use_char = use_char_ngrams

    @property
    def dim(self) -> int:
        """Vector width."""
        return self._dim

    @property
    def id(self) -> str:
        """Cache key identifying this configuration."""
        chars = self._char_ngram if self._use_char else 0
        return f"hashing-{self._dim}-w{self._word_ngrams}-c{chars}"

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        features: Counter[str] = Counter(tokens)
        for order in range(2, self._word_ngrams + 1):
            for start in range(len(tokens) - order + 1):
                features["_".join(tokens[start : start + order])] += 1
        if self._use_char:
            compact = " ".join(tokens)
            width = self._char_ngram
            for start in range(max(0, len(compact) - width + 1)):
                features["#" + compact[start : start + width]] += 1
        return features

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self._dim, 1.0 if (value >> 63) & 1 else -1.0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for feature, count in self._features(text).items():
            index, sign = self._bucket(feature)
            # Sublinear scaling: the tenth occurrence of a word matters far
            # less than the second.
            vector[index] += sign * (1.0 + math.log(count))
        return l2_normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query. Symmetric with :meth:`embed` for this model."""
        return self._vector(text)

    def __repr__(self) -> str:
        """Show the configuration."""
        return f"HashingEmbedder(dim={self._dim})"
