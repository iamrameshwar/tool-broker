"""OllamaEmbedder against the shared embedder contract, on a live server."""

from __future__ import annotations

import pytest
from toolbroker_ollama import OllamaEmbedder

from toolbroker.testing import EmbedderConformanceSuite

from .conftest import requires_ollama

# Built once for the whole module: constructing the embedder costs a probe
# request, and a cold model has to load before it answers.
_cached: dict[str, OllamaEmbedder] = {}


@requires_ollama
class TestOllamaEmbedder(EmbedderConformanceSuite):
    """Runs against a real Ollama server; skipped when one is not available."""

    @pytest.fixture
    def embedder(self, live_model):
        if live_model not in _cached:
            _cached[live_model] = OllamaEmbedder(model=live_model)
        return _cached[live_model]
