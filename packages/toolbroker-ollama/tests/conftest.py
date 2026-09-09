from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("TOOLBROKER_OLLAMA_TEST_MODEL", "")


def _reachable() -> tuple[bool, str]:
    """Return whether a usable Ollama server with an embedding model is up."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as response:
            tags = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return False, ""

    names = [model.get("name", "") for model in tags.get("models", [])]
    if EMBED_MODEL:
        return EMBED_MODEL in names, EMBED_MODEL
    # Any tag that looks like an embedding model. Chat models do respond to
    # /api/embed, but the vectors are not what anyone means by an embedding.
    for name in names:
        if "embed" in name.lower():
            return True, name
    return False, ""


LIVE, LIVE_MODEL = _reachable()

requires_ollama = pytest.mark.skipif(
    not LIVE,
    reason=(
        "no Ollama server with an embedding model at "
        f"{OLLAMA_HOST}; start it and `ollama pull nomic-embed-text`"
    ),
)


@pytest.fixture(scope="session")
def live_model() -> str:
    """Name of an embedding model available on the local server."""
    return LIVE_MODEL
