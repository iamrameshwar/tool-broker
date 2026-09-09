from __future__ import annotations

import sys
from pathlib import Path

import pytest

# bench/ is research tooling, not part of the installed package, but its
# scoring logic is worth testing like anything else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from toolbroker import RiskTier, Tool, ToolBroker
from toolbroker.index.embedders.hashing import HashingEmbedder
from toolbroker.store.memory import InMemoryStore


def make_tool(name: str, description: str = "", **kwargs) -> Tool:
    kwargs.setdefault("namespace", "test")
    return Tool(name=name, description=description, **kwargs)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=128)


@pytest.fixture
def store(embedder: HashingEmbedder) -> InMemoryStore:
    return InMemoryStore(embedder.dim)


@pytest.fixture
def sample_tools() -> list[Tool]:
    return [
        make_tool(
            "search_orders",
            "Find recent orders placed by a customer",
            tags=frozenset({"read", "commerce"}),
        ),
        make_tool(
            "issue_refund",
            "Refund money to a customer for an order",
            tags=frozenset({"write", "commerce"}),
            risk=RiskTier.HIGH,
            required_scopes=frozenset({"payments:write"}),
        ),
        make_tool(
            "check_inventory",
            "Return how many units of a SKU are in stock",
            tags=frozenset({"read", "warehouse"}),
        ),
        make_tool(
            "delete_customer",
            "Permanently erase a customer record",
            tags=frozenset({"write", "destructive"}),
            risk=RiskTier.CRITICAL,
        ),
        make_tool(
            "send_email",
            "Send an email message to a recipient",
            tags=frozenset({"write", "comms"}),
            risk=RiskTier.MEDIUM,
        ),
    ]


@pytest.fixture
def broker(embedder: HashingEmbedder, sample_tools: list[Tool]) -> ToolBroker:
    catalogue = ToolBroker(embedder=embedder, cache_embeddings=False)
    catalogue.index(sample_tools)
    return catalogue
