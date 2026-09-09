"""Contract test suites for plugin authors.

A ``toolbroker-qdrant`` maintainer should not have to guess what "a store" means.
These suites encode the contract as executable tests: subclass, supply the
component, and the shared behaviour is verified for you.

    # tests/test_conformance.py in toolbroker-qdrant
    from toolbroker.testing import StoreConformanceSuite

    class TestQdrantStore(StoreConformanceSuite):
        @pytest.fixture
        def store(self):
            return QdrantStore(dim=self.DIM, location=":memory:")

This is what keeps "everything is swappable" honest across packages the core
team does not maintain.
"""

from .conformance import (
    AdapterConformanceSuite,
    EmbedderConformanceSuite,
    RerankerConformanceSuite,
    RetrieverConformanceSuite,
    StoreConformanceSuite,
    TracerConformanceSuite,
)
from .factories import make_records, make_tool, make_tools

__all__ = [
    "AdapterConformanceSuite",
    "EmbedderConformanceSuite",
    "RerankerConformanceSuite",
    "RetrieverConformanceSuite",
    "StoreConformanceSuite",
    "TracerConformanceSuite",
    "make_records",
    "make_tool",
    "make_tools",
]
