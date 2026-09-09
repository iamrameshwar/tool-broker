"""LangGraphAdapter must satisfy the shared adapter contract."""

from __future__ import annotations

import pytest
from toolbroker_langgraph import LangGraphAdapter

from toolbroker.testing import AdapterConformanceSuite


class TestLangGraphAdapter(AdapterConformanceSuite):
    """No overrides, no exemptions."""

    @pytest.fixture
    def adapter(self):
        return LangGraphAdapter()
