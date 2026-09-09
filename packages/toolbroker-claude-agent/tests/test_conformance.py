"""ClaudeAgentAdapter must satisfy the shared adapter contract."""

from __future__ import annotations

import pytest
from toolbroker_claude_agent import ClaudeAgentAdapter

from toolbroker.testing import AdapterConformanceSuite


class TestClaudeAgentAdapter(AdapterConformanceSuite):
    """No overrides, no exemptions."""

    @pytest.fixture
    def adapter(self):
        return ClaudeAgentAdapter()
