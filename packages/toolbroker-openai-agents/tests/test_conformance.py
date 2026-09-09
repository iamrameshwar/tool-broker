"""OpenAIAgentsAdapter must satisfy the shared adapter contract."""

from __future__ import annotations

import pytest
from toolbroker_openai_agents import OpenAIAgentsAdapter

from toolbroker.testing import AdapterConformanceSuite


class TestOpenAIAgentsAdapter(AdapterConformanceSuite):
    """No overrides, no exemptions."""

    @pytest.fixture
    def adapter(self):
        return OpenAIAgentsAdapter()
