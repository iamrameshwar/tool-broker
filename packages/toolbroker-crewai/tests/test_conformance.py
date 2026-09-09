"""CrewAIAdapter must satisfy the shared adapter contract."""

from __future__ import annotations

import pytest
from toolbroker_crewai import CrewAIAdapter

from toolbroker.testing import AdapterConformanceSuite


class TestCrewAIAdapter(AdapterConformanceSuite):
    """No overrides, no exemptions."""

    @pytest.fixture
    def adapter(self):
        return CrewAIAdapter()
