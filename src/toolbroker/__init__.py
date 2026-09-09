"""ToolBroker: give your agent the right 5 tools out of 500.

A retrieval and policy layer between a tool catalogue and an agent's context
window. It never runs an agent loop and never calls a tool; it decides which
tool definitions belong in the prompt and explains why.

    from toolbroker import ToolBroker

    broker = ToolBroker()
    broker.add_functions([search_orders, issue_refund])
    broker.index()
    selection = broker.select("customer wants a refund", k=3)
"""

from __future__ import annotations

from importlib import metadata as _metadata

from . import aio
from .aliases import AliasLearner
from .calibrate import CalibrationReport, calibrate_floor
from .catalog import ToolBroker
from .conversation import Conversation, Turn, contextual_query
from .diagnose import CatalogueDiagnosis, diagnose_catalogue
from .drift import DriftGuard, ToolChange
from .errors import (
    AdapterError,
    ConfigurationError,
    EmbeddingError,
    NotIndexedError,
    PluginError,
    PolicyError,
    PolicyViolationError,
    RetrievalError,
    SourceError,
    StoreError,
    ToolBrokerError,
)
from .hooks import DROP, Event, HookManager
from .index.enrich import EnrichmentConfig
from .index.indexer import IndexReport, RefreshReport
from .policy.engine import AgentPolicy, PolicyEngine
from .policy.rules import (
    AllowTools,
    DenyTags,
    DenyTools,
    MaxCost,
    MaxRisk,
    MaxTools,
    MinScore,
    PredicateRule,
    RequireScopes,
    RequireTags,
    TenantIsolation,
)
from .protocols import (
    Adapter,
    Embedder,
    Policy,
    Reranker,
    Retriever,
    Source,
    Store,
    Tracer,
)
from .reachability import AgentReach, PolicyDiff, diff_policies, reachable_tools
from .refresh import PeriodicRefresher
from .retrieve.resilient import ResilientRetriever
from .risk import RiskClassifier, classify_by_name
from .savings import SavingsReport, SavingsTally
from .types import (
    CostTier,
    Decision,
    Exclusion,
    Filters,
    Hit,
    RiskTier,
    RuleFiring,
    Selection,
    Stage,
    Tool,
    ToolRecord,
)
from .usage import UsageTracker

# Read from the installed metadata rather than repeated here. Two sources of
# truth drift silently and in the worst possible direction: `pip show` said
# 0.1.0 while `toolbroker --version` said 0.1.0.dev0, which is exactly the sort
# of thing nobody notices until a bug report quotes the wrong one.
try:
    __version__ = _metadata.version("toolbroker")
except _metadata.PackageNotFoundError:  # pragma: no cover - source tree, uninstalled
    __version__ = "0.0.0+unknown"

__all__ = [
    "DROP",
    "Adapter",
    "AdapterError",
    "AgentPolicy",
    "AgentReach",
    "AliasLearner",
    "AllowTools",
    "CalibrationReport",
    "CatalogueDiagnosis",
    "ConfigurationError",
    "Conversation",
    "CostTier",
    "Decision",
    "DenyTags",
    "DenyTools",
    "DriftGuard",
    "Embedder",
    "EmbeddingError",
    "EnrichmentConfig",
    "Event",
    "Exclusion",
    "Filters",
    "Hit",
    "HookManager",
    "IndexReport",
    "MaxCost",
    "MaxRisk",
    "MaxTools",
    "MinScore",
    "NotIndexedError",
    "PeriodicRefresher",
    "PluginError",
    "Policy",
    "PolicyDiff",
    "PolicyEngine",
    "PolicyError",
    "PolicyViolationError",
    "PredicateRule",
    "RefreshReport",
    "RequireScopes",
    "RequireTags",
    "Reranker",
    "ResilientRetriever",
    "RetrievalError",
    "Retriever",
    "RiskClassifier",
    "RiskTier",
    "RuleFiring",
    "SavingsReport",
    "SavingsTally",
    "Selection",
    "Source",
    "SourceError",
    "Stage",
    "Store",
    "StoreError",
    "TenantIsolation",
    "Tool",
    "ToolBroker",
    "ToolBrokerError",
    "ToolChange",
    "ToolRecord",
    "Tracer",
    "Turn",
    "UsageTracker",
    "__version__",
    "aio",
    "calibrate_floor",
    "classify_by_name",
    "contextual_query",
    "diagnose_catalogue",
    "diff_policies",
    "reachable_tools",
]
