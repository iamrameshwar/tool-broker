from __future__ import annotations

import e2e
import pytest

from toolbroker import Tool, ToolBroker
from toolbroker.bench import BenchmarkCase
from toolbroker.index.embedders import HashingEmbedder


class StubCaller:
    """A caller that returns whatever it was scripted to return."""

    def __init__(self, chosen, *, max_tools=None, error=None):
        self._chosen = chosen
        self._error = error
        self.max_tools = max_tools
        self.name = "stub"
        self.seen: list[int] = []

    def call(self, query, tools):
        self.seen.append(len(tools))
        return e2e.CallOutcome(
            chosen=self._chosen(query) if callable(self._chosen) else self._chosen,
            prompt_tokens=e2e.estimate_tokens(tools),
            error=self._error,
        )


@pytest.fixture
def broker():
    catalogue = ToolBroker(embedder=HashingEmbedder(dim=128), cache_embeddings=False)
    catalogue.index(
        [
            Tool(name="issue_refund", namespace="billing", description="Refund a payment"),
            Tool(name="send_email", namespace="messaging", description="Send an email"),
            Tool(name="query_logs", namespace="obs", description="Search application logs"),
        ]
    )
    return catalogue


CASES = [BenchmarkCase(query="refund a payment", expected=["billing/issue_refund"])]


def test_full_condition_sends_the_whole_catalogue(broker):
    caller = StubCaller("billing__issue_refund")
    e2e.evaluate(caller, broker, CASES, condition="full", size=3, k=1)
    assert caller.seen == [3]


def test_toolbroker_condition_sends_only_k(broker):
    caller = StubCaller("billing__issue_refund")
    e2e.evaluate(caller, broker, CASES, condition="toolbroker", size=3, k=1)
    assert caller.seen == [1]


def test_correct_choice_is_scored(broker):
    result = e2e.evaluate(
        StubCaller("billing__issue_refund"), broker, CASES, condition="full", size=3, k=1
    )
    assert result.correct == 1
    assert result.accuracy == 1.0


def test_wrong_choice_is_recorded_with_the_failure(broker):
    result = e2e.evaluate(
        StubCaller("messaging__send_email"), broker, CASES, condition="full", size=3, k=1
    )
    assert result.wrong == 1
    assert result.failures[0]["got"] == "messaging__send_email"


def test_abstention_is_distinguished_from_a_wrong_answer(broker):
    result = e2e.evaluate(StubCaller(None), broker, CASES, condition="full", size=3, k=1)
    assert result.abstained == 1
    assert result.wrong == 0


def test_negative_case_is_correct_when_nothing_is_called(broker):
    negatives = [BenchmarkCase(query="who won the 1974 world cup", expected=[])]
    result = e2e.evaluate(StubCaller(None), broker, negatives, condition="full", size=3, k=1)
    assert result.correct == 1


def test_negative_case_is_wrong_when_a_tool_is_called(broker):
    negatives = [BenchmarkCase(query="who won the 1974 world cup", expected=[])]
    result = e2e.evaluate(
        StubCaller("messaging__send_email"), broker, negatives, condition="full", size=3, k=1
    )
    assert result.wrong == 1
    assert result.failures[0]["expected"] == "(no tool)"


def test_provider_limit_marks_the_full_condition_unrunnable(broker):
    # At 200 and 1000 tools OpenAI simply refuses the request. Reporting that
    # is the point; it is a finding, not an error to swallow.
    result = e2e.evaluate(
        StubCaller("x", max_tools=2), broker, CASES, condition="full", size=3, k=1
    )
    assert result.runnable is False
    assert "exceeds the provider limit" in result.skip_reason


def test_provider_limit_does_not_block_the_retrieval_condition(broker):
    result = e2e.evaluate(
        StubCaller("billing__issue_refund", max_tools=2),
        broker,
        CASES,
        condition="toolbroker",
        size=3,
        k=1,
    )
    assert result.runnable is True
    assert result.correct == 1


def test_api_errors_are_counted_not_scored(broker):
    result = e2e.evaluate(
        StubCaller(None, error="rate limited"), broker, CASES, condition="full", size=3, k=1
    )
    assert result.errors == 1
    assert result.attempted == 0
    assert result.accuracy == 0.0


def test_expected_names_are_matched_after_flattening(broker):
    # The model sees `billing__issue_refund`; the label says `billing/issue_refund`.
    result = e2e.evaluate(
        StubCaller("billing__issue_refund"), broker, CASES, condition="full", size=3, k=1
    )
    assert result.correct == 1


def test_lexical_caller_picks_the_overlapping_tool():
    tools = [
        {"function": {"name": "send_email", "description": "Send an email message"}},
        {"function": {"name": "query_logs", "description": "Search application logs"}},
    ]
    assert e2e.LexicalCaller().call("search the logs", tools).chosen == "query_logs"


def test_lexical_caller_abstains_when_nothing_overlaps():
    tools = [{"function": {"name": "send_email", "description": "Send an email message"}}]
    assert e2e.LexicalCaller().call("zzzz qqqq", tools).chosen is None


def test_token_estimate_grows_with_the_tool_block():
    small = [{"function": {"name": "a", "description": "x"}}]
    large = small * 50
    assert e2e.estimate_tokens(large) > e2e.estimate_tokens(small) * 10


def test_table_marks_unrunnable_rows(broker):
    row = e2e.ConditionResult(caller="c", size=1000, condition="full", runnable=False)
    assert "not runnable" in e2e.render_table([row], k=5)


def test_unknown_caller_is_rejected():
    with pytest.raises(SystemExit):
        e2e.build_caller("telepathy", None)
