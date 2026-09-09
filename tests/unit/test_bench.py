from __future__ import annotations

from toolbroker.bench import BenchmarkCase, run_benchmark


def test_perfect_retrieval_scores_one(broker):
    cases = [BenchmarkCase(query="how many units are in stock", expected=["test/check_inventory"])]
    result = run_benchmark(broker, cases, k=5)
    assert result.recall_at_k == 1.0
    assert result.hit_rate == 1.0


def test_missed_retrieval_is_reported_as_a_failure(broker):
    cases = [BenchmarkCase(query="stock levels", expected=["test/nonexistent"])]
    result = run_benchmark(broker, cases, k=1)
    assert result.hit_rate == 0.0
    assert result.to_dict()["failures"][0]["expected"] == ["test/nonexistent"]


def test_mrr_rewards_a_higher_rank(broker):
    cases = [BenchmarkCase(query="how many units are in stock", expected=["test/check_inventory"])]
    top = run_benchmark(broker, cases, k=1).mrr
    assert top == 1.0


def test_context_reduction_reflects_the_catalogue_size(broker):
    cases = [BenchmarkCase(query="stock", expected=["test/check_inventory"])]
    result = run_benchmark(broker, cases, k=1)
    assert result.context_reduction == 0.8  # 1 of 5 tools sent


def test_negative_cases_expect_an_empty_selection(broker):
    from toolbroker import MinScore, PolicyEngine

    broker.set_policy(PolicyEngine(selection_rules=[MinScore(0.99)]))
    cases = [BenchmarkCase(query="entirely unrelated aerospace telemetry", expected=[])]
    result = run_benchmark(broker, cases, k=5)
    assert result.hit_rate == 1.0


def test_negative_case_fails_when_something_is_returned(broker):
    cases = [BenchmarkCase(query="stock", expected=[])]
    assert run_benchmark(broker, cases, k=3).hit_rate == 0.0


def test_summary_lists_failures(broker):
    cases = [BenchmarkCase(query="stock", expected=["test/nope"])]
    text = run_benchmark(broker, cases, k=1).summary()
    assert "failures (1)" in text
    assert "recall@1" in text


def test_per_case_agent_overrides_the_default(broker):
    from toolbroker import AgentPolicy, DenyTools, PolicyEngine

    broker.set_policy(AgentPolicy({"locked": PolicyEngine([DenyTools(["test/*"])])}))
    cases = [BenchmarkCase(query="stock", expected=[], agent="locked")]
    assert run_benchmark(broker, cases, k=5).hit_rate == 1.0


def test_negative_case_that_returns_tools_scores_zero_mrr(broker):
    # Crediting a negative case that returned five wrong tools as a perfect
    # answer made the aggregate look better than the system was.
    cases = [BenchmarkCase(query="stock", expected=[])]
    result = run_benchmark(broker, cases, k=3)
    assert result.cases[0].hit is False
    assert result.cases[0].reciprocal_rank == 0.0
    assert result.mrr == 0.0


def test_negative_case_that_returns_nothing_scores_one(broker):
    from toolbroker import MinScore, PolicyEngine

    broker.set_policy(PolicyEngine(selection_rules=[MinScore(0.99)]))
    cases = [BenchmarkCase(query="entirely unrelated aerospace telemetry", expected=[])]
    result = run_benchmark(broker, cases, k=5)
    assert result.cases[0].reciprocal_rank == 1.0


def test_categories_are_reported_separately(broker):
    cases = [
        BenchmarkCase(
            query="how many units are in stock",
            expected=["test/check_inventory"],
            category="paraphrase",
        ),
        BenchmarkCase(query="zzz nothing", expected=[], category="negative"),
    ]
    groups = run_benchmark(broker, cases, k=3).categories()
    assert set(groups) == {"paraphrase", "negative"}
    assert groups["paraphrase"].hit_rate == 1.0


def test_confidence_interval_narrows_with_sample_size():
    from toolbroker.bench import wilson_interval

    narrow = wilson_interval(237, 300)
    wide = wilson_interval(33, 42)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_confidence_interval_stays_within_bounds():
    from toolbroker.bench import wilson_interval

    for successes, total in [(0, 10), (10, 10), (1, 1), (0, 1)]:
        low, high = wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0


def test_empty_sample_has_a_zero_interval():
    from toolbroker.bench import wilson_interval

    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_summary_shows_the_category_table(broker):
    cases = [
        BenchmarkCase(
            query="how many units are in stock",
            expected=["test/check_inventory"],
            category="paraphrase",
        ),
        BenchmarkCase(query="zzz", expected=[], category="negative"),
    ]
    text = run_benchmark(broker, cases, k=3).summary()
    assert "category" in text
    assert "paraphrase" in text
    assert "95% CI" in text
