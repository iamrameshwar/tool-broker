from __future__ import annotations

import pytest

from toolbroker import MinScore, PolicyEngine, ToolBroker, calibrate_floor
from toolbroker.calibrate import DEFAULT_NOISE, CalibrationReport, _fraction_below, _percentile
from toolbroker.index.embedders import HashingEmbedder


def test_default_noise_is_not_empty():
    assert len(DEFAULT_NOISE) >= 20


def test_report_covers_every_requested_percentile(broker):
    report = calibrate_floor(broker, percentiles=(50, 90, 100))
    assert [s.percentile for s in report.suggestions] == [50, 90, 100]


def test_floors_are_non_decreasing_with_percentile(broker):
    floors = [s.floor for s in calibrate_floor(broker).suggestions]
    assert floors == sorted(floors)


def test_higher_floors_block_more_noise(broker):
    suggestions = calibrate_floor(broker).suggestions
    rejected = [s.noise_rejected for s in suggestions]
    assert rejected == sorted(rejected)


def test_records_the_catalogue_and_embedder(broker):
    report = calibrate_floor(broker)
    assert report.catalogue_size == len(broker)
    assert report.embedder == broker.embedder.id


def test_custom_noise_is_used(broker):
    report = calibrate_floor(broker, noise=["one", "two", "three"])
    assert len(report.noise_scores) == 3


def test_empty_noise_is_rejected(broker):
    with pytest.raises(ValueError, match="at least one noise query"):
        calibrate_floor(broker, noise=[])


def test_without_samples_the_report_says_so(broker):
    text = calibrate_floor(broker).summary()
    assert "Pass sample queries" in text
    assert calibrate_floor(broker).overlaps is False


def test_samples_produce_both_cost_columns(broker):
    report = calibrate_floor(broker, samples=["refund the customer", "how much stock"])
    for suggestion in report.suggestions:
        assert suggestion.samples_emptied is not None
        assert suggestion.samples_thinned is not None


def test_thinning_is_never_less_than_emptying(broker):
    # A query that lost everything also lost at least one; the reverse does not
    # hold, and conflating them is what makes a floor look free.
    report = calibrate_floor(
        broker, samples=["refund the customer", "how much stock", "send an email"], k=3
    )
    for suggestion in report.suggestions:
        assert suggestion.samples_thinned >= suggestion.samples_emptied


def test_recommended_returns_the_matching_floor(broker):
    report = calibrate_floor(broker, percentiles=(50, 90))
    assert report.recommended(percentile=90) == report.suggestions[-1].floor


def test_recommended_falls_back_to_computing_the_percentile(broker):
    report = calibrate_floor(broker, percentiles=(50,))
    assert 0.0 <= report.recommended(percentile=99) <= 1.0


def test_overlap_is_reported_when_noise_outscores_a_sample():
    report = CalibrationReport(
        noise_scores=(0.4, 0.6), sample_scores=(0.5,), noise_ceiling=0.6, sample_floor=0.5
    )
    assert report.overlaps is True
    assert "overlap" in report.summary()


def test_no_overlap_is_reported_when_they_separate():
    report = CalibrationReport(
        noise_scores=(0.1, 0.2), sample_scores=(0.8,), noise_ceiling=0.2, sample_floor=0.8
    )
    assert report.overlaps is False
    assert "do not overlap" in report.summary()


def test_the_suggested_floor_actually_blocks_noise(broker):
    # End to end: take the recommendation, apply it, confirm the noise queries
    # come back empty.
    report = calibrate_floor(broker, percentiles=(100,))
    broker.set_policy(PolicyEngine(selection_rules=[MinScore(report.recommended(percentile=100))]))
    blocked = sum(1 for query in DEFAULT_NOISE if len(broker.select(query, k=5)) == 0)
    assert blocked >= len(DEFAULT_NOISE) // 2


def test_calibration_ignores_an_existing_policy(broker):
    # Measuring through a policy that already caps or filters would distort the
    # score distribution being measured.
    before = calibrate_floor(broker).noise_scores
    broker.set_policy(PolicyEngine(selection_rules=[MinScore(0.99)]))
    assert calibrate_floor(broker).noise_scores == before


def test_works_on_an_empty_catalogue():
    empty = ToolBroker(embedder=HashingEmbedder(dim=64), cache_embeddings=False)
    empty.index([])
    report = calibrate_floor(empty)
    assert set(report.noise_scores) == {0.0}


@pytest.mark.parametrize(
    ("values", "percentile", "expected"),
    [
        ([0.1, 0.2, 0.3], 0, 0.1),
        ([0.1, 0.2, 0.3], 50, 0.2),
        ([0.1, 0.2, 0.3], 100, 0.3),
        ([], 50, 0.0),
    ],
)
def test_percentile(values, percentile, expected):
    assert _percentile(values, percentile) == expected


@pytest.mark.parametrize(
    ("values", "floor", "expected"),
    [
        ([0.1, 0.2, 0.3], 0.25, pytest.approx(2 / 3)),
        ([0.1, 0.2], 0.0, 0.0),
        ([], 0.5, 0.0),
    ],
)
def test_fraction_below(values, floor, expected):
    assert _fraction_below(values, floor) == expected
