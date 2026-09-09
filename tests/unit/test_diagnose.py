from __future__ import annotations

import pytest

from toolbroker import CatalogueDiagnosis, Tool, ToolBroker, diagnose_catalogue
from toolbroker.diagnose import SHADOWED, THIN, TWINNED, UNDESCRIBED
from toolbroker.index.embedders import HashingEmbedder


def _catalogue(*tools: Tool) -> ToolBroker:
    broker = ToolBroker(embedder=HashingEmbedder(dim=256), cache_embeddings=False)
    broker.index(list(tools))
    return broker


def _tool(name: str, description: str, namespace: str = "svc") -> Tool:
    return Tool(name=name, namespace=namespace, description=description)


def test_reports_every_tool(broker):
    report = diagnose_catalogue(broker)
    assert isinstance(report, CatalogueDiagnosis)
    assert len(report.findings) == report.catalogue_size == len(broker)


def test_a_distinct_catalogue_flags_nothing():
    broker = _catalogue(
        _tool("issue_refund", "Return money to a customer for a completed order."),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
        _tool("scale_deployment", "Change the replica count for a running service."),
    )
    report = diagnose_catalogue(broker)
    assert all(finding.healthy for finding in report.findings)
    assert report.unhealthy == ()


def test_every_tool_finds_itself_first_when_descriptions_differ():
    broker = _catalogue(
        _tool("issue_refund", "Return money to a customer for a completed order."),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
        _tool("scale_deployment", "Change the replica count for a running service."),
    )
    report = diagnose_catalogue(broker)
    assert {finding.self_rank for finding in report.findings} == {1}


def test_empty_description_is_reported():
    broker = _catalogue(
        _tool("mystery", ""),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
    )
    finding = next(f for f in diagnose_catalogue(broker).findings if f.tool_id.endswith("mystery"))
    assert UNDESCRIBED in finding.issues
    assert THIN not in finding.issues  # empty is its own, worse, category


def test_short_description_is_thin_but_not_undescribed():
    broker = _catalogue(
        _tool("get_it", "Gets it."),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
    )
    finding = next(f for f in diagnose_catalogue(broker).findings if f.tool_id.endswith("get_it"))
    assert finding.issues == (THIN,)


def test_a_tool_buried_by_near_duplicates_is_shadowed():
    # Five tools sharing one description, plus the victim. With k=2 the victim
    # cannot be in its own top 2, because its clones score as well as it does.
    clones = [_tool(f"sync_report_v{n}", "Sync a report.") for n in range(6)]
    broker = _catalogue(*clones)
    report = diagnose_catalogue(broker, k=2)
    assert any(SHADOWED in finding.issues for finding in report.findings)


def test_shadowed_names_who_is_in_the_way():
    clones = [_tool(f"sync_report_v{n}", "Sync a report.") for n in range(6)]
    report = diagnose_catalogue(_catalogue(*clones), k=2)
    shadowed = [f for f in report.findings if SHADOWED in f.issues]
    assert shadowed, "expected at least one shadowed tool"
    assert all(finding.shadowed_by for finding in shadowed)


def test_no_margin_flags_no_twins_however_close():
    """The default must not invent a threshold.

    Identical descriptions sit ~0.02 apart under one embedder and ~0.20 under
    another, and no catalogue-internal statistic distinguishes the two cases,
    so shipping a default margin would be silently wrong somewhere.
    """
    clones = [_tool(f"sync_report_v{n}", "Sync a report.") for n in range(4)]
    report = diagnose_catalogue(_catalogue(*clones))
    assert report.margin is None
    assert not any(TWINNED in finding.issues for finding in report.findings)


def test_an_explicit_margin_flags_twins():
    clones = [_tool(f"sync_report_v{n}", "Sync a report.") for n in range(4)]
    report = diagnose_catalogue(_catalogue(*clones), margin=0.9)
    assert report.margin == 0.9
    assert any(TWINNED in finding.issues for finding in report.findings)


def test_closest_pairs_are_ranked_tightest_first():
    broker = _catalogue(
        _tool("sync_report", "Sync a report."),
        _tool("sync_report_v1", "Sync a report."),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
    )
    pairs = diagnose_catalogue(broker).closest_pairs
    assert [pair.gap for pair in pairs] == sorted(pair.gap for pair in pairs)


def test_identical_descriptions_are_marked_as_certain():
    broker = _catalogue(
        _tool("sync_report", "Sync a report."),
        _tool("sync_report_v1", "Sync a report."),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
    )
    tightest = diagnose_catalogue(broker).closest_pairs[0]
    assert tightest.identical_description
    assert {tightest.left, tightest.right} == {"svc/sync_report", "svc/sync_report_v1"}


def test_a_pair_is_reported_once_not_from_both_ends():
    broker = _catalogue(
        _tool("sync_report", "Sync a report."),
        _tool("sync_report_v1", "Sync a report."),
    )
    pairs = diagnose_catalogue(broker).closest_pairs
    assert len(pairs) == 1


def test_pairs_limit_is_respected():
    clones = [_tool(f"sync_report_v{n}", "Sync a report.") for n in range(10)]
    assert len(diagnose_catalogue(_catalogue(*clones), pairs=3).closest_pairs) == 3


def test_samples_report_which_tools_nothing_reaches():
    broker = _catalogue(
        _tool("issue_refund", "Return money to a customer for a completed order."),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
        _tool("scale_deployment", "Change the replica count for a running service."),
    )
    report = diagnose_catalogue(broker, k=1, samples=["refund the customer their money"])
    assert report.sample_count == 1
    assert "svc/issue_refund" not in report.unreached
    assert report.unreached


def test_no_samples_means_no_unreached_claim():
    """Absence of samples must not be reported as full coverage."""
    broker = _catalogue(
        _tool("issue_refund", "Return money to a customer for a completed order."),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
    )
    report = diagnose_catalogue(broker)
    assert report.unreached == ()
    assert report.sample_count == 0


def test_summary_always_states_what_it_cannot_detect(broker):
    """The vocabulary gap is the biggest failure mode and the one this misses."""
    assert "phrased unlike anything a user would type" in diagnose_catalogue(broker).summary()


def test_summary_renders_for_a_clean_catalogue():
    broker = _catalogue(
        _tool("issue_refund", "Return money to a customer for a completed order."),
        _tool("track_shipment", "Where a parcel is right now, by tracking number."),
    )
    assert "Nothing flagged" in diagnose_catalogue(broker).summary()


def test_summary_renders_every_section():
    broker = _catalogue(
        _tool("a", ""),
        *[_tool(f"sync_report_v{n}", "Sync a report.") for n in range(6)],
    )
    summary = diagnose_catalogue(broker, k=2, margin=0.9, samples=["sync a report"]).summary()
    for heading in ("SHADOWED", "TWINNED", "THIN", "CLOSEST PAIRS", "UNREACHED"):
        assert heading in summary


def test_an_empty_catalogue_does_not_explode():
    broker = ToolBroker(embedder=HashingEmbedder(dim=256), cache_embeddings=False)
    broker.index([])
    report = diagnose_catalogue(broker)
    assert report.catalogue_size == 0
    assert report.findings == ()
    assert report.summary()


def test_a_single_tool_has_no_neighbour():
    broker = _catalogue(_tool("alone", "The only tool in the catalogue."))
    finding = diagnose_catalogue(broker).findings[0]
    assert finding.nearest is None
    assert finding.nearest_gap is None
    assert finding.healthy


def test_report_round_trips_through_json(broker):
    report = diagnose_catalogue(broker)
    assert CatalogueDiagnosis.model_validate(report.model_dump(mode="json")) == report


@pytest.mark.parametrize("k", [1, 3, 10])
def test_k_is_recorded(broker, k):
    assert diagnose_catalogue(broker, k=k).k == k
