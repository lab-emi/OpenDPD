"""The Markdown results report reads recomputed rows; it never invents or reorders evidence."""

import math
from pathlib import Path
import sys

import pytest

from opendpd.core import arena
from opendpd.schemas.arena import ArenaRow
from tests.unit.test_arena_scoring import calibrated, cases  # noqa: F401 - shared frozen-evidence fixtures

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
import report_arena_results as reporting  # noqa: E402


def row(records, backbone, board, qualities=(8., 8., 8.), power=0., **changes):
    names = {m.key: m for m in arena.bundled_backbones()}
    evidence = cases(records, board, qualities, backbone)
    for case in evidence:
        for judge in case["judges"]:
            judge["power_error_db"] = power
    summary = arena.summarize_cases(backbone, board, evidence, arena.sweep(backbone))
    return ArenaRow.model_validate({**dict(
        entry_id=f"official-{board}-{backbone}", board_id=board, backbone=backbone,
        display_name=names[backbone].display_name, origin="official", status="succeeded",
        protocol_sha256="a" * 64, cases=evidence, evidence_type="measured_data_simulation",
        execution_semantics="streaming_stateful" if backbone.endswith("_stream") else reporting.OFFLINE),
        **summary, **changes})


@pytest.fixture
def rows(calibrated):  # noqa: F811
    return [row(calibrated, "gru", "apa-200mhz-b", (9., 9., 9.)), row(calibrated, "mcldnn", "apa-200mhz-b", (9., 9., 9.)),
            row(calibrated, "tcn", "apa-200mhz-b", (2., 2., 2.), power=.8),      # linearity bought with output power
            row(calibrated, "gru_stream", "apa-200mhz-b", (12., 12., 12.)),
            row(calibrated, "mp_ls", "apa-200mhz-b", (6.,))]


def test_report_lists_leaders_per_ranking_inside_the_offline_cohort(rows):
    text = reporting.report(rows, arena.protocol(), "f" * 64)
    leaders = {line.split("|")[1].strip(" *"): [cell.strip() for cell in line.split("|")[2:-1]]
               for line in text.split("## Leaders")[1].split("## Overall")[0].splitlines() if line.startswith("| **")}
    assert set(leaders) == {ranking["title"] for ranking in arena.RANKINGS}
    # The streaming entry has the best quality on the board and still leads nothing here.
    assert all(cells[0].startswith("GRU ·") for cells in leaders.values())
    assert all(len(cells) == 1 for cells in leaders.values())
    assert "ILC-DPD" not in text


def test_all_configuration_costs_and_output_aclr_are_reported(rows):
    text=reporting.report(rows,arena.protocol(),'f'*64)
    assert 'OPs/sample' in text and 'EVM (dB)' in text and 'ACLR (dBc)' in text
    assert 'unavailable configurations are not zero scores' in text
    for r in rows:
        for p in r.budgets:
            if p.available:
                assert f'| {reporting.name(r)} | {p.parameters} | {p.ops} |' in text
                assert f'| {p.metrics.aclr_db:.2f} |' in text
    assert '| Unranked |' in text


def test_no_quality_gain_is_unranked_and_does_not_receive_a_zero_score(calibrated):
    small,none=row(calibrated,'gru','apa-200mhz-b',(1.,)*3),row(calibrated,'lstm','apa-200mhz-b',(-1.,)*3)
    text=reporting.report([small,none],arena.protocol(),'f'*64)
    assert small.eligible and not none.eligible and none.score is None
    assert reporting.weighted_fom(none,1) is None
    assert '½ · (ΔEVM + ΔACLR)' in text


def test_weighting_sensitivity_recovers_the_protocol_score_at_one_to_one(rows):
    for item in rows:
        if item.eligible:
            assert math.isclose(reporting.weighted_fom(item, 1), item.rankings["overall"].score, abs_tol=1e-9)
    # A model whose operations are mostly multiplications loses under a heavier multiplier.
    gru = rows[0]
    heavy, light = reporting.weighted_fom(gru, 16), reporting.weighted_fom(gru, 1)
    assert abs(heavy - light) < 0.2   # a GRU needs about one ADD per MUL, so the weighting barely matters


def test_spearman_is_one_for_identical_orders_and_minus_one_for_reversed():
    assert reporting.spearman([3., 2., 1.], [30., 20., 10.]) == pytest.approx(1.0)
    assert reporting.spearman([3., 2., 1.], [1., 2., 3.]) == pytest.approx(-1.0)
    assert reporting.spearman([1., 2.], [2., 1.]) is None
