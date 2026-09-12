"""S20: the leaderboard contract — open tracks only, evidence per track, traceability, the label from the entries."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from opendpd.schemas import BenchmarkReport, Leaderboard, SubmissionCard
from opendpd.schemas.leaderboard import MIN_EXTERNAL_ACCEPTED, MIN_INDEPENDENT_RECOMPUTATIONS, TRACK_GATES

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs" / "community" / "submission-example.json"
BOARDS = ROOT / "docs" / "leaderboard" / "v2026.09"
REPORT = ROOT / "benchmark" / "regression" / "cpu-regression-dpa-200mhz" / "report.json"


def example() -> dict:
    return json.loads(EXAMPLE.read_text())


def test_the_example_submission_is_a_valid_card_with_three_seeds_and_a_package_per_seed():
    card = SubmissionCard.model_validate(example())
    assert card.track == "pa_modeling" and len(card.result.seeds) == 3 and len(card.packages) == 3
    assert card.submitter.kind == "external" and "TODO" not in json.dumps(example())


def test_closed_tracks_and_mismatched_evidence_are_refused():
    for track in ("standard_evaluation", "robustness", "deployment"):
        with pytest.raises(ValidationError, match="not open"):
            SubmissionCard.model_validate({**example(), "track": track})
        assert track in TRACK_GATES
    with pytest.raises(ValidationError, match="takes dpd_surrogate evidence"):
        SubmissionCard.model_validate({**example(), "track": "dpd_surrogate"})


def test_an_entry_is_traceable_to_packages_or_a_report_and_the_aggregate_matches_the_seeds():
    data = example()
    data["packages"] = []
    with pytest.raises(ValidationError, match="traceable"):
        SubmissionCard.model_validate(data)
    data["benchmark_report_sha256"] = "a" * 64
    with pytest.raises(ValidationError, match="maintainers' path"):
        SubmissionCard.model_validate(data)
    data["submitter"]["kind"] = "maintainer"
    assert SubmissionCard.model_validate(data).benchmark_report_sha256 == "a" * 64
    data = example()
    data["result"]["metrics"]["NMSE"]["n"] = 2
    with pytest.raises(ValidationError, match="aggregates 2 values"):
        SubmissionCard.model_validate(data)
    data = example()
    data["packages"][0]["run_id"] = "run-not-a-seed"
    with pytest.raises(ValidationError, match="not one of the card's seeds"):
        SubmissionCard.model_validate(data)


def test_the_checked_in_boards_are_intact_and_copy_the_regression_report():
    report = BenchmarkReport.model_validate(json.loads(REPORT.read_text()))
    assert report.intact
    seen = set()
    for path in sorted(BOARDS.glob("*.json")):
        board = Leaderboard.model_validate(json.loads(path.read_text()))
        assert board.intact, f"{path.name} was edited after it was sealed"
        assert board.version == "v2026.09" and board.label == "reference benchmark"
        assert path.with_suffix(".md").read_text().startswith(f"# {board.board_id} v2026.09 — reference benchmark")
        for entry in board.entries:
            source = next(e for e in report.entries if e.entry_id == entry.entry_id)
            assert entry.submission.result.seeds == source.seeds and entry.submission.result.metrics == source.aggregate
            assert entry.submission.benchmark_report_sha256 == report.report_sha256
            assert entry.evidence_grade == "self_reported" and entry.submission.submitter.kind == "maintainer"
            seen.add(entry.entry_id)
    assert seen == {e.entry_id for e in report.entries}
    assert {p.stem for p in BOARDS.glob("*.json")} == {"pa_modeling", "dpd_surrogate", "dpd_measured"}


def test_the_label_is_computed_from_external_accepted_and_recomputed_entries():
    board = Leaderboard.model_validate(json.loads((BOARDS / "pa_modeling.json").read_text()))
    assert board.external_accepted == 0 and board.independent_recomputations == 0
    assert MIN_EXTERNAL_ACCEPTED == 3 and MIN_INDEPENDENT_RECOMPUTATIONS == 2
    with pytest.raises(ValidationError, match="not open"):
        Leaderboard(board_id="x", version="v1", track="deployment")
