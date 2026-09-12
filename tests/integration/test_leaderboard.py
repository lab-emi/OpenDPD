"""S20: submissions drafted from runs, checked with a recomputation in a fresh workspace, boards seeded from the
hash-bound report, reviews, corrections and retractions with their history, and the label from the entries."""

import json
from pathlib import Path

import pytest

from opendpd.schemas import BenchmarkReport, Recomputation, Review, RunStatus, SubmissionCard, SubmissionCheck
from opendpd.services import leaderboard as lb
from opendpd.services.experiments import create_run, execute_run
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "benchmark" / "regression" / "cpu-regression-dpa-200mhz" / "report.json"
STATEMENTS = {"method": {"description": "GRU of the smoke recipe", "licence": "Apache-2.0"},
              "licence": {"code": "Apache-2.0", "weights": "Apache-2.0", "data": "built-in", "redistribution_allowed": True,
                          "statement": "everything redistributable"},
              "conflict_of_interest": "none", "citation": "Test Group 2026", "isolated_validation": "a machine without lab access"}


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    ws = Workspace.create(tmp_path_factory.mktemp("ws"))
    ws.register_builtin_dataset("DPA_200MHz")
    return ws


@pytest.fixture(scope="module")
def runs(ws):
    ids = []
    for seed in (0, 1):
        record = execute_run(ws, create_run(ws, instantiate("pa-gru-smoke-v1", "dpa-200mhz", seed=seed)).run_id)
        assert record.status == RunStatus.succeeded, record.error
        ids.append(record.run_id)
    return ids


@pytest.fixture(scope="module")
def submission(ws, runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("submission")
    card, path = lb.prepare(ws, runs, out, submission_id="test-group-gru", submitter="Test Group")
    return card, path


def filled(card: SubmissionCard, **changes) -> SubmissionCard:
    data = card.model_dump(mode="json")
    for key, value in STATEMENTS.items():
        data[key] = {**data[key], **value} if isinstance(value, dict) else value
    for key, value in changes.items():
        data[key] = {**data[key], **value} if isinstance(value, dict) else value
    return SubmissionCard.model_validate(data)


def test_prepare_drafts_the_card_from_the_runs_and_the_check_blocks_on_todo_statements(submission, runs):
    card, path = submission
    assert card.track == "pa_modeling" and [s.run_id for s in card.result.seeds] == runs and [s.seed for s in card.result.seeds] == [0, 1]
    assert len(card.packages) == 2 and all((path.parent / p.path).exists() for p in card.packages)
    assert card.result.metrics["NMSE"].n == 2 and card.data.availability == "public" and card.data.raw_sha256
    assert card.model.n_parameters == 1911 and card.result.seeds[0].checkpoint_sha256
    result = lb.check(card, path.parent)
    assert not result.passed
    assert next(i for i in result.items if i.name == "statements").status == "fail"
    assert next(i for i in result.items if i.name == "packages").status == "ok"
    assert next(i for i in result.items if i.name == "seeds").status == "warn"
    good = lb.check(filled(card), path.parent)
    assert good.passed and {i.name: i.status for i in good.items}["licence"] == "ok"


def test_the_check_recomputes_every_package_in_a_fresh_workspace_and_catches_an_edited_number(submission, tmp_path):
    card, path = submission
    card = filled(card)
    rec = lb.recompute(card, path.parent, by="Reviewer", kind="external", workspace=tmp_path / "fresh")
    assert rec.within_tolerance and rec.max_abs_delta == 0.0 and "2 package(s)" in rec.note
    assert (tmp_path / "fresh" / "runs" / card.packages[0].run_id / "run.json").exists()
    data = card.model_dump(mode="json")
    data["result"]["seeds"][0]["metrics"]["NMSE"] -= 0.5
    edited = SubmissionCard.model_validate(data)
    bad = lb.recompute(edited, path.parent, by="Reviewer", kind="external", workspace=tmp_path / "fresh2")
    assert not bad.within_tolerance and bad.max_abs_delta == pytest.approx(0.5)
    assert not lb.check(edited, path.parent, recomputation=bad).passed
    # a package that is not the card's is caught before any recomputation
    data = card.model_dump(mode="json")
    data["packages"][0]["sha256"] = "0" * 64
    assert "sha256 differs" in next(i for i in lb.check(SubmissionCard.model_validate(data), path.parent).items if i.name == "packages").detail


def test_a_board_seeded_from_the_report_is_a_reference_benchmark_whose_entries_trace_to_the_report(tmp_path):
    report = BenchmarkReport.model_validate(json.loads(REPORT.read_text()))
    board = lb.seed_board(report, board_id="pa", version="v-test", track="pa_modeling")
    assert board.label == "reference benchmark" and {e.entry_id for e in board.entries} == {"pa-gru", "pa-mp-ls"}
    entry = board.entry("pa-gru")
    assert entry.status == "accepted" and entry.evidence_grade == "self_reported" and entry.reviews == []
    assert entry.submission.benchmark_report_sha256 == report.report_sha256 and len(entry.submission.result.seeds) == 3
    assert entry.submission.result.seeds[0].checkpoint_sha256 and entry.submission.data.raw_sha256 == report.dataset.raw_sha256
    surrogate = lb.seed_board(report, board_id="dpd", version="v-test", track="dpd_surrogate")
    assert {e.entry_id for e in surrogate.entries} == {"dpd-gru", "dpd-mp-ila"}
    assert any("surrogate" in f for f in surrogate.entry("dpd-gru").submission.failure_conditions)
    with pytest.raises(ValueError, match="not open"):
        lb.seed_board(report, board_id="x", version="v-test", track="deployment")
    edited = report.model_copy(update={"seeds": [0, 1, 2, 3]})
    with pytest.raises(ValueError, match="not intact"):
        lb.seed_board(edited, board_id="x", version="v-test", track="pa_modeling")
    written = lb.write_board(board, tmp_path / "boards" / "pa.json")
    assert written.intact and lb.load_board(tmp_path / "boards" / "pa.json").board_sha256 == written.board_sha256
    assert (tmp_path / "boards" / "pa.md").read_text().startswith("# pa v-test — reference benchmark")


def test_review_correction_and_retraction_keep_history_and_the_bar_counts_external_entries_only(submission):
    card, path = submission
    report = BenchmarkReport.model_validate(json.loads(REPORT.read_text()))
    board = lb.seed_board(report, board_id="pa", version="v-test", track="pa_modeling")
    ok = Recomputation(by="Independent Reviewer", kind="external", within_tolerance=True, max_abs_delta=0.0, note="reproduced")
    board = lb.add_entry(board, filled(card, submission_id="ext-0", submitter={"name": "Group 0", "kind": "external"}),
                         lb.check(filled(card, submission_id="ext-0"), path.parent, board=board), by="Group 0")
    assert board.entry("ext-0").status == "submitted"
    with pytest.raises(ValueError, match="already on the board"):
        lb.add_entry(board, filled(card, submission_id="ext-1"), lb.check(filled(card, submission_id="ext-1"), path.parent, board=board), by="x")
    # the other external groups are simulated with a passed checklist: one workspace has one set of packages
    passed = SubmissionCheck(submission_id="simulated", items=[])
    for i in (1, 2):
        this = filled(card, submission_id=f"ext-{i}", submitter={"name": f"Group {i}", "kind": "external"})
        board = lb.add_entry(board, this, passed, by=f"Group {i}")
    with pytest.raises(ValueError, match="outside the tolerance"):
        lb.review_entry(board, "ext-0", Review(reviewer="R", kind="maintainer", decision="accepted", notes="no",
                                               recomputation=ok.model_copy(update={"within_tolerance": False})))
    board = lb.review_entry(board, "ext-0", Review(reviewer="Maintainer", kind="maintainer", decision="accepted", notes="checklist"))
    assert board.entry("ext-0").evidence_grade == "reviewed" and board.label == "reference benchmark"
    board = lb.review_entry(board, "ext-1", Review(reviewer="Independent Reviewer", kind="external", decision="accepted",
                                                   notes="recomputed", recomputation=ok))
    board = lb.review_entry(board, "ext-2", Review(reviewer="Group 2", kind="external", decision="accepted", notes="self",
                                                   recomputation=ok.model_copy(update={"by": "Group 2"})))
    assert board.entry("ext-1").evidence_grade == "independently_recomputed"
    assert board.entry("ext-2").evidence_grade == "reviewed", "a recomputation by the submitter is not independent"
    assert board.external_accepted == 3 and board.independent_recomputations == 1 and board.label == "reference benchmark"
    with pytest.raises(ValueError, match="cannot become accepted"):
        lb.review_entry(board, "ext-2", Review(reviewer="Maintainer", kind="maintainer", decision="accepted", notes="again"))
    this = filled(card, submission_id="ext-3", submitter={"name": "Group 3", "kind": "external"})
    board = lb.add_entry(board, this, passed, by="Group 3")
    board = lb.review_entry(board, "ext-3", Review(reviewer="Independent Reviewer", kind="external", decision="accepted",
                                                   notes="recomputed", recomputation=ok))
    assert board.independent_recomputations == 2 and board.label == "community leaderboard"
    # corrections keep the previous numbers; retractions un-rank; both stay in the history
    corrected = filled(card, submission_id="ext-3", submitter={"name": "Group 3", "kind": "external"},
                       failure_conditions=["fails above 30 dBm"])
    board = lb.amend_entry(board, "ext-3", action="correct", by="Group 3", reason="failure conditions added", corrected=corrected)
    assert board.entry("ext-3").status == "corrected" and board.entry("ext-3").ranked
    assert board.entry("ext-3").history[-1].previous_metrics["NMSE"].mean == card.result.metrics["NMSE"].mean
    board = lb.amend_entry(board, "ext-1", action="retract", by="Group 1", reason="wrong data version")
    assert not board.entry("ext-1").ranked and board.label == "reference benchmark"
    with pytest.raises(ValueError, match="cannot become corrected"):
        lb.amend_entry(board, "ext-1", action="correct", by="Group 1", reason="x", corrected=corrected)
    with pytest.raises(ValueError, match="keeps the submission id"):
        lb.amend_entry(board, "ext-0", action="correct", by="Group 0", reason="x", corrected=corrected)
    md = lb.board_markdown(board.sealed())
    assert "previous metrics kept" in md and "| ext-1 | retracted |" in md and "fails above 30 dBm" in md
    assert [h.action for h in board.entry("ext-3").history] == ["submitted", "accepted", "corrected"]


def test_non_comparable_entries_are_grouped_apart_and_the_rendering_carries_uncertainty_resources_and_failures(submission):
    card, path = submission
    report = BenchmarkReport.model_validate(json.loads(REPORT.read_text()))
    board = lb.seed_board(report, board_id="pa", version="v-test", track="pa_modeling")
    small = filled(card, submission_id="small-budget", failure_conditions=["not tried on other data"])
    small = SubmissionCard.model_validate({**small.model_dump(mode="json"),
                                           "result": {**small.model_dump(mode="json")["result"],
                                                      "resources": {**small.result.resources.model_dump(), "budget_class": "params<=2000"}}})
    board = lb.add_entry(board, small, lb.check(small, path.parent, board=board), by="Test Group")
    board = lb.review_entry(board, "small-budget", Review(reviewer="Maintainer", kind="maintainer", decision="accepted", notes="ok"))
    md = lb.board_markdown(board.sealed())
    groups = [line for line in md.splitlines() if line.startswith("## Group: ")]
    assert len(groups) == 2 and any("budget params<=2000" in g for g in groups) and any("budget unbounded" in g for g in groups)
    assert "Ordered by NMSE (lower is better); the other columns are not a tie-break." in md
    assert "± " in md and "(n=3)" in md and "(n=2)" in md and "on cpu" in md and "not tried on other data" in md
    assert "cpu_regression tier" in md and "self_reported" in md and "reviewed" in md
    assert "**reference benchmark**" in md and "1 of the 3 required external submissions" in md
