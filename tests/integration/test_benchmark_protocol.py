"""benchmark-v1 (plan S12): pre-registered plans with at least three seeds, idempotent execution, hash-bound reports
whose numbers cannot be edited unnoticed, and regression baselines that block only once a human approved them."""

import json
from datetime import date

import pytest
from pydantic import ValidationError

from opendpd.cli import studio_main
from opendpd.schemas import BenchmarkPlan, RunStatus
from opendpd.services import benchmark as bm
from opendpd.services.experiments import list_runs
from opendpd.services.workspace import Workspace, WorkspaceError, write_json_atomic

pytestmark = pytest.mark.integration

DATASET = "DPA_200MHz"           # measured built-in data: real signals, smoke budgets


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    ws = Workspace.create(tmp_path_factory.mktemp("ws"))
    ws.register_builtin_dataset(DATASET)
    return ws


@pytest.fixture(scope="module")
def plan():
    entries = [e for e in bm.default_entries("cpu_regression") if e.entry_id in ("pa-gru", "dpd-mp-ila")]
    return bm.make_plan("dpa-200mhz", tier="cpu_regression", seeds=(0, 1, 2), entries=entries)


@pytest.fixture(scope="module")
def executed(ws, plan):
    runs = bm.run_plan(ws, plan)
    assert {k for k in runs} == {(e, s) for e in ("pa-gru", "dpd-mp-ila") for s in (0, 1, 2)}
    assert all(r.status == RunStatus.succeeded for r in runs.values())
    return runs


def test_a_plan_needs_three_pre_registered_seeds_and_is_frozen_by_its_hash(tmp_path):
    with pytest.raises(ValidationError, match="at least 3"):
        bm.make_plan("dpa-200mhz", seeds=(0, 1))
    with pytest.raises(ValidationError, match="distinct"):
        bm.make_plan("dpa-200mhz", seeds=(0, 1, 1))
    plan = bm.make_plan("dpa-200mhz")
    assert plan.protocol_id == "benchmark-v1" and [e.entry_id for e in plan.entries] == ["pa-gru", "pa-mp-ls", "dpd-gru", "dpd-mp-ila"]
    assert plan.plan_sha256 == plan.compute_sha256()
    assert bm.make_plan("dpa-200mhz").plan_sha256 == plan.plan_sha256          # created_at is not part of the identity
    assert bm.make_plan("dpa-200mhz", seeds=(0, 1, 3)).plan_sha256 != plan.plan_sha256
    path = tmp_path / "plan.json"
    bm.write_plan(plan, path)
    assert bm.load_plan(path).plan_sha256 == plan.plan_sha256
    edited = json.loads(path.read_text())
    edited["seeds"] = [0, 1, 2, 3]
    path.write_text(json.dumps(edited))
    with pytest.raises(WorkspaceError, match="edited after registration"):
        bm.load_plan(path)
    with pytest.raises(ValidationError, match="pa_entry"):
        BenchmarkPlan(tier="cpu_regression", dataset={"id": "x"}, seeds=[0, 1, 2],
                      entries=[{"entry_id": "d", "task": "train_dpd", "model": {"key": "gru"}, "training": {}}])
    assert len(bm.default_entries("gpu_full")) == 6


def test_runs_are_keyed_by_the_plan_and_reused_on_a_second_execution(ws, plan, executed):
    before = {r.run_id for r in list_runs(ws)}
    again = bm.run_plan(ws, plan)
    assert {k: r.run_id for k, r in again.items()} == {k: r.run_id for k, r in executed.items()}
    assert {r.run_id for r in list_runs(ws)} == before
    assert executed[("pa-gru", 1)].idempotency_key == bm.run_key(plan, "pa-gru", 1)
    assert bm.find_runs(ws, plan).keys() == executed.keys()


def test_report_binds_every_number_to_runs_and_hashes_and_reports_each_seed(ws, plan, executed):
    report = bm.build_report(ws, plan)
    assert report.intact and report.plan_sha256 == plan.plan_sha256 and report.seeds == [0, 1, 2]
    pa = next(e for e in report.entries if e.entry_id == "pa-gru")
    dpd = next(e for e in report.entries if e.entry_id == "dpd-mp-ila")
    assert [s.seed for s in pa.seeds] == [0, 1, 2] and pa.missing_seeds == [] and pa.training_path == "gradient"
    assert dpd.training_path == "ila_least_squares" and dpd.fit and dpd.fit["rank"] == dpd.fit["n_coefficients"] == 100
    for s in pa.seeds + dpd.seeds:
        assert s.run_id == executed[(("pa-gru" if s in pa.seeds else "dpd-mp-ila"), s.seed)].run_id
        assert s.config_sha256 and s.checkpoint_sha256 and s.wall_clock_s is not None and "NMSE" in s.metrics
    assert {s.surrogate_run_id for s in dpd.seeds} == {executed[("pa-gru", s)].run_id for s in (0, 1, 2)}
    assert pa.aggregate["NMSE"].n == 3 and pa.aggregate["NMSE"].std is not None
    assert pa.aggregate["NMSE"].min <= pa.aggregate["NMSE"].mean <= pa.aggregate["NMSE"].max
    assert any("not evidence of generality" in n for n in report.notes)
    assert any("equal parameter counts are not equal compute cost" in n for n in report.notes)
    assert report.dataset.raw_sha256 and report.dataset.reported_split == "test" and "physical validation" in report.dataset.statement
    assert "hostname" not in json.dumps(report.machine) and str(ws.root) not in report.model_dump_json()
    md = bm.report_markdown(report)
    assert "| seed | run |" in md and "mean ± std" in md and executed[("pa-gru", 2)].run_id in md
    assert "ila_least_squares" in md and f"report sha256 `{report.report_sha256}`" in md


def test_an_edited_report_is_detected(ws, plan, executed, tmp_path):
    report = bm.build_report(ws, plan)
    path = tmp_path / "report.json"
    write_json_atomic(path, report)
    assert bm.load_report(path).report_sha256 == report.report_sha256
    payload = json.loads(path.read_text())
    payload["entries"][0]["aggregate"]["NMSE"]["mean"] -= 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(WorkspaceError, match="does not match its report_sha256"):
        bm.load_report(path)


def test_baselines_block_only_when_approved_and_only_for_their_plan(ws, plan, executed):
    report = bm.build_report(ws, plan)
    draft = bm.draft_baseline(report, tolerance_db=0.5)
    assert not draft.approved and draft.plan_sha256 == plan.plan_sha256
    band = draft.entries["pa-gru"]["NMSE"]
    assert band.worse_is == "higher" and band.tolerance >= 0.5
    assert draft.entries["dpd-mp-ila"]["ACLR_AVG"].worse_is == "higher"
    check = bm.check_regression(report, draft)
    assert check.ok and not check.approved and not check.blocking and "advisory" in check.verdict

    approved = draft.model_copy(update={"approved_by": "maintainer", "approved_on": date(2026, 9, 6)})
    assert bm.check_regression(report, approved).ok and not bm.check_regression(report, approved).blocking
    shifted = approved.model_copy(update={"entries": {
        "pa-gru": {"NMSE": band.model_copy(update={"reference": band.reference - 5.0, "tolerance": 0.1})}}})
    check = bm.check_regression(report, shifted)
    assert not check.ok and check.blocking and check.items[0].status == "degraded" and "blocked" in check.verdict
    better = approved.model_copy(update={"entries": {
        "pa-gru": {"NMSE": band.model_copy(update={"reference": band.reference + 5.0, "tolerance": 0.1})}}})
    check = bm.check_regression(report, better)
    assert not check.ok and check.items[0].status == "improved" and "re-approved" in check.verdict
    missing = approved.model_copy(update={"entries": {"pa-gru": {"EVM_X": band}}})
    assert bm.check_regression(report, missing).items[0].status == "missing"
    other = approved.model_copy(update={"plan_sha256": "f" * 64})
    with pytest.raises(WorkspaceError, match="approved for one pre-registered plan"):
        bm.check_regression(report, other)


def test_the_cli_walks_plan_run_report_check_and_exits_nonzero_on_a_blocked_release(ws, plan, executed, tmp_path, capsys):
    root = str(ws.root)
    rc = studio_main(["benchmark", "plan", "--dataset", "dpa-200mhz", "--seeds", "0,1", "--out", str(tmp_path / "bad.json")])
    assert rc == 2 and "at least 3" in capsys.readouterr().err
    bm.write_plan(plan, tmp_path / "plan.json")                       # the module plan (two entries) keeps the test short
    rc = studio_main(["benchmark", "run", str(tmp_path / "plan.json"), "--workspace", root, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0 and payload["failed"] == [] and len(payload["runs"]) == 6
    rc = studio_main(["benchmark", "report", str(tmp_path / "plan.json"), "--workspace", root,
                      "--out", str(tmp_path / "report.json"), "--markdown", str(tmp_path / "report.md")])
    assert rc == 0 and (tmp_path / "report.md").read_text().startswith("# benchmark-v1 report")
    capsys.readouterr()
    rc = studio_main(["benchmark", "baseline", str(tmp_path / "report.json"), "--out", str(tmp_path / "baseline.json")])
    assert rc == 0
    rc = studio_main(["benchmark", "check", str(tmp_path / "report.json"), "--baseline", str(tmp_path / "baseline.json")])
    out = capsys.readouterr().out
    assert rc == 0 and "advisory" in out and "within" in out
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    baseline["approved_by"], baseline["approved_on"] = "maintainer", "2026-09-06"
    baseline["entries"]["pa-gru"]["NMSE"]["reference"] -= 5.0
    baseline["entries"]["pa-gru"]["NMSE"]["tolerance"] = 0.1
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))
    rc = studio_main(["benchmark", "check", str(tmp_path / "report.json"), "--baseline", str(tmp_path / "baseline.json"), "--json"])
    check = json.loads(capsys.readouterr().out)
    nmse = next(i for i in check["items"] if i["entry_id"] == "pa-gru" and i["metric"] == "NMSE")
    assert rc == 1 and check["blocking"] and nmse["status"] == "degraded"
