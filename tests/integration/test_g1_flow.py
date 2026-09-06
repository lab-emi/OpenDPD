"""G1 flow on the measured built-in data, machine side (plan S11): import → diagnose → PA → DPD → evaluate → export →
re-evaluation from the package in another workspace, all through the CLI. The human side of the gate (an internal user
not involved in the implementation completing the flow independently) is recorded in docs/releases/studio-progress.md."""

import json

import pytest

from opendpd.cli import studio_main

pytestmark = pytest.mark.integration

REL, ABS = 1e-4, 1e-5          # frozen checkpoint re-evaluation tolerance (docs/protocols/acceptance-thresholds.md)


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _assert_metrics_close(stored, recomputed):
    for m in stored:
        other = next(x for x in recomputed if x["name"] == m["name"])
        assert other["status"] == m["status"], m["name"]
        if m["value"] is not None:
            assert other["value"] == pytest.approx(m["value"], rel=REL, abs=ABS), m["name"]


def test_g1_flow_import_diagnose_pa_dpd_evaluate_export_recompute(tmp_path, capsys):
    ws, ws2 = str(tmp_path / "ws"), str(tmp_path / "ws2")

    # import the measured built-in data and diagnose it
    assert studio_main(["datasets", "import-builtin", "DPA_200MHz", "--workspace", ws]) == 0
    dataset_id = capsys.readouterr().out.split("'")[1]
    assert studio_main(["datasets", "doctor", dataset_id, "--workspace", ws, "--json"]) == 0
    doctor = _json(capsys)
    assert doctor["evaluation_blocked"] is False and doctor["items"]

    # PA surrogate, then DPD through it
    assert studio_main(["run", "--recipe", "pa-gru-smoke-v1", "--dataset", dataset_id, "--workspace", ws, "--json"]) == 0
    pa = _json(capsys)["run"]
    assert pa["status"] == "succeeded", pa.get("error")
    assert studio_main(["run", "--recipe", "dpd-gru-smoke-v1", "--dataset", dataset_id, "--pa-run", pa["run_id"],
                        "--workspace", ws, "--json"]) == 0
    dpd = _json(capsys)["run"]
    assert dpd["status"] == "succeeded", dpd.get("error")

    # evaluate under another registered profile; the stored (primary) result stays the legacy one
    assert studio_main(["evaluate", dpd["run_id"], "--workspace", ws, "--profile", "general-spectral-v1", "--json"]) == 0
    general = _json(capsys)
    assert general["metric_profile_id"] == "general-spectral-v1" and general["evidence_type"] == "dpd_surrogate"
    assert [s["symbol"] for s in general["signal_chain"]] == ["x", "u", "y"]
    stored = json.loads((tmp_path / "ws" / "runs" / dpd["run_id"] / "result.json").read_text())
    assert stored["metric_profile_id"] == "legacy-opendpd-v1"

    # export: the built-in data is not copied, the PA surrogate travels with its checkpoint hash
    out = tmp_path / "dpd-full.zip"
    assert studio_main(["export", dpd["run_id"], "--workspace", ws, "--kind", "full", "--out", str(out), "--json"]) == 0
    manifest = _json(capsys)["manifest"]
    assert manifest["dataset"]["source_kind"] == "builtin" and manifest["dataset"]["included"] is False
    assert [(r["role"], r["run_id"]) for r in manifest["references"]] == [("pa_surrogate", pa["run_id"])]
    assert manifest["missing"] == []

    # another workspace: verified import, built-in data registered and hash-checked, CLI re-evaluation within tolerance
    assert studio_main(["import", str(out), "--workspace", ws2, "--json"]) == 0
    imported = _json(capsys)
    assert imported["dataset_status"] == "registered_builtin"
    assert set(imported["imported_runs"]) == {pa["run_id"], dpd["run_id"]}
    assert studio_main(["evaluate", dpd["run_id"], "--workspace", ws2, "--profile", "legacy-opendpd-v1", "--json"]) == 0
    recomputed = _json(capsys)
    _assert_metrics_close(stored["metrics"], recomputed["metrics"])
    for baseline in stored["baselines"]:
        other = next(b for b in recomputed["baselines"] if b["kind"] == baseline["kind"])
        _assert_metrics_close(baseline["metrics"], other["metrics"])
    assert recomputed["models"][0]["weights_sha256"] == stored["models"][0]["weights_sha256"]
