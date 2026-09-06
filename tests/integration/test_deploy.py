"""S19: a deployment package for a finished GRU run; the C99 reference verified against the software reference."""

import json
import zipfile

import pandas as pd
import pytest

from opendpd.commands import main
from opendpd.schemas import DatasetOrigin, RunStatus, SignalSpec
from opendpd.services import datasets as ds
from opendpd.services import deploy
from opendpd.services.experiments import create_run, execute_run
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace, WorkspaceError
from tests.fixtures.synthetic import Impairments, synthesize

SIGNAL = dict(sample_rate_hz=800e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560, amplitude_units="normalized")


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    x, y = synthesize(16000, 8, impairments=Impairments())
    path = tmp_path_factory.mktemp("src") / "capture.csv"
    pd.DataFrame({"I_in": x[:, 0], "Q_in": x[:, 1], "I_out": y[:, 0], "Q_out": y[:, 1]}).to_csv(path, index=False)
    ws = Workspace.create(tmp_path_factory.mktemp("ws"))
    ds.import_dataset(ws, path, dataset_id="capture", display_name="capture (synthetic)", signal=SignalSpec(**SIGNAL),
                      origin=DatasetOrigin.synthetic, guard_samples=64)
    return ws


def _run(ws, config):
    record = execute_run(ws, create_run(ws, config).run_id)
    assert record.status == RunStatus.succeeded, record.error
    return record


@pytest.fixture(scope="module")
def pa(ws):
    return _run(ws, instantiate("pa-gru-smoke-v1", "capture"))


@pytest.fixture(scope="module")
def package(ws, pa, tmp_path_factory):
    out = tmp_path_factory.mktemp("pkg") / "deploy.zip"
    return deploy.export_deployment(ws, pa.run_id, out), out


def test_the_package_holds_spec_weights_golden_vectors_sources_and_a_hash_per_file(package):
    manifest, out = package
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        listed = DeploymentManifestFiles = json.loads(zf.read("manifest.json"))["files"]
        assert {"spec.json", "weights.json", "README.md", "c/gru_fixed.h", "c/gru_fixed.c", "c/harness.c"} <= names
        assert {f"golden/{g.case_id}/{f}" for g in manifest.golden for f in ("x.i16", "y.i16", "h_final.i16", "h_trace.i16", "meta.json")} <= names
        assert set(listed) == names - {"manifest.json"}
        weights = json.loads(zf.read("weights.json"))
        assert weights["spec_id"] == "fixed-point-v1" and weights["gate_order"] == ["r", "z", "n"] and len(weights["sigmoid_table"]) == 4096
        assert "#define GRU_HIDDEN 23" in zf.read("c/gru_fixed.h").decode()
    assert [g.case_id for g in manifest.golden] == ["normal", "extreme", "saturation", "all_zero", "state_reset", "long_sequence"]
    assert next(g for g in manifest.golden if g.case_id == "state_reset").resets_at == [0, 256, 512, 768]
    assert next(g for g in manifest.golden if g.case_id == "long_sequence").n_samples == deploy.LONG_SEQUENCE
    assert manifest.spec.spec_id == "fixed-point-v1" and manifest.model_key == "gru" and manifest.hidden_size == 23
    assert {t.name for t in manifest.tensors} == {"w_ih", "w_hh", "w_out", "b_ih", "b_hh", "b_out"} and all(t.saturated == 0 for t in manifest.tensors)
    assert manifest.weights_sha256 and DeploymentManifestFiles


def test_the_c_reference_is_verified_bit_for_bit_or_the_package_says_it_was_not(package):
    manifest, _ = package
    v = manifest.verification
    from opendpd.export.c_backend import compiler
    if compiler() is None:
        assert v.status == "not_run" and "no C compiler" in v.detail
    else:
        assert v.status == "bit_exact" and v.cases_checked == 6 and v.compiler
        assert manifest.report.measured_execution is not None and manifest.report.measured_execution.samples_per_second > 0
        assert manifest.report.measured_execution.label == "measured_execution_time"


def test_the_report_labels_every_number_by_how_it_was_obtained(package):
    manifest, _ = package
    r = manifest.report
    names = {d.name for d in r.quality_loss}
    assert "NMSE" in names and r.metric_profile_id == "legacy-opendpd-v1"
    nmse = next(d for d in r.quality_loss if d.name == "NMSE")
    assert nmse.float_value is not None and nmse.fixed_value is not None and nmse.delta == pytest.approx(nmse.fixed_value - nmse.float_value)
    assert abs(nmse.delta) < 3.0                                   # 16-bit words: a fraction of a dB on a smoke model
    assert r.resources.label == "theoretical" and r.resources.mac_per_sample == 3 * 23 * 25 + 2 * 23
    assert r.resources.state_bytes == 46 and r.resources.weight_bytes == (69 * 2 + 69 * 23 + 46) * 2
    assert r.synthesis_estimate is None and r.measured_power is None
    assert any("state" in a and "carried" in a for a in r.execution_assumptions)
    md = deploy.report_markdown(manifest)
    assert "Label **theoretical**" in md and "not available (nothing was synthesised)" in md and "never inferred from MAC" in md


def test_unsupported_models_are_refused_with_the_reason(ws, pa, tmp_path):
    assert deploy.support("gru") is None and deploy.support("gru_stream") is None
    assert "no fixed-point specification" in deploy.support("lstm")
    lstm = _run(ws, instantiate("pa-gru-smoke-v1", "capture").model_copy(update={"model": instantiate("pa-gru-smoke-v1", "capture").model.model_copy(update={"key": "lstm"})}))
    with pytest.raises(WorkspaceError, match="no fixed-point specification"):
        deploy.export_deployment(ws, lstm.run_id, tmp_path / "x.zip")
    assert not (tmp_path / "x.zip").exists()


def test_the_cli_writes_a_package_and_reports_the_verdict(ws, pa, tmp_path, capsys):
    out = tmp_path / "pa-deploy.zip"
    assert main(["deploy", pa.run_id, "--workspace", str(ws.root), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "verification (c99)" in text and "theoretical: 1771 MAC/sample" in text and out.exists()
    assert deploy.read_manifest(out).run_id == pa.run_id


def test_a_dpd_run_exports_with_its_loss_through_the_surrogate(ws, pa, tmp_path):
    dpd = _run(ws, instantiate("dpd-gru-smoke-v1", "capture", pa_run_id=pa.run_id))
    manifest = deploy.export_deployment(ws, dpd.run_id, tmp_path / "dpd.zip")
    assert manifest.hidden_size == 15 and manifest.run_id == dpd.run_id
    assert any(d.name == "ACLR_AVG" and d.delta is not None for d in manifest.report.quality_loss)
