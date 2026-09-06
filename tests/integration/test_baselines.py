"""Least-squares MP/GMP baselines as runs (plan S12): PA fits with a recorded fit, ILA predistorters scored through a
gradient-trained surrogate with their training path stated, refusal as a surrogate, apply, determinism."""

import json

import pandas as pd
import pytest

from opendpd.schemas import DatasetOrigin, RunStatus, SignalSpec
from opendpd.services import datasets as ds
from opendpd.services.config import ConfigError
from opendpd.services.evaluation import compare_results, evaluate_run
from opendpd.services.experiments import create_run, execute_run, lineage, load_artifacts, load_result, training_history
from opendpd.services.recipes import instantiate, run_dpd_config
from opendpd.services.workspace import Workspace
from tests.fixtures.synthetic import Impairments, synthesize

pytestmark = pytest.mark.integration

SIGNAL = dict(sample_rate_hz=800e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560, amplitude_units="normalized")


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    x, y = synthesize(24000, 7, impairments=Impairments(delay_samples=0, n_outliers=0))
    path = tmp_path_factory.mktemp("src") / "capture.csv"
    pd.DataFrame({"I_in": x[:, 0], "Q_in": x[:, 1], "I_out": y[:, 0], "Q_out": y[:, 1]}).to_csv(path, index=False)
    ws = Workspace.create(tmp_path_factory.mktemp("ws"))
    ds.import_dataset(ws, path, dataset_id="capture", display_name="capture (synthetic MP PA)", signal=SignalSpec(**SIGNAL),
                      origin=DatasetOrigin.synthetic, guard_samples=64)
    return ws


def _with_params(config, params):
    return config.model_copy(update={"model": config.model.model_copy(update={"parameters": params})})


def _run(ws, config):
    record = execute_run(ws, create_run(ws, config).run_id)
    assert record.status == RunStatus.succeeded, record.error
    return record


@pytest.fixture(scope="module")
def mp_pa(ws):
    # orders 1/3/5 with memory 3: exactly the structure of the synthetic PA, so the fit is exact up to float32 data
    return _run(ws, _with_params(instantiate("pa-mp-ls-v1", "capture"), {"K": 5, "Q": 3, "rcond": 0.0}))


@pytest.fixture(scope="module")
def gru_pa(ws):
    return _run(ws, instantiate("pa-gru-smoke-v1", "capture"))


@pytest.fixture(scope="module")
def mp_dpd(ws, gru_pa):
    return _run(ws, _with_params(instantiate("dpd-mp-ila-v1", "capture", pa_run_id=gru_pa.run_id),
                                 {"K": 5, "Q": 4, "rcond": 0.0}))


def test_mp_fit_identifies_the_synthetic_pa_and_records_the_fit(ws, mp_pa):
    result = load_result(ws, mp_pa.run_id)
    assert result.evidence_type.value == "pa_modeling"
    nmse = result.metric("NMSE").value
    assert nmse < -50                      # float32 data: the noise floor of an exact-structure fit
    model = result.models[0]
    assert model.training_path == "least_squares" and model.n_parameters == 30 and model.lookahead_samples == 0
    assert result.selected_epoch is None
    assert any(lim.startswith("least-squares fit: rank 15 of 15") for lim in result.limitations)
    assert not any("smoke" in lim for lim in result.limitations)
    manifest = load_artifacts(ws, mp_pa.run_id)
    ids = {a.artifact_id for a in manifest.artifacts}
    assert manifest.complete and ids >= {"checkpoint-best", "log-history", "log-best", "fit-diagnostics", "result", "plot-spectrum"}
    fit = json.loads((ws.run_dir(mp_pa.run_id) / "fit.json").read_text())
    assert fit["role"] == "pa" and fit["diagnostics"]["rank"] == 15 and fit["diagnostics"]["train_nmse_db"] < -50
    history = training_history(ws, mp_pa.run_id)
    assert [(h.epoch, h.split) for h in history] == [(0, "val"), (0, "test")] and "NMSE" in history[0].values
    provenance = json.loads((ws.run_dir(mp_pa.run_id) / "provenance.json").read_text())
    assert provenance["legacy_equivalent_command"] is None
    again = evaluate_run(ws, mp_pa.run_id, "legacy-opendpd-v1")          # deterministic re-evaluation
    assert again.metric("NMSE").value == pytest.approx(nmse, abs=1e-6)


def test_gmp_fit_with_a_cutoff_reports_the_retained_rank_and_its_lookahead(ws):
    params = {"Ka": 3, "La": 3, "Kb": 2, "Lb": 3, "Mb": 1, "Kc": 2, "Lc": 3, "Mc": 1, "rcond": 1e-3}
    record = _run(ws, _with_params(instantiate("pa-gmp-ls-v1", "capture"), params))
    d = json.loads((ws.run_dir(record.run_id) / "fit.json").read_text())["diagnostics"]
    assert d["n_coefficients"] == 3 * 3 + 2 * 3 * 1 + 2 * 3 * 1 and d["rcond"] == 1e-3 and 0 < d["rank"] <= d["n_coefficients"]
    result = load_result(ws, record.run_id)
    assert result.models[0].lookahead_samples == 1 and result.models[0].training_path == "least_squares"
    assert any("cutoff rcond=0.001" in lim for lim in result.limitations)


def test_ila_dpd_is_scored_through_the_gradient_surrogate_and_states_its_path(ws, gru_pa, mp_dpd):
    result = load_result(ws, mp_dpd.run_id)
    assert result.evidence_type.value == "dpd_surrogate"
    assert {m.role: m.training_path for m in result.models} == {"dpd": "ila_least_squares", "pa": "gradient"}
    assert [s.symbol for s in result.signal_chain] == ["x", "u", "y"] and result.signal_chain[2].simulated
    assert any(lim.startswith("indirect learning") for lim in result.limitations)
    assert result.selected_epoch is None and result.metric("NMSE").value is not None
    assert training_history(ws, mp_dpd.run_id) == []           # identification residuals only, no per-epoch metrics
    fit = json.loads((ws.run_dir(mp_dpd.run_id) / "fit.json").read_text())
    assert fit["role"] == "dpd" and fit["reference_gain"] > 0 and fit["method"].startswith("indirect learning")
    assert lineage(ws, mp_dpd.run_id).parents[0].run_id == gru_pa.run_id


def test_compare_states_the_training_path_but_ranks_under_one_protocol(ws, gru_pa, mp_dpd):
    gru_dpd = _run(ws, instantiate("dpd-gru-smoke-v1", "capture", pa_run_id=gru_pa.run_id))
    report = compare_results(ws, [mp_dpd.run_id, gru_dpd.run_id])
    assert report.comparable and report.pairs[0].incompatibilities == []
    paths = {r.run_id: {m.role: m.training_path for m in r.models} for r in report.results}
    assert paths[mp_dpd.run_id]["dpd"] == "ila_least_squares" and paths[gru_dpd.run_id]["dpd"] == "gradient_dla"


def test_a_least_squares_pa_is_refused_as_a_dpd_surrogate(ws, mp_pa):
    with pytest.raises(ConfigError) as info:
        create_run(ws, instantiate("dpd-gru-smoke-v1", "capture", pa_run_id=mp_pa.run_id))
    issue = info.value.issues[0]
    assert "least-squares baseline" in issue.message and "pa-gru" in (issue.hint or "")
    with pytest.raises(ConfigError):
        create_run(ws, instantiate("dpd-mp-ila-v1", "capture", pa_run_id=mp_pa.run_id))


def test_quantisation_is_refused_for_a_fit(ws):
    from opendpd.schemas import QuantizationConfig

    config = instantiate("pa-mp-ls-v1", "capture").model_copy(update={"quantization": QuantizationConfig(enabled=True)})
    with pytest.raises(ConfigError) as info:
        create_run(ws, config)
    assert info.value.issues[0].field == "quantization.enabled"


def test_apply_of_a_fitted_dpd_exports_u_and_reproduces_its_result(ws, mp_dpd):
    record = _run(ws, run_dpd_config("capture", mp_dpd.run_id))
    manifest = load_artifacts(ws, record.run_id)
    csv = next(a for a in manifest.artifacts if a.artifact_id == "dpd-output")
    frame = pd.read_csv(ws.run_dir(record.run_id) / csv.file.path)
    assert list(frame.columns) == ["I", "Q", "I_dpd", "Q_dpd"] and len(frame) > 0
    meta = json.loads((ws.run_dir(record.run_id) / csv.file.path).with_suffix(".meta.json").read_text())
    assert meta["signal_role"] == "pa_input_predistorted" and meta["dpd"]["model"]["key"] == "mp_ls"
    applied, trained = load_result(ws, record.run_id), load_result(ws, mp_dpd.run_id)
    for m in trained.metrics:
        assert applied.metric(m.name).value == pytest.approx(m.value, abs=1e-3), m.name
    assert applied.models[0].training_path == "ila_least_squares" and applied.selected_epoch is None
    assert {link.relation.value for link in lineage(ws, record.run_id).parents} == {"dpd_model", "pa_surrogate"}
