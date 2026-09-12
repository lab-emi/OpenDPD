"""S18: trained weights scored under streaming semantics through the shared executor (gru_stream, gmp_stream)."""

import pandas as pd
import pytest

from opendpd.commands import main
from opendpd.core.registry import get_model, streaming_variant_of
from opendpd.schemas import DatasetOrigin, RunStatus, SignalSpec, TaskType
from opendpd.services import datasets as ds
from opendpd.services.config import ConfigError
from opendpd.services.evaluation import compare_results
from opendpd.services.experiments import create_run, execute_run, lineage, load_resolved, load_result
from opendpd.services.recipes import instantiate
from opendpd.services.streaming import stream_config
from opendpd.services.workspace import Workspace, WorkspaceError
from tests.fixtures.synthetic import Impairments, synthesize

SIGNAL = dict(sample_rate_hz=800e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560, amplitude_units="normalized")


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    x, y = synthesize(16000, 5, impairments=Impairments())
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
def dpd(ws, pa):
    return _run(ws, instantiate("dpd-gru-smoke-v1", "capture", pa_run_id=pa.run_id))


@pytest.fixture(scope="module")
def streamed_pa(ws, pa):
    return _run(ws, stream_config(ws, pa.run_id, chunk_samples=333))


def test_the_registry_names_the_variant_and_it_cannot_be_trained(ws):
    variant = streaming_variant_of("gru")
    assert variant.key == "gru_stream" and variant.weights_from == "gru" and variant.execution_semantics == "streaming_stateful"
    assert variant.status == "experimental" and get_model("gru").execution_semantics == "offline_segmented"
    assert streaming_variant_of("lstm") is None
    config = instantiate("pa-gru-smoke-v1", "capture").model_copy(update={"model": instantiate("pa-gru-smoke-v1", "capture").model.model_copy(update={"key": "gru_stream"})})
    with pytest.raises(ConfigError, match="streaming variant that executes the weights of 'gru'"):
        create_run(ws, config)


def test_a_pa_run_is_rescored_under_streaming_semantics_with_its_evidence(ws, pa, streamed_pa):
    resolved = load_resolved(ws, streamed_pa.run_id)
    assert resolved.task == TaskType.evaluate_pa and resolved.model.key == "gru_stream"
    assert resolved.model.parameters == load_resolved(ws, pa.run_id).model.parameters
    result = load_result(ws, streamed_pa.run_id)
    e = result.execution
    assert e is not None and e.semantics == "streaming_stateful" and e.state == "recurrent" and e.chunk_samples == 333
    assert e.lookahead_samples == 0 and e.lookahead_s == 0.0 and e.history_samples is None
    assert e.consistency.within_tolerance and e.consistency.max_abs_error <= e.consistency.tolerance
    assert e.warmup_samples is not None and e.warmup_samples >= 0
    assert result.models[0].model.key == "gru_stream" and result.models[0].execution_semantics == "streaming_stateful"
    assert any("not comparable with offline_segmented results of gru" in lim for lim in result.limitations)
    offline = load_result(ws, pa.run_id)
    assert offline.execution is None and offline.models[0].execution_semantics == "offline_segmented"
    assert ("pa_model", pa.run_id) in {(p.relation.value, p.run_id) for p in lineage(ws, streamed_pa.run_id).parents}


def test_streaming_and_offline_results_are_never_ranked(ws, pa, streamed_pa):
    report = compare_results(ws, [pa.run_id, streamed_pa.run_id])
    assert not report.comparable
    assert any("execution semantics" in reason for reason in report.pairs[0].incompatibilities)
    nmse = {r.run_id: next(m.value for m in r.metrics if m.name == "NMSE") for r in report.results}
    assert nmse[pa.run_id] != nmse[streamed_pa.run_id]          # re-evaluated, not inherited


def test_a_dpd_run_streams_the_dpd_through_the_offline_surrogate(ws, dpd):
    record = _run(ws, stream_config(ws, dpd.run_id))
    resolved = load_resolved(ws, record.run_id)
    assert resolved.task == TaskType.run_dpd and resolved.model.key == "gru_stream"
    result = load_result(ws, record.run_id)
    assert result.execution.chunk_samples == 1024 and result.execution.consistency.within_tolerance
    roles = {m.role: m for m in result.models}
    assert roles["dpd"].model.key == "gru_stream" and roles["dpd"].execution_semantics == "streaming_stateful"
    assert roles["pa"].model.key == "gru" and roles["pa"].execution_semantics == "offline_segmented"
    parents = {(p.relation.value, p.run_id) for p in lineage(ws, record.run_id).parents}
    assert ("dpd_model", dpd.run_id) in parents


def test_the_windowed_variant_carries_its_history(ws):
    config = instantiate("pa-gru-smoke-v1", "capture").model_copy(update={"model": instantiate("pa-gru-smoke-v1", "capture").model.model_copy(update={"key": "gmp", "parameters": {}})})
    gmp = _run(ws, config)
    record = _run(ws, stream_config(ws, gmp.run_id, chunk_samples=100))
    e = load_result(ws, record.run_id).execution
    # the GMP's envelope windows lag its signal windows: 20 samples of reach; the measured warm-up is the number of
    # those whose contribution exceeds the tolerance on this signal (the farthest lags are the weakest terms)
    assert e.state == "window" and e.history_samples == 20 and e.lookahead_samples == 0 and 15 <= e.warmup_samples <= 20
    assert e.consistency.within_tolerance


def test_the_cli_streams_a_run_and_refuses_what_has_no_variant(ws, pa, tmp_path, capsys):
    assert main(["stream", pa.run_id, "--workspace", str(ws.root), "--chunk", "512"]) == 0
    out = capsys.readouterr().out
    assert "gru_stream: streaming_stateful (recurrent state), chunk 512 samples" in out and "within 0.0001" in out
    lstm = _run(ws, instantiate("pa-gru-smoke-v1", "capture").model_copy(update={"model": instantiate("pa-gru-smoke-v1", "capture").model.model_copy(update={"key": "lstm"})}))
    assert main(["stream", lstm.run_id, "--workspace", str(ws.root)]) == 2
    assert "no registered streaming variant" in capsys.readouterr().err
    with pytest.raises(WorkspaceError, match="starts from a succeeded train_pa or train_dpd run"):
        stream_config(ws, load_resolved(ws, pa.run_id) and pa.run_id if False else _stream_run_id(ws))


def _stream_run_id(ws):
    from opendpd.services.experiments import list_runs
    return next(r.run_id for r in list_runs(ws) if r.task == TaskType.evaluate_pa)
