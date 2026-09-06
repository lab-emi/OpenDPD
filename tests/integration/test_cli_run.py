"""Headless end-to-end runs through the experiments service (L1 integration).

Covers plan S02 acceptance: sequential runs do not leak configuration, the new
path never touches sys.argv, GUI/CLI/API share one resolver and one executor,
and the numbers equal what the legacy ``python main.py`` produces for the same
resolved configuration.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from opendpd.cli import studio_main
from opendpd.schemas import ArtifactKind, RunStatus
from opendpd.services.config import ConfigError
from opendpd.services.experiments import (
    bind_references,
    create_run,
    execute_run,
    load_artifacts,
    load_resolved,
    load_result,
    load_run,
)
from opendpd.services.legacy_adapter import build_namespace, legacy_cli_tokens
from opendpd.services.recipes import instantiate, run_dpd_config
from opendpd.services.workspace import Workspace

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "DPA_200MHz"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = Workspace.create(tmp_path_factory.mktemp("ws"))
    ws.register_builtin_dataset(DATASET)
    return ws


@pytest.fixture(scope="module")
def pa_run(workspace):
    argv_before = list(sys.argv)
    record = create_run(workspace, instantiate("pa-gru-smoke-v1", "dpa-200mhz"), idempotency_key="pa-smoke")
    record = execute_run(workspace, record.run_id)
    assert sys.argv == argv_before, "the explicit-config path must not touch sys.argv"
    assert record.status == RunStatus.succeeded, record.error
    return record


def test_pa_run_produces_verified_artifacts_and_result(workspace, pa_run):
    run_dir = workspace.run_dir(pa_run.run_id)
    for name in ("config.user.json", "config.resolved.json", "provenance.json", "run.json",
                 "artifacts.json", "result.json"):
        assert (run_dir / name).exists(), name
    manifest = load_artifacts(workspace, pa_run.run_id)
    assert manifest.complete
    ckpt = manifest.by_kind(ArtifactKind.checkpoint)[0]
    assert (run_dir / ckpt.file.path).exists() and ckpt.file.sha256
    assert ckpt.file.path.startswith("save/dpa-200mhz/train_pa/PA_S_0_M_GRU_H_23_F_50_P_")
    result = load_result(workspace, pa_run.run_id)
    assert result.evidence_type.value == "pa_modeling" and result.source == "opendpd-studio"
    assert result.metric("NMSE").value < 0 and result.metric("ACLR_AVG").value < 0
    assert result.models[0].weights_sha256 == ckpt.file.sha256
    assert any("smoke" in lim for lim in result.limitations)
    assert pa_run.progress_epoch == 3 and pa_run.worker is not None
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["legacy_equivalent_command"].startswith("python main.py")


def test_idempotency_key_returns_existing_run(workspace, pa_run):
    again = create_run(workspace, instantiate("pa-gru-smoke-v1", "dpa-200mhz"), idempotency_key="pa-smoke")
    assert again.run_id == pa_run.run_id


def test_sequential_runs_do_not_leak_configuration(workspace, pa_run):
    """A second experiment with different parameters must not inherit anything."""
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"parameters": {"hidden_size": 8}})})
    second = execute_run(workspace, create_run(workspace, cfg).run_id)
    assert second.status == RunStatus.succeeded
    first_cfg, second_cfg = load_resolved(workspace, pa_run.run_id), load_resolved(workspace, second.run_id)
    assert first_cfg.model.parameters["hidden_size"] == 23
    assert second_cfg.model.parameters["hidden_size"] == 8
    assert first_cfg.resolution.config_sha256 != second_cfg.resolution.config_sha256
    first_ckpt = load_artifacts(workspace, pa_run.run_id).by_kind(ArtifactKind.checkpoint)[0]
    second_ckpt = load_artifacts(workspace, second.run_id).by_kind(ArtifactKind.checkpoint)[0]
    assert "_H_23_" in first_ckpt.file.path and "_H_8_" in second_ckpt.file.path
    assert (workspace.run_dir(pa_run.run_id) / first_ckpt.file.path).exists()


def test_matches_legacy_cli_numerically(workspace, pa_run, tmp_path):
    """Same resolved config through `python main.py` must give the same log row."""
    resolved = load_resolved(workspace, pa_run.run_id)
    ns = build_namespace(resolved, dataset_dir=workspace.dataset_raw_dir("dpa-200mhz"), dataset_name="dpa-200mhz")
    cmd = [sys.executable, str(REPO_ROOT / "main.py"), *legacy_cli_tokens(ns)]
    env = dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1", KMP_DUPLICATE_LIB_OK="TRUE")
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    import pandas as pd
    legacy = pd.read_csv(next((tmp_path / "log").rglob("best/*.csv"))).iloc[-1]
    studio = load_result(workspace, pa_run.run_id)
    for name in ("NMSE", "EVM", "ACLR_AVG"):
        assert studio.metric(name).value == pytest.approx(float(legacy[f"TEST_{name}"]), abs=1e-6)


@pytest.fixture(scope="module")
def dpd_run(workspace, pa_run):
    cfg = instantiate("dpd-gru-smoke-v1", "dpa-200mhz", pa_run_id=pa_run.run_id)
    record = execute_run(workspace, create_run(workspace, cfg).run_id)
    assert record.status == RunStatus.succeeded, record.error
    return record


def test_dpd_run_binds_pa_surrogate(workspace, pa_run, dpd_run):
    resolved = load_resolved(workspace, dpd_run.run_id)
    pa_ckpt = load_artifacts(workspace, pa_run.run_id).by_kind(ArtifactKind.checkpoint)[0]
    assert resolved.pa_reference.run_id == pa_run.run_id
    assert resolved.pa_reference.checkpoint_sha256 == pa_ckpt.file.sha256
    assert resolved.pa_reference.model.key == "gru" and resolved.pa_reference.model.parameters["hidden_size"] == 23
    assert resolved.evaluation.checkpoint_selection_metric == "ACLR_AVG"
    result = load_result(workspace, dpd_run.run_id)
    assert result.evidence_type.value == "dpd_surrogate"
    assert {m.role for m in result.models} == {"pa", "dpd"}
    assert result.reference.kind == "linear_gain_target" and result.reference.gain_value > 0
    assert any("surrogate" in lim for lim in result.limitations)


def test_dpd_rejects_incompatible_pa(workspace, pa_run):
    cfg = instantiate("dpd-gru-smoke-v1", "dpa-200mhz", pa_run_id=pa_run.run_id, seed=1)
    with pytest.raises(ConfigError) as info:
        bind_references(workspace, cfg)
    assert "seed" in info.value.issues[0].message
    with pytest.raises(ConfigError) as info:
        bind_references(workspace, instantiate("dpd-gru-smoke-v1", "dpa-200mhz", pa_run_id="run-does-not-exist"))
    assert info.value.issues[0].field == "pa_reference.run_id"


def test_run_dpd_exports_predistorted_input(workspace, dpd_run):
    record = execute_run(workspace, create_run(workspace, run_dpd_config("dpa-200mhz", dpd_run.run_id)).run_id)
    assert record.status == RunStatus.succeeded, record.error
    manifest = load_artifacts(workspace, record.run_id)
    out = manifest.by_kind(ArtifactKind.dpd_output)
    assert out and "PA *input*" in out[0].description
    header = (workspace.run_dir(record.run_id) / out[0].file.path).read_text().splitlines()[0]
    assert header == "I,Q,I_dpd,Q_dpd"
    assert load_result(workspace, record.run_id) is None   # generating a signal is not an evaluation


def test_failed_run_records_error_not_ghost_state(workspace, pa_run):
    """Deleting the PA checkpoint after binding must fail the DPD run explicitly."""
    cfg = instantiate("dpd-gru-smoke-v1", "dpa-200mhz", pa_run_id=pa_run.run_id)
    record = create_run(workspace, cfg)
    ckpt = load_artifacts(workspace, pa_run.run_id).by_kind(ArtifactKind.checkpoint)[0]
    path = workspace.run_dir(pa_run.run_id) / ckpt.file.path
    backup = path.read_bytes()
    path.unlink()
    try:
        record = execute_run(workspace, record.run_id)
    finally:
        path.write_bytes(backup)
    assert record.status == RunStatus.failed
    assert record.error.code == "input_missing" and record.finished_at is not None
    assert load_run(workspace, record.run_id).status == RunStatus.failed


def test_cli_run_and_validate(workspace, tmp_path, capsys):
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    cfg_path = tmp_path / "exp.json"
    cfg_path.write_text(cfg.model_dump_json())
    assert studio_main(["validate", "--config", str(cfg_path)]) == 0
    assert "ok: resolved config sha256" in capsys.readouterr().out
    code = studio_main(["run", "--config", str(cfg_path), "--workspace", str(workspace.root),
                        "--idempotency-key", "cli-pa-smoke"])
    out = capsys.readouterr().out
    assert code == 0 and "succeeded" in out and "NMSE" in out and "smoke" in out
    # unknown model through the CLI: structured error, exit 2, nothing executed
    bad = json.loads(cfg_path.read_text()); bad["model"]["key"] = "transformer"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert studio_main(["run", "--config", str(tmp_path / "bad.json"), "--workspace", str(workspace.root)]) == 2
    err = capsys.readouterr().err
    assert "model.key" in err and "unknown model" in err


def test_cli_lists_models_datasets_and_recipes(workspace, capsys):
    assert studio_main(["models"]) == 0
    assert "tres_deltagru" in capsys.readouterr().out
    assert studio_main(["recipes"]) == 0
    assert "pa-gru-smoke-v1" in capsys.readouterr().out
    assert studio_main(["datasets", "list", "--workspace", str(workspace.root)]) == 0
    assert "dpa-200mhz" in capsys.readouterr().out


def test_cli_import_doctor_preprocess_and_train_on_a_version(tmp_path, capsys):
    """J2 headless: import my CSV -> doctor -> preprocess -> train on the new version -> result."""
    from tests.fixtures.synthetic import Impairments, write_dataset

    write_dataset(tmp_path / "src", 20000, 9, 800e6, 200e6, Impairments(delay_samples=6), fmt="csv")
    ws = ["--workspace", str(tmp_path / "ws")]
    assert studio_main(["datasets", "import", str(tmp_path / "src" / "data.csv"), "--id", "mine", "--fs", "800e6",
                        "--bandwidth", "200e6", "--n-sub-ch", "10", "--nperseg", "2560", "--units", "normalized",
                        "--origin", "synthetic", *ws]) == 0
    capsys.readouterr()
    assert studio_main(["datasets", "doctor", "mine", "--json", *ws]) == 0
    report = json.loads(capsys.readouterr().out)
    delay = next(i for i in report["items"] if i["code"] == "time_misalignment")["evidence"]["delay_samples"]
    assert abs(delay - 6) < 0.1
    assert studio_main(["datasets", "preprocess", "mine", "--version", "aligned-v1", "--delay", str(round(delay)), *ws]) == 0
    capsys.readouterr()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"task": "train_pa", "recipe_id": "pa-gru-smoke-v1",
                               "dataset": {"id": "mine", "preprocessing_version": "aligned-v1"},
                               "model": {"key": "gru", "parameters": {"hidden_size": 8, "num_layers": 1}},
                               "training": {"epochs": 1, "frame_length": 50, "frame_stride": 16, "batch_size_eval": 256}}))
    assert studio_main(["run", "--config", str(cfg), "--json", *ws]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert payload["run"]["status"] == "succeeded"
    assert payload["result"]["dataset"]["preprocessing_version"] == "aligned-v1"
    assert payload["result"]["metrics"][0]["value"] is not None
