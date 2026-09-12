"""The adapter must reproduce exactly what the legacy CLI would parse."""

import os
from types import SimpleNamespace

import pytest

from opendpd.schemas import ModelSpec, PAReference
from opendpd.schemas.examples import SHA_A, experiment_train_dpd_smoke, experiment_train_pa_smoke
from opendpd.services.config import resolve
from opendpd.services.legacy_adapter import (
    RunCancelled,
    attach_epoch_hooks,
    build_namespace,
    legacy_cli_tokens,
    legacy_command_line,
    run_in_directory,
)


def test_namespace_round_trips_through_the_legacy_parser(tmp_path):
    from arguments import build_parser

    resolved = resolve(experiment_train_pa_smoke())
    ns = build_namespace(resolved, dataset_dir=tmp_path, dataset_name="dpa-200mhz")
    tokens = legacy_cli_tokens(ns)
    reparsed = build_parser().parse_args(tokens)
    assert vars(reparsed) == vars(ns)
    assert ns.step == "train_pa" and ns.n_epochs == 3 and ns.frame_length == 50 and ns.frame_stride == 16
    assert ns.PA_backbone == "gru" and ns.PA_hidden_size == 23 and ns.accelerator == "cpu"
    assert ns.dataset_path == str(tmp_path) and ns.plot is False
    assert legacy_command_line(ns).startswith("python main.py --dataset_name dpa-200mhz")


def test_dpd_namespace_binds_pa_and_dpd_models(tmp_path):
    cfg = experiment_train_dpd_smoke().model_copy(update={"pa_reference": PAReference(
        run_id="run-pa-0001", checkpoint_sha256=SHA_A,
        model=ModelSpec(key="tres_gru", parameters={"hidden_size": 27, "num_layers": 1}))})
    resolved = resolve(cfg)
    ns = build_namespace(resolved, dataset_dir=tmp_path, dataset_name="ds")
    assert ns.step == "train_dpd"
    assert (ns.PA_backbone, ns.PA_hidden_size) == ("tres_gru", 27)
    assert (ns.DPD_backbone, ns.DPD_hidden_size) == ("gru", 15)


def test_run_in_directory_restores_cwd(tmp_path):
    before = os.getcwd()
    with run_in_directory(tmp_path / "run") as run_dir:
        assert os.getcwd() == str(run_dir.resolve())
    assert os.getcwd() == before
    with pytest.raises(RuntimeError):
        with run_in_directory(tmp_path / "run2"):
            raise RuntimeError("boom")
    assert os.getcwd() == before


class _FakeLogger:
    def __init__(self):
        self.rows = []

    def write_log(self, log_stat):
        self.rows.append(dict(log_stat))


def _fake_project():
    project = SimpleNamespace()

    def build_logger(model_id):
        project.logger = _FakeLogger()

    project.build_logger = build_logger
    return project


def test_epoch_hooks_observe_without_changing_logs():
    project = _fake_project()
    seen = []
    attach_epoch_hooks(project, seen.append)
    project.build_logger("m")
    project.logger.write_log({"EPOCH": 0, "VAL_NMSE": -1.0})
    assert seen == [{"EPOCH": 0, "VAL_NMSE": -1.0}]
    assert project.logger.rows == seen


def test_cancel_is_raised_at_epoch_boundary():
    project = _fake_project()
    attach_epoch_hooks(project, lambda row: None, should_cancel=lambda: True)
    project.build_logger("m")
    with pytest.raises(RunCancelled):
        project.logger.write_log({"EPOCH": 4})
    assert project.logger.rows == [{"EPOCH": 4}]   # the epoch was still recorded
