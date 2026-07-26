"""Tests for the public opendpd Python API (opendpd/api.py)."""

import inspect
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import opendpd
from conftest import REPO_ROOT, SMOKE_DATASET


@pytest.fixture
def preserved_argv():
    """The API passes parameters by rewriting sys.argv — keep tests isolated."""
    argv = sys.argv[:]
    yield
    sys.argv = argv


class TestPublicSurface:
    def test_documented_functions_exposed(self):
        for name in (
            "train_pa",
            "train_dpd",
            "run_dpd",
            "plot_dpd",
            "load_dataset",
            "create_dataset",
            "OpenDPDTrainer",
        ):
            assert callable(getattr(opendpd, name)), f"opendpd.{name} missing"

    def test_version_defined(self):
        assert isinstance(opendpd.__version__, str) and opendpd.__version__


class TestLoadDataset:
    def test_loads_builtin_split_dataset(self):
        data = opendpd.load_dataset(str(REPO_ROOT / "datasets" / SMOKE_DATASET))
        assert set(data) == {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
        for key, array in data.items():
            assert array.shape[-1] == 2, f"{key} must be I/Q pairs"
            assert len(array) > 0
        assert len(data["X_train"]) == len(data["y_train"])


class TestCreateDataset:
    @pytest.fixture
    def measurement_csv(self, tmp_path):
        n = 100
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(size=(n, 4)), columns=["I_in", "Q_in", "I_out", "Q_out"]
        )
        csv_path = tmp_path / "measurements.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.mark.parametrize("dataset_format", ["single_csv", "split_csv"])
    def test_create_and_load_roundtrip(self, tmp_path, measurement_csv, dataset_format):
        dataset_dir = opendpd.create_dataset(
            csv_path=str(measurement_csv),
            output_dir=str(tmp_path / "datasets"),
            dataset_name="TestPA",
            dataset_format=dataset_format,
            input_signal_fs=800e6,
            bw_main_ch=200e6,
        )
        data = opendpd.load_dataset(dataset_dir)
        assert len(data["X_train"]) == 60  # 0.6 * 100
        assert len(data["X_val"]) == 20
        assert len(data["X_test"]) == 20

    def test_rejects_wrong_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1.0], "b": [2.0]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="must contain columns"):
            opendpd.create_dataset(
                csv_path=str(bad_csv),
                output_dir=str(tmp_path),
                dataset_name="Bad",
            )


class TestApiTraining:
    @pytest.mark.parametrize("function_name", ["train_pa", "train_dpd"])
    def test_benchmark_recipe_defaults_are_public_api_defaults(self, function_name):
        signature = inspect.signature(getattr(opendpd, function_name))

        assert signature.parameters["n_epochs"].default == 300
        assert signature.parameters["batch_size"].default == 64
        assert signature.parameters["lr"].default == 5e-3

    def test_adamw_and_scheduler_recipe_is_explicit(self, monkeypatch):
        import project as project_module

        calls = {}
        optimizer_sentinel = object()
        scheduler_sentinel = object()

        def capture_adamw(params, **kwargs):
            calls["optimizer_params"] = tuple(params)
            calls["optimizer_kwargs"] = kwargs
            return optimizer_sentinel

        def capture_scheduler(**kwargs):
            calls["scheduler_kwargs"] = kwargs
            return scheduler_sentinel

        monkeypatch.setattr(project_module.optim, "AdamW", capture_adamw)
        monkeypatch.setattr(
            project_module.optim.lr_scheduler,
            "ReduceLROnPlateau",
            capture_scheduler,
        )

        project = project_module.Project.__new__(project_module.Project)
        project.opt_type = "adamw"
        project.lr = 5e-3
        project.decay_factor = 0.5
        project.patience = 5
        project.lr_end = 5e-5
        net = torch.nn.Linear(2, 2)

        optimizer, scheduler = project.build_optimizer(net)

        assert optimizer is optimizer_sentinel
        assert scheduler is scheduler_sentinel
        assert calls["optimizer_params"] == tuple(net.parameters())
        assert calls["optimizer_kwargs"] == {
            "lr": 5e-3,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }
        assert calls["scheduler_kwargs"] == {
            "optimizer": optimizer_sentinel,
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 5e-5,
            "eps": 1e-8,
        }

    def test_cuda_graph_training_flag_is_explicit_opt_in(self, preserved_argv):
        from arguments import get_arguments

        sys.argv = ['opendpd']
        assert get_arguments().cuda_graph_training is False
        sys.argv = ['opendpd', '--cuda_graph_training']
        assert get_arguments().cuda_graph_training is True

    def test_boolean_kwargs_are_emitted_as_flags(self, preserved_argv):
        from opendpd import api

        sys.argv = ['opendpd']
        api._append_cli_kwargs({
            "collect_delta_stats": True,
            "cuda_graph_training": True,
            "plot": False,
            "frame_stride": 16,
            "unused": None,
        })
        assert sys.argv == [
            'opendpd', '--collect_delta_stats', '--cuda_graph_training',
            '--frame_stride', '16'
        ]

    def test_train_pa_smoke(self, tmp_path, monkeypatch, preserved_argv):
        from pathlib import Path

        monkeypatch.chdir(tmp_path)
        result = opendpd.train_pa(
            dataset_name=SMOKE_DATASET,
            n_epochs=1,
            frame_length=50,
            PA_hidden_size=8,
            batch_size=64,
            accelerator="cpu",
            frame_stride=16,
        )
        assert result["status"] == "completed"
        assert Path(result["model_path"]).is_file()
        assert Path(result["log_path"]).is_file()
