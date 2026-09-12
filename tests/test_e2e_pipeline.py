"""End-to-end smoke tests for the full OpenDPD pipeline.

Chains the four CLI steps on the smallest dataset (DPA_200MHz) with tiny
hyperparameters: train_pa -> train_dpd -> (quantized train_dpd) ->
run_dpd -> plot. Each stage asserts on the artifacts a real run must
produce, so a regression anywhere in the data pipeline, training loop,
model saving, or metric computation fails the build.
"""

import math

import numpy as np
import pandas as pd
import torch
import pytest

from conftest import SMOKE_DATASET, run_main

METRIC_COLUMNS = ["TRAIN_LOSS", "VAL_NMSE", "TEST_NMSE", "TEST_ACLR_AVG"]


def _read_history(workdir, step, pattern="*.csv"):
    history_dir = workdir / "log" / SMOKE_DATASET / step
    files = sorted(history_dir.rglob(pattern))
    assert files, f"no log CSV produced under {history_dir}"
    return pd.read_csv(files[0])


class TestTrainPA:
    def test_produces_checkpoint(self, pa_trained):
        ckpts = list((pa_trained / "save" / SMOKE_DATASET / "train_pa").glob("PA_*.pt"))
        assert ckpts, "train_pa did not save a PA model checkpoint"

    def test_logs_finite_metrics(self, pa_trained):
        history = _read_history(pa_trained, "train_pa")
        assert len(history) >= 1
        for col in METRIC_COLUMNS:
            assert col in history.columns, f"missing metric column {col}"
            value = float(history[col].iloc[-1])
            assert math.isfinite(value), f"{col} is not finite: {value}"


class TestTrainDPD:
    def test_produces_checkpoint(self, dpd_trained):
        save_dir = dpd_trained / "save" / SMOKE_DATASET / "train_dpd"
        ckpts = list(save_dir.rglob("DPD_*.pt"))
        assert ckpts, "train_dpd did not save a DPD model checkpoint"

    def test_logs_finite_metrics(self, dpd_trained):
        history = _read_history(dpd_trained, "train_dpd")
        assert len(history) >= 1
        for col in METRIC_COLUMNS:
            value = float(history[col].iloc[-1])
            assert math.isfinite(value), f"{col} is not finite: {value}"


class TestQuantizedTrainDPD:
    def test_quantization_aware_training(self, dpd_trained):
        """W16A16 quantization-aware DPD learning as documented in the README: a float ``qgru``
        DPD is trained first, then fine-tuned with quantized weights and activations."""
        save_dir = dpd_trained / "save" / SMOKE_DATASET / "train_dpd"
        run_main(dpd_trained, "train_dpd", "--DPD_backbone", "qgru")
        pretrained = sorted(
            p for p in save_dir.rglob("DPD_*_M_QGRU_*.pt") if "ci_quant" not in p.parts
        )
        assert pretrained, "float qgru train_dpd did not save a checkpoint"
        run_main(
            dpd_trained,
            "train_dpd",
            "--DPD_backbone", "qgru",
            "--quant",
            "--n_bits_w", "16",
            "--n_bits_a", "16",
            "--pretrained_model", str(pretrained[0]),
            "--quant_dir_label", "ci_quant",
        )
        quant_ckpts = list(save_dir.rglob("ci_quant/DPD_*.pt"))
        assert quant_ckpts, "quantized train_dpd did not save a checkpoint"
        # The checkpoint must come from the quantized cell-based GRU, not from the float fallback
        # that the quantization environment uses when the pretrained checkpoint cannot be loaded.
        keys = torch.load(quant_ckpts[0], map_location="cpu").keys()
        assert any("rnn_cell_list" in key for key in keys), "quantization did not engage"


class TestRunDPD:
    def test_generates_predistorted_output(self, dpd_trained):
        run_main(dpd_trained, "run_dpd")
        out_files = list((dpd_trained / "dpd_out").glob("*.csv"))
        assert out_files, "run_dpd did not write any CSV to dpd_out/"

        output = pd.read_csv(out_files[0])
        for col in ("I", "Q", "I_dpd", "Q_dpd"):
            assert col in output.columns, f"dpd_out CSV missing column {col}"
        assert len(output) > 0
        assert np.isfinite(output[["I_dpd", "Q_dpd"]].to_numpy()).all(), (
            "predistorted output contains NaN/Inf"
        )


class TestPlotStep:
    def test_generates_comparison_plots(self, dpd_trained):
        run_main(dpd_trained, "plot")
        pngs = list((dpd_trained / "plots" / SMOKE_DATASET).rglob("*.png"))
        assert pngs, "plot step did not produce any PNG figures"


@pytest.mark.extended
class TestAllDatasetsE2E:
    """Weekly job: run the PA-modeling smoke test on every built-in dataset."""

    @pytest.mark.parametrize(
        "dataset", ["DPA_200MHz", "DPA_160MHz", "APA_200MHz", "APA_200MHz_b"]
    )
    def test_train_pa_on_dataset(self, tmp_path, dataset):
        run_main(tmp_path, "train_pa", dataset=dataset)
        ckpts = list((tmp_path / "save" / dataset / "train_pa").glob("PA_*.pt"))
        assert ckpts, f"train_pa produced no checkpoint for {dataset}"


# Backbones that are both accepted by the CLI (--PA_backbone choices) and
# implemented in models.CoreModel. The CLI accepts more names, but those raise
# ValueError at model construction; see tests/test_backbones.py.
TRAINABLE_CLI_BACKBONES = [
    "gmp",
    "gru",
    "dgru",
    "qgru",
    "qgru_amp1",
    "lstm",
    "vdlstm",
    "rvtdcnn",
    "tcn",
    "deltagru",
    "deltajanet",
    "pgjanet",
    "dvrjanet",
    "bojanet",
    "mcldnn",
]


@pytest.mark.extended
class TestAllBackbonesTraining:
    """Weekly job: every supported backbone must survive one training epoch
    through the real CLI (BOJANET requires hidden_size <= 18)."""

    @pytest.mark.parametrize("backbone", TRAINABLE_CLI_BACKBONES)
    def test_train_pa_with_backbone(self, tmp_path, backbone):
        run_main(
            tmp_path,
            "train_pa",
            "--PA_backbone", backbone,
            "--PA_hidden_size", "8",
        )
        ckpts = list((tmp_path / "save" / SMOKE_DATASET / "train_pa").glob("PA_*.pt"))
        assert ckpts, f"train_pa produced no checkpoint for backbone {backbone}"
