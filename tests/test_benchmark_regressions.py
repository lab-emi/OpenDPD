"""Focused regression tests for the benchmark workflow."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import warnings

import numpy as np
import pytest

from project import Project


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "benchmark" / "benchmark_volterra.py"
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_volterra", BENCHMARK_PATH
)
benchmark_volterra = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(benchmark_volterra)
COLLECTOR_PATH = REPO_ROOT / "benchmark" / "collect_benchmark_report.py"
COLLECTOR_SPEC = importlib.util.spec_from_file_location(
    "collect_benchmark_report", COLLECTOR_PATH
)
collect_benchmark_report = importlib.util.module_from_spec(COLLECTOR_SPEC)
COLLECTOR_SPEC.loader.exec_module(collect_benchmark_report)


def parse_recorded_commands(path):
    blocks = {}
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# "):
            label = line[2:]
            assert label not in blocks, f"duplicate command label: {label}"
            blocks[label] = shlex.split(lines[index + 1])
    return blocks


def option(command, name):
    assert command.count(name) == 1, f"expected one {name} in {command}"
    return command[command.index(name) + 1]


def polynomial_metrics():
    return {
        "NMSE": -39.0,
        "EVM": -47.0,
        "ACLR_L": -42.0,
        "ACLR_R": -40.0,
        "ACLR_AVG": -41.0,
    }


def polynomial_result(task="pa_modeling", model="mp"):
    configuration = collect_benchmark_report.POLYNOMIAL_CONFIGURATIONS[task][model]
    method = (
        "direct_least_squares"
        if task == "pa_modeling"
        else "indirect_learning_architecture"
    )
    result = {
        "schema_version": 4,
        "dataset": "APA_200MHz",
        "task": task,
        "method": method,
        "model": model,
        "solver": "torch.linalg.lstsq",
        "solver_mode": "gels",
        "solver_driver": "gels",
        "dtype": "torch.complex64",
        "device": "cuda:0",
        "solver_device": "cuda:0",
        "solver_dtype": "torch.complex64",
        "column_scaling": "l2",
        "column_scale_min": 1.0,
        "column_scale_max": 10.0,
        "training_relative_residual": 0.1,
        "coefficient_l2_norm": 1.0,
        "regularization": None,
        "svd_rcond": None,
        "singular_value_max": None,
        "singular_value_min": None,
        "singular_value_cutoff": None,
        "retained_singular_value_min": None,
        "condition_number": None,
        "full_rank_assumption": True,
        "least_squares_rank": None,
        "basis_configuration": configuration["basis_configuration"],
        "complex_coefficients": configuration["complex_coefficients"],
        "real_parameters": configuration["real_parameters"],
        "parameter_count_convention": (
            "two real degrees of freedom per complex coefficient"
        ),
        "target_gain": 1.2,
        "sample_rate_hz": 983_040_000.0,
        "nperseg": 19_662,
        "validation": polynomial_metrics(),
        "test": polynomial_metrics(),
    }
    if task == "pa_modeling" and model == "gmp":
        result.update(
            {
                "solver": "torch.linalg.svd",
                "solver_mode": "truncated_svd",
                "solver_driver": "gesvdj",
                "full_rank_assumption": False,
                "least_squares_rank": 650,
                "regularization": "truncated_svd",
                "svd_rcond": 1e-4,
                "singular_value_max": 10.0,
                "singular_value_min": 1e-6,
                "singular_value_cutoff": 1e-3,
                "retained_singular_value_min": 1.1e-3,
                "condition_number": 1e7,
                "coefficient_l2_norm": 8.0,
            }
        )
    if task == "dpd_ila":
        result.update(
            {
                "pa_checkpoint": (
                    "save/APA_200MHz/train_pa/"
                    "PA_S_0_M_TRES_GRU_H_27_F_200_P_2751.pt"
                ),
                "pa_checkpoint_sha256": "a" * 64,
                "pa_evaluation_device": "cuda:0",
            }
        )
    return result


def test_volterra_cli_selects_distinct_mp_and_gmp_bases():
    parser = benchmark_volterra.build_arg_parser()
    mp_args = parser.parse_args(["--dataset-name", "fixture", "--model", "mp"])
    gmp_args = parser.parse_args(["--dataset-name", "fixture", "--model", "gmp"])

    mp_builder, mp_coefficients, mp_banner = benchmark_volterra.select_basis(mp_args)
    gmp_builder, gmp_coefficients, gmp_banner = benchmark_volterra.select_basis(
        gmp_args
    )

    signal = np.linspace(0.1, 1.0, 64) * np.exp(
        1j * np.linspace(0.0, 2.0 * np.pi, 64)
    )
    mp_basis = mp_builder(signal)
    gmp_basis = gmp_builder(signal)

    assert mp_coefficients == mp_args.K * mp_args.Q == 250
    assert gmp_coefficients == (
        gmp_args.Ka * gmp_args.La
        + gmp_args.Kb * gmp_args.Lb * gmp_args.Mb
        + gmp_args.Kc * gmp_args.Lc * gmp_args.Mc
    ) == 255
    assert mp_basis.shape == (signal.size, mp_coefficients)
    assert gmp_basis.shape == (signal.size, gmp_coefficients)
    assert not np.allclose(mp_basis, gmp_basis[:, :mp_coefficients])
    assert " MP " in mp_banner
    assert " GMP " in gmp_banner


def test_segmented_polynomial_basis_resets_delay_state():
    signal = np.arange(1, 9, dtype=np.float64).astype(np.complex128)
    builder = lambda values: benchmark_volterra.build_mp_basis(values, K=1, Q=2)

    full = builder(signal)
    segmented = benchmark_volterra.build_segmented_numpy_basis(
        signal, builder, segment_length=4
    )

    assert full[4, 1] == signal[3]
    assert segmented[4, 1] == 0
    assert np.array_equal(segmented[:4], full[:4])


def test_truncated_svd_stabilizes_rank_deficient_gmp_basis():
    import torch

    signal = np.ones(64, dtype=np.complex128)
    target = 1.5 * signal
    configuration = {
        "Ka": 2,
        "La": 2,
        "Kb": 1,
        "Lb": 2,
        "Mb": 1,
        "Kc": 1,
        "Lc": 2,
        "Mc": 1,
    }

    coefficients, diagnostics = (
        benchmark_volterra.fit_complex_least_squares(
            signal,
            target,
            polynomial_model="gmp",
            configuration=configuration,
            segment_length=64,
            device=torch.device("cpu"),
            dtype=torch.complex128,
            solver_mode="truncated_svd",
            svd_rcond=1e-4,
        )
    )
    prediction = benchmark_volterra.apply_polynomial(
        signal,
        coefficients,
        polynomial_model="gmp",
        configuration=configuration,
        segment_length=64,
    )

    assert diagnostics["solver_mode"] == "truncated_svd"
    assert diagnostics["solver_implementation"] == "torch.linalg.svd"
    assert diagnostics["regularization"] == "truncated_svd"
    assert 0 < diagnostics["least_squares_rank"] < diagnostics["columns"]
    assert diagnostics["svd_rcond"] == 1e-4
    assert np.all(np.isfinite(coefficients.numpy()))
    assert np.linalg.norm(prediction - target) / np.linalg.norm(target) < 1e-12


def test_solver_mode_rejects_incompatible_svd_cutoff():
    import torch

    signal = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError, match="only valid for truncated_svd"):
        benchmark_volterra.fit_complex_least_squares(
            signal,
            signal,
            polynomial_model="mp",
            configuration={"K": 1, "Q": 1},
            segment_length=16,
            device=torch.device("cpu"),
            dtype=torch.complex128,
            solver_mode="gels",
            svd_rcond=1e-4,
        )


def test_build_logger_warns_before_existing_recipe_artifacts_are_overwritten(
    tmp_path,
):
    project = Project.__new__(Project)
    project.path_dir_save = str(tmp_path / "save")
    project.path_dir_log_hist = str(tmp_path / "history")
    project.path_dir_log_best = str(tmp_path / "best")
    project.log_precision = 8
    for directory in (
        project.path_dir_save,
        project.path_dir_log_hist,
        project.path_dir_log_best,
    ):
        Path(directory).mkdir()

    model_id = "PA_S_0_M_GRU_H_28_F_200_P_2746"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        project.build_logger(model_id)
    assert caught == []

    Path(project.path_save_file_best).touch()
    with pytest.warns(RuntimeWarning, match="does not encode the full training recipe"):
        project.build_logger(model_id)


def test_benchmark_report_runner_dry_run_records_complete_recipe(tmp_path):
    output_dir = tmp_path / "benchmark-run"
    runner = REPO_ROOT / "benchmark" / "reproduce_benchmark_report.sh"
    environment = dict(os.environ, OPENDPD_PYTHON=sys.executable)

    result = subprocess.run(
        [
            "bash",
            str(runner),
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    commands = parse_recorded_commands(output_dir / "commands.log")
    expected_labels = {"snapshot_context", "collect_report"}
    expected_labels.update(
        f"train_pa_{slug}_{model}"
        for slug in ("apa", "dpa")
        for model in ("gru", "tres_gru", "tres_deltagru")
    )
    expected_labels.update(
        f"train_dpd_{slug}_{experiment}"
        for slug in ("apa", "dpa")
        for experiment in (
            "gru_via_tres_gru_pa",
            "tres_gru_via_tres_gru_pa",
            "tres_deltagru_via_tres_gru_pa",
            "tres_deltagru_via_tres_deltagru_pa",
        )
    )
    expected_labels.update(
        f"pa_model_{slug}_{model}"
        for slug in ("apa", "dpa")
        for model in ("mp", "gmp")
    )
    expected_labels.update(
        f"dpd_ila_{slug}_{model}"
        for slug in ("apa", "dpa")
        for model in ("mp", "gmp")
    )
    assert set(commands) == expected_labels

    for slug, dataset in (("apa", "APA_200MHz"), ("dpa", "DPA_160MHz")):
        for backbone, hidden_size in {
            "gru": "28",
            "tres_gru": "27",
            "tres_deltagru": "27",
        }.items():
                command = commands[f"train_pa_{slug}_{backbone}"]
                assert command[1] == "main.py"
                assert option(command, "--dataset_name") == dataset
                assert option(command, "--step") == "train_pa"
                assert option(command, "--frame_length") == "200"
                assert option(command, "--frame_stride") == "1"
                assert option(command, "--n_epochs") == "300"
                assert option(command, "--opt_type") == "adamw"
                assert option(command, "--batch_size") == "64"
                assert option(command, "--batch_size_eval") == "64"
                assert option(command, "--lr") == "5e-3"
                assert option(command, "--lr_schedule") == "1"
                assert option(command, "--lr_end") == "5e-5"
                assert option(command, "--decay_factor") == "0.5"
                assert option(command, "--patience") == "5"
                assert option(command, "--loss_type") == "l2"
                assert option(command, "--grad_clip_val") == "200"
                assert option(command, "--thx") == "0"
                assert option(command, "--thh") == "0"
                assert option(command, "--seed") == "0"
                assert option(command, "--re_level") == "soft"
                assert option(command, "--accelerator") == "cuda"
                assert option(command, "--devices") == "0"
                assert option(command, "--PA_backbone") == backbone
                assert option(command, "--PA_hidden_size") == hidden_size

        dpd_experiments = {
            "gru_via_tres_gru_pa": ("gru", "16", "tres_gru"),
            "tres_gru_via_tres_gru_pa": (
                "tres_gru",
                "15",
                "tres_gru",
            ),
            "tres_deltagru_via_tres_gru_pa": (
                "tres_deltagru",
                "15",
                "tres_gru",
            ),
            "tres_deltagru_via_tres_deltagru_pa": (
                "tres_deltagru",
                "15",
                "tres_deltagru",
            ),
        }
        for experiment, (backbone, hidden_size, pa_backbone) in (
            dpd_experiments.items()
        ):
            command = commands[f"train_dpd_{slug}_{experiment}"]
            assert command[1] == "main.py"
            assert option(command, "--dataset_name") == dataset
            assert option(command, "--step") == "train_dpd"
            assert option(command, "--frame_length") == "200"
            assert option(command, "--frame_stride") == "1"
            assert option(command, "--n_epochs") == "300"
            assert option(command, "--opt_type") == "adamw"
            assert option(command, "--batch_size") == "64"
            assert option(command, "--batch_size_eval") == "64"
            assert option(command, "--lr") == "5e-3"
            assert option(command, "--lr_schedule") == "1"
            assert option(command, "--lr_end") == "5e-5"
            assert option(command, "--decay_factor") == "0.5"
            assert option(command, "--patience") == "5"
            assert option(command, "--loss_type") == "l2"
            assert option(command, "--grad_clip_val") == "200"
            assert option(command, "--thx") == "0"
            assert option(command, "--thh") == "0"
            assert option(command, "--seed") == "0"
            assert option(command, "--re_level") == "soft"
            assert option(command, "--accelerator") == "cuda"
            assert option(command, "--devices") == "0"
            assert option(command, "--PA_backbone") == pa_backbone
            assert option(command, "--PA_hidden_size") == "27"
            assert option(command, "--DPD_backbone") == backbone
            assert option(command, "--DPD_hidden_size") == hidden_size

    for slug, dataset in (("apa", "APA_200MHz"), ("dpa", "DPA_160MHz")):
        pa_mp = commands[f"pa_model_{slug}_mp"]
        pa_gmp = commands[f"pa_model_{slug}_gmp"]
        dpd_mp = commands[f"dpd_ila_{slug}_mp"]
        dpd_gmp = commands[f"dpd_ila_{slug}_gmp"]
        for command, task in (
            (pa_mp, "pa_modeling"),
            (pa_gmp, "pa_modeling"),
            (dpd_mp, "dpd_ila"),
            (dpd_gmp, "dpd_ila"),
        ):
            assert command[1:3] == ["-m", "benchmark.benchmark_volterra"]
            assert option(command, "--task") == task
            assert option(command, "--dataset-name") == dataset
            assert option(command, "--solver-device") == "cuda:0"
            assert option(command, "--solver-dtype") == "complex64"
        assert option(pa_mp, "--solver-mode") == "gels"
        assert option(pa_gmp, "--solver-mode") == "truncated_svd"
        assert option(pa_gmp, "--svd-rcond") == "1e-4"
        assert option(dpd_mp, "--solver-mode") == "gels"
        assert option(dpd_gmp, "--solver-mode") == "gels"
        assert "--svd-rcond" not in pa_mp
        assert "--svd-rcond" not in dpd_mp
        assert "--svd-rcond" not in dpd_gmp
        assert option(pa_mp, "--K") == "9"
        assert option(pa_mp, "--Q") == "150"
        assert option(pa_gmp, "--La") == "30"
        assert option(pa_gmp, "--Mb") == "5"
        assert option(pa_gmp, "--Mc") == "5"
        assert option(dpd_mp, "--K") == "5"
        assert option(dpd_mp, "--Q") == "100"
        assert option(dpd_gmp, "--La") == "20"
        assert option(dpd_gmp, "--Mb") == "3"
        assert option(dpd_gmp, "--Mc") == "2"
        for command in (dpd_mp, dpd_gmp):
            assert option(command, "--pa-backbone") == "tres_gru"
            assert option(command, "--pa-hidden-size") == "27"

    assert "--snapshot-context" in commands["snapshot_context"]
    assert "--snapshot-context" not in commands["collect_report"]
    collect_benchmark_report.validate_neural_commands(
        output_dir / "commands.log"
    )
    collect_benchmark_report.validate_runner_commands(
        output_dir / "commands.log",
        output_dir=output_dir,
        device=0,
    )

    recipe = json.loads((output_dir / "recipe.json").read_text())
    assert recipe == collect_benchmark_report.expected_runner_recipe()
    assert recipe["schema_version"] == 6
    assert recipe["neural"]["epochs"] == 300
    assert recipe["neural"]["batch_size"] == 64
    assert recipe["neural"]["initial_learning_rate"] == 5e-3
    assert recipe["neural"]["scheduler_patience"] == 5
    assert recipe["neural"]["minimum_learning_rate"] == 5e-5
    assert recipe["neural"]["optimizer_weight_decay"] == 0.01
    assert recipe["neural"]["optimizer_betas"] == [0.9, 0.999]
    assert recipe["neural"]["scheduler_threshold_mode"] == "rel"
    assert recipe["pa_models"]["mp"]["real_parameters"] == 2700
    assert recipe["pa_models"]["tres_deltagru"] == {
        "hidden_size": 27,
        "real_parameters": 2751,
        "delta_thresholds": {"input": 0.0, "hidden": 0.0},
    }
    assert recipe["dpd_models"]["gmp"]["real_parameters"] == 1000
    assert recipe["polynomial_solvers"]["gmp_pa"] == {
        "implementation": "torch.linalg.svd",
        "mode": "truncated_svd",
        "driver": "gesvdj",
        "dtype": "torch.complex64",
        "column_scaling": "l2",
        "regularization": {
            "type": "truncated_svd",
            "relative_cutoff": 1e-4,
        },
    }
    assert recipe["dpd_tres_deltagru_pa_surrogate_comparison"] == {
        "model": {
            "hidden_size": 15,
            "real_parameters": 999,
            "delta_thresholds": {"input": 0.0, "hidden": 0.0},
        },
        "pa_surrogates": {
            "tres_gru": {
                "hidden_size": 27,
                "real_parameters": 2751,
                "delta_thresholds": None,
            },
            "tres_deltagru": {
                "hidden_size": 27,
                "real_parameters": 2751,
                "delta_thresholds": {"input": 0.0, "hidden": 0.0},
            },
        },
    }


def test_distributed_signal_papr_is_stable():
    apa_record = collect_benchmark_report.signal_record("APA_200MHz")
    dpa_record = collect_benchmark_report.signal_record("DPA_160MHz")
    apa = apa_record["papr"]
    dpa = dpa_record["papr"]

    assert apa_record["samples"] == 98_304
    assert dpa_record["samples"] == 491_520
    assert apa["ccdf_probability"] == 1e-5
    assert apa["quantile_method"] == "numpy linear"
    assert apa["ccdf_db"] == pytest.approx(10.00798, abs=1e-5)
    assert apa["absolute_peak_db"] == pytest.approx(10.01054, abs=1e-5)
    assert dpa["ccdf_db"] == pytest.approx(10.13155, abs=1e-5)
    assert dpa["absolute_peak_db"] == pytest.approx(10.13976, abs=1e-5)


def test_pa_checkpoint_bindings_require_eight_unique_rows(tmp_path):
    experiments = (
        ("gru", "tres_gru"),
        ("tres_gru", "tres_gru"),
        ("tres_deltagru", "tres_gru"),
        ("tres_deltagru", "tres_deltagru"),
    )
    rows = [
        (
            dataset,
            dpd_model,
            pa_model,
            f"save/{dataset}/{pa_model}.pt",
            digest * 64,
        )
        for dataset, digest in (("APA_200MHz", "a"), ("DPA_160MHz", "b"))
        for dpd_model, pa_model in experiments
    ]
    path = tmp_path / "pa_checkpoint_bindings.tsv"
    path.write_text(
        "dataset\tdpd_model_key\tpa_model_key\tcheckpoint\tsha256\n"
        + "".join("\t".join(row) + "\n" for row in rows)
    )

    bindings = collect_benchmark_report.read_pa_bindings(tmp_path)
    assert set(bindings) == {
        (dataset, dpd_model, pa_model)
        for dataset in ("APA_200MHz", "DPA_160MHz")
        for dpd_model, pa_model in experiments
    }

    rows[-1] = rows[0]
    path.write_text(
        "dataset\tdpd_model_key\tpa_model_key\tcheckpoint\tsha256\n"
        + "".join("\t".join(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="Duplicate PA checkpoint binding"):
        collect_benchmark_report.read_pa_bindings(tmp_path)


def test_polynomial_metric_validation_rejects_nonfinite_and_bad_average():
    metrics = polynomial_metrics()
    assert collect_benchmark_report.normalize_polynomial_metrics(metrics) == {
        "nmse_db": -39.0,
        "evm_db": -47.0,
        "aclr_left_db": -42.0,
        "aclr_right_db": -40.0,
        "aclr_avg_db": -41.0,
    }

    with pytest.raises(ValueError, match="not finite"):
        collect_benchmark_report.normalize_polynomial_metrics(
            dict(metrics, NMSE=float("nan"))
        )
    with pytest.raises(ValueError, match="ACLR average"):
        collect_benchmark_report.normalize_polynomial_metrics(
            dict(metrics, ACLR_AVG=-40.5)
        )


def benchmark_context():
    return {
        "git": {"commit": "abc", "branch": "benchmark-fix", "dirty": False},
        "environment": {"python": "3.13"},
        "source_file_sha256": {"main.py": "1" * 64},
        "datasets": {"APA_200MHz": {"spec_sha256": "2" * 64}},
    }


def test_context_stability_accepts_identical_context():
    initial = benchmark_context()
    collect_benchmark_report.validate_context_stability(
        initial, copy.deepcopy(initial)
    )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("git", "commit"), "def", "Git commit"),
        (("git", "branch"), "other", "Git branch"),
        (("environment", "python"), "3.14", "software and hardware"),
        (("source_file_sha256", "main.py"), "3" * 64, "source files"),
        (
            ("datasets", "APA_200MHz", "spec_sha256"),
            "4" * 64,
            "dataset files",
        ),
    ],
)
def test_context_stability_rejects_drift(path, replacement, message):
    initial = benchmark_context()
    changed = copy.deepcopy(initial)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(ValueError, match=message):
        collect_benchmark_report.validate_context_stability(initial, changed)


def test_context_loader_requires_pre_run_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pre-run context snapshot"):
        collect_benchmark_report.load_and_validate_context(tmp_path, 0)


def test_source_snapshot_covers_runtime_sources():
    sources = set(collect_benchmark_report.source_files())
    assert {
        "arguments.py",
        "models.py",
        "project.py",
        "backbones/gru.py",
        "backbones/tres_deltagru.py",
        "backbones/tres_gru.py",
        "benchmark/benchmark_volterra.py",
        "benchmark/collect_benchmark_report.py",
        "modules/train_funcs.py",
        "steps/train_dpd.py",
        "steps/train_pa.py",
        "utils/metrics.py",
    } <= sources


def test_polynomial_pa_artifact_contract_rejects_recipe_drift(tmp_path):
    canonical = polynomial_result(task="pa_modeling", model="mp")
    output_path = tmp_path / "benchmark_report_apa_pa_mp.json"
    output_path.write_text(json.dumps(canonical))

    result = collect_benchmark_report.collect_polynomial_model(
        tmp_path,
        dataset="APA_200MHz",
        slug="apa",
        task="pa_modeling",
        model_name="mp",
    )
    assert result["real_parameters"] == 2700
    assert result["least_squares_rank"] is None
    assert result["pa_checkpoint_binding"] is None

    mutations = [
        ("solver", "other", "polynomial solver"),
        ("basis_configuration", {"K": 9, "Q": 149}, "polynomial basis"),
        ("complex_coefficients", 1349, "complex coefficient count"),
        ("real_parameters", 2698, "parameter count"),
        ("sample_rate_hz", 800_000_000.0, "sample rate"),
        ("nperseg", 16_384, "segment length"),
        ("column_scale_min", 0.0, "must be positive"),
    ]
    for field, replacement, message in mutations:
        changed = copy.deepcopy(canonical)
        changed[field] = replacement
        output_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            collect_benchmark_report.collect_polynomial_model(
                tmp_path,
                dataset="APA_200MHz",
                slug="apa",
                task="pa_modeling",
                model_name="mp",
            )


def test_gmp_pa_artifact_contract_requires_truncated_svd_evidence(tmp_path):
    canonical = polynomial_result(task="pa_modeling", model="gmp")
    output_path = tmp_path / "benchmark_report_apa_pa_gmp.json"
    output_path.write_text(json.dumps(canonical))

    result = collect_benchmark_report.collect_polynomial_model(
        tmp_path,
        dataset="APA_200MHz",
        slug="apa",
        task="pa_modeling",
        model_name="gmp",
    )
    assert result["solver"] == "torch.linalg.svd"
    assert result["solver_mode"] == "truncated_svd"
    assert result["svd_rcond"] == 1e-4
    assert result["least_squares_rank"] == 650
    assert result["coefficient_l2_norm"] == 8.0

    mutations = [
        ("solver_mode", "gels", "solver mode"),
        ("svd_rcond", 1e-6, "SVD relative cutoff"),
        ("least_squares_rank", 1350, "rank is outside"),
        ("retained_singular_value_min", 5e-4, "does not exceed"),
        ("condition_number", 1e3, "inconsistent"),
    ]
    for field, replacement, message in mutations:
        changed = copy.deepcopy(canonical)
        changed[field] = replacement
        output_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            collect_benchmark_report.collect_polynomial_model(
                tmp_path,
                dataset="APA_200MHz",
                slug="apa",
                task="pa_modeling",
                model_name="gmp",
            )


def test_polynomial_dpd_artifact_requires_reference_pa_hash(tmp_path):
    canonical = polynomial_result(task="dpd_ila", model="mp")
    output_path = tmp_path / "benchmark_report_apa_dpd_mp.json"
    output_path.write_text(json.dumps(canonical))

    result = collect_benchmark_report.collect_polynomial_model(
        tmp_path,
        dataset="APA_200MHz",
        slug="apa",
        task="dpd_ila",
        model_name="mp",
        expected_pa_hash="a" * 64,
    )
    assert result["real_parameters"] == 1000
    assert result["pa_checkpoint_binding"]["sha256"] == "a" * 64

    changed = dict(canonical, pa_checkpoint_sha256="b" * 64)
    output_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="checkpoint hash"):
        collect_benchmark_report.collect_polynomial_model(
            tmp_path,
            dataset="APA_200MHz",
            slug="apa",
            task="dpd_ila",
            model_name="mp",
            expected_pa_hash="a" * 64,
        )


def test_checkpoint_validation_checks_loadability_and_parameter_count(tmp_path):
    import torch

    checkpoint = tmp_path / "model.pt"
    torch.save({"weight": torch.ones(3), "bias": torch.zeros(2)}, checkpoint)
    collect_benchmark_report.validate_checkpoint(checkpoint, 5)

    with pytest.raises(ValueError, match="checkpoint parameter count"):
        collect_benchmark_report.validate_checkpoint(checkpoint, 4)

    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="cannot be loaded safely"):
        collect_benchmark_report.validate_checkpoint(checkpoint, 5)


def test_run_manifest_uses_v6_schema():
    assert collect_benchmark_report.RUN_SCHEMA_VERSION == 6


def report_manifest():
    values = {
        "APA_200MHz": {
            "pa": [-37.0, -38.7, -38.8, -39.1, -39.2],
            "dpd": [-45.2, -43.6, -51.4, -53.4],
            "delta_dpd": [-54.1, -55.2],
        },
        "DPA_160MHz": {
            "pa": [-38.4, -39.4, -39.5, -39.7, -39.8],
            "dpd": [-50.7, -52.8, -56.5, -59.5],
            "delta_dpd": [-60.1, -61.3],
        },
    }
    display_names = {
        "mp": "MP",
        "gmp": "GMP",
        "gru": "GRU-H28",
        "tres_gru": "TRes-GRU-H27",
        "tres_deltagru": "TRes-DeltaGRU-H27 (THX=THH=0)",
    }
    datasets = {}
    for dataset, dataset_values in values.items():
        pa_models = {}
        dpd_models = {}
        delta_dpd_by_pa_surrogate = {}
        for index, key in enumerate(collect_benchmark_report.PA_MODEL_ORDER):
            polynomial = key in {"mp", "gmp"}
            pa_models[key] = {
                "method": (
                    "direct_least_squares"
                    if polynomial
                    else "supervised_training"
                ),
                "model": {
                    "display_name": display_names[key],
                    "parameters": 2700 + index,
                },
                "selected_epoch_zero_based": None if polynomial else 10 + index,
                "metrics": {
                    "validation": {
                        "nmse_db": dataset_values["pa"][index] + 0.1,
                        "evm_db": -40.0 - index,
                        "aclr_left_db": -30.0 - index,
                        "aclr_right_db": -31.0 - index,
                        "aclr_avg_db": -30.5 - index,
                    },
                    "test": {
                        "nmse_db": dataset_values["pa"][index],
                        "evm_db": -40.2 - index,
                        "aclr_left_db": -30.2 - index,
                        "aclr_right_db": -31.2 - index,
                        "aclr_avg_db": -30.7 - index,
                    },
                },
            }
            if key == "gmp":
                pa_models[key]["training_relative_residual"] = 0.0112
                pa_models[key]["regularization"] = "truncated_svd"
                pa_models[key]["svd_rcond"] = 1e-4
                pa_models[key]["least_squares_rank"] = 650
                pa_models[key]["complex_coefficients"] = 1350
            if key == "tres_deltagru":
                pa_models[key]["delta_thresholds"] = {
                    "input": 0.0,
                    "hidden": 0.0,
                }

        for index, key in enumerate(collect_benchmark_report.DPD_MODEL_ORDER):
            polynomial = key in {"mp", "gmp"}
            dpd_models[key] = {
                "method": (
                    "indirect_learning_architecture"
                    if polynomial
                    else "direct_learning_architecture"
                ),
                "model": {
                    "display_name": display_names[key],
                    "parameters": 1000 + index,
                },
                "selected_epoch_zero_based": None if polynomial else 20 + index,
                "metrics": {
                    "validation": {
                        "nmse_db": -42.0 - index,
                        "evm_db": -45.0 - index,
                        "aclr_left_db": dataset_values["dpd"][index] - 0.4,
                        "aclr_right_db": dataset_values["dpd"][index] + 0.2,
                        "aclr_avg_db": dataset_values["dpd"][index] - 0.1,
                    },
                    "test": {
                        "nmse_db": -42.2 - index,
                        "evm_db": -45.2 - index,
                        "aclr_left_db": dataset_values["dpd"][index] - 0.3,
                        "aclr_right_db": dataset_values["dpd"][index] + 0.3,
                        "aclr_avg_db": dataset_values["dpd"][index],
                    },
                },
            }
        for index, pa_model_key in enumerate(("tres_gru", "tres_deltagru")):
            aclr = dataset_values["delta_dpd"][index]
            delta_dpd_by_pa_surrogate[pa_model_key] = {
                "method": "direct_learning_architecture",
                "model": {
                    "display_name": "TRes-DeltaGRU-H15 (THX=THH=0)",
                    "parameters": 999,
                },
                "selected_epoch_zero_based": 30 + index,
                "metrics": {
                    "validation": {
                        "nmse_db": -48.0 - index,
                        "evm_db": -50.0 - index,
                        "aclr_left_db": aclr - 0.3,
                        "aclr_right_db": aclr + 0.1,
                        "aclr_avg_db": aclr - 0.1,
                    },
                    "test": {
                        "nmse_db": -48.2 - index,
                        "evm_db": -50.2 - index,
                        "aclr_left_db": aclr - 0.2,
                        "aclr_right_db": aclr + 0.2,
                        "aclr_avg_db": aclr,
                    },
                },
                "pa_surrogate": {"model_key": pa_model_key},
            }
        datasets[dataset] = {
            "pa_models": pa_models,
            "dpd_models": dpd_models,
            "dpd_tres_deltagru_by_pa_surrogate": (
                delta_dpd_by_pa_surrogate
            ),
        }
    return {
        "generated_at_utc": "2026-07-24T00:00:00+00:00",
        "git": {"commit": "abc", "branch": "benchmark-fix", "dirty": False},
        "environment": {
            "python": "3.13",
            "torch": "2.0",
            "cuda": "13.0",
            "gpu": "Fixture GPU",
        },
        "datasets": datasets,
    }


def test_report_is_latest_only_and_documents_gmp_stability_and_temporal_context():
    markdown = collect_benchmark_report.render_markdown(
        report_manifest(),
        canonical=True,
    )

    assert "benchmark_results.png" in markdown
    assert "benchmark_delta_dpd_results.png" in markdown
    assert "APA GMP stability" in markdown
    assert "rcond=1e-04" in markdown
    assert "650/1,350 singular directions" in markdown
    assert "Truncated SVD (rank 650/1,350)" in markdown
    assert "16-sample right context" in markdown
    assert "wraps the final position" in markdown
    assert "not demodulated constellation EVM" in markdown
    assert "reproduce_benchmark_report.sh" in markdown
    assert "commands.log" not in markdown
    assert "previous report" not in markdown.lower()
    assert "DGRU" not in markdown
    assert "TRes-DeltaGRU" in markdown
    assert "THX=THH=0" in markdown
    assert "dense zero-threshold recurrence" in markdown
    assert "TRes-DeltaGRU DPD by PA surrogate" in markdown
    assert "separately trained" in markdown
    for dataset in ("APA_200MHz", "DPA_160MHz"):
        dataset_section = markdown.split(f"## {dataset}", 1)[1].split(
            "\n## ",
            1,
        )[0]
        primary_dpd = dataset_section.split("### DPD\n", 1)[1].split(
            "### TRes-DeltaGRU DPD by PA surrogate",
            1,
        )[0]
        assert "TRes-DeltaGRU" not in primary_dpd


def test_results_figure_is_a_nonempty_png(tmp_path):
    output = tmp_path / "benchmark_results.png"

    collect_benchmark_report.write_results_figure(report_manifest(), output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000


def test_delta_dpd_figure_is_a_nonempty_png(tmp_path):
    output = tmp_path / "benchmark_delta_dpd_results.png"

    collect_benchmark_report.write_delta_dpd_figure(
        report_manifest(),
        output,
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000


def test_source_snapshot_roundtrip_matches_pre_run_hashes(tmp_path):
    paths = ["arguments.py", "benchmark/collect_benchmark_report.py"]
    expected = {
        path: collect_benchmark_report.sha256_file(REPO_ROOT / path)
        for path in paths
    }
    output = tmp_path / "source_snapshot.tar.gz"

    collect_benchmark_report.write_source_snapshot(paths, output)
    metadata = collect_benchmark_report.validate_source_snapshot(
        output,
        expected,
    )

    assert metadata["path"] == "source_snapshot.tar.gz"
    assert metadata["file_count"] == len(paths)
    assert metadata["sha256"] == collect_benchmark_report.sha256_file(output)
