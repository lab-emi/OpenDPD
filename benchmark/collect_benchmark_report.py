"""Collect one benchmark-report reproduction run into JSON and Markdown.

The training entry point writes checkpoints and CSV logs to architecture-based
paths under ``save/`` and ``log/``.  The benchmark runner archives those files
into a run-specific directory, then calls this module to validate the expected
recipes and assemble a portable evidence bundle.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tarfile
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCHEMA_VERSION = 6
POLYNOMIAL_SCHEMA_VERSION = 4
GMP_PA_SVD_RCOND = 1e-4
PA_MODEL_ORDER = ("mp", "gmp", "gru", "tres_gru", "tres_deltagru")
DPD_MODEL_ORDER = ("mp", "gmp", "gru", "tres_gru")

DATASETS = {
    "APA_200MHz": {
        "slug": "apa",
        "sample_rate_hz": 983_040_000.0,
        "nperseg": 19_662,
    },
    "DPA_160MHz": {
        "slug": "dpa",
        "sample_rate_hz": 640_000_000.0,
        "nperseg": 16_384,
    },
}

NEURAL_PA_MODELS = {
    "gru": {
        "display_name": "GRU-H28",
        "backbone": "gru",
        "hidden_size": 28,
        "parameters": 2746,
        "model_id": "PA_S_0_M_GRU_H_28_F_200_P_2746",
    },
    "tres_gru": {
        "display_name": "TRes-GRU-H27",
        "backbone": "tres_gru",
        "hidden_size": 27,
        "parameters": 2751,
        "model_id": "PA_S_0_M_TRES_GRU_H_27_F_200_P_2751",
    },
    "tres_deltagru": {
        "display_name": "TRes-DeltaGRU-H27 (THX=THH=0)",
        "backbone": "tres_deltagru",
        "hidden_size": 27,
        "parameters": 2751,
        "model_id": "PA_S_0_M_TRES_DELTAGRU_H_27_F_200_P_2751",
        "delta_thresholds": {
            "input": 0.0,
            "hidden": 0.0,
        },
    },
}

REFERENCE_PA_KEY = "tres_gru"
REFERENCE_PA_PARENT = "PA_S_0_M_TRES_GRU_H_27_F_200"
ALTERNATE_PA_KEY = "tres_deltagru"
ALTERNATE_PA_PARENT = "PA_S_0_M_TRES_DELTAGRU_H_27_F_200"

NEURAL_DPD_MODELS = {
    "gru": {
        "display_name": "GRU-H16",
        "backbone": "gru",
        "hidden_size": 16,
        "parameters": 994,
        "model_id": "DPD_S_0_M_GRU_H_16_F_200_P_994",
    },
    "tres_gru": {
        "display_name": "TRes-GRU-H15",
        "backbone": "tres_gru",
        "hidden_size": 15,
        "parameters": 999,
        "model_id": "DPD_S_0_M_TRES_GRU_H_15_F_200_P_999",
    },
}

TRES_DELTAGRU_DPD_MODEL = {
    "display_name": "TRes-DeltaGRU-H15 (THX=THH=0)",
    "backbone": "tres_deltagru",
    "hidden_size": 15,
    "parameters": 999,
    "model_id": (
        "DPD_S_0_M_TRES_DELTAGRU_H_15_F_200_P_999_"
        "THX_0.000_THH_0.000"
    ),
    "delta_thresholds": {
        "input": 0.0,
        "hidden": 0.0,
    },
}

NEURAL_DPD_EXPERIMENTS = {
    **{
        f"{model_key}_via_{REFERENCE_PA_KEY}_pa": {
            "model_key": model_key,
            "pa_model_key": REFERENCE_PA_KEY,
            "pa_parent": REFERENCE_PA_PARENT,
        }
        for model_key in NEURAL_DPD_MODELS
    },
    "tres_deltagru_via_tres_gru_pa": {
        "model_key": "tres_deltagru",
        "pa_model_key": REFERENCE_PA_KEY,
        "pa_parent": REFERENCE_PA_PARENT,
    },
    "tres_deltagru_via_tres_deltagru_pa": {
        "model_key": "tres_deltagru",
        "pa_model_key": ALTERNATE_PA_KEY,
        "pa_parent": ALTERNATE_PA_PARENT,
    },
}

POLYNOMIAL_CONFIGURATIONS = {
    "pa_modeling": {
        "mp": {
            "basis_configuration": {"K": 9, "Q": 150},
            "complex_coefficients": 1350,
            "real_parameters": 2700,
        },
        "gmp": {
            "basis_configuration": {
                "Ka": 5,
                "La": 30,
                "Kb": 4,
                "Lb": 30,
                "Mb": 5,
                "Kc": 4,
                "Lc": 30,
                "Mc": 5,
            },
            "complex_coefficients": 1350,
            "real_parameters": 2700,
        },
    },
    "dpd_ila": {
        "mp": {
            "basis_configuration": {"K": 5, "Q": 100},
            "complex_coefficients": 500,
            "real_parameters": 1000,
        },
        "gmp": {
            "basis_configuration": {
                "Ka": 5,
                "La": 20,
                "Kb": 4,
                "Lb": 20,
                "Mb": 3,
                "Kc": 4,
                "Lc": 20,
                "Mc": 2,
            },
            "complex_coefficients": 500,
            "real_parameters": 1000,
        },
    },
}

NEURAL_RECIPE = {
    "epochs": 300,
    "batch_size": 64,
    "initial_learning_rate": 5e-3,
    "lr_schedule": "ReduceLROnPlateau",
    "minimum_learning_rate": 5e-5,
    "decay_factor": 0.5,
    "patience": 5,
    "frame_length": 200,
    "frame_stride": 1,
    "optimizer": "AdamW",
    "optimizer_weight_decay": 0.01,
    "optimizer_betas": [0.9, 0.999],
    "optimizer_eps": 1e-8,
    "loss": "MSE",
    "scheduler_threshold": 1e-4,
    "scheduler_threshold_mode": "rel",
    "scheduler_cooldown": 0,
    "scheduler_eps": 1e-8,
    "seed": 0,
}

SOURCE_PATTERNS = (
    "*.py",
    "backbones/**/*.py",
    "benchmark/**/*.py",
    "modules/**/*.py",
    "opendpd/**/*.py",
    "quant/**/*.py",
    "steps/**/*.py",
    "utils/**/*.py",
)
SOURCE_DOCUMENTS = (
    "benchmark/benchmark_report.md",
    "benchmark/reproduce_benchmark_report.sh",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_snapshot(paths: list[str], destination: Path) -> None:
    """Write an atomic, deterministic archive of the exact benchmark sources."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with temporary_path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                with tarfile.open(
                    fileobj=gzip_handle,
                    mode="w",
                    format=tarfile.GNU_FORMAT,
                ) as archive:
                    for relative_path in sorted(paths):
                        source = REPO_ROOT / relative_path
                        if not source.is_file() or source.is_symlink():
                            raise ValueError(
                                "Source snapshot entries must be regular files: "
                                f"{relative_path}"
                            )
                        content = source.read_bytes()
                        information = tarfile.TarInfo(relative_path)
                        information.size = len(content)
                        information.mode = source.stat().st_mode & 0o777
                        information.mtime = 0
                        information.uid = 0
                        information.gid = 0
                        information.uname = ""
                        information.gname = ""
                        archive.addfile(information, io.BytesIO(content))
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_source_snapshot(
    path: Path,
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    """Verify every archive member against the pre-run source hash map."""
    if not path.is_file():
        raise FileNotFoundError(f"Source snapshot is missing: {path}")
    observed: dict[str, str] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError(
                    f"Source snapshot contains a non-file member: {member.name}"
                )
            if member.name in observed:
                raise ValueError(
                    f"Source snapshot contains a duplicate member: {member.name}"
                )
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(
                    f"Cannot read source snapshot member: {member.name}"
                )
            observed[member.name] = hashlib.sha256(handle.read()).hexdigest()
    require_equal(observed, expected_hashes, "source snapshot members")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "file_count": len(observed),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def as_int(row: dict[str, str], key: str) -> int:
    value = as_float(row, key)
    if not value.is_integer():
        raise ValueError(f"{key} is not an integer: {value}")
    return int(value)


def as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite: {value}")
    return value


def require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{description} does not match the benchmark recipe: "
            f"expected {expected!r}, got {actual!r}"
        )


def require_close(
    actual: float,
    expected: float,
    description: str,
    *,
    tolerance: float = 1e-12,
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"{description} does not match the benchmark recipe: "
            f"expected {expected}, got {actual}"
        )


def metric_block(row: dict[str, str]) -> dict[str, dict[str, float]]:
    def split(prefix: str) -> dict[str, float]:
        metrics = {
            "loss": as_float(row, f"{prefix}_LOSS"),
            "nmse_db": as_float(row, f"{prefix}_NMSE"),
            "evm_db": as_float(row, f"{prefix}_EVM"),
            "aclr_left_db": as_float(row, f"{prefix}_ACLR_L"),
            "aclr_right_db": as_float(row, f"{prefix}_ACLR_R"),
            "aclr_avg_db": as_float(row, f"{prefix}_ACLR_AVG"),
        }
        require_close(
            metrics["aclr_avg_db"],
            (metrics["aclr_left_db"] + metrics["aclr_right_db"]) / 2.0,
            f"{prefix} logged ACLR average",
            tolerance=1e-8,
        )
        return metrics

    return {"validation": split("VAL"), "test": split("TEST")}


def source_files() -> tuple[str, ...]:
    paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for pattern in SOURCE_PATTERNS
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    }
    paths.update(SOURCE_DOCUMENTS)
    missing = [path for path in paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Expected benchmark source files are missing: {sorted(missing)}"
        )
    return tuple(sorted(paths))


def validate_checkpoint(path: Path, expected_parameters: int) -> None:
    import torch

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Checkpoint cannot be loaded safely: {path}") from exc
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Checkpoint is not a non-empty state dictionary: {path}")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError(f"Checkpoint contains non-tensor state entries: {path}")
    parameter_count = sum(value.numel() for value in state_dict.values())
    require_equal(
        parameter_count,
        expected_parameters,
        f"checkpoint parameter count for {path}",
    )
    if not all(torch.isfinite(value).all().item() for value in state_dict.values()):
        raise ValueError(f"Checkpoint contains a non-finite tensor: {path}")


def archived_artifact(output_dir: Path, source: Path) -> dict[str, Any]:
    absolute_source = REPO_ROOT / source
    archived_path = output_dir / "artifacts" / source
    if not absolute_source.is_file():
        raise FileNotFoundError(f"Expected benchmark artifact is missing: {source}")
    if not archived_path.is_file():
        raise FileNotFoundError(
            f"Benchmark artifact was not archived into the run directory: {source}"
        )
    source_hash = sha256_file(absolute_source)
    archived_hash = sha256_file(archived_path)
    require_equal(archived_hash, source_hash, f"archived hash for {source}")
    return {
        "source_path": source.as_posix(),
        "archived_path": archived_path.relative_to(output_dir).as_posix(),
        "sha256": source_hash,
        "size_bytes": absolute_source.stat().st_size,
    }


def read_pa_bindings(
    output_dir: Path,
) -> dict[tuple[str, str, str], dict[str, str]]:
    path = output_dir / "pa_checkpoint_bindings.tsv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require_equal(
        len(rows),
        len(DATASETS) * len(NEURAL_DPD_EXPERIMENTS),
        "neural DPD PA checkpoint binding count",
    )

    bindings: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["dpd_model_key"],
            row["pa_model_key"],
        )
        if key in bindings:
            raise ValueError(f"Duplicate PA checkpoint binding: {key}")
        if row["dataset"] not in DATASETS:
            raise ValueError(f"Unknown dataset in PA checkpoint binding: {key}")
        if (
            row["dpd_model_key"] not in NEURAL_DPD_MODELS
            and row["dpd_model_key"] != "tres_deltagru"
        ):
            raise ValueError(f"Unknown model in PA checkpoint binding: {key}")
        if row["pa_model_key"] not in {REFERENCE_PA_KEY, ALTERNATE_PA_KEY}:
            raise ValueError(f"Unknown PA surrogate in checkpoint binding: {key}")
        if len(row["sha256"]) != 64:
            raise ValueError(f"Invalid PA checkpoint hash for {key}")
        bindings[key] = row

    expected_keys = {
        (
            dataset,
            experiment["model_key"],
            experiment["pa_model_key"],
        )
        for dataset in DATASETS
        for experiment in NEURAL_DPD_EXPERIMENTS.values()
    }
    require_equal(set(bindings), expected_keys, "neural DPD PA checkpoint bindings")
    return bindings


def read_recorded_commands(path: Path) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    current_label: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            current_label = line[2:]
            if current_label in commands:
                raise ValueError(f"Duplicate command label in {path}: {current_label}")
            continue
        if not line:
            continue
        if current_label is None:
            raise ValueError(f"Unlabelled command in {path}: {line}")
        commands[current_label] = shlex.split(line)
        current_label = None
    if current_label is not None:
        raise ValueError(f"Command label has no command in {path}: {current_label}")
    return commands


def cli_options(command: list[str], *, label: str) -> dict[str, str]:
    try:
        main_index = command.index("main.py")
    except ValueError as exc:
        raise ValueError(f"Recorded training command lacks main.py: {label}") from exc
    tokens = command[main_index + 1 :]
    if len(tokens) % 2:
        raise ValueError(f"Recorded training command has an unpaired option: {label}")
    options: dict[str, str] = {}
    for option, value in zip(tokens[::2], tokens[1::2]):
        if not option.startswith("--"):
            raise ValueError(
                f"Unexpected positional token in recorded training command "
                f"{label}: {option}"
            )
        key = option[2:]
        if key in options:
            raise ValueError(f"Duplicate --{key} in recorded command: {label}")
        options[key] = value
    return options


def validate_neural_commands(path: Path, device: int | None = None) -> None:
    commands = read_recorded_commands(path)
    training_commands = {
        label: cli_options(command, label=label)
        for label, command in commands.items()
        if "main.py" in command
    }
    require_equal(
        len(training_commands),
        len(DATASETS) * (len(NEURAL_PA_MODELS) + len(NEURAL_DPD_EXPERIMENTS)),
        "recorded neural training command count",
    )

    expected_matrix = {
        *(
            (
                "train_pa",
                dataset,
                model["backbone"],
                model["hidden_size"],
                None,
                None,
            )
            for dataset in DATASETS
            for model in NEURAL_PA_MODELS.values()
        ),
        *(
            (
                "train_dpd",
                dataset,
                (
                    NEURAL_DPD_MODELS[experiment["model_key"]]
                    if experiment["model_key"] in NEURAL_DPD_MODELS
                    else TRES_DELTAGRU_DPD_MODEL
                )["backbone"],
                (
                    NEURAL_DPD_MODELS[experiment["model_key"]]
                    if experiment["model_key"] in NEURAL_DPD_MODELS
                    else TRES_DELTAGRU_DPD_MODEL
                )["hidden_size"],
                NEURAL_PA_MODELS[experiment["pa_model_key"]]["backbone"],
                NEURAL_PA_MODELS[experiment["pa_model_key"]]["hidden_size"],
            )
            for dataset in DATASETS
            for experiment in NEURAL_DPD_EXPERIMENTS.values()
        ),
    }
    actual_matrix: set[tuple[str, str, str, int, str | None, int | None]] = set()
    for label, options in training_commands.items():
        step = options.get("step")
        if step not in {"train_pa", "train_dpd"}:
            raise ValueError(f"Unexpected training step in {label}: {step!r}")
        backbone_key = "PA_backbone" if step == "train_pa" else "DPD_backbone"
        hidden_key = "PA_hidden_size" if step == "train_pa" else "DPD_hidden_size"
        try:
            matrix_key = (
                step,
                options["dataset_name"],
                options[backbone_key],
                int(options[hidden_key]),
                options.get("PA_backbone") if step == "train_dpd" else None,
                (
                    int(options["PA_hidden_size"])
                    if step == "train_dpd"
                    else None
                ),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Incomplete model recipe in command {label}") from exc
        if matrix_key in actual_matrix:
            raise ValueError(f"Duplicate neural benchmark command: {matrix_key}")
        actual_matrix.add(matrix_key)

        integer_expectations = {
            "n_epochs": NEURAL_RECIPE["epochs"],
            "batch_size": NEURAL_RECIPE["batch_size"],
            "batch_size_eval": NEURAL_RECIPE["batch_size"],
            "lr_schedule": 1,
            "frame_length": NEURAL_RECIPE["frame_length"],
            "frame_stride": NEURAL_RECIPE["frame_stride"],
            "PA_num_layers": 1,
            "DPD_num_layers": 1,
            "log_precision": 8,
            "eval_val": 1,
            "eval_test": 1,
            "seed": NEURAL_RECIPE["seed"],
        }
        if device is not None:
            integer_expectations["devices"] = device
        for option, expected in integer_expectations.items():
            try:
                actual = int(options[option])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Missing or invalid --{option} in command {label}"
                ) from exc
            require_equal(actual, expected, f"--{option} in command {label}")

        float_expectations = {
            "lr": NEURAL_RECIPE["initial_learning_rate"],
            "lr_end": NEURAL_RECIPE["minimum_learning_rate"],
            "decay_factor": NEURAL_RECIPE["decay_factor"],
            "patience": NEURAL_RECIPE["patience"],
            "grad_clip_val": 200.0,
            "thx": 0.0,
            "thh": 0.0,
        }
        for option, expected in float_expectations.items():
            try:
                actual = float(options[option])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Missing or invalid --{option} in command {label}"
                ) from exc
            require_close(actual, expected, f"--{option} in command {label}")

        require_equal(options.get("opt_type"), "adamw", f"optimizer in {label}")
        require_equal(options.get("loss_type"), "l2", f"loss in {label}")
        require_equal(options.get("accelerator"), "cuda", f"accelerator in {label}")
        require_equal(
            options.get("re_level"),
            "soft",
            f"reproducibility level in {label}",
        )
    require_equal(
        actual_matrix,
        expected_matrix,
        "recorded neural model matrix",
    )


def module_cli_options(
    command: list[str],
    *,
    label: str,
    module: str,
) -> dict[str, str]:
    try:
        module_index = command.index("-m")
    except ValueError as exc:
        raise ValueError(f"Recorded command lacks -m {module}: {label}") from exc
    if (
        module_index + 1 >= len(command)
        or command[module_index + 1] != module
    ):
        raise ValueError(f"Recorded command uses the wrong module: {label}")
    tokens = command[module_index + 2 :]
    if len(tokens) % 2:
        raise ValueError(f"Recorded module command has an unpaired option: {label}")
    options: dict[str, str] = {}
    for option, value in zip(tokens[::2], tokens[1::2]):
        if not option.startswith("--"):
            raise ValueError(
                f"Unexpected positional token in recorded command {label}: "
                f"{option}"
            )
        key = option[2:]
        if key in options:
            raise ValueError(f"Duplicate --{key} in recorded command: {label}")
        options[key] = value
    return options


def validate_runner_commands(
    path: Path,
    *,
    output_dir: Path,
    device: int,
) -> None:
    """Validate the complete recorded neural and polynomial experiment matrix."""
    validate_neural_commands(path, device=device)
    commands = read_recorded_commands(path)
    expected_labels = {"snapshot_context", "collect_report"}
    expected_labels.update(
        f"train_pa_{configuration['slug']}_{model}"
        for configuration in DATASETS.values()
        for model in NEURAL_PA_MODELS
    )
    expected_labels.update(
        f"train_dpd_{configuration['slug']}_{experiment_key}"
        for configuration in DATASETS.values()
        for experiment_key in NEURAL_DPD_EXPERIMENTS
    )
    expected_labels.update(
        f"pa_model_{configuration['slug']}_{model}"
        for configuration in DATASETS.values()
        for model in ("mp", "gmp")
    )
    expected_labels.update(
        f"dpd_ila_{configuration['slug']}_{model}"
        for configuration in DATASETS.values()
        for model in ("mp", "gmp")
    )
    require_equal(set(commands), expected_labels, "recorded benchmark job matrix")

    for label in ("snapshot_context", "collect_report"):
        command = commands[label]
        require_equal(
            command.count("benchmark.collect_benchmark_report"),
            1,
            f"collector module in {label}",
        )
        require_equal(
            command[command.index("--output-dir") + 1],
            str(output_dir),
            f"collector output directory in {label}",
        )
        require_equal(
            int(command[command.index("--device") + 1]),
            device,
            f"collector device in {label}",
        )
    require_equal(
        "--snapshot-context" in commands["snapshot_context"],
        True,
        "pre-run context command",
    )
    require_equal(
        "--snapshot-context" in commands["collect_report"],
        False,
        "final collector command",
    )

    expected_solver_device = f"cuda:{device}"
    for dataset, dataset_configuration in DATASETS.items():
        slug = dataset_configuration["slug"]
        for task, label_prefix, filename_task in (
            ("pa_modeling", "pa_model", "pa"),
            ("dpd_ila", "dpd_ila", "dpd"),
        ):
            for model_name in ("mp", "gmp"):
                label = f"{label_prefix}_{slug}_{model_name}"
                options = module_cli_options(
                    commands[label],
                    label=label,
                    module="benchmark.benchmark_volterra",
                )
                expectations = {
                    "task": task,
                    "dataset-name": dataset,
                    "model": model_name,
                    "solver-device": expected_solver_device,
                    "solver-dtype": "complex64",
                    "device": str(device),
                    "json-out": str(
                        output_dir
                        / (
                            f"benchmark_report_{slug}_{filename_task}_"
                            f"{model_name}.json"
                        )
                    ),
                }
                if task == "pa_modeling" and model_name == "gmp":
                    expectations.update(
                        {
                            "solver-mode": "truncated_svd",
                            "svd-rcond": "1e-4",
                        }
                    )
                else:
                    expectations["solver-mode"] = "gels"
                    require_equal(
                        "svd-rcond" in options,
                        False,
                        f"unexpected --svd-rcond in command {label}",
                    )
                if task == "dpd_ila":
                    expectations.update(
                        {
                            "pa-backbone": "tres_gru",
                            "pa-hidden-size": "27",
                            "pa-num-layers": "1",
                            "pa-checkpoint": (
                                f"save/{dataset}/train_pa/"
                                "PA_S_0_M_TRES_GRU_H_27_F_200_P_2751.pt"
                            ),
                        }
                    )
                for key, value in expectations.items():
                    require_equal(
                        options.get(key),
                        value,
                        f"--{key} in command {label}",
                    )

                configuration = POLYNOMIAL_CONFIGURATIONS[task][model_name][
                    "basis_configuration"
                ]
                for key, value in configuration.items():
                    require_equal(
                        int(options[key]),
                        value,
                        f"--{key} in command {label}",
                    )


def expected_runner_recipe() -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "datasets": list(DATASETS),
        "parameter_count_convention": (
            "Neural parameters are real trainable scalars; each complex "
            "polynomial coefficient counts as two real degrees of freedom."
        ),
        "neural": {
            "optimizer": "AdamW",
            "optimizer_weight_decay": NEURAL_RECIPE["optimizer_weight_decay"],
            "optimizer_betas": NEURAL_RECIPE["optimizer_betas"],
            "optimizer_eps": NEURAL_RECIPE["optimizer_eps"],
            "loss": "MSE",
            "epochs": NEURAL_RECIPE["epochs"],
            "batch_size": NEURAL_RECIPE["batch_size"],
            "evaluation_batch_size": NEURAL_RECIPE["batch_size"],
            "initial_learning_rate": NEURAL_RECIPE["initial_learning_rate"],
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": NEURAL_RECIPE["decay_factor"],
            "scheduler_patience": NEURAL_RECIPE["patience"],
            "scheduler_threshold": NEURAL_RECIPE["scheduler_threshold"],
            "scheduler_threshold_mode": NEURAL_RECIPE[
                "scheduler_threshold_mode"
            ],
            "scheduler_cooldown": NEURAL_RECIPE["scheduler_cooldown"],
            "scheduler_eps": NEURAL_RECIPE["scheduler_eps"],
            "minimum_learning_rate": NEURAL_RECIPE[
                "minimum_learning_rate"
            ],
            "frame_length": NEURAL_RECIPE["frame_length"],
            "frame_stride": NEURAL_RECIPE["frame_stride"],
            "gradient_clip": 200,
            "seed": NEURAL_RECIPE["seed"],
            "reproducibility": "soft",
            "pa_scheduler_and_selection_metric": "validation NMSE",
            "dpd_scheduler_and_selection_metric": "validation ACLR_AVG",
        },
        "pa_models": {
            "mp": {"K": 9, "Q": 150, "real_parameters": 2700},
            "gmp": {
                **POLYNOMIAL_CONFIGURATIONS["pa_modeling"]["gmp"][
                    "basis_configuration"
                ],
                "real_parameters": 2700,
            },
            "gru": {"hidden_size": 28, "real_parameters": 2746},
            "tres_gru": {"hidden_size": 27, "real_parameters": 2751},
            "tres_deltagru": {
                "hidden_size": 27,
                "real_parameters": 2751,
                "delta_thresholds": {
                    "input": 0.0,
                    "hidden": 0.0,
                },
            },
        },
        "dpd_models": {
            "mp": {
                "K": 5,
                "Q": 100,
                "real_parameters": 1000,
                "method": "ILA",
            },
            "gmp": {
                **POLYNOMIAL_CONFIGURATIONS["dpd_ila"]["gmp"][
                    "basis_configuration"
                ],
                "real_parameters": 1000,
                "method": "ILA",
            },
            "gru": {"hidden_size": 16, "real_parameters": 994},
            "tres_gru": {"hidden_size": 15, "real_parameters": 999},
        },
        "dpd_tres_deltagru_pa_surrogate_comparison": {
            "model": {
                "hidden_size": 15,
                "real_parameters": 999,
                "delta_thresholds": {
                    "input": 0.0,
                    "hidden": 0.0,
                },
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
                    "delta_thresholds": {
                        "input": 0.0,
                        "hidden": 0.0,
                    },
                },
            },
        },
        "dpd_pa_surrogate": {
            "backbone": "tres_gru",
            "hidden_size": 27,
            "real_parameters": 2751,
        },
        "polynomial_solvers": {
            "default": {
                "implementation": "torch.linalg.lstsq",
                "mode": "gels",
                "driver": "gels",
                "dtype": "torch.complex64",
                "column_scaling": "l2",
                "regularization": None,
            },
            "gmp_pa": {
                "implementation": "torch.linalg.svd",
                "mode": "truncated_svd",
                "driver": "gesvdj",
                "dtype": "torch.complex64",
                "column_scaling": "l2",
                "regularization": {
                    "type": "truncated_svd",
                    "relative_cutoff": GMP_PA_SVD_RCOND,
                },
            },
            "segment_boundary_policy": (
                "zero delay state at every nperseg boundary"
            ),
        },
    }


def validate_recipe_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark recipe is missing: {path}")
    with path.open() as handle:
        recipe = json.load(handle)
    require_equal(recipe, expected_runner_recipe(), "recorded benchmark recipe")
    return recipe


def validate_learning_rate_schedule(
    history_rows: list[dict[str, str]],
    *,
    model_id: str,
    initial_learning_rate: float,
    minimum_learning_rate: float,
    decay_factor: float,
) -> None:
    """Validate the observable portion of the configured plateau schedule."""
    learning_rates = [as_float(row, "LR") for row in history_rows]
    require_close(
        learning_rates[0],
        initial_learning_rate,
        f"initial learning rate for {model_id}",
        tolerance=1e-10,
    )
    for epoch, learning_rate in enumerate(learning_rates):
        if learning_rate < minimum_learning_rate - 1e-10:
            raise ValueError(
                f"Learning rate at epoch {epoch} for {model_id} is below "
                f"the configured minimum: {learning_rate} < "
                f"{minimum_learning_rate}"
            )
        if learning_rate > initial_learning_rate + 1e-10:
            raise ValueError(
                f"Learning rate at epoch {epoch} for {model_id} exceeds "
                f"the configured initial value: {learning_rate} > "
                f"{initial_learning_rate}"
            )
    for epoch, (previous, current) in enumerate(
        zip(learning_rates, learning_rates[1:]), start=1
    ):
        if current > previous + 1e-10:
            raise ValueError(
                f"Learning rate increased at epoch {epoch} for {model_id}: "
                f"{previous} -> {current}"
            )
        if not math.isclose(current, previous, rel_tol=0.0, abs_tol=1e-10):
            expected = max(previous * decay_factor, minimum_learning_rate)
            require_close(
                current,
                expected,
                f"scheduled learning rate at epoch {epoch} for {model_id}",
                tolerance=1e-8,
            )


def collect_training_model(
    output_dir: Path,
    *,
    dataset: str,
    step: str,
    parent: str | None,
    model_id: str,
    expected_epochs: int,
    expected_batch_size: int,
    expected_initial_lr: float,
    expected_backbone: str,
    expected_hidden_size: int,
    expected_parameters: int,
    selection_metric: str,
    expected_delta_thresholds: dict[str, float] | None = None,
    expected_pa_parameters: int | None = None,
) -> dict[str, Any]:
    base = Path("log") / dataset / step
    save_base = Path("save") / dataset / step
    if parent is not None:
        base /= parent
        save_base /= parent

    best_path = base / "best" / f"{model_id}.csv"
    history_path = base / "history" / f"{model_id}.csv"
    checkpoint_path = save_base / f"{model_id}.pt"
    validate_checkpoint(REPO_ROOT / checkpoint_path, expected_parameters)

    best_rows = read_csv(REPO_ROOT / best_path)
    history_rows = read_csv(REPO_ROOT / history_path)
    require_equal(len(best_rows), 1, f"best-row count for {model_id}")
    require_equal(
        len(history_rows), expected_epochs, f"history-row count for {model_id}"
    )
    require_equal(
        [as_int(row, "EPOCH") for row in history_rows],
        list(range(expected_epochs)),
        f"epoch sequence for {model_id}",
    )

    best = best_rows[0]
    require_equal(as_int(best, "N_EPOCH"), expected_epochs, f"epochs for {model_id}")
    require_equal(
        as_int(best, "BATCH_SIZE"),
        expected_batch_size,
        f"batch size for {model_id}",
    )
    require_equal(as_int(best, "FRAME_LENGTH"), 200, f"frame length for {model_id}")
    require_equal(best["BACKBONE"], expected_backbone, f"backbone for {model_id}")
    require_equal(
        as_int(best, "HIDDEN_SIZE"),
        expected_hidden_size,
        f"hidden size for {model_id}",
    )
    expected_logged_parameters = expected_parameters
    if step == "train_dpd":
        if expected_pa_parameters is None:
            raise ValueError(
                f"Expected PA parameter count is required for DPD model {model_id}"
            )
        expected_logged_parameters += expected_pa_parameters
    require_equal(
        as_int(best, "N_PARAM"),
        expected_logged_parameters,
        f"logged network parameter count for {model_id}",
    )
    if expected_delta_thresholds is not None:
        for row_index, row in enumerate(history_rows):
            require_close(
                as_float(row, "THX"),
                expected_delta_thresholds["input"],
                f"input delta threshold at row {row_index} for {model_id}",
            )
            require_close(
                as_float(row, "THH"),
                expected_delta_thresholds["hidden"],
                f"hidden delta threshold at row {row_index} for {model_id}",
            )
        require_close(
            as_float(best, "THX"),
            expected_delta_thresholds["input"],
            f"selected input delta threshold for {model_id}",
        )
        require_close(
            as_float(best, "THH"),
            expected_delta_thresholds["hidden"],
            f"selected hidden delta threshold for {model_id}",
        )
    validate_learning_rate_schedule(
        history_rows,
        model_id=model_id,
        initial_learning_rate=expected_initial_lr,
        minimum_learning_rate=NEURAL_RECIPE["minimum_learning_rate"],
        decay_factor=NEURAL_RECIPE["decay_factor"],
    )

    best_epoch = as_int(best, "EPOCH")
    logged_selection_values = [
        as_float(row, selection_metric) for row in history_rows
    ]
    minimum_logged_value = min(logged_selection_values)
    tied_best_epochs = {
        index
        for index, value in enumerate(logged_selection_values)
        if value == minimum_logged_value
    }
    if best_epoch not in tied_best_epochs:
        raise ValueError(
            f"Selected epoch {best_epoch} for {model_id} is not tied for the "
            f"minimum logged {selection_metric}; tied epochs are "
            f"{sorted(tied_best_epochs)}"
        )
    matching_history_rows = [
        row for row in history_rows if as_int(row, "EPOCH") == best_epoch
    ]
    require_equal(
        len(matching_history_rows), 1, f"selected epoch in history for {model_id}"
    )
    for key, value in best.items():
        require_equal(
            matching_history_rows[0].get(key),
            value,
            f"best/history value {key} for {model_id}",
        )

    result = {
        "selected_epoch_zero_based": best_epoch,
        "completed_epochs": len(history_rows),
        "selected_learning_rate": as_float(best, "LR"),
        "train_loss": as_float(best, "TRAIN_LOSS"),
        "metrics": metric_block(best),
        "artifacts": {
            "checkpoint": archived_artifact(output_dir, checkpoint_path),
            "best_csv": archived_artifact(output_dir, best_path),
            "history_csv": archived_artifact(output_dir, history_path),
        },
    }
    if expected_delta_thresholds is not None:
        result["delta_thresholds"] = dict(expected_delta_thresholds)
    return result


def normalize_polynomial_metrics(
    values: dict[str, float],
) -> dict[str, float]:
    normalized = {
        "nmse_db": float(values["NMSE"]),
        "evm_db": float(values["EVM"]),
        "aclr_left_db": float(values["ACLR_L"]),
        "aclr_right_db": float(values["ACLR_R"]),
        "aclr_avg_db": float(values["ACLR_AVG"]),
    }
    for key, value in normalized.items():
        if not math.isfinite(value):
            raise ValueError(f"Polynomial metric {key} is not finite: {value}")
    require_close(
        normalized["aclr_avg_db"],
        (normalized["aclr_left_db"] + normalized["aclr_right_db"]) / 2.0,
        "polynomial-model ACLR average",
        tolerance=1e-10,
    )
    return normalized


def finite_number(
    raw: dict[str, Any],
    key: str,
    *,
    source_path: Path,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    try:
        value = float(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} is missing or non-numeric in {source_path}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite in {source_path}: {value}")
    if positive and value <= 0.0:
        raise ValueError(f"{key} must be positive in {source_path}: {value}")
    if nonnegative and value < 0.0:
        raise ValueError(f"{key} must be nonnegative in {source_path}: {value}")
    return value


def collect_polynomial_model(
    output_dir: Path,
    *,
    dataset: str,
    slug: str,
    task: str,
    model_name: str,
    expected_pa_hash: str | None = None,
    expected_device: str | None = None,
    expected_target_gain: float | None = None,
) -> dict[str, Any]:
    if task not in POLYNOMIAL_CONFIGURATIONS:
        raise ValueError(f"Unsupported polynomial benchmark task: {task}")
    filename_task = "pa" if task == "pa_modeling" else "dpd"
    source_path = (
        output_dir / f"benchmark_report_{slug}_{filename_task}_{model_name}.json"
    )
    with source_path.open() as handle:
        raw = json.load(handle)

    configuration = POLYNOMIAL_CONFIGURATIONS[task][model_name]
    expected_method = (
        "direct_least_squares"
        if task == "pa_modeling"
        else "indirect_learning_architecture"
    )
    uses_truncated_svd = task == "pa_modeling" and model_name == "gmp"

    require_equal(
        raw["schema_version"],
        POLYNOMIAL_SCHEMA_VERSION,
        f"polynomial schema in {source_path}",
    )
    require_equal(raw["dataset"], dataset, f"polynomial dataset in {source_path}")
    require_equal(raw["task"], task, f"polynomial task in {source_path}")
    require_equal(raw["model"], model_name, f"polynomial model in {source_path}")
    require_equal(raw["method"], expected_method, f"method in {source_path}")
    expected_solver = (
        "torch.linalg.svd" if uses_truncated_svd else "torch.linalg.lstsq"
    )
    expected_solver_mode = "truncated_svd" if uses_truncated_svd else "gels"
    expected_solver_driver = "gesvdj" if uses_truncated_svd else "gels"
    require_equal(raw["solver"], expected_solver, f"polynomial solver in {source_path}")
    require_equal(
        raw["solver_mode"],
        expected_solver_mode,
        f"polynomial solver mode in {source_path}",
    )
    require_equal(
        raw["solver_driver"],
        expected_solver_driver,
        f"least-squares driver in {source_path}",
    )
    require_equal(
        raw["dtype"],
        "torch.complex64",
        f"least-squares dtype in {source_path}",
    )
    require_equal(
        raw["column_scaling"],
        "l2",
        f"least-squares column scaling in {source_path}",
    )
    if not isinstance(raw["device"], str) or not raw["device"].startswith("cuda:"):
        raise ValueError(
            f"Polynomial least-squares device must be a CUDA device in "
            f"{source_path}: {raw['device']!r}"
        )
    require_equal(
        raw.get("solver_device"),
        raw["device"],
        f"least-squares device alias in {source_path}",
    )
    if expected_device is not None:
        require_equal(
            raw["device"],
            expected_device,
            f"least-squares device in {source_path}",
        )
    require_equal(
        raw["basis_configuration"],
        configuration["basis_configuration"],
        f"polynomial basis in {source_path}",
    )
    require_equal(
        raw["complex_coefficients"],
        configuration["complex_coefficients"],
        f"least-squares complex coefficient count in {source_path}",
    )
    require_equal(
        raw["real_parameters"],
        configuration["real_parameters"],
        f"least-squares parameter count in {source_path}",
    )
    require_equal(
        raw["parameter_count_convention"],
        "two real degrees of freedom per complex coefficient",
        f"parameter-count convention in {source_path}",
    )
    target_gain = finite_number(
        raw,
        "target_gain",
        source_path=source_path,
        positive=True,
    )
    if expected_target_gain is not None:
        require_close(
            target_gain,
            expected_target_gain,
            f"target gain in {source_path}",
            tolerance=1e-6,
        )
    rank = raw.get("least_squares_rank")
    if uses_truncated_svd:
        require_equal(
            raw.get("full_rank_assumption"),
            False,
            f"full-rank assumption for truncated SVD in {source_path}",
        )
        require_equal(
            raw.get("regularization"),
            "truncated_svd",
            f"regularization in {source_path}",
        )
        svd_rcond = finite_number(
            raw,
            "svd_rcond",
            source_path=source_path,
            positive=True,
        )
        require_close(
            svd_rcond,
            GMP_PA_SVD_RCOND,
            f"SVD relative cutoff in {source_path}",
            tolerance=1e-12,
        )
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(
                f"least-squares rank is not an integer in {source_path}: "
                f"{rank!r}"
            )
        if not 0 < rank < configuration["complex_coefficients"]:
            raise ValueError(
                f"truncated-SVD rank is outside the expected range in "
                f"{source_path}: {rank}"
            )
        singular_value_max = finite_number(
            raw,
            "singular_value_max",
            source_path=source_path,
            positive=True,
        )
        singular_value_min = finite_number(
            raw,
            "singular_value_min",
            source_path=source_path,
            positive=True,
        )
        singular_value_cutoff = finite_number(
            raw,
            "singular_value_cutoff",
            source_path=source_path,
            positive=True,
        )
        retained_singular_value_min = finite_number(
            raw,
            "retained_singular_value_min",
            source_path=source_path,
            positive=True,
        )
        condition_number = finite_number(
            raw,
            "condition_number",
            source_path=source_path,
            positive=True,
        )
        require_close(
            singular_value_cutoff,
            singular_value_max * svd_rcond,
            f"singular-value cutoff in {source_path}",
            tolerance=1e-5,
        )
        if not (
            singular_value_min
            <= retained_singular_value_min
            <= singular_value_max
        ):
            raise ValueError(
                f"retained singular-value bounds are invalid in {source_path}"
            )
        if retained_singular_value_min <= singular_value_cutoff:
            raise ValueError(
                f"retained singular value does not exceed the cutoff in "
                f"{source_path}"
            )
        if not math.isclose(
            condition_number,
            singular_value_max / singular_value_min,
            rel_tol=1e-6,
        ):
            raise ValueError(
                f"condition number is inconsistent with singular values in "
                f"{source_path}"
            )
    else:
        svd_rcond = None
        singular_value_max = None
        singular_value_min = None
        singular_value_cutoff = None
        retained_singular_value_min = None
        condition_number = None
        require_equal(
            raw.get("regularization"),
            None,
            f"regularization in {source_path}",
        )
        require_equal(
            raw.get("svd_rcond"),
            None,
            f"SVD relative cutoff in {source_path}",
        )
        for key in (
            "singular_value_max",
            "singular_value_min",
            "singular_value_cutoff",
            "retained_singular_value_min",
            "condition_number",
        ):
            require_equal(
                raw.get(key),
                None,
                f"{key} in {source_path}",
            )
        require_equal(
            raw.get("full_rank_assumption"),
            True,
            f"full-rank assumption for CUDA gels in {source_path}",
        )
        require_equal(rank, None, f"least-squares rank in {source_path}")

    residual = finite_number(
        raw,
        "training_relative_residual",
        source_path=source_path,
        nonnegative=True,
    )
    column_scale_min = finite_number(
        raw,
        "column_scale_min",
        source_path=source_path,
        positive=True,
    )
    column_scale_max = finite_number(
        raw,
        "column_scale_max",
        source_path=source_path,
        positive=True,
    )
    if column_scale_max < column_scale_min:
        raise ValueError(
            f"column_scale_max is below column_scale_min in {source_path}: "
            f"{column_scale_max} < {column_scale_min}"
        )
    coefficient_l2_norm = finite_number(
        raw,
        "coefficient_l2_norm",
        source_path=source_path,
        positive=True,
    )

    expected_sample_rate = DATASETS[dataset]["sample_rate_hz"]
    expected_nperseg = DATASETS[dataset]["nperseg"]
    require_close(
        float(raw["sample_rate_hz"]),
        expected_sample_rate,
        f"sample rate in {source_path}",
    )
    require_equal(
        raw["nperseg"],
        expected_nperseg,
        f"segment length in {source_path}",
    )
    pa_checkpoint_binding: dict[str, str] | None = None
    pa_evaluation_device: str | None = None
    if task == "dpd_ila":
        if expected_pa_hash is None:
            raise ValueError(
                f"Expected PA hash is required to collect DPD ILA result: "
                f"{source_path}"
            )
        expected_checkpoint = (
            Path("save")
            / dataset
            / "train_pa"
            / f"{NEURAL_PA_MODELS[REFERENCE_PA_KEY]['model_id']}.pt"
        ).as_posix()
        require_equal(
            raw["pa_checkpoint"],
            expected_checkpoint,
            f"PA checkpoint path in {source_path}",
        )
        require_equal(
            raw["pa_checkpoint_sha256"],
            expected_pa_hash,
            f"PA checkpoint hash in {source_path}",
        )
        pa_evaluation_device = raw.get("pa_evaluation_device")
        if expected_device is not None:
            require_equal(
                pa_evaluation_device,
                expected_device,
                f"PA evaluation device in {source_path}",
            )
        pa_checkpoint_binding = {
            "checkpoint": raw["pa_checkpoint"],
            "sha256": raw["pa_checkpoint_sha256"],
        }
    else:
        unexpected_binding_fields = {
            key
            for key in ("pa_checkpoint", "pa_checkpoint_sha256")
            if key in raw
        }
        if unexpected_binding_fields:
            raise ValueError(
                f"Direct PA polynomial result must not contain a PA checkpoint "
                f"binding in {source_path}: {sorted(unexpected_binding_fields)}"
            )

    validation_metrics = normalize_polynomial_metrics(raw["validation"])
    test_metrics = normalize_polynomial_metrics(raw["test"])

    return {
        "task": raw["task"],
        "method": raw["method"],
        "solver": raw["solver"],
        "solver_mode": raw["solver_mode"],
        "solver_driver": raw["solver_driver"],
        "dtype": raw["dtype"],
        "device": raw["device"],
        "column_scaling": raw["column_scaling"],
        "column_scale_min": column_scale_min,
        "column_scale_max": column_scale_max,
        "training_relative_residual": residual,
        "regularization": raw.get("regularization"),
        "svd_rcond": svd_rcond,
        "singular_value_max": singular_value_max,
        "singular_value_min": singular_value_min,
        "singular_value_cutoff": singular_value_cutoff,
        "retained_singular_value_min": retained_singular_value_min,
        "condition_number": condition_number,
        "coefficient_l2_norm": coefficient_l2_norm,
        "full_rank_assumption": raw.get("full_rank_assumption"),
        "basis_configuration": raw["basis_configuration"],
        "complex_coefficients": int(raw["complex_coefficients"]),
        "real_parameters": int(raw["real_parameters"]),
        "parameter_count_convention": raw["parameter_count_convention"],
        "target_gain": target_gain,
        "least_squares_rank": int(rank) if rank is not None else None,
        "sample_rate_hz": float(raw["sample_rate_hz"]),
        "nperseg": int(raw["nperseg"]),
        "pa_evaluation_device": pa_evaluation_device,
        "pa_checkpoint_binding": pa_checkpoint_binding,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "artifact": {
            "path": source_path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
        },
    }


def signal_record(dataset: str) -> dict[str, Any]:
    import numpy as np

    arrays = []
    dataset_files: dict[str, str] = {}
    train_input: Any = None
    train_output: Any = None
    for split in ("train", "val", "test"):
        input_path = Path("datasets") / dataset / f"{split}_input.csv"
        absolute_path = REPO_ROOT / input_path
        input_iq = np.loadtxt(
            absolute_path,
            delimiter=",",
            skiprows=1,
            dtype=np.float64,
        )
        arrays.append(input_iq)
        dataset_files[input_path.as_posix()] = sha256_file(absolute_path)
        output_path = Path("datasets") / dataset / f"{split}_output.csv"
        absolute_output_path = REPO_ROOT / output_path
        dataset_files[output_path.as_posix()] = sha256_file(absolute_output_path)
        if split == "train":
            train_input = input_iq
            train_output = np.loadtxt(
                absolute_output_path,
                delimiter=",",
                skiprows=1,
                dtype=np.float64,
            )

    iq = np.vstack(arrays)
    power = np.square(iq[:, 0]) + np.square(iq[:, 1])
    input_peak = float(
        np.max(np.hypot(train_input[:, 0], train_input[:, 1]))
    )
    output_peak = float(
        np.max(np.hypot(train_output[:, 0], train_output[:, 1]))
    )
    mean_power = float(np.mean(power))
    ccdf_probability = 1e-5
    ccdf_power = float(
        np.quantile(power, 1.0 - ccdf_probability, method="linear")
    )
    return {
        "samples": int(power.size),
        "target_gain": output_peak / input_peak,
        "papr": {
            "ccdf_probability": ccdf_probability,
            "quantile_method": "numpy linear",
            "ccdf_db": float(10.0 * np.log10(ccdf_power / mean_power)),
            "absolute_peak_db": float(
                10.0 * np.log10(float(np.max(power)) / mean_power)
            ),
        },
        "dataset_file_sha256": dataset_files,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def environment_record(device: int) -> dict[str, Any]:
    import torch

    gpu_name = (
        torch.cuda.get_device_name(device)
        if torch.cuda.is_available() and device < torch.cuda.device_count()
        else None
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "pandas": package_version("pandas"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_name,
        "gpu_device_index": device,
        "training_reproducibility_level": "soft",
        "training_deterministic_algorithms": False,
        "training_cudnn_benchmark": True,
    }


def git_record() -> dict[str, Any]:
    status = command_output(["git", "status", "--short", "--untracked-files=all"])
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "branch": command_output(["git", "branch", "--show-current"]),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def runner_start_git_record(output_dir: Path) -> dict[str, Any]:
    commit_path = output_dir / "git_commit_before.txt"
    branch_path = output_dir / "git_branch_before.txt"
    status_path = output_dir / "git_status_before.txt"
    if not all(path.is_file() for path in (commit_path, branch_path, status_path)):
        raise FileNotFoundError(
            "Runner start-state files are missing; use "
            "benchmark/reproduce_benchmark_report.sh."
        )
    status = status_path.read_text().strip()
    return {
        "commit": commit_path.read_text().strip(),
        "branch": branch_path.read_text().strip(),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def capture_context(
    output_dir: Path,
    device: int,
    *,
    use_runner_start_git: bool,
) -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": (
            runner_start_git_record(output_dir)
            if use_runner_start_git
            else git_record()
        ),
        "environment": environment_record(device),
        "source_file_sha256": {
            path: sha256_file(REPO_ROOT / path) for path in source_files()
        },
        "datasets": {
            dataset: {
                "spec_sha256": sha256_file(
                    REPO_ROOT / "datasets" / dataset / "spec.json"
                ),
                "signal": signal_record(dataset),
            }
            for dataset in DATASETS
        },
    }


def validate_context_stability(
    initial: dict[str, Any],
    completion: dict[str, Any],
) -> None:
    require_equal(
        completion["git"]["commit"],
        initial["git"]["commit"],
        "Git commit across the benchmark run",
    )
    require_equal(
        completion["git"]["branch"],
        initial["git"]["branch"],
        "Git branch across the benchmark run",
    )
    require_equal(
        completion["environment"],
        initial["environment"],
        "software and hardware environment across the benchmark run",
    )
    require_equal(
        completion["source_file_sha256"],
        initial["source_file_sha256"],
        "source files across the benchmark run",
    )
    require_equal(
        completion["datasets"],
        initial["datasets"],
        "dataset files and signal evidence across the benchmark run",
    )


def load_and_validate_context(
    output_dir: Path,
    device: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context_path = output_dir / "context_before.json"
    if not context_path.is_file():
        raise FileNotFoundError(
            "Pre-run context snapshot is missing; run the collector once with "
            "--snapshot-context before training."
        )
    with context_path.open() as handle:
        initial = json.load(handle)
    snapshot_metadata = validate_source_snapshot(
        output_dir / "source_snapshot.tar.gz",
        initial["source_file_sha256"],
    )
    require_equal(
        initial.get("source_snapshot"),
        snapshot_metadata,
        "source snapshot metadata",
    )
    completion = capture_context(
        output_dir,
        device,
        use_runner_start_git=False,
    )
    validate_context_stability(initial, completion)
    return initial, completion


def collect_manifest(
    output_dir: Path,
    device: int,
    initial_context: dict[str, Any],
    completion_context: dict[str, Any],
) -> dict[str, Any]:
    pa_bindings = read_pa_bindings(output_dir)
    commands_path = output_dir / "commands.log"
    validate_runner_commands(
        commands_path,
        output_dir=output_dir,
        device=device,
    )
    recipe_path = output_dir / "recipe.json"
    validate_recipe_file(recipe_path)
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_kind": "opendpd_benchmark_reproduction_run",
        "started_at_utc": initial_context["captured_at_utc"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_report": "benchmark/benchmark_report.md",
        "git": initial_context["git"],
        "environment": initial_context["environment"],
        "completion_context": {
            "captured_at_utc": completion_context["captured_at_utc"],
            "git_commit": completion_context["git"]["commit"],
            "git_branch": completion_context["git"]["branch"],
            "validated_unchanged": True,
        },
        "run_file_sha256": {
            "commands.log": sha256_file(commands_path),
            "context_before.json": sha256_file(
                output_dir / "context_before.json"
            ),
            "git_commit_before.txt": sha256_file(
                output_dir / "git_commit_before.txt"
            ),
            "git_branch_before.txt": sha256_file(
                output_dir / "git_branch_before.txt"
            ),
            "git_status_before.txt": sha256_file(
                output_dir / "git_status_before.txt"
            ),
            "pa_checkpoint_bindings.tsv": sha256_file(
                output_dir / "pa_checkpoint_bindings.tsv"
            ),
            "recipe.json": sha256_file(recipe_path),
            "source_snapshot.tar.gz": sha256_file(
                output_dir / "source_snapshot.tar.gz"
            ),
        },
        "source_snapshot": initial_context["source_snapshot"],
        "source_file_sha256": initial_context["source_file_sha256"],
        "recipes": {
            "neural_pa": dict(NEURAL_RECIPE),
            "neural_dpd": dict(NEURAL_RECIPE),
            "neural_dpd_pa_surrogate_sensitivity": {
                **NEURAL_RECIPE,
                "dpd_model": dict(TRES_DELTAGRU_DPD_MODEL),
                "pa_surrogates": [
                    REFERENCE_PA_KEY,
                    ALTERNATE_PA_KEY,
                ],
            },
            "polynomial_pa": {
                "method": "direct_least_squares",
                "column_scaling": "l2",
                "solvers": {
                    "mp": {
                        "implementation": "torch.linalg.lstsq",
                        "mode": "gels",
                        "driver": "gels",
                        "regularization": None,
                    },
                    "gmp": {
                        "implementation": "torch.linalg.svd",
                        "mode": "truncated_svd",
                        "driver": "gesvdj",
                        "regularization": {
                            "type": "truncated_svd",
                            "relative_cutoff": GMP_PA_SVD_RCOND,
                        },
                    },
                },
                "parameter_convention": (
                    "two real degrees of freedom per complex coefficient"
                ),
                "models": POLYNOMIAL_CONFIGURATIONS["pa_modeling"],
            },
            "polynomial_dpd": {
                "method": "indirect_learning_architecture",
                "solver": "torch.linalg.lstsq",
                "solver_driver": "gels",
                "column_scaling": "l2",
                "parameter_convention": (
                    "two real degrees of freedom per complex coefficient"
                ),
                "models": POLYNOMIAL_CONFIGURATIONS["dpd_ila"],
            },
        },
        "metric_note": (
            "EVM is the repository-specific normalized mean absolute complex-"
            "spectrum error across configured main-channel subchannels, not "
            "demodulated constellation EVM. DPD metrics are simulated through "
            "a frozen learned PA surrogate."
        ),
        "datasets": {},
    }

    for dataset, configuration in DATASETS.items():
        pa_models: dict[str, Any] = {}
        for model_name in ("mp", "gmp"):
            result = collect_polynomial_model(
                output_dir,
                dataset=dataset,
                slug=configuration["slug"],
                task="pa_modeling",
                model_name=model_name,
                expected_device=f"cuda:{device}",
                expected_target_gain=initial_context["datasets"][dataset][
                    "signal"
                ]["target_gain"],
            )
            result["model"] = {
                "display_name": model_name.upper(),
                "backbone": model_name,
                "parameters": result["real_parameters"],
                "parameter_convention": (
                    "two real degrees of freedom per complex coefficient"
                ),
            }
            pa_models[model_name] = result

        for key, model_configuration in NEURAL_PA_MODELS.items():
            result = collect_training_model(
                output_dir,
                dataset=dataset,
                step="train_pa",
                parent=None,
                model_id=model_configuration["model_id"],
                expected_epochs=NEURAL_RECIPE["epochs"],
                expected_batch_size=NEURAL_RECIPE["batch_size"],
                expected_initial_lr=NEURAL_RECIPE["initial_learning_rate"],
                expected_backbone=model_configuration["backbone"],
                expected_hidden_size=model_configuration["hidden_size"],
                expected_parameters=model_configuration["parameters"],
                selection_metric="VAL_NMSE",
                expected_delta_thresholds=model_configuration.get(
                    "delta_thresholds"
                ),
            )
            result["task"] = "pa_modeling"
            result["method"] = "supervised_training"
            result["model"] = dict(model_configuration)
            pa_models[key] = result

        reference_pa = pa_models[REFERENCE_PA_KEY]
        pa_hash = reference_pa["artifacts"]["checkpoint"]["sha256"]
        expected_pa_checkpoint = (
            Path("save")
            / dataset
            / "train_pa"
            / f"{NEURAL_PA_MODELS[REFERENCE_PA_KEY]['model_id']}.pt"
        ).as_posix()

        dpd_models: dict[str, Any] = {}
        for model_name in ("mp", "gmp"):
            result = collect_polynomial_model(
                output_dir,
                dataset=dataset,
                slug=configuration["slug"],
                task="dpd_ila",
                model_name=model_name,
                expected_pa_hash=pa_hash,
                expected_device=f"cuda:{device}",
                expected_target_gain=initial_context["datasets"][dataset][
                    "signal"
                ]["target_gain"],
            )
            result["model"] = {
                "display_name": model_name.upper(),
                "backbone": model_name,
                "parameters": result["real_parameters"],
                "parameter_convention": (
                    "two real degrees of freedom per complex coefficient"
                ),
            }
            dpd_models[model_name] = result

        for key, model_configuration in NEURAL_DPD_MODELS.items():
            result = collect_training_model(
                output_dir,
                dataset=dataset,
                step="train_dpd",
                parent=REFERENCE_PA_PARENT,
                model_id=model_configuration["model_id"],
                expected_epochs=NEURAL_RECIPE["epochs"],
                expected_batch_size=NEURAL_RECIPE["batch_size"],
                expected_initial_lr=NEURAL_RECIPE["initial_learning_rate"],
                expected_backbone=model_configuration["backbone"],
                expected_hidden_size=model_configuration["hidden_size"],
                expected_parameters=model_configuration["parameters"],
                selection_metric="VAL_ACLR_AVG",
                expected_delta_thresholds=model_configuration.get(
                    "delta_thresholds"
                ),
                expected_pa_parameters=reference_pa["model"]["parameters"],
            )
            result["task"] = "dpd_training"
            result["method"] = "direct_learning_architecture"
            result["model"] = dict(model_configuration)
            binding = pa_bindings.get((dataset, key, REFERENCE_PA_KEY))
            if binding is None:
                raise ValueError(
                    f"Missing PA checkpoint binding for "
                    f"{dataset}/{key}/{REFERENCE_PA_KEY}"
                )
            require_equal(
                binding["checkpoint"],
                expected_pa_checkpoint,
                f"PA checkpoint path binding for {dataset}/{key}",
            )
            require_equal(
                binding["sha256"],
                pa_hash,
                f"PA checkpoint hash binding for {dataset}/{key}",
            )
            result["pa_checkpoint_binding"] = binding
            dpd_models[key] = result

        delta_dpd_by_pa_surrogate: dict[str, Any] = {}
        for pa_model_key, pa_parent in (
            (REFERENCE_PA_KEY, REFERENCE_PA_PARENT),
            (ALTERNATE_PA_KEY, ALTERNATE_PA_PARENT),
        ):
            pa_result = pa_models[pa_model_key]
            pa_checkpoint = (
                Path("save")
                / dataset
                / "train_pa"
                / f"{NEURAL_PA_MODELS[pa_model_key]['model_id']}.pt"
            ).as_posix()
            result = collect_training_model(
                output_dir,
                dataset=dataset,
                step="train_dpd",
                parent=pa_parent,
                model_id=TRES_DELTAGRU_DPD_MODEL["model_id"],
                expected_epochs=NEURAL_RECIPE["epochs"],
                expected_batch_size=NEURAL_RECIPE["batch_size"],
                expected_initial_lr=NEURAL_RECIPE["initial_learning_rate"],
                expected_backbone=TRES_DELTAGRU_DPD_MODEL["backbone"],
                expected_hidden_size=TRES_DELTAGRU_DPD_MODEL["hidden_size"],
                expected_parameters=TRES_DELTAGRU_DPD_MODEL["parameters"],
                selection_metric="VAL_ACLR_AVG",
                expected_delta_thresholds=TRES_DELTAGRU_DPD_MODEL[
                    "delta_thresholds"
                ],
                expected_pa_parameters=pa_result["model"]["parameters"],
            )
            result["task"] = "dpd_training"
            result["method"] = "direct_learning_architecture"
            result["model"] = dict(TRES_DELTAGRU_DPD_MODEL)
            binding = pa_bindings.get(
                (dataset, "tres_deltagru", pa_model_key)
            )
            if binding is None:
                raise ValueError(
                    f"Missing PA checkpoint binding for "
                    f"{dataset}/tres_deltagru/{pa_model_key}"
                )
            require_equal(
                binding["checkpoint"],
                pa_checkpoint,
                (
                    "PA checkpoint path binding for "
                    f"{dataset}/tres_deltagru/{pa_model_key}"
                ),
            )
            require_equal(
                binding["sha256"],
                pa_result["artifacts"]["checkpoint"]["sha256"],
                (
                    "PA checkpoint hash binding for "
                    f"{dataset}/tres_deltagru/{pa_model_key}"
                ),
            )
            result["pa_checkpoint_binding"] = binding
            result["pa_surrogate"] = {
                "model_key": pa_model_key,
                "model": dict(NEURAL_PA_MODELS[pa_model_key]),
                "checkpoint": pa_checkpoint,
                "checkpoint_sha256": binding["sha256"],
                "selection_metric": "validation NMSE",
            }
            delta_dpd_by_pa_surrogate[pa_model_key] = result

        manifest["datasets"][dataset] = {
            "spec_sha256": initial_context["datasets"][dataset]["spec_sha256"],
            "signal": initial_context["datasets"][dataset]["signal"],
            "reference_pa": {
                "model_key": REFERENCE_PA_KEY,
                "checkpoint": expected_pa_checkpoint,
                "checkpoint_sha256": pa_hash,
                "selection_metric": "validation NMSE",
            },
            "pa_models": pa_models,
            "dpd_models": dpd_models,
            "dpd_tres_deltagru_by_pa_surrogate": (
                delta_dpd_by_pa_surrogate
            ),
        }

    return manifest


def publish_manifest_from_run(
    manifest: dict[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    """Bind post-collection source/job evidence into a canonical manifest."""
    source_schema_version = manifest["schema_version"]
    source_result_count = sum(
        len(dataset["pa_models"])
        + len(dataset["dpd_models"])
        + len(dataset["dpd_tres_deltagru_by_pa_surrogate"])
        for dataset in manifest["datasets"].values()
    )
    if source_schema_version != RUN_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported source manifest schema for publication: "
            f"{source_schema_version}"
        )
    context_path = run_directory / "context_before.json"
    original_manifest_path = run_directory / "benchmark_report_results.json"
    jobs_path = run_directory / "jobs.tsv"
    recipe_path = run_directory / "recipe.json"
    commands_path = run_directory / "commands.log"
    with context_path.open() as handle:
        context = json.load(handle)

    snapshot_path = run_directory / "source_snapshot.tar.gz"
    snapshot_metadata = validate_source_snapshot(
        snapshot_path,
        context["source_file_sha256"],
    )
    snapshot_time = datetime.fromtimestamp(
        snapshot_path.stat().st_mtime,
        timezone.utc,
    )
    started = datetime.fromisoformat(manifest["started_at_utc"])
    collected = datetime.fromisoformat(manifest["generated_at_utc"])
    if not started <= snapshot_time <= collected:
        raise ValueError(
            "Source snapshot was not created during the recorded benchmark run: "
            f"{snapshot_time.isoformat()} is outside "
            f"{started.isoformat()} .. {collected.isoformat()}"
        )

    with jobs_path.open(newline="") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    command_labels = set(read_recorded_commands(commands_path))
    job_labels = [row["label"] for row in jobs]
    require_equal(len(job_labels), len(set(job_labels)), "unique job labels")
    require_equal(set(job_labels), command_labels, "completed benchmark jobs")
    for row in jobs:
        require_equal(
            int(row["exit_status"]),
            0,
            f"exit status for job {row['label']}",
        )

    published = copy.deepcopy(manifest)
    published["schema_version"] = RUN_SCHEMA_VERSION
    for dataset in published["datasets"].values():
        dataset["pa_models"] = {
            key: dataset["pa_models"][key] for key in PA_MODEL_ORDER
        }
        dataset["dpd_models"] = {
            key: dataset["dpd_models"][key] for key in DPD_MODEL_ORDER
        }
        dataset["dpd_tres_deltagru_by_pa_surrogate"] = {
            key: dataset["dpd_tres_deltagru_by_pa_surrogate"][key]
            for key in (REFERENCE_PA_KEY, ALTERNATE_PA_KEY)
        }
    published["source_snapshot"] = snapshot_metadata
    published["run_file_sha256"].update(
        {
            "jobs.tsv": sha256_file(jobs_path),
            "recipe.json": sha256_file(recipe_path),
            "source_snapshot.tar.gz": sha256_file(snapshot_path),
        }
    )
    for stage in ("neural_pa", "neural_dpd"):
        published["recipes"][stage].update(copy.deepcopy(NEURAL_RECIPE))
    published["metric_note"] = (
        "EVM is the repository-specific normalized mean absolute complex-"
        "spectrum error across configured main-channel subchannels, not "
        "demodulated constellation EVM. DPD metrics are simulated through a "
        "frozen learned PA surrogate."
    )
    published["publication"] = {
        "source_run_id": run_directory.name,
        "original_schema_version": source_schema_version,
        "original_collector_manifest_sha256": sha256_file(
            original_manifest_path
        ),
        "source_snapshot_created_during_run_utc": snapshot_time.isoformat(),
        "source_snapshot_member_hashes_match_pre_run_context": True,
        "source_job_count": len(jobs),
        "source_result_count": source_result_count,
        "published_result_count": (
            len(published["datasets"])
            * (len(PA_MODEL_ORDER) + len(DPD_MODEL_ORDER) + 2)
        ),
        "published_models": list(PA_MODEL_ORDER),
        "published_models_by_stage": {
            "pa_modeling": list(PA_MODEL_ORDER),
            "dpd": list(DPD_MODEL_ORDER),
            "dpd_pa_surrogate_sensitivity": ["tres_deltagru"],
        },
        "published_dpd_pa_surrogates": [
            REFERENCE_PA_KEY,
            ALTERNATE_PA_KEY,
        ],
        "all_jobs_exit_zero": True,
        "total_job_duration_seconds": sum(
            int(row["duration_seconds"]) for row in jobs
        ),
        "note": (
            "The canonical manifest contains the published model subset. "
            "Source-run hashes, job totals, and archives cover the complete "
            "source run and can include experiments outside that subset; "
            "source-archive members were verified against the pre-run source "
            "hash map."
        ),
    }
    return published


def pair(metrics: dict[str, dict[str, float]], key: str) -> str:
    return (
        f"{metrics['validation'][key]:.4f} / "
        f"{metrics['test'][key]:.4f}"
    )


def aclr_triplet(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['aclr_left_db']:.4f} / "
        f"{metrics['aclr_right_db']:.4f} / "
        f"{metrics['aclr_avg_db']:.4f}"
    )


def result_method(result: dict[str, Any]) -> str:
    if result["method"] == "supervised_training":
        return "Supervised, AdamW"
    if result["method"] == "direct_learning_architecture":
        return "DLA, AdamW"
    if result["method"] == "direct_least_squares":
        if result.get("regularization") == "truncated_svd":
            return (
                "Truncated SVD "
                f"(rank {result['least_squares_rank']:,}/"
                f"{result['complex_coefficients']:,})"
            )
        return "Direct least squares"
    if result["method"] == "indirect_learning_architecture":
        return "ILA, least squares"
    raise ValueError(f"Unknown benchmark method: {result['method']}")


def selected_epoch(result: dict[str, Any]) -> str:
    epoch = result.get("selected_epoch_zero_based")
    return "N/A" if epoch is None else str(epoch)


def result_table(
    results: dict[str, dict[str, Any]],
    model_order: tuple[str, ...],
) -> list[str]:
    lines = [
        "| Model | Parameters | Method | Selected epoch | "
        "NMSE, val / test (dB) | EVM, val / test (dB) | "
        "Validation ACLR L / R / avg (dB) | "
        "Test ACLR L / R / avg (dB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key in model_order:
        result = results[key]
        lines.append(
            f"| {result['model']['display_name']} | "
            f"{result['model']['parameters']:,} | "
            f"{result_method(result)} | {selected_epoch(result)} | "
            f"{pair(result['metrics'], 'nmse_db')} | "
            f"{pair(result['metrics'], 'evm_db')} | "
            f"{aclr_triplet(result['metrics']['validation'])} | "
            f"{aclr_triplet(result['metrics']['test'])} |"
        )
    return lines


def delta_dpd_surrogate_table(values: dict[str, Any]) -> list[str]:
    lines = [
        "| PA surrogate | PA NMSE, val / test (dB) | DPD parameters | "
        "Selected epoch | DPD NMSE, val / test (dB) | "
        "DPD EVM, val / test (dB) | DPD ACLR avg, val / test (dB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    results = values["dpd_tres_deltagru_by_pa_surrogate"]
    for pa_model_key in (REFERENCE_PA_KEY, ALTERNATE_PA_KEY):
        result = results[pa_model_key]
        pa_result = values["pa_models"][pa_model_key]
        lines.append(
            f"| {pa_result['model']['display_name']} | "
            f"{pair(pa_result['metrics'], 'nmse_db')} | "
            f"{result['model']['parameters']:,} | "
            f"{selected_epoch(result)} | "
            f"{pair(result['metrics'], 'nmse_db')} | "
            f"{pair(result['metrics'], 'evm_db')} | "
            f"{pair(result['metrics'], 'aclr_avg_db')} |"
        )
    return lines


def write_results_figure(manifest: dict[str, Any], path: Path) -> None:
    """Render a compact test-set comparison directly from the manifest."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {
        "mp": "#4477AA",
        "gmp": "#66CCEE",
        "gru": "#228833",
        "tres_gru": "#EE6677",
        "tres_deltagru": "#AA3377",
    }
    panels = (
        (
            "APA_200MHz",
            "pa_models",
            "nmse_db",
            "PA test NMSE (dB)",
            PA_MODEL_ORDER,
        ),
        (
            "DPA_160MHz",
            "pa_models",
            "nmse_db",
            "PA test NMSE (dB)",
            PA_MODEL_ORDER,
        ),
        (
            "APA_200MHz",
            "dpd_models",
            "aclr_avg_db",
            "DPD test ACLR average (dB)",
            DPD_MODEL_ORDER,
        ),
        (
            "DPA_160MHz",
            "dpd_models",
            "aclr_avg_db",
            "DPD test ACLR average (dB)",
            DPD_MODEL_ORDER,
        ),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9.0),
        constrained_layout=True,
    )
    figure.suptitle("OpenDPD benchmark — test split", fontsize=17, weight="bold")
    for axis, (dataset, result_key, metric, title, model_order) in zip(
        axes.flat,
        panels,
    ):
        results = manifest["datasets"][dataset][result_key]
        values = [
            results[key]["metrics"]["test"][metric] for key in model_order
        ]
        validation_winner = min(
            model_order,
            key=lambda key: results[key]["metrics"]["validation"][metric],
        )
        span = max(values) - min(values)
        label_offset = max(0.22, span * 0.025)
        left_padding = max(0.4, span * 0.035)
        right_padding = max(1.0, span * 0.16)
        for y_position, (key, value) in enumerate(zip(model_order, values)):
            axis.scatter(
                value,
                y_position,
                s=115,
                color=colors[key],
                edgecolor="black" if key == validation_winner else "white",
                linewidth=1.8 if key == validation_winner else 0.8,
                zorder=3,
            )
            axis.text(
                value + label_offset,
                y_position,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        axis.set_xlim(
            min(values) - left_padding,
            max(values) + right_padding,
        )
        if min(values) < 0.0 < max(values):
            axis.axvline(0.0, color="#777777", linestyle="--", linewidth=1)
        axis.set_yticks(
            range(len(model_order)),
            [results[key]["model"]["display_name"] for key in model_order],
        )
        axis.invert_yaxis()
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(f"{dataset}\n{title}", fontsize=12)
        axis.set_xlabel("More negative is better  ←")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    figure.text(
        0.5,
        -0.01,
        "Outlined markers are validation leaders. DPD is simulated through "
        "the dataset-specific validation-selected TRes-GRU-H27 PA surrogate.",
        ha="center",
        fontsize=10,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".png",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        figure.savefig(
            temporary_path,
            dpi=160,
            facecolor="white",
            bbox_inches="tight",
        )
        os.replace(temporary_path, path)
    finally:
        plt.close(figure)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_delta_dpd_figure(manifest: dict[str, Any], path: Path) -> None:
    """Render the TRes-DeltaGRU DPD PA-surrogate sensitivity comparison."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pa_order = (REFERENCE_PA_KEY, ALTERNATE_PA_KEY)
    colors = {
        REFERENCE_PA_KEY: "#EE6677",
        ALTERNATE_PA_KEY: "#AA3377",
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.8),
        constrained_layout=True,
    )
    figure.suptitle(
        "TRes-DeltaGRU-H15 DPD — PA surrogate sensitivity (test split)",
        fontsize=16,
        weight="bold",
    )
    for axis, dataset in zip(axes, DATASETS):
        dataset_values = manifest["datasets"][dataset]
        results = dataset_values["dpd_tres_deltagru_by_pa_surrogate"]
        values = [
            results[key]["metrics"]["test"]["aclr_avg_db"] for key in pa_order
        ]
        validation_winner = min(
            pa_order,
            key=lambda key: results[key]["metrics"]["validation"]["aclr_avg_db"],
        )
        span = max(values) - min(values)
        label_offset = max(0.12, span * 0.08)
        for y_position, (key, value) in enumerate(zip(pa_order, values)):
            axis.scatter(
                value,
                y_position,
                s=125,
                color=colors[key],
                edgecolor="black" if key == validation_winner else "white",
                linewidth=1.8 if key == validation_winner else 0.8,
                zorder=3,
            )
            axis.text(
                value + label_offset,
                y_position,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=10,
            )
        left_padding = max(0.35, span * 0.25)
        right_padding = max(0.8, span * 0.55)
        axis.set_xlim(min(values) - left_padding, max(values) + right_padding)
        axis.set_yticks(
            range(len(pa_order)),
            [
                f"{dataset_values['pa_models'][key]['model']['display_name']} PA"
                for key in pa_order
            ],
        )
        axis.invert_yaxis()
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(
            f"{dataset}\nDPD test ACLR average (dB)",
            fontsize=12,
        )
        axis.set_xlabel("More negative is better  ←")
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    figure.text(
        0.5,
        -0.02,
        "Each point is an independently trained TRes-DeltaGRU-H15 DPD. "
        "Outlined markers are selected by validation ACLR average.",
        ha="center",
        fontsize=10,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".png",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        figure.savefig(
            temporary_path,
            dpi=160,
            facecolor="white",
            bbox_inches="tight",
        )
        os.replace(temporary_path, path)
    finally:
        plt.close(figure)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def bind_derived_artifacts(
    manifest: dict[str, Any],
    *,
    report_path: Path,
    figure_path: Path,
    delta_dpd_figure_path: Path,
) -> None:
    manifest["derived_artifacts"] = {
        report_path.name: {
            "sha256": sha256_file(report_path),
            "size_bytes": report_path.stat().st_size,
        },
        figure_path.name: {
            "sha256": sha256_file(figure_path),
            "size_bytes": figure_path.stat().st_size,
        },
        delta_dpd_figure_path.name: {
            "sha256": sha256_file(delta_dpd_figure_path),
            "size_bytes": delta_dpd_figure_path.stat().st_size,
        },
    }


def render_markdown(
    manifest: dict[str, Any],
    *,
    canonical: bool = False,
) -> str:
    lines = [
        "# OpenDPD PA Modeling and DPD Benchmark",
        "",
        "## Technical summary",
        "",
        "This benchmark evaluates MP, GMP, GRU, TRes-GRU, and "
        "TRes-DeltaGRU (THX=THH=0) for PA modeling on APA_200MHz and "
        "DPA_160MHz. The DPD comparison evaluates MP, GMP, GRU, and TRes-GRU. "
        "PA models use approximately 2,700 real parameters; DPD models use "
        "approximately 1,000. Every neural run uses the same 300-epoch "
        "optimization recipe. MP PA uses direct least squares and GMP PA uses "
        "rank-controlled truncated SVD; their predistorters use the indirect "
        "learning architecture (ILA). The four-model DPD comparison uses the "
        "dataset-specific "
        "TRes-GRU-H27 PA checkpoint selected by validation NMSE. A separate "
        "sensitivity experiment trains TRes-DeltaGRU-H15 DPD independently "
        "through both the TRes-GRU-H27 and zero-threshold "
        "TRes-DeltaGRU-H27 PA surrogates.",
        "",
        "## Key findings",
        "",
    ]

    for dataset, values in manifest["datasets"].items():
        best_pa = min(
            values["pa_models"].values(),
            key=lambda result: result["metrics"]["validation"]["nmse_db"],
        )
        best_dpd = min(
            values["dpd_models"].values(),
            key=lambda result: result["metrics"]["validation"]["aclr_avg_db"],
        )
        delta_dpd_results = values["dpd_tres_deltagru_by_pa_surrogate"]
        best_delta_pa_key = min(
            delta_dpd_results,
            key=lambda key: delta_dpd_results[key]["metrics"]["validation"][
                "aclr_avg_db"
            ],
        )
        best_delta_dpd = delta_dpd_results[best_delta_pa_key]
        best_delta_pa = values["pa_models"][best_delta_pa_key]
        lines.append(
            f"- **{dataset}:** the validation PA leader is "
            f"{best_pa['model']['display_name']} at "
            f"{best_pa['metrics']['validation']['nmse_db']:.4f} dB NMSE "
            f"({best_pa['metrics']['test']['nmse_db']:.4f} dB test). The "
            f"validation DPD leader is {best_dpd['model']['display_name']} at "
            f"{best_dpd['metrics']['validation']['aclr_avg_db']:.4f} dB ACLR "
            f"average ({best_dpd['metrics']['test']['aclr_avg_db']:.4f} dB test). "
            f"For TRes-DeltaGRU-H15 DPD, the better validation result uses the "
            f"{best_delta_pa['model']['display_name']} PA surrogate at "
            f"{best_delta_dpd['metrics']['validation']['aclr_avg_db']:.4f} dB "
            f"ACLR average "
            f"({best_delta_dpd['metrics']['test']['aclr_avg_db']:.4f} dB test)."
        )

    apa_gmp = manifest["datasets"]["APA_200MHz"]["pa_models"]["gmp"]
    lines.extend(
        [
            (
                "- **APA GMP stability:** column-normalized truncated SVD at "
                f"`rcond={apa_gmp['svd_rcond']:.0e}` retains "
                f"{apa_gmp['least_squares_rank']:,}/"
                f"{apa_gmp['complex_coefficients']:,} singular directions. "
                "Validation/test NMSE is "
                f"{apa_gmp['metrics']['validation']['nmse_db']:.2f}/"
                f"{apa_gmp['metrics']['test']['nmse_db']:.2f} dB. The fixed "
                "cutoff suppresses ill-conditioned delayed-envelope directions "
                "instead of applying CUDA `gels`'s invalid full-rank "
                "assumption."
            ),
            "",
            "![Test-set PA modeling and DPD results](benchmark_results.png)",
            "",
            "*Test split; more negative is better. Outlined points are the "
            "models selected by validation. DPD is simulated through the "
            "dataset-specific TRes-GRU-H27 PA surrogate.*",
            "",
            "![TRes-DeltaGRU DPD PA-surrogate sensitivity]"
            "(benchmark_delta_dpd_results.png)",
            "",
            "*Test split; each point is an independently trained "
            "TRes-DeltaGRU-H15 DPD using the named frozen PA surrogate. "
            "These results compare surrogate sensitivity, not measurements "
            "from one shared physical PA.*",
        ]
    )

    for dataset, values in manifest["datasets"].items():
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                "### PA modeling",
                "",
            ]
        )
        lines.extend(result_table(values["pa_models"], PA_MODEL_ORDER))
        lines.extend(
            [
                "",
                "### DPD",
                "",
            ]
        )
        lines.extend(result_table(values["dpd_models"], DPD_MODEL_ORDER))
        lines.extend(
            [
                "",
                "### TRes-DeltaGRU DPD by PA surrogate",
                "",
                "Both rows use TRes-DeltaGRU-H15 with 999 parameters and "
                "THX=THH=0. They are separately trained through the indicated "
                "frozen PA surrogate.",
                "",
            ]
        )
        lines.extend(delta_dpd_surrogate_table(values))

    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- **NMSE:** normalized mean-square error in dB; more negative is "
            "better. The implementation averages per-segment dB ratios rather "
            "than pooling all samples into one ratio.",
            "- **EVM:** the repository-specific mean absolute complex-spectrum "
            "error within the configured main channel, normalized within each "
            "subchannel by reference-spectrum magnitude and converted with "
            "`20 log10`. It is not demodulated constellation EVM; more negative "
            "is better.",
            "- **ACLR L / R / avg:** adjacent-subchannel power normalized by the "
            "strongest configured main-channel subchannel, plus the arithmetic "
            "mean of left and right, in dB. More negative is better.",
            "- **Parameters:** neural checkpoint tensor elements. MP/GMP count "
            "two real degrees of freedom for every complex coefficient.",
            "- **Selected epoch:** zero-based neural checkpoint epoch. It is N/A "
            "for closed-form least-squares fits.",
            "",
            "## Model configurations",
            "",
            "| Stage | MP | GMP | GRU | TRes-GRU | TRes-DeltaGRU |",
            "|---|---|---|---|---|---|",
            "| PA modeling | K=9, Q=150 | Ka/La=5/30; "
            "Kb/Lb/Mb=4/30/5; Kc/Lc/Mc=4/30/5 | H28 | H27 | "
            "H27, THX=THH=0 |",
            "| DPD | K=5, Q=100 | Ka/La=5/20; "
            "Kb/Lb/Mb=4/20/3; Kc/Lc/Mc=4/20/2 | H16 | H15 | "
            "H15, THX=THH=0 (PA-surrogate sensitivity) |",
            "",
            "## Temporal context and sequence boundaries",
            "",
            "In sequence interiors, PA GMP uses up to five future samples and "
            "DPD GMP uses up to two through their leading-envelope terms. "
            "TRes-GRU and TRes-DeltaGRU use one-sample right context in their "
            "recurrent features and 16-sample right context in their dilated "
            "residual convolution. MP and GRU use no explicit future samples.",
            "",
            "These are offline segmented evaluations. GMP delay accesses are "
            "zero-filled and reset at each `nperseg` boundary. In both TRes "
            "models, `torch.roll(..., shifts=-1)` wraps the final position to "
            "the first sample of the same supplied sequence; the convolution "
            "zero-pads both boundaries, and recurrent state resets for each "
            "sequence. Neural optimization uses overlapping 200-sample frames "
            "with stride 1, while validation and test use independent "
            "`nperseg` segments.",
            "",
            "## Methodology",
            "",
            "Neural PA and DPD models use batch size 64, 300 epochs, AdamW with "
            "MSE loss, initial learning rate 5e-3, and ReduceLROnPlateau with "
            "factor 0.5, patience 5, and minimum learning rate 5e-5. Frame "
            "length is 200, frame stride is 1, and seed is 0. PA checkpoints "
            "are selected by minimum validation NMSE; neural DPD checkpoints "
            "are selected by minimum validation ACLR average.",
            "",
            "The TRes-DeltaGRU PA and DPD runs use input-delta threshold THX=0 "
            "and hidden-state-delta threshold THH=0. This disables "
            "threshold-induced temporal pruning; exact arithmetic deltas may "
            "still naturally be zero. The configuration therefore evaluates "
            "the dense zero-threshold recurrence, not a sparsity or efficiency "
            "claim.",
            "",
            "AdamW uses weight decay 0.01, betas (0.9, 0.999), and epsilon "
            "1e-8. The scheduler uses relative threshold 1e-4, cooldown 0, and "
            "epsilon 1e-8.",
            "",
            "MP and GMP are complex polynomial models fit after L2 column "
            "scaling. MP and both ILA-DPD fits use `torch.linalg.lstsq` "
            "(`gels`). GMP PA modeling uses `torch.linalg.svd` (`gesvdj`) "
            f"with a fixed relative cutoff of {GMP_PA_SVD_RCOND:.0e}; the "
            "effective retained rank is reported with each GMP PA result. "
            "These closed-form fits do not use the neural batch, epoch, "
            "optimizer, or learning-rate settings. PA polynomial fits map "
            "measured PA input to output directly. DPD polynomial fits use "
            "ILA, fitting a postdistorter and copying its coefficients to the "
            "predistorter. No ridge penalty or validation-tuned regularization "
            "is applied.",
            "",
            "Each dataset uses its independently trained TRes-GRU-H27 PA "
            "surrogate for the four-model DPD comparison. The two "
            "TRes-DeltaGRU-H15 sensitivity rows are separate DPD training runs, "
            "one through that TRes-GRU PA and one through the independently "
            "trained TRes-DeltaGRU-H27 PA. Training data supply all gradient "
            "and least-squares fits. Neural validation metrics drive the "
            "learning-rate schedule and checkpoint selection; test data are "
            "not used for fitting, scheduling, or selection.",
            "",
            "## Limitations and robustness",
            "",
            "- Neural execution uses soft reproducibility with cuDNN benchmark "
            "enabled, so repeated runs can differ slightly.",
            "- DPD scores measure simulated performance through a learned PA "
            "surrogate, not a fresh over-the-air or bench measurement.",
            "- One seed is evaluated. The reported table is not a distribution "
            "over training runs.",
            "- The five PA candidates and four primary DPD candidates are "
            "matched approximately by real parameter count, not by FLOPs, "
            "latency, memory traffic, or fit time. The PA-surrogate sensitivity "
            "experiment adds two independently trained TRes-DeltaGRU DPD runs "
            "per dataset.",
            "- Results obtained through different learned PA surrogates are "
            "simulator-sensitivity evidence; they are not a controlled ranking "
            "against one shared physical PA response.",
            "- CUDA `gels` assumes a full-rank design matrix and does not "
            "return numerical rank. It remains in use for MP and ILA-DPD; the "
            "GMP PA path records its SVD spectrum, cutoff, and retained rank.",
            "- GMP PA has 1,350 stored complex coefficients (2,700 nominal real "
            "parameters), but truncated SVD reduces its effective rank. The "
            "comparison is matched by stored coefficient count, not effective "
            "degrees of freedom.",
            "- Look-ahead is an input dependency, not measured inference "
            "latency. Streaming reformulations and continuous boundary/state "
            "handling are not evaluated.",
            "",
            "## Provenance",
            "",
            f"- Generated: `{manifest['generated_at_utc']}`",
            f"- Git commit: `{manifest['git']['commit']}`",
            f"- Git branch: `{manifest['git']['branch']}`",
            f"- Git working tree at launch: "
            f"`{'dirty' if manifest['git']['dirty'] else 'clean'}`; exact "
            "source hashes and start status are retained in the machine "
            "evidence.",
            f"- Python: `{manifest['environment']['python']}`",
            f"- PyTorch: `{manifest['environment']['torch']}`",
            f"- CUDA: `{manifest['environment']['cuda']}`",
            f"- GPU: `{manifest['environment']['gpu']}`",
        ]
    )
    if canonical:
        publication = manifest.get("publication")
        if publication is not None:
            lines.append(
                "- Canonical evidence schema: "
                f"`{manifest['schema_version']}` (published from collector "
                f"schema `{publication['original_schema_version']}`); the "
                "source archive and completed job ledger are cryptographically "
                "bound in the machine-readable evidence."
            )
        lines.extend(
            [
                "- Reproduce the full matrix: "
                "[`reproduce_benchmark_report.sh`](reproduce_benchmark_report.sh)",
                "- Machine-readable evidence: "
                "[`results/benchmark_report_results.json`]"
                "(results/benchmark_report_results.json)",
                "- Each reproduction run writes exact commands, a verified "
                "source snapshot, raw polynomial results, checkpoints, and CSV "
                "logs to its timestamped evidence directory.",
            ]
        )
    else:
        lines.extend(
            [
                "- Exact launch commands: [`commands.log`](commands.log)",
                "- Machine-readable evidence: "
                "[`benchmark_report_results.json`](benchmark_report_results.json)",
                "- Source snapshot: "
                "[`source_snapshot.tar.gz`](source_snapshot.tar.gz)",
                "- Archived checkpoints and CSV logs: [`artifacts/`](artifacts/)",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_atomic(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory containing raw least-squares JSON and archived artifacts.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index used by the benchmark run.",
    )
    parser.add_argument(
        "--snapshot-context",
        action="store_true",
        help="Capture pre-run source, dataset, Git, and environment evidence.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace an existing collector snapshot or final manifest.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.snapshot_context:
        context_path = output_dir / "context_before.json"
        snapshot_path = output_dir / "source_snapshot.tar.gz"
        if (
            (context_path.exists() or snapshot_path.exists())
            and not args.overwrite_output
        ):
            raise FileExistsError(
                f"Context or source snapshot already exists in: {output_dir}"
            )
        context = capture_context(
            output_dir,
            args.device,
            use_runner_start_git=True,
        )
        if context["environment"]["gpu"] is None:
            raise ValueError(
                "CUDA device evidence is missing from the pre-run context"
            )
        write_source_snapshot(
            list(context["source_file_sha256"]),
            snapshot_path,
        )
        context["source_snapshot"] = validate_source_snapshot(
            snapshot_path,
            context["source_file_sha256"],
        )
        write_atomic(
            context_path,
            json.dumps(
                context,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        print(f"Wrote pre-run context snapshot to {context_path}")
        return

    result_path = output_dir / "benchmark_report_results.json"
    markdown_path = output_dir / "benchmark_report.md"
    figure_path = output_dir / "benchmark_results.png"
    delta_dpd_figure_path = output_dir / "benchmark_delta_dpd_results.png"
    if (
        (
            result_path.exists()
            or markdown_path.exists()
            or figure_path.exists()
            or delta_dpd_figure_path.exists()
        )
        and not args.overwrite_output
    ):
        raise FileExistsError(
            "Collector output already exists; use a new run directory or "
            "--overwrite-output."
        )

    initial_context, completion_context = load_and_validate_context(
        output_dir, args.device
    )
    manifest = collect_manifest(
        output_dir,
        args.device,
        initial_context,
        completion_context,
    )
    write_results_figure(manifest, figure_path)
    write_delta_dpd_figure(manifest, delta_dpd_figure_path)
    write_atomic(
        markdown_path,
        render_markdown(manifest),
    )
    bind_derived_artifacts(
        manifest,
        report_path=markdown_path,
        figure_path=figure_path,
        delta_dpd_figure_path=delta_dpd_figure_path,
    )
    write_atomic(
        result_path,
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    print(f"Validated benchmark evidence in {output_dir}")
    print(f"Wrote {output_dir / 'benchmark_report_results.json'}")
    print(f"Wrote {output_dir / 'benchmark_report.md'}")
    print(f"Wrote {output_dir / 'benchmark_results.png'}")
    print(f"Wrote {output_dir / 'benchmark_delta_dpd_results.png'}")


if __name__ == "__main__":
    main()
