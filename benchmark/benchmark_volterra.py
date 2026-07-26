"""Polynomial PA-modeling and indirect-learning DPD benchmarks.

The benchmark supports two complex polynomial structures:

* MP: a memory polynomial.
* GMP: a generalized memory polynomial with aligned, lagging, and leading
  envelope terms.

For PA modeling, the complex coefficients are identified directly from
``Phi(x) w ~= y``.  For DPD, an indirect learning architecture (ILA) first
identifies the postdistorter from ``Phi(y / G) w ~= x``, copies those
coefficients to the predistorter, and evaluates the result through a trained
PA model.

Large benchmark systems are solved on CUDA after L2 column scaling.  The
well-conditioned MP and ILA systems use ``torch.linalg.lstsq`` with ``gels``.
GMP PA modeling can instead use a rank-controlled truncated SVD, because its
correlated cross-term basis does not satisfy CUDA ``gels``'s full-rank
assumption on every dataset.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

import models as model
from modules.data_collector import load_dataset
from utils import metrics
from utils.util import count_net_params, set_target_gain


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# NumPy basis implementations (small inputs, reference tests, and API users)
# ---------------------------------------------------------------------------

def build_mp_basis(x_complex, K, Q):
    """Build an MP basis with columns ``x(n-q) * |x(n-q)|**k``."""
    _validate_positive_configuration({"K": K, "Q": Q})
    x_complex = np.asarray(x_complex, dtype=np.complex128)
    n_samples = len(x_complex)
    phi = np.zeros((n_samples, K * Q), dtype=np.complex128)
    column = 0
    for k in range(K):
        for q in range(Q):
            delayed = _delay(x_complex, q)
            phi[:, column] = delayed * np.abs(delayed) ** k
            column += 1
    return phi


def _delay(x, delay):
    """Shift a one-dimensional complex signal and zero-fill the boundary."""
    x = np.asarray(x)
    n_samples = len(x)
    output = np.zeros(n_samples, dtype=x.dtype)
    if 0 <= delay < n_samples:
        output[delay:] = x[:n_samples - delay]
    elif -n_samples < delay < 0:
        output[:n_samples + delay] = x[-delay:]
    return output


def build_gmp_basis(x_complex, Ka, La, Kb, Lb, Mb, Kc, Lc, Mc):
    """Build the aligned, lagging, and leading GMP basis."""
    configuration = {
        "Ka": Ka,
        "La": La,
        "Kb": Kb,
        "Lb": Lb,
        "Mb": Mb,
        "Kc": Kc,
        "Lc": Lc,
        "Mc": Mc,
    }
    _validate_positive_configuration(configuration)
    x_complex = np.asarray(x_complex, dtype=np.complex128)
    n_samples = len(x_complex)
    n_columns = gmp_coefficient_count(configuration)
    phi = np.zeros((n_samples, n_columns), dtype=np.complex128)
    column = 0

    for k in range(Ka):
        for q in range(La):
            xq = _delay(x_complex, q)
            phi[:, column] = xq * np.abs(xq) ** k
            column += 1

    for k in range(1, Kb + 1):
        for q in range(Lb):
            for lag in range(1, Mb + 1):
                xq = _delay(x_complex, q)
                envelope = _delay(x_complex, q + lag)
                phi[:, column] = xq * np.abs(envelope) ** k
                column += 1

    for k in range(1, Kc + 1):
        for q in range(Lc):
            for lead in range(1, Mc + 1):
                xq = _delay(x_complex, q)
                envelope = _delay(x_complex, q - lead)
                phi[:, column] = xq * np.abs(envelope) ** k
                column += 1

    return phi


def build_segmented_numpy_basis(
    x_complex: np.ndarray,
    basis_builder: Callable[[np.ndarray], np.ndarray],
    segment_length: int,
) -> np.ndarray:
    """Build a reference basis while resetting delays at every segment."""
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    x_complex = np.asarray(x_complex)
    blocks = [
        basis_builder(x_complex[start:start + segment_length])
        for start in range(0, len(x_complex), segment_length)
    ]
    if not blocks:
        raise ValueError("cannot build a basis for an empty signal")
    return np.concatenate(blocks, axis=0)


# ---------------------------------------------------------------------------
# Memory-conscious Torch basis implementations used by the benchmark
# ---------------------------------------------------------------------------

def _torch_delay(x: torch.Tensor, delay: int) -> torch.Tensor:
    output = torch.zeros_like(x)
    n_samples = x.numel()
    if 0 <= delay < n_samples:
        output[delay:] = x[:n_samples - delay]
    elif -n_samples < delay < 0:
        output[:n_samples + delay] = x[-delay:]
    return output


def _torch_delay_columns(x: torch.Tensor, delays: list[int]) -> torch.Tensor:
    return torch.stack([_torch_delay(x, delay) for delay in delays], dim=1)


def _build_mp_basis_torch_block(
    x: torch.Tensor,
    configuration: dict[str, int],
) -> torch.Tensor:
    k_count = configuration["K"]
    memory_depth = configuration["Q"]
    delayed = _torch_delay_columns(x, list(range(memory_depth)))
    magnitudes = torch.abs(delayed)
    phi = torch.empty(
        (x.numel(), k_count * memory_depth),
        dtype=x.dtype,
        device=x.device,
    )
    for k in range(k_count):
        start = k * memory_depth
        phi[:, start:start + memory_depth] = delayed * magnitudes.pow(k)
    return phi


def _build_gmp_basis_torch_block(
    x: torch.Tensor,
    configuration: dict[str, int],
) -> torch.Tensor:
    Ka, La = configuration["Ka"], configuration["La"]
    Kb, Lb, Mb = configuration["Kb"], configuration["Lb"], configuration["Mb"]
    Kc, Lc, Mc = configuration["Kc"], configuration["Lc"], configuration["Mc"]
    phi = torch.empty(
        (x.numel(), gmp_coefficient_count(configuration)),
        dtype=x.dtype,
        device=x.device,
    )
    column = 0

    aligned = _torch_delay_columns(x, list(range(La)))
    aligned_magnitude = torch.abs(aligned)
    for k in range(Ka):
        phi[:, column:column + La] = aligned * aligned_magnitude.pow(k)
        column += La

    lag_base = _torch_delay_columns(x, list(range(Lb)))
    lag_delays = [q + lag for q in range(Lb) for lag in range(1, Mb + 1)]
    lag_envelopes = torch.abs(_torch_delay_columns(x, lag_delays)).reshape(
        x.numel(), Lb, Mb
    )
    for k in range(1, Kb + 1):
        block = lag_base.unsqueeze(-1) * lag_envelopes.pow(k)
        width = Lb * Mb
        phi[:, column:column + width] = block.reshape(x.numel(), width)
        column += width

    lead_base = _torch_delay_columns(x, list(range(Lc)))
    lead_delays = [q - lead for q in range(Lc) for lead in range(1, Mc + 1)]
    lead_envelopes = torch.abs(_torch_delay_columns(x, lead_delays)).reshape(
        x.numel(), Lc, Mc
    )
    for k in range(1, Kc + 1):
        block = lead_base.unsqueeze(-1) * lead_envelopes.pow(k)
        width = Lc * Mc
        phi[:, column:column + width] = block.reshape(x.numel(), width)
        column += width

    if column != phi.shape[1]:
        raise RuntimeError("internal GMP column-count mismatch")
    return phi


@torch.inference_mode()
def build_torch_basis(
    x: torch.Tensor,
    polynomial_model: str,
    configuration: dict[str, int],
    segment_length: int,
) -> torch.Tensor:
    """Build a design matrix on ``x.device`` with per-segment delay resets."""
    if x.ndim != 1 or not torch.is_complex(x):
        raise ValueError("x must be a one-dimensional complex Torch tensor")
    if x.numel() == 0:
        raise ValueError("cannot build a basis for an empty signal")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    _validate_positive_configuration(configuration)

    n_columns = coefficient_count(polynomial_model, configuration)
    phi = torch.empty(
        (x.numel(), n_columns),
        dtype=x.dtype,
        device=x.device,
    )
    build_block = (
        _build_mp_basis_torch_block
        if polynomial_model == "mp"
        else _build_gmp_basis_torch_block
    )
    for start in range(0, x.numel(), segment_length):
        stop = min(start + segment_length, x.numel())
        phi[start:stop] = build_block(x[start:stop], configuration)
    return phi


# ---------------------------------------------------------------------------
# Configuration and least-squares helpers
# ---------------------------------------------------------------------------

def _validate_positive_configuration(configuration: dict[str, int]) -> None:
    for name, value in configuration.items():
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def gmp_coefficient_count(configuration: dict[str, int]) -> int:
    return (
        configuration["Ka"] * configuration["La"]
        + configuration["Kb"] * configuration["Lb"] * configuration["Mb"]
        + configuration["Kc"] * configuration["Lc"] * configuration["Mc"]
    )


def coefficient_count(
    polynomial_model: str,
    configuration: dict[str, int],
) -> int:
    if polynomial_model == "mp":
        return configuration["K"] * configuration["Q"]
    if polynomial_model == "gmp":
        return gmp_coefficient_count(configuration)
    raise ValueError(f"unsupported polynomial model: {polynomial_model!r}")


def basis_configuration(args: argparse.Namespace) -> dict[str, int]:
    if args.model == "mp":
        return {"K": args.K, "Q": args.Q}
    return {
        "Ka": args.Ka,
        "La": args.La,
        "Kb": args.Kb,
        "Lb": args.Lb,
        "Mb": args.Mb,
        "Kc": args.Kc,
        "Lc": args.Lc,
        "Mc": args.Mc,
    }


def select_basis(args):
    """Return the NumPy reference builder, coefficient count, and banner."""
    configuration = basis_configuration(args)
    if args.model == "gmp":
        builder = partial(build_gmp_basis, **configuration)
        banner = (
            "Polynomial GMP "
            + ", ".join(f"{name}={value}" for name, value in configuration.items())
        )
    else:
        builder = partial(build_mp_basis, **configuration)
        banner = f"Polynomial MP K={args.K}, Q={args.Q}"
    return builder, coefficient_count(args.model, configuration), banner


def resolve_solver_device(requested: str, cuda_index: int) -> torch.device:
    requested = requested.strip().lower()
    if requested == "cuda":
        requested = f"cuda:{cuda_index}"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the least-squares solve but is unavailable")
        if device.index is None:
            device = torch.device(f"cuda:{cuda_index}")
    elif device.type != "cpu":
        raise ValueError("solver device must be 'cpu', 'cuda', or 'cuda:INDEX'")
    return device


def resolve_complex_dtype(requested: str) -> torch.dtype:
    mapping = {
        "complex64": torch.complex64,
        "torch.complex64": torch.complex64,
        "complex128": torch.complex128,
        "torch.complex128": torch.complex128,
    }
    try:
        return mapping[requested.lower()]
    except KeyError as error:
        raise ValueError("solver dtype must be complex64 or complex128") from error


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def fit_complex_least_squares(
    basis_input: np.ndarray,
    target: np.ndarray,
    *,
    polynomial_model: str,
    configuration: dict[str, int],
    segment_length: int,
    device: torch.device,
    dtype: torch.dtype,
    solver_mode: str = "gels",
    svd_rcond: float | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | str | None]]:
    """Fit complex coefficients after L2 column scaling."""
    basis_input = np.asarray(basis_input)
    target = np.asarray(target)
    if basis_input.ndim != 1 or target.ndim != 1:
        raise ValueError("least-squares input and target must be one-dimensional")
    if basis_input.shape != target.shape:
        raise ValueError("least-squares input and target must have the same shape")
    if not np.all(np.isfinite(basis_input)) or not np.all(np.isfinite(target)):
        raise ValueError("least-squares input and target must contain only finite values")

    n_columns = coefficient_count(polynomial_model, configuration)
    if basis_input.size < n_columns:
        raise ValueError(
            f"least-squares system is underdetermined: "
            f"{basis_input.size} observations for {n_columns} coefficients"
        )
    if solver_mode not in {"gels", "truncated_svd"}:
        raise ValueError(f"unsupported solver mode: {solver_mode!r}")
    if solver_mode == "gels" and svd_rcond is not None:
        raise ValueError("svd_rcond is only valid for truncated_svd")
    if solver_mode == "truncated_svd":
        if svd_rcond is None or not math.isfinite(svd_rcond):
            raise ValueError(
                "truncated_svd requires a finite svd_rcond"
            )
        if not 0.0 < svd_rcond < 1.0:
            raise ValueError("svd_rcond must be between zero and one")

    _synchronize(device)
    started = time.perf_counter()
    x_tensor = torch.as_tensor(basis_input, dtype=dtype, device=device)
    target_tensor = torch.as_tensor(target, dtype=dtype, device=device)
    phi = build_torch_basis(
        x_tensor,
        polynomial_model,
        configuration,
        segment_length,
    )
    _synchronize(device)
    basis_seconds = time.perf_counter() - started

    column_norms = torch.linalg.vector_norm(phi, dim=0)
    if not bool(torch.all(torch.isfinite(column_norms)).item()):
        raise RuntimeError("basis column norms are not finite")
    if bool(torch.any(column_norms == 0).item()):
        zero_count = int(torch.count_nonzero(column_norms == 0).item())
        raise RuntimeError(f"basis contains {zero_count} all-zero columns")
    smallest_norm = float(column_norms.min().item())
    largest_norm = float(column_norms.max().item())
    phi.div_(column_norms)

    _synchronize(device)
    solve_started = time.perf_counter()
    least_squares_rank: int | None = None
    singular_value_max: float | None = None
    singular_value_min: float | None = None
    singular_value_cutoff: float | None = None
    retained_singular_value_min: float | None = None
    condition_number: float | None = None
    if solver_mode == "gels":
        solver_implementation = "torch.linalg.lstsq"
        solver_driver = "gels"
        regularization = None
        try:
            solution_scaled = torch.linalg.lstsq(
                phi,
                target_tensor.unsqueeze(1),
                driver=solver_driver,
            ).solution.squeeze(1)
        except RuntimeError as error:
            raise RuntimeError(
                "torch.linalg.lstsq failed for the polynomial benchmark. "
                "The CUDA gels driver assumes a full-rank design matrix; "
                "verify the basis configuration and available accelerator "
                "memory."
            ) from error
    else:
        solver_implementation = "torch.linalg.svd"
        solver_driver = "gesvdj" if device.type == "cuda" else "default"
        regularization = "truncated_svd"
        svd_driver = "gesvdj" if device.type == "cuda" else None
        try:
            left, singular_values, right_h = torch.linalg.svd(
                phi,
                full_matrices=False,
                driver=svd_driver,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "torch.linalg.svd failed for the polynomial benchmark; "
                "verify the basis configuration and available accelerator "
                "memory."
            ) from error
        singular_value_max = float(singular_values[0].item())
        singular_value_min = float(singular_values[-1].item())
        singular_value_cutoff = singular_value_max * float(svd_rcond)
        retained = singular_values > singular_value_cutoff
        least_squares_rank = int(torch.count_nonzero(retained).item())
        if least_squares_rank == 0:
            raise RuntimeError(
                "truncated SVD rejected every singular direction"
            )
        retained_singular_value_min = float(
            singular_values[retained][-1].item()
        )
        condition_number = (
            singular_value_max / singular_value_min
            if singular_value_min > 0.0
            else None
        )
        projected_target = left[:, retained].mH @ target_tensor
        solution_scaled = right_h[retained].mH @ (
            projected_target / singular_values[retained]
        )
    _synchronize(device)
    solve_seconds = time.perf_counter() - solve_started

    fitted = phi @ solution_scaled
    residual_norm = torch.linalg.vector_norm(fitted - target_tensor)
    target_norm = torch.linalg.vector_norm(target_tensor)
    relative_residual = residual_norm / target_norm
    coefficients = solution_scaled / column_norms

    if not bool(torch.all(torch.isfinite(coefficients)).item()):
        raise RuntimeError("least-squares coefficients are not finite")
    relative_residual_value = float(relative_residual.item())
    if not math.isfinite(relative_residual_value):
        raise RuntimeError("least-squares training relative residual is not finite")
    coefficient_l2_norm = float(torch.linalg.vector_norm(coefficients).item())
    if not math.isfinite(coefficient_l2_norm):
        raise RuntimeError("least-squares coefficient norm is not finite")

    diagnostics: dict[str, float | int | str | None] = {
        "observations": int(basis_input.size),
        "columns": int(n_columns),
        # The gels driver does not estimate numerical rank. Its documented
        # contract assumes that the design matrix is full rank.
        "least_squares_rank": least_squares_rank,
        "solver_mode": solver_mode,
        "solver_implementation": solver_implementation,
        "solver_driver": solver_driver,
        "regularization": regularization,
        "svd_rcond": svd_rcond,
        "singular_value_max": singular_value_max,
        "singular_value_min": singular_value_min,
        "singular_value_cutoff": singular_value_cutoff,
        "retained_singular_value_min": retained_singular_value_min,
        "condition_number": condition_number,
        "coefficient_l2_norm": coefficient_l2_norm,
        "training_relative_residual": relative_residual_value,
        "training_residual_nmse_db": float(
            20.0 * math.log10(max(relative_residual_value, np.finfo(float).tiny))
        ),
        "basis_build_seconds": float(basis_seconds),
        "solve_seconds": float(solve_seconds),
        "column_norm_min": smallest_norm,
        "column_norm_max": largest_norm,
    }
    return coefficients, diagnostics


@torch.inference_mode()
def apply_polynomial(
    input_complex: np.ndarray,
    coefficients: torch.Tensor,
    *,
    polynomial_model: str,
    configuration: dict[str, int],
    segment_length: int,
) -> np.ndarray:
    """Apply fitted polynomial coefficients with the training boundary policy."""
    input_tensor = torch.as_tensor(
        np.asarray(input_complex),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    phi = build_torch_basis(
        input_tensor,
        polynomial_model,
        configuration,
        segment_length,
    )
    prediction = phi @ coefficients
    if not bool(torch.all(torch.isfinite(prediction)).item()):
        raise RuntimeError("polynomial prediction contains non-finite values")
    return prediction.cpu().numpy()


# ---------------------------------------------------------------------------
# Data, PA, and metric helpers
# ---------------------------------------------------------------------------

def iq_to_complex(iq):
    iq = np.asarray(iq)
    return iq[:, 0] + 1j * iq[:, 1]


def complex_to_iq(complex_signal):
    complex_signal = np.asarray(complex_signal)
    return np.stack(
        [complex_signal.real, complex_signal.imag],
        axis=-1,
    ).astype(np.float32)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pa_model_id(
    *,
    seed: int,
    backbone: str,
    hidden_size: int,
    frame_length: int,
    parameter_count: int,
) -> str:
    return (
        f"PA_S_{seed}_M_{backbone.upper()}_H_{hidden_size}_"
        f"F_{frame_length}_P_{parameter_count}"
    )


def resolve_pa_checkpoint(
    args: argparse.Namespace,
    parameter_count: int,
) -> tuple[Path, str]:
    model_id = pa_model_id(
        seed=args.seed,
        backbone=args.pa_backbone,
        hidden_size=args.pa_hidden_size,
        frame_length=args.frame_length,
        parameter_count=parameter_count,
    )
    if args.pa_checkpoint:
        checkpoint = Path(args.pa_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
    else:
        checkpoint = (
            REPO_ROOT
            / "save"
            / args.dataset_name
            / "train_pa"
            / f"{model_id}.pt"
        )
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PA checkpoint does not exist: {checkpoint}")
    return checkpoint, model_id


def _load_state_dict(path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older supported Torch releases.
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"PA checkpoint is not a state dictionary: {path}")
    return state


def load_pa_model(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, torch.device, Path, str, int]:
    if args.device < 0:
        pa_device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PA evaluation was requested but CUDA is unavailable")
        pa_device = torch.device(f"cuda:{args.device}")

    pa_net = model.CoreModel(
        input_size=2,
        hidden_size=args.pa_hidden_size,
        num_layers=args.pa_num_layers,
        backbone_type=args.pa_backbone,
    )
    parameter_count = count_net_params(pa_net)
    checkpoint, model_id = resolve_pa_checkpoint(args, parameter_count)
    pa_net.load_state_dict(_load_state_dict(checkpoint), strict=True)
    pa_net = pa_net.to(pa_device)
    pa_net.eval()
    return pa_net, pa_device, checkpoint, model_id, parameter_count


@torch.inference_mode()
def run_pa_model(iq_signal, pa_net, device, nperseg):
    """Evaluate the PA with one independent recurrent state per segment."""
    iq_signal = np.asarray(iq_signal, dtype=np.float32)
    n_samples = len(iq_signal)
    segments = []
    for start in range(0, n_samples, nperseg):
        segment = iq_signal[start:start + nperseg]
        if len(segment) < nperseg:
            segment = np.vstack(
                [
                    segment,
                    np.zeros((nperseg - len(segment), 2), dtype=np.float32),
                ]
            )
        segments.append(segment)
    if not segments:
        raise ValueError("cannot evaluate an empty signal")
    tensor = torch.as_tensor(np.stack(segments), dtype=torch.float32, device=device)
    output = pa_net(tensor).cpu().numpy()
    return output.reshape(-1, 2)[:n_samples]


def compute_metrics(prediction_iq, ground_truth_iq, spec):
    """Compute metrics with the same segmentation and definitions as training."""
    nperseg = int(spec["nperseg"])
    n_samples = min(len(prediction_iq), len(ground_truth_iq))
    n_segments = n_samples // nperseg
    if n_segments == 0:
        raise ValueError(
            f"metric input needs at least one complete {nperseg}-sample segment"
        )
    usable = n_segments * nperseg
    prediction = np.asarray(prediction_iq[:usable]).reshape(n_segments, nperseg, 2)
    ground_truth = np.asarray(ground_truth_iq[:usable]).reshape(
        n_segments, nperseg, 2
    )
    nmse = metrics.NMSE(prediction, ground_truth)
    evm = metrics.EVM(
        prediction,
        ground_truth,
        sample_rate=int(spec["input_signal_fs"]),
        bw_main_ch=spec["bw_main_ch"],
        n_sub_ch=spec["n_sub_ch"],
        nperseg=nperseg,
    )
    aclr_left, aclr_right = metrics.ACLR(
        prediction,
        fs=spec["input_signal_fs"],
        nperseg=nperseg,
        bw_main_ch=spec["bw_main_ch"],
        n_sub_ch=spec["n_sub_ch"],
    )
    return nmse, evm, aclr_left, aclr_right


def metric_dict(prediction_iq, ground_truth_iq, spec) -> dict[str, float]:
    nmse, evm, aclr_left, aclr_right = compute_metrics(
        prediction_iq,
        ground_truth_iq,
        spec,
    )
    result = {
        "ACLR_L": float(aclr_left),
        "ACLR_R": float(aclr_right),
        "ACLR_AVG": float((aclr_left + aclr_right) / 2.0),
        "EVM": float(evm),
        "NMSE": float(nmse),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"evaluation produced non-finite metrics: {result}")
    return result


def display_metrics(split: str, values: dict[str, float]) -> None:
    print(f"\n{split} results")
    print(f"{'Metric':<12} {'Value':>12}")
    print("-" * 26)
    for name in ("ACLR_L", "ACLR_R", "ACLR_AVG", "EVM", "NMSE"):
        print(f"{name:<12} {values[name]:>12.2f} dB")


def _portable_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# CLI and benchmark execution
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fit and evaluate MP/GMP PA or ILA-DPD benchmarks."
    )
    parser.add_argument(
        "--task",
        choices=("pa_modeling", "dpd_ila"),
        default="dpd_ila",
        help="Benchmark task (default preserves the historical DPD behavior).",
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        dest="dataset_name",
        required=True,
    )
    parser.add_argument("--model", choices=("mp", "gmp"), default="mp")

    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--Q", type=int, default=50)
    parser.add_argument("--Ka", type=int, default=5)
    parser.add_argument("--La", type=int, default=15)
    parser.add_argument("--Kb", type=int, default=4)
    parser.add_argument("--Lb", type=int, default=15)
    parser.add_argument("--Mb", type=int, default=2)
    parser.add_argument("--Kc", type=int, default=4)
    parser.add_argument("--Lc", type=int, default=15)
    parser.add_argument("--Mc", type=int, default=1)

    parser.add_argument(
        "--solver-device",
        default="cuda",
        help="'cuda', 'cuda:INDEX', or 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--solver-dtype",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument(
        "--solver-mode",
        choices=("gels", "truncated_svd"),
        default="gels",
    )
    parser.add_argument(
        "--svd-rcond",
        type=float,
        help="Relative singular-value cutoff for truncated_svd.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA index for PA evaluation; -1 selects CPU.",
    )

    parser.add_argument(
        "--pa-backbone",
        "--pa_backbone",
        dest="pa_backbone",
        default="tres_gru",
    )
    parser.add_argument(
        "--pa-hidden-size",
        "--pa_hidden_size",
        dest="pa_hidden_size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--pa-num-layers",
        "--pa_num_layers",
        dest="pa_num_layers",
        type=int,
        default=1,
    )
    parser.add_argument("--pa-checkpoint", help="Explicit trained PA checkpoint.")
    parser.add_argument("--frame-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", help="Write metrics and provenance as JSON.")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict:
    spec_path = REPO_ROOT / "datasets" / args.dataset_name / "spec.json"
    with open(spec_path, encoding="utf-8") as file:
        spec = json.load(file)
    nperseg = int(spec["nperseg"])
    configuration = basis_configuration(args)
    _validate_positive_configuration(configuration)
    n_complex_coefficients = coefficient_count(args.model, configuration)
    n_real_parameters = 2 * n_complex_coefficients
    solver_device = resolve_solver_device(args.solver_device, args.device)
    solver_dtype = resolve_complex_dtype(args.solver_dtype)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        dataset_name=args.dataset_name
    )
    target_gain = float(set_target_gain(X_train, y_train))
    print(
        f"\n=== {args.task}: {args.model.upper()} "
        f"({n_complex_coefficients} complex / {n_real_parameters} real) ==="
    )
    if args.solver_mode == "truncated_svd":
        solver_description = (
            f"torch.linalg.svd truncated at rcond={args.svd_rcond:g}"
        )
    else:
        solver_description = "torch.linalg.lstsq gels"
    print(f"Solver: {solver_description} on {solver_device} ({solver_dtype})")

    common_fit_arguments = {
        "polynomial_model": args.model,
        "configuration": configuration,
        "segment_length": nperseg,
        "device": solver_device,
        "dtype": solver_dtype,
        "solver_mode": args.solver_mode,
        "svd_rcond": args.svd_rcond,
    }

    if args.task == "pa_modeling":
        coefficients, diagnostics = fit_complex_least_squares(
            iq_to_complex(X_train),
            iq_to_complex(y_train),
            **common_fit_arguments,
        )

        validation_prediction = complex_to_iq(
            apply_polynomial(
                iq_to_complex(X_val),
                coefficients,
                polynomial_model=args.model,
                configuration=configuration,
                segment_length=nperseg,
            )
        )
        validation_metrics = metric_dict(validation_prediction, y_val, spec)
        del validation_prediction

        test_prediction = complex_to_iq(
            apply_polynomial(
                iq_to_complex(X_test),
                coefficients,
                polynomial_model=args.model,
                configuration=configuration,
                segment_length=nperseg,
            )
        )
        test_metrics = metric_dict(test_prediction, y_test, spec)
        method = "direct_least_squares"
        pa_binding = {}
    else:
        coefficients, diagnostics = fit_complex_least_squares(
            iq_to_complex(y_train) / target_gain,
            iq_to_complex(X_train),
            **common_fit_arguments,
        )
        pa_net, pa_device, pa_path, pa_id, pa_parameters = load_pa_model(args)

        validation_dpd = complex_to_iq(
            apply_polynomial(
                iq_to_complex(X_val),
                coefficients,
                polynomial_model=args.model,
                configuration=configuration,
                segment_length=nperseg,
            )
        )
        validation_prediction = run_pa_model(
            validation_dpd,
            pa_net,
            pa_device,
            nperseg,
        )
        validation_target = (target_gain * X_val).astype(np.float32)
        validation_metrics = metric_dict(
            validation_prediction,
            validation_target,
            spec,
        )
        del validation_dpd, validation_prediction

        test_dpd = complex_to_iq(
            apply_polynomial(
                iq_to_complex(X_test),
                coefficients,
                polynomial_model=args.model,
                configuration=configuration,
                segment_length=nperseg,
            )
        )
        test_prediction = run_pa_model(test_dpd, pa_net, pa_device, nperseg)
        test_target = (target_gain * X_test).astype(np.float32)
        test_metrics = metric_dict(test_prediction, test_target, spec)
        method = "indirect_learning_architecture"
        pa_binding = {
            "pa_backbone": args.pa_backbone,
            "pa_hidden_size": int(args.pa_hidden_size),
            "pa_num_layers": int(args.pa_num_layers),
            "pa_model_id": pa_id,
            "pa_parameters": int(pa_parameters),
            "pa_evaluation_device": str(pa_device),
            "pa_checkpoint": _portable_path(pa_path),
            "pa_checkpoint_sha256": sha256_file(pa_path),
        }

    display_metrics("Validation", validation_metrics)
    display_metrics("Test", test_metrics)
    dtype_name = str(solver_dtype)
    device_name = str(solver_device)
    result = {
        "schema_version": 4,
        "task": args.task,
        "method": method,
        "dataset": args.dataset_name,
        "model": args.model,
        "basis_configuration": configuration,
        "complex_coefficients": int(n_complex_coefficients),
        "real_parameters": int(n_real_parameters),
        "parameter_count_convention": "two real degrees of freedom per complex coefficient",
        "target_gain": target_gain,
        "sample_rate_hz": float(spec["input_signal_fs"]),
        "nperseg": nperseg,
        "segment_boundary_policy": "zero delay state at every nperseg boundary",
        "solver": diagnostics["solver_implementation"],
        "solver_mode": diagnostics["solver_mode"],
        "solver_driver": diagnostics["solver_driver"],
        "device": device_name,
        "dtype": dtype_name,
        # Retain explicit aliases so downstream consumers cannot confuse the
        # solver device with the separate neural-PA evaluation device.
        "solver_device": device_name,
        "solver_dtype": dtype_name,
        "column_scaling": "l2",
        "column_scale_min": diagnostics["column_norm_min"],
        "column_scale_max": diagnostics["column_norm_max"],
        "full_rank_assumption": args.solver_mode == "gels",
        "least_squares_rank": diagnostics["least_squares_rank"],
        "regularization": diagnostics["regularization"],
        "svd_rcond": diagnostics["svd_rcond"],
        "singular_value_max": diagnostics["singular_value_max"],
        "singular_value_min": diagnostics["singular_value_min"],
        "singular_value_cutoff": diagnostics["singular_value_cutoff"],
        "retained_singular_value_min": diagnostics[
            "retained_singular_value_min"
        ],
        "condition_number": diagnostics["condition_number"],
        "coefficient_l2_norm": diagnostics["coefficient_l2_norm"],
        "training_relative_residual": diagnostics["training_relative_residual"],
        "training_diagnostics": diagnostics,
        "determinism": {
            "stochastic_fitting": False,
            "seed": int(args.seed),
            "note": (
                "The design matrix and targets are deterministic; final low-order "
                "floating-point bits can vary across CUDA library versions."
            ),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        **pa_binding,
    }
    return result


def main():
    args = build_arg_parser().parse_args()
    result = run_benchmark(args)
    if args.json_out:
        output_path = Path(args.json_out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, sort_keys=True)
            file.write("\n")
        print(f"\nRaw results written to {output_path}")


if __name__ == "__main__":
    main()
