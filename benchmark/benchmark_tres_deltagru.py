#!/usr/bin/env python3
"""Reproducible training benchmark for the TRes-DeltaGRU backbone.

This benchmark intentionally keeps synthetic input resident on the selected
device.  It measures the model/optimizer path rather than CSV parsing,
DataLoader workers, or host-to-device transfers.  The default shape matches
``bash_scripts/OpenDPDv2.sh``: B=64, T=200, H=15, thx=.01, thh=.05.

Examples
--------
    .venv/bin/python benchmark/benchmark_tres_deltagru.py
    .venv/bin/python benchmark/benchmark_tres_deltagru.py --batch-size 256
    .venv/bin/python benchmark/benchmark_tres_deltagru.py --reference-only
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from models import CoreModel  # noqa: E402
from backbones.triton_deltagru import triton  # noqa: E402


@dataclass
class TimingResult:
    implementation: str
    steps_per_block: int
    median_ms: float
    iqr_ms: float
    block_ms: list[float]
    batches_per_second: float
    sequence_positions_per_second: float
    peak_allocated_mib: float


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_model(
    state: dict[str, torch.Tensor],
    hidden_size: int,
    thx: float,
    thh: float,
    device: torch.device,
    fused: bool,
) -> CoreModel:
    with contextlib.redirect_stdout(io.StringIO()):
        model = CoreModel(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=1,
            backbone_type="tres_deltagru",
            thx=thx,
            thh=thh,
        ).to(device)
    model.load_state_dict(state)
    model.backbone.rnn.use_triton = fused
    model.backbone.set_debug(0)
    if fused:
        probe = next(model.parameters()).new_empty((1, 1, model.backbone.rnn.input_size))
        if not model.backbone.rnn._can_use_triton(probe):
            raise RuntimeError(
                "the fused TRes-DeltaGRU kernel does not support this device, "
                "dtype, hidden size, or module configuration"
            )
    model.train()
    return model


def one_step(
    model: CoreModel,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    grad_clip: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(features), targets)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


def time_implementation(
    name: str,
    fused: bool,
    state: dict[str, torch.Tensor],
    features: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
    steps_per_block: int,
) -> TimingResult:
    device = features.device
    model = build_model(state, args.hidden_size, args.thx, args.thh, device, fused)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for _ in range(args.warmup):
        one_step(model, optimizer, features, targets, args.grad_clip)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    block_times: list[float] = []
    for _ in range(args.blocks):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(steps_per_block):
                one_step(model, optimizer, features, targets, args.grad_clip)
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / steps_per_block
        else:
            started = time.perf_counter()
            for _ in range(steps_per_block):
                one_step(model, optimizer, features, targets, args.grad_clip)
            elapsed_ms = (time.perf_counter() - started) * 1_000 / steps_per_block
        block_times.append(elapsed_ms)

    median_ms = statistics.median(block_times)
    quartiles = (
        statistics.quantiles(block_times, n=4, method="inclusive")
        if len(block_times) > 1
        else [block_times[0]] * 3
    )
    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else float("nan")
    )
    return TimingResult(
        implementation=name,
        steps_per_block=steps_per_block,
        median_ms=median_ms,
        iqr_ms=quartiles[2] - quartiles[0],
        block_ms=block_times,
        batches_per_second=1_000 / median_ms,
        sequence_positions_per_second=(
            features.shape[0] * features.shape[1] * 1_000 / median_ms
        ),
        peak_allocated_mib=peak_mib,
    )


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = (reference - candidate).detach()
    denominator = reference.detach().norm().clamp_min(torch.finfo(reference.dtype).tiny)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (difference.norm() / denominator).item(),
    }


def check_equivalence(
    state: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    # A modest batch checks all equations without making the eager preflight a
    # substantial fraction of the benchmark runtime.
    batch_size = min(args.batch_size, args.check_batch_size)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    x_reference = torch.randn(
        batch_size, args.sequence_length, 2, generator=generator, device=device
    ).requires_grad_()
    x_fused = x_reference.detach().clone().requires_grad_()
    target = torch.randn(
        batch_size, args.sequence_length, 2, generator=generator, device=device
    )
    reference = build_model(
        state, args.hidden_size, args.thx, args.thh, device, fused=False
    )
    fused = build_model(state, args.hidden_size, args.thx, args.thh, device, fused=True)

    reference_output = reference(x_reference)
    reference_loss = torch.nn.functional.mse_loss(reference_output, target)
    reference_loss.backward()
    fused_output = fused(x_fused)
    fused_loss = torch.nn.functional.mse_loss(fused_output, target)
    fused_loss.backward()
    synchronize(device)

    parameter_errors = {}
    for (name, reference_parameter), (candidate_name, candidate_parameter) in zip(
        reference.named_parameters(), fused.named_parameters(), strict=True
    ):
        if name != candidate_name:
            raise RuntimeError(f"parameter order mismatch: {name} != {candidate_name}")
        parameter_errors[name] = error_metrics(
            reference_parameter.grad, candidate_parameter.grad
        )

    return {
        "output": error_metrics(reference_output, fused_output),
        "input_gradient": error_metrics(x_reference.grad, x_fused.grad),
        "loss_abs": abs(reference_loss.item() - fused_loss.item()),
        "parameter_gradients": parameter_errors,
    }


def enforce_equivalence(equivalence: dict[str, object]) -> None:
    """Fail before timing if the candidate exceeds calibrated FP32 tolerances."""

    limits = {
        "output": (2e-5, 1e-4),
        "input_gradient": (2e-6, 1e-3),
    }
    for name, (max_abs_limit, relative_limit) in limits.items():
        metrics = equivalence[name]
        if metrics["max_abs"] > max_abs_limit or metrics["relative_l2"] > relative_limit:
            raise RuntimeError(f"{name} equivalence check failed: {metrics}")
    for name, metrics in equivalence["parameter_gradients"].items():
        if metrics["max_abs"] > 2e-5 or metrics["relative_l2"] > 1e-4:
            raise RuntimeError(f"{name} gradient equivalence check failed: {metrics}")
    if equivalence["loss_abs"] > 1e-5:
        raise RuntimeError(f"loss equivalence check failed: {equivalence['loss_abs']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=15)
    parser.add_argument("--thx", type=float, default=0.01)
    parser.add_argument("--thh", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=200.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--reference-steps-per-block", type=int, default=5)
    parser.add_argument("--fused-steps-per-block", type=int, default=100)
    parser.add_argument("--check-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive_values = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "blocks": args.blocks,
        "reference_steps_per_block": args.reference_steps_per_block,
        "fused_steps_per_block": args.fused_steps_per_block,
        "check_batch_size": args.check_batch_size,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid or args.warmup < 0:
        raise ValueError(f"benchmark counts must be positive (warmup may be zero): {invalid}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    # The fused recurrence accumulates in FP32.  Disable TF32 in the reference
    # so the equivalence check measures operation ordering, not lower precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with contextlib.redirect_stdout(io.StringIO()):
        template = CoreModel(
            input_size=2,
            hidden_size=args.hidden_size,
            num_layers=1,
            backbone_type="tres_deltagru",
            thx=args.thx,
            thh=args.thh,
        ).to(device)
    state = copy.deepcopy(template.state_dict())
    generator = torch.Generator(device=device).manual_seed(args.seed)
    features = torch.randn(
        args.batch_size,
        args.sequence_length,
        2,
        generator=generator,
        device=device,
    )
    targets = torch.randn(
        args.batch_size,
        args.sequence_length,
        2,
        generator=generator,
        device=device,
    )

    system = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__ if triton is not None else None,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "compute_capability": (
            ".".join(map(str, torch.cuda.get_device_capability(device)))
            if device.type == "cuda"
            else None
        ),
        "driver": (
            command_output([
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ])
            if device.type == "cuda"
            else None
        ),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    print(json.dumps(system, indent=2))
    print(
        f"shape=B{args.batch_size}xT{args.sequence_length}x2, H={args.hidden_size}, "
        f"thresholds=({args.thx}, {args.thh}), trainable_params="
        f"{sum(parameter.numel() for parameter in template.parameters()):,}"
    )

    equivalence = None
    if device.type == "cuda" and not args.reference_only:
        equivalence = check_equivalence(state, args, device)
        enforce_equivalence(equivalence)
        print("equivalence:")
        print(json.dumps(equivalence, indent=2))

    results = [
        time_implementation(
            "eager-clean",
            False,
            state,
            features,
            targets,
            args,
            args.reference_steps_per_block,
        )
    ]
    if device.type == "cuda" and not args.reference_only:
        results.append(
            time_implementation(
                "triton-fused",
                True,
                state,
                features,
                targets,
                args,
                args.fused_steps_per_block,
            )
        )

    baseline_ms = results[0].median_ms
    print(
        "\nimplementation       median ms/step  IQR(block means)  "
        "steps/block  sequence positions/s    peak MiB    speedup"
    )
    for result in results:
        speedup = baseline_ms / result.median_ms
        print(
            f"{result.implementation:<20} {result.median_ms:>12.3f} "
            f"{result.iqr_ms:>17.3f} {result.steps_per_block:>12d} "
            f"{result.sequence_positions_per_second:>20,.0f} "
            f"{result.peak_allocated_mib:>11.1f} {speedup:>9.2f}x"
        )
        print(
            f"  block means ({result.implementation}, ms/step): "
            + ", ".join(f"{value:.3f}" for value in result.block_ms)
        )

    payload = {
        "system": system,
        "config": vars(args) | {"json_out": str(args.json_out) if args.json_out else None},
        "equivalence": equivalence,
        "results": [asdict(result) for result in results],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
