#!/usr/bin/env python3
"""Controlled end-to-end benchmark for TRes-DeltaGRU DPD training.

The same script can execute the committed reference tree and a working tree:

    # Create one deterministic, trained PA fixture with the reference code.
    python benchmark/benchmark_end_to_end.py prepare \
      --repo-root /tmp/OpenDPD-e2e-baseline \
      --fixture benchmark/fixtures/dpa200_dgru_h23.pt

    # Run identical batches through the two implementations.
    python benchmark/benchmark_end_to_end.py run \
      --repo-root /tmp/OpenDPD-e2e-baseline --implementation original \
      --fixture benchmark/fixtures/dpa200_dgru_h23.pt \
      --output-dir benchmark/results/e2e_original
    python benchmark/benchmark_end_to_end.py run \
      --repo-root . --implementation optimized \
      --fixture benchmark/fixtures/dpa200_dgru_h23.pt \
      --output-dir benchmark/results/e2e_optimized

    python benchmark/benchmark_end_to_end.py compare \
      --reference-dir benchmark/results/e2e_original \
      --candidate-dir benchmark/results/e2e_optimized \
      --json-out benchmark/results/e2e_comparison.json

The fixture contains identical PA and DPD initial state dictionaries.  Every
run also records the exact shuffled frame indices used in each epoch.  The run
includes CSV loading, frame construction, host-to-device transfer, one isolated
compile/warm-up step, training, validation, testing, the repository's ACLR,
NMSE and EVM functions, best-checkpoint selection, and artifact serialization.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRIC_NAMES = ("NMSE", "EVM", "ACLR_L", "ACLR_R", "ACLR_AVG")


class FixedOrderSampler(Sampler[int]):
    """Yield a recorded permutation without consuming the global RNG."""

    def __init__(self, indices: Iterable[int]):
        self.indices = tuple(int(index) for index in indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _command_output(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command, cwd=cwd, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _load_repo(repo_root: Path) -> SimpleNamespace:
    repo_root = repo_root.resolve()
    if not (repo_root / "models.py").is_file():
        raise FileNotFoundError(f"not an OpenDPD tree: {repo_root}")
    sys.path.insert(0, str(repo_root))
    models = importlib.import_module("models")
    data = importlib.import_module("modules.data_collector")
    train_funcs = importlib.import_module("modules.train_funcs")
    util = importlib.import_module("utils.util")
    return SimpleNamespace(
        root=repo_root,
        models=models,
        data=data,
        train_funcs=train_funcs,
        util=util,
    )


def _load_spec(repo_root: Path, dataset_name: str) -> dict[str, Any]:
    path = repo_root / "datasets" / dataset_name / "spec.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _build_core(
    repo: SimpleNamespace,
    backbone: str,
    hidden_size: int,
    thx: float = 0.0,
    thh: float = 0.0,
):
    with contextlib.redirect_stdout(io.StringIO()):
        return repo.models.CoreModel(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=1,
            backbone_type=backbone,
            thx=thx,
            thh=thh,
        )


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _build_cascade(
    repo: SimpleNamespace,
    fixture: dict[str, Any],
    device: torch.device,
):
    config = fixture["config"]
    pa = _build_core(repo, config["pa_backbone"], config["pa_hidden_size"])
    dpd = _build_core(
        repo,
        config["dpd_backbone"],
        config["dpd_hidden_size"],
        config["thx"],
        config["thh"],
    )
    pa.load_state_dict(fixture["pa_state"])
    dpd.load_state_dict(fixture["dpd_state"])
    cascade = repo.models.CascadedModel(dpd_model=dpd, pa_model=pa)
    cascade.freeze_pa_model()
    return cascade.to(device)


def _metric_args(spec: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        input_signal_fs=spec["input_signal_fs"],
        bw_main_ch=spec["bw_main_ch"],
        n_sub_ch=spec["n_sub_ch"],
        nperseg=spec["nperseg"],
    )


def _metrics(
    train_funcs: Any,
    metric_args: SimpleNamespace,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    result = train_funcs.calculate_metrics(
        metric_args, {}, prediction, ground_truth
    )
    return {name: float(result[name]) for name in METRIC_NAMES}


def _environment(repo_root: Path, device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    source_files = [
        "backbones/tres_deltagru.py",
        "models.py",
        "modules/train_funcs.py",
        "modules/cuda_graph_training.py",
        "modules/data_collector.py",
        "project.py",
    ]
    hashes = {
        relative: _sha256_file(repo_root / relative)
        for relative in source_files
        if (repo_root / relative).is_file()
    }
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"], repo_root),
        "git_dirty": bool(_command_output(["git", "status", "--short"], repo_root)),
        "source_sha256": hashes,
    }


def prepare(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    repo = _load_repo(args.repo_root)
    device = torch.device(args.device)
    _configure_reproducibility(args.seed)

    arrays = repo.data.load_dataset(dataset_name=args.dataset_name)
    x_train, y_train = arrays[0], arrays[1]
    pa_dataset = repo.data.IQFrameDataset(
        x_train, y_train, frame_length=args.frame_length, stride=args.pa_frame_stride
    )

    # Model initialization is intentionally done once.  The resulting tensors,
    # rather than a seed assumption, are the fixture contract used by every arm.
    pa = _build_core(repo, args.pa_backbone, args.pa_hidden_size).to(device)
    generator = torch.Generator().manual_seed(args.seed + 10_000)
    loader = DataLoader(
        pa_dataset,
        batch_size=args.pa_batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(pa.parameters(), lr=args.pa_lr)
    criterion = torch.nn.MSELoss()
    epoch_losses: list[float] = []
    pa.train()
    for _ in range(args.pa_pretrain_epochs):
        losses: list[float] = []
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pa(features), targets)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(pa.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(float(np.mean(losses)))

    # Re-seed before DPD construction so its state is independent of PA epoch
    # count and can be regenerated deliberately if the fixture format changes.
    torch.manual_seed(args.seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + 1)
    dpd = _build_core(
        repo,
        args.dpd_backbone,
        args.dpd_hidden_size,
        args.thx,
        args.thh,
    )
    fixture = {
        "format_version": 1,
        "config": {
            "dataset_name": args.dataset_name,
            "frame_length": args.frame_length,
            "pa_backbone": args.pa_backbone,
            "pa_hidden_size": args.pa_hidden_size,
            "dpd_backbone": args.dpd_backbone,
            "dpd_hidden_size": args.dpd_hidden_size,
            "thx": args.thx,
            "thh": args.thh,
            "seed": args.seed,
        },
        "pa_pretraining": {
            "epochs": args.pa_pretrain_epochs,
            "frame_stride": args.pa_frame_stride,
            "batch_size": args.pa_batch_size,
            "lr": args.pa_lr,
            "epoch_losses": epoch_losses,
            "surrogate_notice": (
                "Deterministically trained benchmark surrogate; the repository "
                "does not contain a published PA checkpoint."
            ),
        },
        "pa_state": _cpu_state(pa),
        "dpd_state": _cpu_state(dpd),
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.fixture)
    print(
        json.dumps(
            {
                "fixture": str(args.fixture),
                "sha256": _sha256_file(args.fixture),
                "pa_final_loss": epoch_losses[-1] if epoch_losses else None,
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
    )


def _loader(
    dataset: Any,
    batch_size: int,
    device: torch.device,
    optimized: bool,
    indices: Iterable[int] | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=FixedOrderSampler(indices) if indices is not None else None,
        pin_memory=optimized and device.type == "cuda",
    )


def _time_eval(
    repo: SimpleNamespace,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    _sync(device)
    started = time.perf_counter()
    log: dict[str, Any] = {}
    _, prediction, truth = repo.train_funcs.net_eval(
        log=log,
        net=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
    )
    _sync(device)
    return time.perf_counter() - started, float(log["loss"]), prediction, truth


def _prediction_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    difference = candidate64 - reference64
    denominator = max(np.linalg.norm(reference64.ravel()), np.finfo(np.float64).tiny)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
    }


def run(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    repo = _load_repo(args.repo_root)
    device = torch.device(args.device)
    _configure_reproducibility(args.seed)
    optimized = args.implementation != "original"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if not args.progress:
        repo.train_funcs.tqdm = lambda iterable: iterable

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    config = fixture["config"]
    if args.dataset_name != config["dataset_name"]:
        raise ValueError(
            f"fixture dataset {config['dataset_name']} != {args.dataset_name}"
        )
    if args.frame_length != config["frame_length"]:
        raise ValueError(
            f"fixture frame length {config['frame_length']} != {args.frame_length}"
        )
    spec = _load_spec(repo.root, args.dataset_name)
    metric_args = _metric_args(spec)

    setup_started = time.perf_counter()
    x_train, y_train, x_val, _, x_test, _ = repo.data.load_dataset(
        dataset_name=args.dataset_name
    )
    target_gain = float(repo.util.set_target_gain(x_train, y_train))
    train_set = repo.data.IQFrameDataset(
        x_train,
        target_gain * x_train,
        frame_length=args.frame_length,
        stride=args.frame_stride,
    )
    val_set = repo.data.IQSegmentDataset(
        x_val, target_gain * x_val, nperseg=spec["nperseg"]
    )
    test_set = repo.data.IQSegmentDataset(
        x_test, target_gain * x_test, nperseg=spec["nperseg"]
    )

    order_generator = torch.Generator().manual_seed(args.seed + 20_000)
    orders = [
        torch.randperm(len(train_set), generator=order_generator).numpy()
        for _ in range(args.epochs)
    ]
    val_loader = _loader(val_set, args.eval_batch_size, device, optimized)
    test_loader = _loader(test_set, args.eval_batch_size, device, optimized)
    setup_data_s = time.perf_counter() - setup_started

    # Compile and warm all kernels with a throw-away model so the measured
    # epochs start from the immutable fixture state.
    warm_started = time.perf_counter()
    model = _build_cascade(repo, fixture, device)
    warm_loader = _loader(
        train_set,
        args.batch_size,
        device,
        optimized,
        orders[0][: args.batch_size],
    )
    warm_features, warm_targets = next(iter(warm_loader))
    warm_features = warm_features.to(device, non_blocking=optimized)
    warm_targets = warm_targets.to(device, non_blocking=optimized)
    warm_loss = torch.nn.functional.mse_loss(model(warm_features), warm_targets)
    warm_loss.backward()
    _sync(device)
    warmup_s = time.perf_counter() - warm_started
    model.zero_grad(set_to_none=True)
    del warm_features, warm_targets, warm_loss

    # Whole-cascade graph capture must not inherit AccumulateGrad nodes that a
    # prior default-stream backward kept alive.  The throw-away warm model has
    # already populated global Triton/cuDNN caches; build an untouched model
    # with the immutable fixture state for the captured training trajectory.
    if args.cuda_graph_training:
        del model
        model = _build_cascade(repo, fixture, device)

    model_started = time.perf_counter()
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer_parameters = trainable if optimized else model.parameters()
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.decay_factor,
        patience=args.patience,
        threshold=1e-4,
        min_lr=args.lr_end,
    )
    criterion = torch.nn.MSELoss()
    model_optimizer_setup_s = time.perf_counter() - model_started
    trajectory: list[dict[str, Any]] = []
    epoch_times: list[dict[str, float]] = []
    best_val_aclr = float("inf")
    best_epoch = -1
    best_dpd_state: dict[str, torch.Tensor] | None = None

    for epoch, order in enumerate(orders):
        train_loader = _loader(
            train_set, args.batch_size, device, optimized, order
        )
        _sync(device)
        train_started = time.perf_counter()
        train_log: dict[str, Any] = {}
        train_kwargs = dict(
            log=train_log,
            net=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            grad_clip_val=args.grad_clip,
            device=device,
        )
        # The detached reference worktree predates this opt-in keyword, so add
        # it only for runs that explicitly request whole-step capture.
        if args.cuda_graph_training:
            train_kwargs["cuda_graph_training"] = True
        repo.train_funcs.net_train(**train_kwargs)
        _sync(device)
        train_s = time.perf_counter() - train_started

        val_s, val_loss, val_prediction, val_truth = _time_eval(
            repo, model, val_loader, criterion, device
        )
        metric_started = time.perf_counter()
        val_metrics = _metrics(repo.train_funcs, metric_args, val_prediction, val_truth)
        val_metric_s = time.perf_counter() - metric_started

        test_s, test_loss, test_prediction, test_truth = _time_eval(
            repo, model, test_loader, criterion, device
        )
        metric_started = time.perf_counter()
        test_metrics = _metrics(repo.train_funcs, metric_args, test_prediction, test_truth)
        test_metric_s = time.perf_counter() - metric_started

        checkpoint_started = time.perf_counter()
        if val_metrics["ACLR_AVG"] < best_val_aclr:
            best_val_aclr = val_metrics["ACLR_AVG"]
            best_epoch = epoch
            best_dpd_state = _cpu_state(model.dpd_model)
        checkpoint_s = time.perf_counter() - checkpoint_started
        if args.lr_schedule:
            scheduler.step(val_metrics["ACLR_AVG"])

        trajectory.append(
            {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(train_log["loss"]),
                "val_loss": val_loss,
                "test_loss": test_loss,
                "val": val_metrics,
                "test": test_metrics,
            }
        )
        epoch_times.append(
            {
                "train_s": train_s,
                "val_inference_s": val_s,
                "val_metrics_s": val_metric_s,
                "test_inference_s": test_s,
                "test_metrics_s": test_metric_s,
                "checkpoint_s": checkpoint_s,
                "total_s": (
                    train_s
                    + val_s
                    + val_metric_s
                    + test_s
                    + test_metric_s
                    + checkpoint_s
                ),
            }
        )

    if best_dpd_state is None:
        raise RuntimeError("no best checkpoint was selected")
    model.dpd_model.load_state_dict(best_dpd_state)
    best_test_s, best_test_loss, best_prediction, best_truth = _time_eval(
        repo, model, test_loader, criterion, device
    )
    metric_started = time.perf_counter()
    best_metrics = _metrics(repo.train_funcs, metric_args, best_prediction, best_truth)
    best_metric_s = time.perf_counter() - metric_started

    model.dpd_model.eval()
    dpd_inference_started = time.perf_counter()
    dpd_outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for features, _ in test_loader:
            outputs = model.dpd_model(
                features.to(device, non_blocking=optimized)
            )
            dpd_outputs.append(outputs.cpu())
    best_dpd_prediction = torch.cat(dpd_outputs, dim=0).numpy()
    _sync(device)
    best_dpd_inference_s = time.perf_counter() - dpd_inference_started

    serialization_started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_path = args.output_dir / "batch_orders.npz"
    np.savez_compressed(order_path, **{f"epoch_{i}": order for i, order in enumerate(orders)})
    output_path = args.output_dir / "outputs.npz"
    np.savez_compressed(
        output_path,
        cascade=best_prediction,
        dpd=best_dpd_prediction,
        ground_truth=best_truth,
    )
    checkpoint_path = args.output_dir / "best_dpd.pt"
    torch.save(best_dpd_state, checkpoint_path)
    _sync(device)
    artifact_serialization_s = time.perf_counter() - serialization_started

    total_s = time.perf_counter() - total_started
    result = {
        "format_version": 1,
        "implementation": args.implementation,
        "repo_root": str(repo.root),
        "fixture": {
            "path": str(args.fixture.resolve()),
            "sha256": _sha256_file(args.fixture),
            "pa_pretraining": fixture["pa_pretraining"],
        },
        "config": {
            **config,
            "frame_stride": args.frame_stride,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "lr_end": args.lr_end,
            "decay_factor": args.decay_factor,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "hard_determinism": True,
            "tf32": False,
            "progress_display": args.progress,
            "cuda_graph_training": args.cuda_graph_training,
            "train_frames": len(train_set),
            "train_batches": (len(train_set) + args.batch_size - 1) // args.batch_size,
        },
        "environment": _environment(repo.root, device),
        "timing": {
            "setup_data_s": setup_data_s,
            "compile_and_warmup_s": warmup_s,
            "model_optimizer_setup_s": model_optimizer_setup_s,
            "epochs": epoch_times,
            "best_test_inference_s": best_test_s,
            "best_test_metrics_s": best_metric_s,
            "best_dpd_inference_s": best_dpd_inference_s,
            "artifact_serialization_s": artifact_serialization_s,
            "total_s": total_s,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
        },
        "trajectory": trajectory,
        "best": {
            "epoch": best_epoch,
            "test_loss": best_test_loss,
            "test": best_metrics,
        },
        "artifacts": {
            "orders": {"path": order_path.name, "sha256": _sha256_file(order_path)},
            "outputs": {"path": output_path.name, "sha256": _sha256_file(output_path)},
            "checkpoint": {
                "path": checkpoint_path.name,
                "sha256": _sha256_file(checkpoint_path),
            },
        },
    }
    _write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, indent=2, default=_json_default))


def _state_error(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]):
    names = sorted(reference)
    if names != sorted(candidate):
        raise ValueError("checkpoint parameter names differ")
    per_parameter: dict[str, dict[str, float]] = {}
    for name in names:
        per_parameter[name] = _prediction_error(
            reference[name].detach().cpu().numpy(),
            candidate[name].detach().cpu().numpy(),
        )
    return {
        "max_abs": max(item["max_abs"] for item in per_parameter.values()),
        "max_relative_l2": max(
            item["relative_l2"] for item in per_parameter.values()
        ),
        "per_parameter": per_parameter,
    }


def compare(args: argparse.Namespace) -> None:
    reference_result = json.loads((args.reference_dir / "result.json").read_text())
    candidate_result = json.loads((args.candidate_dir / "result.json").read_text())
    if reference_result["fixture"]["sha256"] != candidate_result["fixture"]["sha256"]:
        raise ValueError("runs did not use the same fixture")
    reference_orders = np.load(args.reference_dir / "batch_orders.npz")
    candidate_orders = np.load(args.candidate_dir / "batch_orders.npz")
    if reference_orders.files != candidate_orders.files or any(
        not np.array_equal(reference_orders[name], candidate_orders[name])
        for name in reference_orders.files
    ):
        raise ValueError("runs did not use identical batch permutations")

    reference_outputs = np.load(args.reference_dir / "outputs.npz")
    candidate_outputs = np.load(args.candidate_dir / "outputs.npz")
    output_errors = {
        name: _prediction_error(reference_outputs[name], candidate_outputs[name])
        for name in ("cascade", "dpd", "ground_truth")
    }
    reference_state = torch.load(
        args.reference_dir / "best_dpd.pt", map_location="cpu", weights_only=True
    )
    candidate_state = torch.load(
        args.candidate_dir / "best_dpd.pt", map_location="cpu", weights_only=True
    )
    state_errors = _state_error(reference_state, candidate_state)

    reference_train_s = sum(
        epoch["train_s"] for epoch in reference_result["timing"]["epochs"]
    )
    candidate_train_s = sum(
        epoch["train_s"] for epoch in candidate_result["timing"]["epochs"]
    )
    metric_deltas = {
        name: (
            candidate_result["best"]["test"][name]
            - reference_result["best"]["test"][name]
        )
        for name in METRIC_NAMES
    }
    trajectory_deltas = []
    for reference_epoch, candidate_epoch in zip(
        reference_result["trajectory"], candidate_result["trajectory"], strict=True
    ):
        trajectory_deltas.append(
            {
                "epoch": reference_epoch["epoch"],
                "train_loss": candidate_epoch["train_loss"] - reference_epoch["train_loss"],
                "val": {
                    name: candidate_epoch["val"][name] - reference_epoch["val"][name]
                    for name in METRIC_NAMES
                },
                "test": {
                    name: candidate_epoch["test"][name] - reference_epoch["test"][name]
                    for name in METRIC_NAMES
                },
            }
        )

    limits = {
        "prediction_relative_l2": args.prediction_relative_l2_limit,
        "metric_abs_db": args.metric_abs_db_limit,
    }
    checks = {
        "same_fixture": True,
        "same_batch_orders": True,
        "cascade_prediction_relative_l2": (
            output_errors["cascade"]["relative_l2"]
            <= args.prediction_relative_l2_limit
        ),
        "dpd_prediction_relative_l2": (
            output_errors["dpd"]["relative_l2"]
            <= args.prediction_relative_l2_limit
        ),
        "all_metric_deltas": all(
            abs(delta) <= args.metric_abs_db_limit for delta in metric_deltas.values()
        ),
    }
    comparison = {
        "format_version": 1,
        "reference_dir": str(args.reference_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "speed": {
            "total_reference_s": reference_result["timing"]["total_s"],
            "total_candidate_s": candidate_result["timing"]["total_s"],
            "total_speedup": (
                reference_result["timing"]["total_s"]
                / candidate_result["timing"]["total_s"]
            ),
            "train_reference_s": reference_train_s,
            "train_candidate_s": candidate_train_s,
            "train_speedup": reference_train_s / candidate_train_s,
        },
        "best_epoch": {
            "reference": reference_result["best"]["epoch"],
            "candidate": candidate_result["best"]["epoch"],
        },
        "metric_deltas_db": metric_deltas,
        "output_errors": output_errors,
        "checkpoint_errors": state_errors,
        "trajectory_deltas": trajectory_deltas,
        "acceptance_limits": limits,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _write_json(args.json_out, comparison)
    print(json.dumps(comparison, indent=2, default=_json_default))
    if args.fail_on_mismatch and not comparison["passed"]:
        raise SystemExit(1)


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_ROOT)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--dataset-name", default="DPA_200MHz")
    parser.add_argument("--frame-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=DEFAULT_DEVICE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="create immutable model fixture")
    _common_config(prepare_parser)
    prepare_parser.add_argument("--pa-backbone", default="dgru")
    prepare_parser.add_argument("--pa-hidden-size", type=int, default=23)
    prepare_parser.add_argument("--dpd-backbone", default="tres_deltagru")
    prepare_parser.add_argument("--dpd-hidden-size", type=int, default=15)
    prepare_parser.add_argument("--thx", type=float, default=0.01)
    prepare_parser.add_argument("--thh", type=float, default=0.05)
    prepare_parser.add_argument("--pa-pretrain-epochs", type=int, default=20)
    prepare_parser.add_argument("--pa-frame-stride", type=int, default=16)
    prepare_parser.add_argument("--pa-batch-size", type=int, default=256)
    prepare_parser.add_argument("--pa-lr", type=float, default=5e-4)
    prepare_parser.add_argument("--grad-clip", type=float, default=200.0)

    run_parser = subparsers.add_parser("run", help="run one implementation")
    _common_config(run_parser)
    run_parser.add_argument(
        "--implementation",
        choices=(
            "original", "eager_optimized", "fused_optimized",
            "graph_optimized", "optimized",
        ),
        required=True,
    )
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--frame-stride", type=int, default=16)
    run_parser.add_argument("--batch-size", type=int, default=64)
    run_parser.add_argument("--eval-batch-size", type=int, default=256)
    run_parser.add_argument("--epochs", type=int, default=3)
    run_parser.add_argument("--lr", type=float, default=5e-3)
    run_parser.add_argument("--lr-schedule", action="store_true")
    run_parser.add_argument("--lr-end", type=float, default=1e-4)
    run_parser.add_argument("--decay-factor", type=float, default=0.5)
    run_parser.add_argument("--patience", type=int, default=10)
    run_parser.add_argument("--grad-clip", type=float, default=200.0)
    run_parser.add_argument("--progress", action="store_true")
    run_parser.add_argument("--cuda-graph-training", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="compare two run directories")
    compare_parser.add_argument("--reference-dir", type=Path, required=True)
    compare_parser.add_argument("--candidate-dir", type=Path, required=True)
    compare_parser.add_argument("--json-out", type=Path, required=True)
    compare_parser.add_argument("--prediction-relative-l2-limit", type=float, default=1e-4)
    compare_parser.add_argument("--metric-abs-db-limit", type=float, default=0.05)
    compare_parser.add_argument("--fail-on-mismatch", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
