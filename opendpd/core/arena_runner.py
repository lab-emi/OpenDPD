"""Isolated Arena evaluation: parameter sweep and one frozen-PA cascade.

Only registered native models and validated data-only templates are accepted.
The request is written by the server, never an uploaded Python program. No
host is timed: cost is the analytic operation count in arena_ops.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback

import numpy as np
import torch

from opendpd.core import arena
# Re-exported for the auditor and tests, which address the evaluator by this module.
from opendpd.core.arena_engine import (  # noqa: F401
    ArenaCascade,
    Condition,
    _asset,
    _load_weights,
    _save_weights,
    build_model,
    calibrate_linear_baseline,
    dpd_output,
    fit_case,
    fit_polynomial,
    judge_case,
    limit_array,
    limit_tensor,
    load_frozen,
    offline_output,
    pa_output,
    parameter_count,
    signal_metrics,
    train_gradient,
)
from opendpd.core.registry import get_model
from opendpd.schemas.arena import ArenaRow
from opendpd.services.workspace import read_json, write_json_atomic


class TrainingCondition:
    """A fitting context with no test arrays or test-evaluation methods.

    The training data, frozen PA and engine are unchanged. Loading only these
    four arrays makes the orchestration boundary explicit without changing
    the checkpoint's training fingerprint.
    """

    def __init__(self, identifier, device):
        self.identifier, self.device = identifier, device
        self.manifest = arena.calibration()[identifier]
        m = self.manifest
        with np.load(_asset(m["data_file"], m["data_sha256"]), allow_pickle=False) as data:
            self.data = {key: np.asarray(data[key], dtype=np.float32)
                         for key in ("x_train", "y_train", "x_val", "y_val")}
        self.peak = float(m["peak_limit"])
        self.gain = float(m["reference_gain"])
        self.teacher = load_frozen(m["teacher"], device)


def runtime_environment(device):
    """Provenance only. Arena scores do not depend on the host that produced them."""
    return dict(
        processor=platform.processor() or platform.machine(),
        architecture=platform.machine(),
        os=platform.system(),
        torch=torch.__version__,
        numpy=np.__version__,
        device=device,
        precision="FP32, TF32 disabled; LS complex128",
    )


def default_device(base_key, device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def configure_runtime(base_key=None):
    # A 200-step complex recurrent backward can exceed Inductor's default
    # Python recursion limit during topological sorting. This is compiler
    # bookkeeping only; no model, optimizer or data setting changes.
    sys.setrecursionlimit(max(sys.getrecursionlimit(),20000))
    torch.set_num_threads(arena.TRAINING["threads"])
    if base_key == "apnrru":
        # Bound the size of generated kernels for the complex 200-step cell.
        # This execution profile passed full-batch/tail/zero-IQ forward,
        # gradient and AdamW checks at every supported Arena parameter budget.
        import torch._inductor.config as compiler_config
        compiler_config.max_fusion_size = 8


def cache_folder(cache, base_key, parameters, condition_id, seed):
    model_hash = arena.canonical_hash(
        dict(key=base_key, parameters=parameters, training=arena.training_fingerprint())
    )
    return Path(cache) / model_hash / condition_id / f"seed-{seed}"


def prefetch(key, budget, condition_id, cache, *, device=None):
    """Train or fit one sweep unit into the cache; judging and scoring come later."""
    base_key = get_model(key).weights_from or key
    parameters = arena.model_parameters(key, budget)
    if parameters is None:
        return 0
    device = default_device(base_key, device)
    configure_runtime(base_key)
    seeds = arena.SEEDS[:1] if base_key in arena.DETERMINISTIC else arena.SEEDS
    from opendpd.core.arena_cuda import ARENA_REPLAY_BACKBONES
    parallel = (device == "cuda" and base_key in ARENA_REPLAY_BACKBONES
                and os.getenv("OPENDPD_ARENA_PARALLEL_SEEDS", "0") == "1")
    condition = TrainingCondition(condition_id, device)
    pending = []
    for seed in seeds:
        folder = cache_folder(cache, base_key, parameters, condition_id, seed)
        if parallel and not ((folder / "training.json").exists() and (folder / "weights.npz").exists()):
            pending.append((seed, folder))
        else:
            fit_case(condition, key, parameters, seed, folder, lambda *_: None)
    if pending:
        # Each seed owns its PA, model, optimizer, shuffle generator and stream.
        # Initialize and capture every graph before allowing any updates. This
        # keeps CUDA capture and torch.compile autotuning away from live graphs.
        contexts = [condition] + [TrainingCondition(condition_id, device) for _ in pending[1:]]
        streams = [torch.cuda.Stream() for _ in pending]
        torch.cuda.synchronize()
        barrier, initialization = threading.Barrier(len(pending)), threading.Lock()

        def train(item):
            (seed, folder), context, stream = item
            owned = False
            def ready():
                nonlocal owned
                stream.synchronize()
                initialization.release()
                owned = False
                barrier.wait()
            try:
                initialization.acquire()
                owned = True
                with torch.cuda.stream(stream):
                    fit_case(context, key, parameters, seed, folder, lambda *_: None, ready=ready)
                stream.synchronize()
            except BaseException:
                barrier.abort()
                raise
            finally:
                if owned:
                    initialization.release()

        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            list(pool.map(train, zip(pending, contexts, streams)))
    return len(seeds)


def evaluate_request(request, output, cache=None, *, device=None, cached_only=False):
    start_time = time.monotonic()
    current = arena.protocol()
    if request["protocol_sha256"] != current.protocol_sha256:
        raise ValueError("Arena request protocol is stale")
    selected = arena.board(request["board_id"])
    key = request["backbone"]
    descriptor = get_model(key)
    entry = next((item for item in arena.bundled_backbones() if item.key == key), None)
    if entry is None:
        raise ValueError("Backbone is excluded from the Arena catalogue")
    display_name = entry.display_name
    if "dpd" not in descriptor.roles:
        raise ValueError("Backbone does not support DPD")
    supplied = request.get("model_parameters") or {}
    if supplied and (key != "user_template" or set(supplied) != {"definition"}):
        raise ValueError("Arena accepts only server-verified template definitions")
    points = arena.sweep(key, supplied or None)
    base_key = descriptor.weights_from or key
    seeds = arena.SEEDS[:1] if key in arena.DETERMINISTIC else arena.SEEDS
    cases = []
    available = [point for point in points if point["model_parameters"] is not None]
    expected = len(available) * len(selected.conditions) * len(seeds)
    cache = Path(cache or output.parent / "evaluation")
    cache.mkdir(parents=True, exist_ok=True)
    device = default_device(base_key, device)
    configure_runtime(base_key)
    semantics = (
        "streaming_stateful" if descriptor.weights_from else "offline_overlap_200_100"
    )

    def emit(phase, epoch, message):
        write_json_atomic(
            output.with_suffix(".progress.json"),
            dict(
                phase=phase,
                epoch=epoch,
                epochs=arena.TRAINING["epochs"],
                completed_cases=len(cases),
                expected_cases=expected,
                message=message,
            ),
        )

    common = dict(
        entry_id="arena-local-result",
        board_id=selected.board_id,
        backbone=key,
        display_name=display_name,
        origin="workspace",
        protocol_sha256=current.protocol_sha256,
        evidence_type=selected.evidence_type,
        execution_semantics=semantics,
    )
    try:
        if not available:
            raise ValueError("This backbone has no configuration inside the Arena parameter sweep")
        # Complete every fit/checkpoint selection before any test input is loaded.
        # At most four small configurations per seed are retained on CPU.
        frozen = []
        for identifier in selected.conditions:
            emit("preparing", 0, f"Verifying frozen PA and training/validation data: {identifier}")
            condition = TrainingCondition(identifier, device)
            for point in available:
                budget, parameters = point["budget"], point["model_parameters"]
                for seed in seeds:
                    label = f"{identifier} · ≤{budget} parameters · seed {seed}"
                    folder = cache_folder(cache, base_key, parameters, identifier, seed)
                    if cached_only and not all((folder/name).is_file() for name in ('weights.npz','training.json')):
                        raise ValueError('Official test evaluation requires every frozen checkpoint; training is disabled')
                    model, info = fit_case(
                        condition,
                        key,
                        parameters,
                        seed,
                        folder,
                        lambda phase, epoch, _, label=label: emit(
                            phase, epoch, f"{label} · epoch {epoch}/{arena.TRAINING['epochs']}"
                            if phase == "training" else f"{label} · deterministic fit completed"
                        ),
                    )
                    model.to("cpu").eval().requires_grad_(False)
                    frozen.append((identifier, budget, seed, model, info, folder,
                                   arena.file_hash(folder / "weights.npz")))
            del condition
        for identifier in selected.conditions:
            condition = Condition(identifier, device)
            for condition_id, budget, seed, model, info, folder, checkpoint_sha in frozen:
                if condition_id != identifier:
                    continue
                if arena.file_hash(folder / "weights.npz") != checkpoint_sha:
                    raise ValueError("Frozen checkpoint changed before test evaluation")
                model.to(device)
                label = f"{identifier} · ≤{budget} parameters · seed {seed}"
                emit("evaluating", arena.TRAINING["epochs"], f"Frozen PA cascade test: {label}")
                cases.append(
                    dict(
                        budget=budget,
                        condition_id=identifier,
                        seed=seed,
                        parameters=parameter_count(model),
                        **judge_case(condition, model, key),
                        **info,
                        evaluation_split="test",
                        evaluation_device=condition.device,
                        checkpoint_sha256=checkpoint_sha,
                    )
                )
                write_json_atomic(output.parent / "completed-cases.json", cases)
                model.to("cpu")
            del condition
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row = ArenaRow.model_validate(
            dict(
                **common,
                status="succeeded",
                cases=cases,
                provenance=dict(
                    protocol_id=current.protocol_id,
                    training_sha256=current.training_sha256,
                    model_parameters=supplied,
                    environment=runtime_environment(device),
                    elapsed_seconds=time.monotonic() - start_time,
                    model_provenance=request.get("model_provenance", {}),
                    data_access=dict(training_splits=["train", "val"], evaluation_split="test",
                                     all_checkpoints_frozen_before_test=True),
                    note="Public reproducible reference tests; no hidden-holdout or measured hardware claim.",
                ),
                **arena.summarize_cases(key, selected.board_id, cases, points),
            )
        )
    except Exception as exc:
        traceback.print_exc()
        row = ArenaRow.model_validate(
            dict(
                **common,
                status="failed",
                eligible=False,
                cases=cases,
                seeds=seeds,
                budgets=[dict(point, available=point["model_parameters"] is not None) for point in points],
                available_budgets=len(available),
                expected_cases=expected,
                completed_cases=len(cases),
                error=f"{type(exc).__name__}: {exc}"[:1000],
                provenance=dict(
                    training_sha256=current.training_sha256,
                    model_parameters=supplied,
                    environment=runtime_environment(device),
                    elapsed_seconds=time.monotonic() - start_time,
                ),
            )
        )
    write_json_atomic(output, row)
    emit(
        "complete",
        arena.TRAINING["epochs"],
        "Evaluation complete" if row.status == "succeeded" else row.error,
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--cached-only", action="store_true", help="Refuse training during the official test phase")
    parser.add_argument("--prefetch", nargs=3, metavar=("BACKBONE", "BUDGET", "CONDITION"),
                        help="Only train or fit this sweep unit into --cache")
    args = parser.parse_args()
    if args.prefetch:
        if args.cache is None:
            parser.error("--prefetch requires --cache")
        key, budget, condition_id = args.prefetch
        prefetch(key, int(budget), condition_id, args.cache, device=args.device)
        return
    if args.request is None or args.output is None:
        parser.error("--request and --output are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate_request(
        read_json(args.request), args.output, args.cache, device=args.device,
        **({"cached_only": True} if args.cached_only else {}),
    )


if __name__ == "__main__":
    main()
