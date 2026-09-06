"""Bridge from a resolved experiment config to the legacy trainer.

The legacy pipeline (``project.Project`` + ``steps/*``) reads an
``argparse.Namespace`` and writes ``save/``, ``log/`` and ``dpd_out/``
relative to the current directory. This adapter builds that namespace
explicitly (never from ``sys.argv``), and ``run_in_directory`` confines the
relative paths to one run directory. Nothing here changes numerics.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from opendpd.core.registry import get_model
from opendpd.schemas import ResolvedExperimentConfig, TaskType
from opendpd.services.workspace import WorkspaceError


def build_namespace(resolved: ResolvedExperimentConfig, *, dataset_dir: Path,
                    dataset_name: str) -> argparse.Namespace:
    """Legacy namespace equivalent to ``resolved``.

    ``dataset_name`` is only used for the ``save/<name>/...`` layout; data is
    always read from ``dataset_dir`` (which must contain ``spec.json``).
    """
    from arguments import build_parser

    ns = build_parser().parse_args([])           # single source of legacy defaults
    ns.step = resolved.task.value
    ns.dataset_name = dataset_name
    ns.dataset_path = str(dataset_dir)
    ns.plot = False

    t = resolved.training
    ns.n_epochs = t.epochs
    ns.batch_size = t.batch_size
    ns.batch_size_eval = t.batch_size_eval
    ns.lr = t.learning_rate
    ns.lr_end = t.lr_end
    ns.lr_schedule = int(t.lr_schedule)
    ns.decay_factor = t.decay_factor
    ns.patience = t.patience
    ns.opt_type = t.optimizer
    ns.loss_type = t.loss
    ns.grad_clip_val = t.grad_clip
    ns.frame_length = t.frame_length
    ns.frame_stride = t.frame_stride
    ns.seed = t.seed
    ns.re_level = t.reproducibility
    ns.eval_val = int(t.eval_val)
    ns.eval_test = int(t.eval_test)

    e = resolved.execution
    ns.accelerator = e.device
    ns.devices = e.device_index
    ns.cuda_graph_training = e.cuda_graph_training

    role = "pa" if resolved.task in (TaskType.train_pa, TaskType.evaluate_pa) else "dpd"
    _apply_model(ns, resolved.model.key, resolved.model.parameters, role)
    if role == "dpd":
        assert resolved.pa_reference is not None and resolved.pa_reference.model is not None
        pa = resolved.pa_reference.model
        _apply_model(ns, pa.key, pa.parameters, "pa", skip_shared=True)

    q = resolved.quantization
    if q is not None and q.enabled:
        ns.quant = True
        ns.n_bits_w = q.n_bits_w
        ns.n_bits_a = q.n_bits_a
        ns.quant_dir_label = q.label
    return ns


def _apply_model(ns: argparse.Namespace, key: str, parameters: Dict, role: str, skip_shared: bool = False) -> None:
    model = get_model(key)
    setattr(ns, "PA_backbone" if role == "pa" else "DPD_backbone", model.legacy_backbone)
    for spec in model.params:
        target = spec.legacy_arg.get(role)
        if target is None:
            continue
        shared = spec.legacy_arg.get("pa") == spec.legacy_arg.get("dpd")
        if skip_shared and shared:
            continue   # thx/thh/num_dvr_units are global in the legacy parser; the primary model wins
        setattr(ns, target, parameters.get(spec.name, spec.default))


def legacy_cli_tokens(ns: argparse.Namespace) -> List[str]:
    """``python main.py`` tokens that reproduce ``ns`` (for exports and tests)."""
    from arguments import build_parser

    defaults = vars(build_parser().parse_args([]))
    tokens: List[str] = []
    for key, value in vars(ns).items():
        if key not in defaults or value == defaults[key]:
            continue
        if isinstance(value, bool):
            if value:
                tokens.append(f"--{key}")
            continue
        tokens.extend([f"--{key}", str(value)])
    return tokens


def legacy_command_line(ns: argparse.Namespace) -> str:
    return "python main.py " + " ".join(shlex.quote(t) for t in legacy_cli_tokens(ns))


@contextlib.contextmanager
def run_in_directory(run_dir: Path) -> Iterator[Path]:
    """Confine the legacy relative paths (save/, log/, dpd_out/) to ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    os.chdir(run_dir)
    try:
        yield run_dir
    finally:
        os.chdir(previous)


class RunCancelled(Exception):
    """Raised inside the training loop at a safe stop point (end of an epoch)."""


def attach_epoch_hooks(project, on_epoch: Callable[[dict], None],
                       should_cancel: Optional[Callable[[], bool]] = None) -> None:
    """Observe the legacy logger without changing what it writes.

    ``on_epoch`` receives the per-epoch log dict after it has been written to
    the CSV history; ``should_cancel`` is polled at the same safe point.
    """
    original_build_logger = project.build_logger

    def build_logger(model_id: str):
        original_build_logger(model_id)
        logger = project.logger
        original_write_log = logger.write_log

        def write_log(log_stat):
            original_write_log(log_stat)
            on_epoch(dict(log_stat))
            if should_cancel is not None and should_cancel():
                raise RunCancelled(f"cancel requested after epoch {log_stat.get('EPOCH')}")

        logger.write_log = write_log

    project.build_logger = build_logger


def run_step(ns: argparse.Namespace, on_epoch: Optional[Callable[[dict], None]] = None,
             should_cancel: Optional[Callable[[], bool]] = None):
    """Execute one legacy step in the current directory and return the Project."""
    from project import Project

    project = Project(args=ns)
    if on_epoch is not None or should_cancel is not None:
        attach_epoch_hooks(project, on_epoch or (lambda _: None), should_cancel)
    if ns.step == "train_pa":
        from steps import train_pa as step
    elif ns.step == "train_dpd":
        from steps import train_dpd as step
    elif ns.step == "run_dpd":
        from steps import run_dpd as step
    else:
        raise ValueError(f"unsupported step {ns.step}")
    step.main(project)
    return project


def load_checkpoint(path):
    """Tensors and plain containers only (``weights_only=True``): a checkpoint never runs code.

    Legacy checkpoints written by ``main.py`` are plain ``state_dict`` files and load under the same
    restriction, so there is no trusted or unrestricted loading path anywhere in OpenDPD.
    """
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as err:  # noqa: BLE001 - UnpicklingError, RuntimeError or zip errors from the restricted loader
        first = (str(err).splitlines() or [""])[0][:240]
        raise CheckpointRefused(f"checkpoint {Path(path).name} could not be loaded as a plain state_dict and was not "
                                f"executed ({type(err).__name__}: {first}); OpenDPD never unpickles arbitrary objects "
                                "from checkpoints") from None


class CheckpointRefused(WorkspaceError):
    """The restricted loader rejected a checkpoint (not a plain tensor container)."""
