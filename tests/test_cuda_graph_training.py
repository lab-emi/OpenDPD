"""Tests for guarded whole-cascade CUDA-graph training."""

import contextlib
import copy
import io

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import modules.cuda_graph_training as graph_training
from backbones.cuda_graph_frozen_dgru import disable_cuda_graph_frozen_dgru
from models import CascadedModel, CoreModel
from modules.cuda_graph_training import (
    clear_cuda_graph_training_cache,
    force_clean_eager_tres,
    try_cuda_graph_training_step,
)
from modules.train_funcs import net_train


requires_nvidia_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or getattr(torch.version, "hip", None) is not None,
    reason="requires NVIDIA CUDA",
)


def _build_cascade(device="cpu"):
    with contextlib.redirect_stdout(io.StringIO()):
        dpd = CoreModel(
            2, 15, 1, "tres_deltagru", thx=0.01, thh=0.05
        )
        pa = CoreModel(2, 23, 1, "dgru")
    cascade = CascadedModel(dpd, pa)
    cascade.freeze_pa_model()
    return cascade.to(device).train()


def _trainable(module):
    return tuple(
        parameter for parameter in module.parameters()
        if parameter.requires_grad
    )


def test_cpu_opt_in_falls_back_cleanly_and_restores_dispatch():
    torch.manual_seed(301)
    clear_cuda_graph_training_cache()
    reference = _build_cascade()
    candidate = copy.deepcopy(reference)
    state_keys = tuple(candidate.state_dict())
    module_names = tuple(name for name, _ in candidate.named_modules())
    features = torch.randn(4, 5, 2)
    targets = torch.randn(4, 5, 2)
    loader = DataLoader(
        TensorDataset(features, targets), batch_size=2, shuffle=False
    )
    reference_optimizer = torch.optim.AdamW(_trainable(reference), lr=5e-4)
    candidate_optimizer = torch.optim.AdamW(_trainable(candidate), lr=5e-4)
    criterion = torch.nn.MSELoss()

    reference_log = {}
    with force_clean_eager_tres(reference):
        net_train(
            reference_log, reference, loader, reference_optimizer,
            criterion, 200.0, torch.device("cpu")
        )
    candidate_log = {}
    original_dispatch = candidate.dpd_model.backbone.rnn.use_triton
    observed_dispatch = []
    hook = candidate.dpd_model.backbone.rnn.register_forward_pre_hook(
        lambda recurrent, _inputs: observed_dispatch.append(
            recurrent.use_triton
        )
    )
    net_train(
        candidate_log, candidate, loader, candidate_optimizer,
        criterion, 200.0, torch.device("cpu"),
        cuda_graph_training=True,
    )
    hook.remove()

    assert candidate_log == reference_log
    assert observed_dispatch and not any(observed_dispatch)
    assert candidate.dpd_model.backbone.rnn.use_triton is original_dispatch
    assert tuple(candidate.state_dict()) == state_keys
    assert tuple(name for name, _ in candidate.named_modules()) == module_names
    for expected, actual in zip(
        reference.parameters(), candidate.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)


def _snapshot(module, parameters, optimizer, output, loss):
    return {
        "output": output.detach().clone(),
        "loss": loss.detach().clone(),
        "gradients": [
            parameter.grad.detach().clone() for parameter in parameters
        ],
        "parameters": [parameter.detach().clone() for parameter in parameters],
        "optimizer": [
            {
                key: (
                    value.detach().clone()
                    if torch.is_tensor(value) else value
                )
                for key, value in optimizer.state[parameter].items()
            }
            for parameter in parameters
        ],
        "state_keys": tuple(module.state_dict()),
    }


def _run_eager_trajectory(module, batches, targets):
    parameters = _trainable(module)
    optimizer = torch.optim.AdamW(parameters, lr=5e-4)
    criterion = torch.nn.MSELoss()
    output = loss = None
    for features, target in zip(batches, targets, strict=True):
        optimizer.zero_grad(set_to_none=True)
        with force_clean_eager_tres(module), disable_cuda_graph_frozen_dgru():
            output = module(features)
            loss = criterion(output, target)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 200.0)
        optimizer.step()
    return _snapshot(module, parameters, optimizer, output, loss)


def _run_graph_trajectory(module, batches, targets):
    parameters = _trainable(module)
    optimizer = torch.optim.AdamW(parameters, lr=5e-4)
    criterion = torch.nn.MSELoss()
    output = loss = None
    original_dispatch = module.dpd_model.backbone.rnn.use_triton
    for features, target in zip(batches, targets, strict=True):
        result = try_cuda_graph_training_step(
            module, features, target, criterion, parameters
        )
        assert result is not None
        assert module.dpd_model.backbone.rnn.use_triton is original_dispatch
        output = result.output.clone()
        loss = result.loss
        torch.nn.utils.clip_grad_norm_(parameters, 200.0)
        optimizer.step()
    return _snapshot(module, parameters, optimizer, output, loss)


def _max_abs(left, right):
    return max(
        (first - second).abs().max().item()
        for first, second in zip(left, right, strict=True)
    )


def _optimizer_max_abs(left, right):
    differences = []
    for first, second in zip(left, right, strict=True):
        assert first.keys() == second.keys()
        for key in first:
            if torch.is_tensor(first[key]):
                differences.append(
                    (first[key] - second[key]).abs().max().item()
                )
            else:
                assert first[key] == second[key]
    return max(differences, default=0.0)


@requires_nvidia_cuda
def test_cuda_ten_step_drift_is_within_eager_repeat_envelope(monkeypatch):
    torch.manual_seed(401)
    clear_cuda_graph_training_cache()
    base = _build_cascade("cuda")
    eager_first = copy.deepcopy(base)
    eager_second = copy.deepcopy(base)
    graphed = copy.deepcopy(base)
    for module in (eager_first, eager_second, graphed):
        module.pa_model.backbone.rnn.flatten_parameters()
    generator = torch.Generator(device="cuda").manual_seed(402)
    batch_sizes = [4, 3] * 5
    batches = [
        torch.randn(size, 17, 2, device="cuda", generator=generator)
        for size in batch_sizes
    ]
    targets = [
        torch.randn(size, 17, 2, device="cuda", generator=generator)
        for size in batch_sizes
    ]

    first = _run_eager_trajectory(eager_first, batches, targets)
    second = _run_eager_trajectory(eager_second, batches, targets)
    capture_count = 0
    original_init = graph_training._CapturedTrainingStep.__init__

    def counted_init(instance, *args, **kwargs):
        nonlocal capture_count
        capture_count += 1
        original_init(instance, *args, **kwargs)

    monkeypatch.setattr(
        graph_training._CapturedTrainingStep, "__init__", counted_init
    )
    candidate = _run_graph_trajectory(graphed, batches, targets)
    torch.cuda.synchronize()

    assert candidate["state_keys"] == first["state_keys"]
    assert capture_count == 2
    comparisons = (
        (
            (first["output"] - candidate["output"]).abs().max().item(),
            (first["output"] - second["output"]).abs().max().item(),
        ),
        (
            abs((first["loss"] - candidate["loss"]).item()),
            abs((first["loss"] - second["loss"]).item()),
        ),
        (
            _max_abs(first["gradients"], candidate["gradients"]),
            _max_abs(first["gradients"], second["gradients"]),
        ),
        (
            _max_abs(first["parameters"], candidate["parameters"]),
            _max_abs(first["parameters"], second["parameters"]),
        ),
        (
            _optimizer_max_abs(first["optimizer"], candidate["optimizer"]),
            _optimizer_max_abs(first["optimizer"], second["optimizer"]),
        ),
    )
    for candidate_error, eager_repeat_error in comparisons:
        assert candidate_error <= eager_repeat_error
