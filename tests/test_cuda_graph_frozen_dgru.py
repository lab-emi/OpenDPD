"""Correctness and fallback tests for frozen-DGRU CUDA-graph replay."""

import copy

import pytest
import torch

from backbones.cuda_graph_frozen_dgru import (
    clear_cuda_graph_frozen_dgru_cache,
)
from backbones.dgru import DGRU


requires_nvidia_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or getattr(torch.version, "hip", None) is not None,
    reason="requires NVIDIA CUDA",
)


def _build_frozen(device="cpu"):
    model = DGRU(hidden_size=23, output_size=2, num_layers=1)
    model = model.to(device).train()
    model.reset_parameters()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _is_graph_replay(output):
    return "FrozenDGRUReplayFunction" in type(output.grad_fn).__name__


def test_cpu_fallback_preserves_module_and_state_dict(monkeypatch):
    clear_cuda_graph_frozen_dgru_cache()
    monkeypatch.delenv("OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU", raising=False)
    model = _build_frozen()
    state_keys = tuple(model.state_dict())
    module_names = tuple(name for name, _ in model.named_modules())
    features = torch.randn(2, 7, 2, requires_grad=True)
    initial_state = torch.zeros(1, 2, 23)

    output = model(features, initial_state)
    output.square().mean().backward()

    assert not _is_graph_replay(output)
    assert tuple(model.state_dict()) == state_keys
    assert tuple(name for name, _ in model.named_modules()) == module_names
    assert torch.isfinite(features.grad).all()


def test_trainable_dgru_uses_eager_fallback():
    model = DGRU(hidden_size=5, output_size=2, num_layers=1).train()
    features = torch.randn(2, 4, 2, requires_grad=True)
    output = model(features, torch.zeros(1, 2, 5))
    assert not _is_graph_replay(output)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@requires_nvidia_cuda
def test_cuda_replay_is_bitwise_exact_and_state_dict_stable(monkeypatch):
    torch.manual_seed(123)
    clear_cuda_graph_frozen_dgru_cache()
    model = _build_frozen("cuda")
    state_before = copy.deepcopy(model.state_dict())
    module_names = tuple(name for name, _ in model.named_modules())
    initial_state = torch.randn(1, 4, 23, device="cuda") * 0.1

    for _ in range(2):
        features = torch.randn(4, 37, 2, device="cuda")
        output_gradient = torch.randn(4, 37, 2, device="cuda")
        monkeypatch.setenv("OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU", "1")
        reference_input = features.clone().requires_grad_()
        reference_output = model(reference_input, initial_state)
        reference_output.backward(output_gradient)

        monkeypatch.delenv(
            "OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU", raising=False
        )
        replay_input = features.clone().requires_grad_()
        replay_output = model(replay_input, initial_state)
        replay_output.backward(output_gradient)
        torch.cuda.synchronize()

        assert _is_graph_replay(replay_output)
        assert torch.equal(replay_output, reference_output)
        assert torch.equal(replay_input.grad, reference_input.grad)

    assert tuple(model.state_dict()) == tuple(state_before)
    assert tuple(name for name, _ in model.named_modules()) == module_names
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])


@requires_nvidia_cuda
def test_busy_replay_and_opt_out_fall_back_safely(monkeypatch):
    torch.manual_seed(321)
    clear_cuda_graph_frozen_dgru_cache()
    monkeypatch.delenv("OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU", raising=False)
    model = _build_frozen("cuda")
    initial_state = torch.zeros(1, 2, 23, device="cuda")

    first_input = torch.randn(2, 17, 2, device="cuda", requires_grad=True)
    first_output = model(first_input, initial_state)
    assert _is_graph_replay(first_output)

    # The first graph still owns its training reserve until backward.  A
    # second in-flight call must use independent eager buffers.
    second_input = torch.randn(2, 17, 2, device="cuda", requires_grad=True)
    second_output = model(second_input, initial_state)
    assert not _is_graph_replay(second_output)
    second_output.sum().backward()
    first_output.sum().backward()

    monkeypatch.setenv("OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU", "true")
    opt_out_input = torch.randn(2, 17, 2, device="cuda", requires_grad=True)
    opt_out_output = model(opt_out_input, initial_state)
    assert not _is_graph_replay(opt_out_output)
    opt_out_output.sum().backward()
    assert torch.isfinite(first_input.grad).all()
    assert torch.isfinite(second_input.grad).all()
    assert torch.isfinite(opt_out_input.grad).all()


@requires_nvidia_cuda
def test_autocast_and_eval_use_eager_fallback(monkeypatch):
    clear_cuda_graph_frozen_dgru_cache()
    monkeypatch.delenv("OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU", raising=False)
    model = _build_frozen("cuda")
    initial_state = torch.zeros(1, 2, 23, device="cuda")
    features = torch.randn(2, 7, 2, device="cuda", requires_grad=True)

    with torch.autocast("cuda", dtype=torch.float16):
        autocast_output = model(features, initial_state)
    assert not _is_graph_replay(autocast_output)

    noncontiguous = torch.randn(
        2, 2, 7, device="cuda", requires_grad=True
    ).transpose(1, 2)
    noncontiguous_output = model(noncontiguous, initial_state)
    assert not _is_graph_replay(noncontiguous_output)

    with torch.no_grad():
        no_grad_output = model(features, initial_state)
    assert not no_grad_output.requires_grad

    model.eval()
    eval_output = model(features, initial_state)
    assert not _is_graph_replay(eval_output)


@requires_nvidia_cuda
def test_higher_order_gradient_fails_loudly(monkeypatch):
    clear_cuda_graph_frozen_dgru_cache()
    monkeypatch.delenv("OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU", raising=False)
    model = _build_frozen("cuda")
    features = torch.randn(1, 3, 2, device="cuda", requires_grad=True)
    initial_state = torch.zeros(1, 1, 23, device="cuda")

    with pytest.raises(RuntimeError, match="first-order training only"):
        torch.autograd.grad(
            model(features, initial_state).sum(),
            features,
            create_graph=True,
        )
