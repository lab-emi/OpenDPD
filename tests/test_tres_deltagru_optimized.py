"""Correctness tests for the fused TRes-DeltaGRU recurrence."""

import contextlib
import copy
import io
from types import SimpleNamespace

import pytest
import torch

from models import CoreModel
from backbones.tres_deltagru import DeltaGRULayer
from backbones.triton_deltagru import triton
from modules.paths import gen_log_stat


requires_fused_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="requires CUDA and Triton",
)


def build_model(device, *, fused, thx=0.01, thh=0.05):
    with contextlib.redirect_stdout(io.StringIO()):
        model = CoreModel(
            input_size=2,
            hidden_size=15,
            num_layers=1,
            backbone_type="tres_deltagru",
            thx=thx,
            thh=thh,
        ).to(device)
    model.backbone.rnn.use_triton = fused
    model.backbone.set_debug(0)
    return model


def test_debug_setting_reaches_recurrent_layer():
    model = build_model("cpu", fused=False)
    model.backbone.set_debug(1)
    assert model.backbone.debug == 1
    assert model.backbone.rnn.debug == 1
    model.backbone.set_debug(0)
    assert model.backbone.debug == 0
    assert model.backbone.rnn.debug == 0


def test_h15_matches_the_approximately_1000_parameter_dpd_budget():
    model = build_model("cpu", fused=False, thx=0.0, thh=0.0)

    assert sum(parameter.numel() for parameter in model.parameters()) == 999


def test_pa_delta_thresholds_are_recorded():
    model = build_model("cpu", fused=False, thx=0.0, thh=0.0)
    args = SimpleNamespace(
        step="train_pa",
        n_epochs=2,
        batch_size=2,
        frame_length=3,
        PA_backbone="tres_deltagru",
        PA_hidden_size=15,
    )

    log = gen_log_stat(args, 0.0, model, None, 0)

    assert log["THX"] == 0.0
    assert log["THH"] == 0.0


def test_opt_in_statistics_are_logged_and_reset():
    model = build_model("cpu", fused=False)
    model.backbone.set_debug(1)
    model(torch.randn(2, 3, 2))
    wrapper = SimpleNamespace(
        dpd_model=model,
        named_parameters=model.named_parameters,
    )
    args = SimpleNamespace(
        step="train_dpd",
        n_epochs=2,
        batch_size=2,
        frame_length=3,
        DPD_backbone="tres_deltagru",
        DPD_hidden_size=15,
    )

    log = gen_log_stat(args, 0.0, wrapper, None, 0)

    assert log["THX"] == 0.01
    assert log["THH"] == 0.05
    assert {"SP_T_DX", "SP_T_DH", "SP_T_DV", "HW_PARAM"} <= log.keys()
    assert model.backbone.rnn.statistics == {
        "num_dx_zeros": 0,
        "num_dx_numel": 0,
        "num_dh_zeros": 0,
        "num_dh_numel": 0,
    }


@requires_fused_cuda
@pytest.mark.parametrize("thresholds", [(0.0, 0.0), (0.01, 0.05)])
@pytest.mark.parametrize("sequence_length", [1, 17, 200])
def test_fused_output_and_gradients_match_eager(thresholds, sequence_length):
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    thx, thh = thresholds
    reference = build_model("cuda", fused=False, thx=thx, thh=thh)
    fused = build_model("cuda", fused=True, thx=thx, thh=thh)
    fused.load_state_dict(copy.deepcopy(reference.state_dict()))

    reference_input = torch.randn(
        2, sequence_length, 2, device="cuda", requires_grad=True
    )
    fused_input = reference_input.detach().clone().requires_grad_()
    target = torch.randn(2, sequence_length, 2, device="cuda")

    reference_output = reference(reference_input)
    reference_loss = torch.nn.functional.mse_loss(reference_output, target)
    reference_loss.backward()
    fused_output = fused(fused_input)
    fused_loss = torch.nn.functional.mse_loss(fused_output, target)
    fused_loss.backward()

    torch.testing.assert_close(fused_output, reference_output, rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(fused_input.grad, reference_input.grad, rtol=5e-4, atol=2e-6)
    for (name, reference_parameter), (fused_name, fused_parameter) in zip(
        reference.named_parameters(), fused.named_parameters(), strict=True
    ):
        assert name == fused_name
        torch.testing.assert_close(
            fused_parameter.grad,
            reference_parameter.grad,
            rtol=5e-4,
            atol=2e-6,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )


@requires_fused_cuda
def test_fused_statistics_match_eager():
    torch.manual_seed(11)
    reference = build_model("cuda", fused=False)
    fused = build_model("cuda", fused=True)
    fused.load_state_dict(copy.deepcopy(reference.state_dict()))
    reference.backbone.set_debug(1)
    fused.backbone.set_debug(1)
    features = torch.randn(4, 17, 2, device="cuda")

    reference(features)
    fused(features)

    for key in ("num_dx_zeros", "num_dx_numel", "num_dh_zeros", "num_dh_numel"):
        reference_value = torch.as_tensor(reference.backbone.rnn.statistics[key])
        fused_value = torch.as_tensor(fused.backbone.rnn.statistics[key])
        torch.testing.assert_close(fused_value.cpu(), reference_value.cpu())


@requires_fused_cuda
def test_custom_initial_state_gradients_match_eager():
    torch.manual_seed(13)
    reference = DeltaGRULayer(6, 15, 1, thx=0.01, thh=0.05).cuda()
    fused = copy.deepcopy(reference)
    reference.use_triton = False
    fused.use_triton = True
    reference.debug = 0
    fused.debug = 0

    reference_input = torch.randn(2, 17, 6, device="cuda", requires_grad=True)
    fused_input = reference_input.detach().clone().requires_grad_()

    def make_state(width):
        return (torch.randn(1, 2, width, device="cuda") * 0.1).requires_grad_()

    reference_states = [
        make_state(15),  # x_p_0 uses x_p_length=max(input_size, hidden_size)
        make_state(15),
        make_state(15),
        make_state(15),
        make_state(45),
    ]
    fused_states = [
        state.detach().clone().requires_grad_() for state in reference_states
    ]
    output_gradient = torch.randn(2, 17, 15, device="cuda")

    reference_output = reference(reference_input, *reference_states)
    (reference_output * output_gradient).sum().backward()
    fused_output = fused(fused_input, *fused_states)
    (fused_output * output_gradient).sum().backward()

    torch.testing.assert_close(fused_output, reference_output, rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(fused_input.grad, reference_input.grad, rtol=5e-4, atol=2e-6)
    for reference_state, fused_state in zip(
        reference_states, fused_states, strict=True
    ):
        torch.testing.assert_close(
            fused_state.grad, reference_state.grad, rtol=5e-4, atol=2e-6
        )


@requires_fused_cuda
def test_inference_does_not_save_training_workspace():
    model = build_model("cuda", fused=True).eval()
    features = torch.randn(4, 512, 2, device="cuda")
    with torch.inference_mode():
        output = model(features)
    assert output.shape == (4, 512, 2)
    assert not output.requires_grad
    assert torch.isfinite(output).all()


@requires_fused_cuda
def test_cuda_autocast_uses_equivalent_eager_fallback():
    torch.manual_seed(17)
    reference = build_model("cuda", fused=False)
    candidate = build_model("cuda", fused=True)
    candidate.load_state_dict(copy.deepcopy(reference.state_dict()))
    features = torch.randn(2, 17, 2, device="cuda")

    with torch.autocast("cuda", dtype=torch.float16):
        reference_output = reference(features)
        candidate_output = candidate(features)
        probe = torch.empty(1, 1, 6, device="cuda")
        assert not candidate.backbone.rnn._can_use_triton(probe)

    torch.testing.assert_close(candidate_output, reference_output, rtol=0, atol=0)


@requires_fused_cuda
def test_quantization_aware_model_uses_eager_fallback():
    from quant.quant_envs import AttrDict, Base_GRUQuantEnv

    float_model = build_model("cpu", fused=True)
    with contextlib.redirect_stdout(io.StringIO()):
        environment = Base_GRUQuantEnv(
            float_model,
            AttrDict(
                n_bits_w=16,
                n_bits_a=16,
                pretrained_model="",
                quant_dir_label="",
            ),
        )
    quantized_model = environment.q_model.cuda()
    probe = torch.empty(1, 1, 6, device="cuda")
    assert not quantized_model.backbone.rnn._can_use_triton(probe)

    features = torch.randn(2, 4, 2, device="cuda")
    output = quantized_model(features)
    output.square().mean().backward()
    assert torch.isfinite(output).all()


@requires_fused_cuda
def test_higher_order_gradient_fails_loudly():
    model = build_model("cuda", fused=True)
    features = torch.randn(1, 2, 2, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="first-order training only"):
        torch.autograd.grad(
            model(features).sum(), features, create_graph=True
        )
