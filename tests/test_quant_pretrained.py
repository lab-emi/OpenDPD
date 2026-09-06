"""Quantization-aware training must fine-tune the float DPD checkpoint it is given.

``train_dpd --quant --pretrained_model <float DPD>`` replaces the ``torch.nn.GRU`` of the float model by the
cell-based GRU of ``quant.modules.gru`` before loading the checkpoint, so the parameter names have to be
translated; without that the quantization environment silently fell back to a float model.
"""
import contextlib
import io

import torch
import torch.nn as nn

from models import CoreModel
from quant.modules.gru import GRU as CellGRU
from quant.quant_envs import AttrDict, Base_GRUQuantEnv, convert_gru_state_dict


def test_convert_gru_state_dict_reproduces_torch_gru():
    torch.manual_seed(0)
    reference = nn.GRU(4, 15, num_layers=2, batch_first=True)
    converted = CellGRU(4, 15, num_layers=2, batch_first=True)
    converted.load_state_dict(convert_gru_state_dict(reference.state_dict()))

    x = torch.randn(3, 20, 4)
    h0 = torch.zeros(2, 3, 15)
    expected, _ = reference(x, h0)
    actual, _ = converted(x, h0)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_convert_gru_state_dict_keeps_other_keys():
    state = {
        "backbone.fc_out.weight": torch.zeros(2, 15),
        "backbone.rnn.rnn_cell_list.0.x2h.weight": torch.zeros(45, 4),
    }
    assert convert_gru_state_dict(state).keys() == state.keys()


def test_quant_env_loads_float_gru_checkpoint(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        float_model = CoreModel(input_size=2, hidden_size=15, num_layers=1, backbone_type="qgru")
    checkpoint = tmp_path / "DPD_float.pt"
    torch.save(float_model.state_dict(), checkpoint)

    with contextlib.redirect_stdout(io.StringIO()):
        env = Base_GRUQuantEnv(
            CoreModel(input_size=2, hidden_size=15, num_layers=1, backbone_type="qgru"),
            AttrDict(n_bits_w=16, n_bits_a=16, pretrained_model=str(checkpoint), quant_dir_label=""),
        )

    loaded = env.pygru_model.state_dict()
    rnn = float_model.backbone.rnn
    torch.testing.assert_close(loaded["backbone.rnn.rnn_cell_list.0.x2h.weight"], rnn.weight_ih_l0)
    torch.testing.assert_close(loaded["backbone.rnn.rnn_cell_list.0.h2h.weight"], rnn.weight_hh_l0)
    torch.testing.assert_close(loaded["backbone.rnn.rnn_cell_list.0.x2h.bias"], rnn.bias_ih_l0)
    torch.testing.assert_close(loaded["backbone.rnn.rnn_cell_list.0.h2h.bias"], rnn.bias_hh_l0)
    torch.testing.assert_close(loaded["backbone.fc_out.weight"], float_model.backbone.fc_out.weight)
    assert any("rnn_cell_list" in key for key in env.q_model.state_dict())
