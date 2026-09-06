"""A checkpoint written by the legacy pipeline must keep loading.

Guards the public checkpoint convention (``state_dict`` of ``models.CoreModel``
saved by ``modules.loggers.PandasLogger``) while Studio adds new code paths.
Loading uses ``weights_only=True`` so the fixture cannot smuggle arbitrary
pickled objects.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from models import CoreModel

FIXTURE_DIR = Path(__file__).with_name("legacy_checkpoints")
CHECKPOINT = FIXTURE_DIR / "PA_S_0_M_GRU_H_23_F_50_P_1911.pt"
SHA256 = "f4e2cac55f1e479070b4b839f084d3892e4b22b4c4f1c15c972e01c3b1540672"


def test_fixture_is_unchanged():
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert digest == SHA256, "legacy checkpoint fixture was modified"


def test_legacy_checkpoint_loads_and_runs():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = CoreModel(input_size=2, hidden_size=23, num_layers=1, backbone_type="gru")
    missing, unexpected = model.load_state_dict(state, strict=True), None
    assert unexpected is None

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 1911, "parameter count encoded in the model id must match"

    torch.manual_seed(0)
    x = torch.randn(1, 256, 2) * 0.2
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 256, 2)
    assert torch.isfinite(y).all()
    # A trained PA model must not be the identity and must not be zero.
    assert float((y - x).abs().mean()) > 1e-3
    assert float(y.abs().mean()) > 1e-3


def test_best_log_row_matches_checkpoint_identity():
    import pandas as pd

    row = pd.read_csv(FIXTURE_DIR / "PA_S_0_M_GRU_H_23_F_50_P_1911.csv").iloc[0]
    assert int(row["N_PARAM"]) == 1911
    assert row["BACKBONE"] == "gru"
    assert int(row["HIDDEN_SIZE"]) == 23
    assert int(row["FRAME_LENGTH"]) == 50
    for col in ("VAL_NMSE", "TEST_NMSE", "VAL_ACLR_AVG"):
        assert np.isfinite(float(row[col]))
