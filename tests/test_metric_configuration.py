from types import SimpleNamespace

import numpy as np

from modules import train_funcs


def test_calculate_metrics_uses_dataset_sample_rate(monkeypatch):
    prediction = np.zeros((1, 8, 2), dtype=np.float32)
    ground_truth = np.zeros_like(prediction)
    args = SimpleNamespace(
        input_signal_fs=983.04e6,
        bw_main_ch=200e6,
        n_sub_ch=5,
        nperseg=8,
    )
    evm_kwargs = {}

    monkeypatch.setattr(train_funcs.metrics, "NMSE", lambda *_: -1.0)

    def fake_evm(*_, **kwargs):
        evm_kwargs.update(kwargs)
        return -2.0

    monkeypatch.setattr(train_funcs.metrics, "EVM", fake_evm)
    monkeypatch.setattr(train_funcs.metrics, "ACLR", lambda *_, **__: (-3.0, -5.0))

    result = train_funcs.calculate_metrics(args, {}, prediction, ground_truth)

    assert evm_kwargs["sample_rate"] == args.input_signal_fs
    assert result == {
        "NMSE": -1.0,
        "EVM": -2.0,
        "ACLR_L": -3.0,
        "ACLR_R": -5.0,
        "ACLR_AVG": -4.0,
    }
