"""Equivalence checks for the optimized data and epoch plumbing."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modules.data_collector import IQFrameDataset
from modules.train_funcs import net_eval, net_train
from project import Project


def _reference_frames(sequence, frame_length, stride):
    num_frames = (len(sequence) - frame_length) // stride + 1
    return np.stack([
        sequence[index * stride:index * stride + frame_length]
        for index in range(num_frames)
    ])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("stride", [1, 2, 4])
def test_vectorized_frames_match_reference_and_do_not_alias(dtype, stride):
    sequence = np.arange(34, dtype=dtype).reshape(17, 2)
    expected = _reference_frames(sequence, frame_length=5, stride=stride)

    actual = IQFrameDataset.get_frames(
        sequence, frame_length=5, stride_length=stride
    )

    assert np.array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, sequence)

    sequence[...] = 0
    assert np.array_equal(actual, expected)


def test_frame_dataset_preserves_values_order_and_independent_storage():
    features = np.arange(40, dtype=np.float64).reshape(20, 2) / 7
    targets = -features.copy()
    expected_features = torch.Tensor(_reference_frames(features, 6, 3))
    expected_targets = torch.Tensor(_reference_frames(targets, 6, 3))

    dataset = IQFrameDataset(features, targets, frame_length=6, stride=3)
    features[...] = 99
    targets[...] = 99

    assert dataset.features.dtype == torch.float32
    assert dataset.targets.dtype == torch.float32
    assert torch.equal(dataset.features, expected_features)
    assert torch.equal(dataset.targets, expected_targets)


class _FrozenTail(nn.Module):
    """Tiny analogue of a trainable DPD followed by a frozen PA."""

    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(3, 4)
        self.frozen = nn.Linear(4, 2)
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs):
        return self.frozen(torch.tanh(self.trainable(inputs)))


def _reference_train(
    log, net, dataloader, optimizer, criterion, grad_clip_val
):
    """The pre-optimization epoch loop, retained only as a test oracle."""
    net.train()
    losses = []
    for features, targets in dataloader:
        features = features.to(torch.device("cpu"))
        targets = targets.to(torch.device("cpu"))
        optimizer.zero_grad()
        loss = criterion(net(features), targets)
        loss.backward()
        if grad_clip_val != 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
        optimizer.step()
        loss.detach()
        losses.append(loss.item())
    log["loss"] = np.mean(losses)
    return net


def _build_project_optimizer(net):
    project = Project.__new__(Project)
    project.opt_type = "adamw"
    project.lr = 2e-3
    project.decay_factor = 0.5
    project.patience = 2
    project.lr_end = 1e-7
    return project.build_optimizer(net)[0]


@pytest.mark.parametrize("num_batches", [1, 4])
def test_train_epoch_matches_original_updates_and_logged_loss(num_batches):
    torch.manual_seed(71)
    features = torch.randn(num_batches * 3, 3)
    targets = torch.randn(num_batches * 3, 2)
    dataloader = DataLoader(
        TensorDataset(features, targets), batch_size=3, shuffle=False
    )

    reference_net = _FrozenTail()
    optimized_net = deepcopy(reference_net)
    reference_optimizer = torch.optim.AdamW(
        reference_net.parameters(), lr=2e-3
    )
    optimized_optimizer = _build_project_optimizer(optimized_net)

    optimized_parameters = {
        id(parameter)
        for group in optimized_optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized_parameters == {
        id(parameter) for parameter in optimized_net.parameters()
        if parameter.requires_grad
    }

    reference_log = {}
    optimized_log = {}
    criterion = nn.MSELoss()
    _reference_train(
        reference_log, reference_net, dataloader, reference_optimizer,
        criterion, grad_clip_val=0.7
    )
    net_train(
        optimized_log, optimized_net, dataloader, optimized_optimizer,
        criterion, grad_clip_val=0.7, device=torch.device("cpu")
    )

    assert optimized_log["loss"] == reference_log["loss"]
    for reference, optimized in zip(
        reference_net.parameters(), optimized_net.parameters(), strict=True
    ):
        assert torch.equal(optimized, reference)


def _reference_eval(log, net, dataloader, criterion):
    net.eval()
    losses = []
    prediction = []
    ground_truth = []
    with torch.no_grad():
        for features, targets in dataloader:
            outputs = net(features)
            loss = criterion(outputs, targets)
            prediction.append(outputs.cpu())
            ground_truth.append(targets.cpu())
            losses.append(loss.item())
    log["loss"] = np.mean(losses)
    return (
        torch.cat(prediction, dim=0).numpy(),
        torch.cat(ground_truth, dim=0).numpy(),
    )


def test_inference_eval_matches_original_outputs_targets_and_loss():
    torch.manual_seed(101)
    net = _FrozenTail()
    features = torch.randn(11, 3)
    targets = torch.randn(11, 2)
    dataloader = DataLoader(
        TensorDataset(features, targets), batch_size=4, shuffle=False
    )
    criterion = nn.L1Loss()

    reference_log = {}
    reference_prediction, reference_targets = _reference_eval(
        reference_log, net, dataloader, criterion
    )
    optimized_log = {}
    _, optimized_prediction, optimized_targets = net_eval(
        optimized_log, net, dataloader, criterion, torch.device("cpu")
    )

    assert optimized_log["loss"] == reference_log["loss"]
    assert np.array_equal(optimized_prediction, reference_prediction)
    assert np.array_equal(optimized_targets, reference_targets)


@pytest.mark.parametrize(
    "device_type, expected", [("cpu", False), ("cuda", True)]
)
def test_cuda_loaders_pin_memory(monkeypatch, device_type, expected):
    features = np.arange(48, dtype=np.float64).reshape(24, 2) / 10
    targets = features * 2

    def fake_load_dataset(**_kwargs):
        return (
            features, targets,
            features[:12], targets[:12],
            features[:12], targets[:12],
        )

    monkeypatch.setattr(
        "modules.data_collector.load_dataset", fake_load_dataset
    )
    project = Project.__new__(Project)
    project.dataset_name = "unused"
    project.step = "train_pa"
    project.frame_length = 4
    project.frame_stride = 2
    project.batch_size = 3
    project.batch_size_eval = 2
    project.args = SimpleNamespace(nperseg=6)
    project.device = torch.device(device_type)

    loaders, _ = project.build_dataloaders()

    assert all(loader.pin_memory is expected for loader in loaders)
