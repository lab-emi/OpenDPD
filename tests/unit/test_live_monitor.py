import types

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opendpd.services.live import LiveMonitor, ObservedLoader


@pytest.mark.parametrize('interval,expected_probes', [(None, 0), (2, 2), (100, 0)])
def test_epoch_preview_reuses_evaluation_and_batch_interval_is_explicit(tmp_path, interval, expected_probes):
    from opendpd.schemas import ExecutionConfig
    from unittest.mock import Mock
    class Net(nn.Module):
        calls = 0
        def forward(self, x):
            Net.calls += 1
            return x
    resolved = types.SimpleNamespace(training=types.SimpleNamespace(epochs=1),
        execution=ExecutionConfig(preview_every_batches=interval), task=types.SimpleNamespace(value='train_pa'))
    dataset = types.SimpleNamespace(signal=types.SimpleNamespace(sample_rate_hz=800e6))
    monitor = LiveMonitor(types.SimpleNamespace(run_dir=lambda _: tmp_path), 'run', resolved, dataset, lambda *a: None)
    monitor.observe_checkpoints = lambda p: None
    monitor.publish = Mock()
    tensor = torch.zeros(10, 4, 2)
    loader = DataLoader(TensorDataset(tensor, tensor), batch_size=2)
    validation = tensor.numpy()
    def train(**kwargs):
        list(kwargs['train_loader'])
        project.on_epoch_evaluation(validation, validation, None, None)
    project = types.SimpleNamespace(train=train, batch_size=2, frame_length=4, frame_stride=1,
                                    device=torch.device('cpu'), args=types.SimpleNamespace(nperseg=4))
    monitor.attach(project)
    project.train(net=Net(), train_loader=loader, val_loader=loader, test_loader=loader)
    assert Net.calls == expected_probes
    assert monitor.publish.call_count == expected_probes + 1
    assert monitor.publish.call_args.args[0].shape == (1, 4, 2)
    assert monitor.state['policy']['mode'] == ('batch' if interval else 'epoch')


@pytest.mark.parametrize('value', [0, -1, 1.5, 1_000_001])
def test_invalid_preview_interval_is_rejected(value):
    from pydantic import ValidationError
    from opendpd.schemas import ExecutionConfig
    with pytest.raises(ValidationError):
        ExecutionConfig(preview_every_batches=value)


def test_observed_loader_yields_original_batches_and_only_reports_after_the_step():
    seen = []
    batches = [object(), object()]
    observer = types.SimpleNamespace(epoch=-1, stage=lambda phase: seen.append(phase),
                                     batch=lambda *args: seen.append(args[1]))
    loader = ObservedLoader(batches, observer, "train")
    iterator = iter(loader)
    assert next(iterator) is batches[0]
    assert seen == ["train"]
    assert next(iterator) is batches[1]
    assert seen == ["train", 1]
    assert list(iterator) == []
    assert seen == ["train", 1, 2] and observer.epoch == 0


def test_batch_updates_are_time_bounded_but_include_first_and_last():
    events = []
    now = [0.0]
    resolved = types.SimpleNamespace(training=types.SimpleNamespace(epochs=3), task=types.SimpleNamespace(value="train_pa"))
    dataset = types.SimpleNamespace(signal=types.SimpleNamespace(sample_rate_hz=800e6))
    monitor = LiveMonitor(None, "run", resolved, dataset, lambda *event: events.append(event), clock=lambda: now[0])
    batch = (types.SimpleNamespace(shape=(64, 50, 2)), None)
    for step in range(1, 101):
        now[0] += .01
        monitor.batch("train", step, 100, batch)
    assert len(events) <= 4
    assert events[0][1]["batch"] == 1 and events[-1][1]["batch"] == 100
    assert monitor.steps == 100


def test_evaluation_batch_reports_padding_from_existing_manifest_boundaries():
    resolved = types.SimpleNamespace(training=types.SimpleNamespace(epochs=1), task=types.SimpleNamespace(value="evaluate_pa"),
                                     dataset=types.SimpleNamespace(preprocessing_version="raw-v1"))
    dataset = types.SimpleNamespace(signal=types.SimpleNamespace(sample_rate_hz=800e6), version=lambda _: None,
                                    split=types.SimpleNamespace(boundaries={"test": (100, 109)}))
    monitor = LiveMonitor(None, "run", resolved, dataset, lambda *event: None)
    monitor.stage("evaluate")
    monitor.batch("evaluate", 1, 2, (types.SimpleNamespace(shape=(2, 4, 2)), None))
    assert monitor.state["last_batch"]["padded_samples"] == 0
    monitor.batch("evaluate", 2, 2, (types.SimpleNamespace(shape=(1, 4, 2)), None))
    assert monitor.state["last_batch"]["valid_samples"] == 1
    assert monitor.state["last_batch"]["padded_samples"] == 3
