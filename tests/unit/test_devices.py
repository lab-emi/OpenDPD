"""CUDA discovery and dispatch identities without requiring multiple physical GPUs."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from opendpd.runtime.supervisor import Supervisor
from opendpd.schemas import ExecutionConfig
from opendpd.schemas.examples import experiment_train_pa_smoke
from opendpd.services import capabilities
from opendpd.services.config import resolve
from opendpd.services.legacy_adapter import build_namespace, legacy_cli_tokens


def test_discovery_reports_each_visible_cuda_device_and_caches(monkeypatch):
    monkeypatch.setattr(capabilities, "_device_cache", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    names = Mock(side_effect=["RTX 4090", "RTX 3090", "RTX 4090"])
    monkeypatch.setattr(torch.cuda, "get_device_name", names)
    info = capabilities.detect_devices()
    assert info["cuda"] == {
        "detected": True, "count": 3, "name": "RTX 4090",
        "instances": [{"index": 0, "name": "RTX 4090"}, {"index": 1, "name": "RTX 3090"},
                      {"index": 2, "name": "RTX 4090"}],
    }
    assert capabilities.detect_devices() is info
    assert [call.args for call in names.call_args_list] == [(0,), (1,), (2,)]


@pytest.mark.parametrize("device,available", [
    ("cpu", True), ("cuda", True), ("cuda:0", True), ("cuda:2", True),
    ("cuda:3", False), ("cuda:-1", False), ("cuda:abc", False), ("cuda:", False),
    ("cuda:1.0", False), ("cuda:²", False), ("mps", False), ("other", False),
])
def test_device_availability_checks_logical_index(device, available):
    detected = {"cuda": {"detected": True, "count": 3}, "mps": {"detected": False}}
    assert capabilities.device_available(device, detected=detected) is available


def test_missing_cuda_is_refused(monkeypatch):
    monkeypatch.setattr(capabilities, "_device_cache", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert capabilities.detect_devices()["cuda"]["instances"] == []
    assert not capabilities.device_available("cuda:0")
    assert capabilities.device_available("cpu")


@pytest.mark.parametrize("device,index,expected", [
    ("cuda", 0, "cuda:0"), ("cuda", 2, "cuda:2"), ("cpu", 0, "cpu"), ("mps", 0, "mps"),
])
def test_device_spec_preserves_configuration_format(device, index, expected):
    execution = ExecutionConfig(device=device, device_index=index)
    assert execution.device_spec == expected
    assert "device_spec" not in execution.model_dump()
    assert execution.model_dump()["device"] == device


@pytest.mark.parametrize("device,key", [("cuda", "cuda:0"), ("cuda:0", "cuda:0"),
                                       ("cuda:2", "cuda:2"), ("cpu", "cpu"), ("mps", "mps")])
def test_queue_keys_share_default_gpu_only(device, key):
    assert Supervisor._device_key(SimpleNamespace(device=device)) == key


def test_nonzero_gpu_reaches_legacy_parser_and_project(tmp_path, monkeypatch):
    from arguments import build_parser
    from project import Project

    config = experiment_train_pa_smoke()
    config.execution = ExecutionConfig(device="cuda", device_index=2)
    namespace = build_namespace(resolve(config), dataset_dir=tmp_path, dataset_name="ds")
    reparsed = build_parser().parse_args(legacy_cli_tokens(namespace))
    assert reparsed.accelerator == "cuda" and reparsed.devices == 2
    project = Project.__new__(Project)
    project.accelerator, project.devices = reparsed.accelerator, reparsed.devices
    project.add_arg = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    name = Mock(return_value="RTX 4090")
    select = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_name", name)
    monkeypatch.setattr(torch.cuda, "set_device", select)
    assert project.set_device() == torch.device("cuda:2")
    name.assert_called_once_with(2)
    select.assert_called_once_with(torch.device("cuda:2"))
