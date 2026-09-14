import subprocess
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from opendpd.services import server_status as status
from opendpd.schemas.system import MachineLoad
from opendpd.services.workspace import Workspace, WorkspaceError


@pytest.mark.parametrize('output,utilization', [('23, 1024, 8192', 23), ('0, 0, 8192', 0), ('[N/A], 0, 8192', None)])
def test_gpu_measurement_preserves_zero_and_unavailable(monkeypatch, output, utilization):
    monkeypatch.setattr(status.shutil, 'which', lambda _: '/usr/bin/nvidia-smi')
    def run(command, **kwargs):
        assert '--id=0' in command and kwargs['timeout'] == 2
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(status.subprocess, 'run', run)
    result = status.gpu_load()
    assert result.utilization_percent == utilization
    assert result.memory_total_bytes == 8 * 2**30


def test_gpu_command_failure_is_unavailable(monkeypatch):
    monkeypatch.setattr(status.shutil, 'which', lambda _: '/usr/bin/nvidia-smi')
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', 2)
    monkeypatch.setattr(status.subprocess, 'run', timeout)
    assert status.gpu_load() is None


def test_cached_samples_age_without_reprobing(monkeypatch):
    sampler = status.ResourceSampler()
    assert sampler.snapshot().stale
    sampler.latest = MachineLoad(sampled_at=datetime.now(timezone.utc), cpu_percent=20)
    sampler.updated = time.monotonic() - 25
    monkeypatch.setattr(status, 'gpu_load', lambda: pytest.fail('HTTP snapshot probed the GPU'))
    result = sampler.snapshot()
    assert result.stale and result.load.cpu_percent == 20 and result.age_seconds >= 25


@pytest.mark.parametrize('identifier', ['../outside', '/tmp/outside', 'a/b', 'a\\b', '.', '..'])
def test_service_level_ids_cannot_escape_workspace(tmp_path, identifier):
    ws = Workspace.open_or_create(tmp_path / 'ws')
    for call in (ws.dataset_dir, ws.run_dir, lambda value: ws.dataset_version_dir('valid', value)):
        with pytest.raises(WorkspaceError):
            call(identifier)
