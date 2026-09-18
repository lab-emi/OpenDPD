"""The hosted GPU advertises capacity only after its training probe succeeds."""
import subprocess
from types import SimpleNamespace

import pytest

from opendpd.web import gpu_agent


def test_failed_training_probe_prevents_job_poll(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_agent, "cleanup_containers", lambda: None)
    image = "sha256:" + "a" * 64
    commands = []

    def failed_probe(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(gpu_agent.subprocess, "run", failed_probe)
    agent = SimpleNamespace(image=image, root=tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        gpu_agent.Agent.loop(agent)
    probe, = commands
    job = gpu_agent.container_command(image, "test-job", tmp_path, "test-run", 60)
    for command in (probe, job):
        assert "--network=none" in command
        assert "--read-only" in command
        assert "--cap-drop=all" in command
        assert "--user=65532:65532" in command
        assert "--security-opt=no-new-privileges" in command
        assert "--tmpfs=/tmp:rw,size=256m,nodev,nosuid,noexec,mode=1777" in command
        assert "--tmpfs=/run/opendpd-triton:rw,size=256m,nodev,nosuid,exec,mode=1777" in command
        assert "--env=TRITON_CACHE_DIR=/run/opendpd-triton" in command
    assert f"{tmp_path}:/workspace:rw,nosuid,nodev,noexec" in job
    assert probe[-2:] == ["-m", "opendpd.web.gpu_probe"]
