"""Hosted Arena uses a fixed worker and bounded, request-independent transfers."""
import base64
import io
import json
import subprocess
import time
import zipfile
from types import SimpleNamespace

import pytest

from opendpd.web import gpu_agent, gpu_broker, gpu_container


IMAGE = "sha256:" + "a" * 64
PROTOCOL = "b" * 64


def archive(files):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as output:
        for name, contents in files.items():
            output.writestr(name, contents)
    return data.getvalue()


@pytest.fixture
def arena_job(tmp_path, monkeypatch):
    from opendpd.core import arena
    monkeypatch.setattr(arena, "protocol", lambda: SimpleNamespace(protocol_sha256=PROTOCOL))
    transfer = tmp_path / "transfer"
    root = transfer / "runs" / "arena-test"
    (root / "logs").mkdir(parents=True)
    (root / "request.json").write_text('{"canonical":true}')
    (root / "submission.json").write_text("must never transfer")
    owner = SimpleNamespace(job_kind="arena", ws=SimpleNamespace(root=tmp_path / "tenant"),
        job_root=lambda identifier: root, job_input_root=lambda identifier: transfer,
        worker_alive=lambda identifier: True, expires_at=time.time() + 600,
        manager=SimpleNamespace(config=SimpleNamespace(max_runtime_seconds=600)))
    broker = gpu_broker.GpuBroker()
    job = broker.enqueue(owner, SimpleNamespace(run_id="arena-test"))
    return broker, job


def test_arena_command_keeps_sandbox_and_fixed_entrypoint(tmp_path):
    normal = gpu_agent.container_command(IMAGE, "job", tmp_path, "arena-test", 60)
    arena = gpu_agent.container_command(IMAGE, "job", tmp_path, "arena-test", 60, "arena")
    assert arena == normal + ["--kind", "arena"]
    assert arena[:len(gpu_agent.container_options(IMAGE, "job", 60))] == gpu_agent.container_options(IMAGE, "job", 60)
    assert normal[-6:] == ["-m", "opendpd.web.gpu_container", "--workspace", "/workspace", "--run-id", "arena-test"]


def test_unknown_kind_is_rejected_before_any_host_work(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        gpu_agent.container_command(IMAGE, "job", tmp_path, "run", 60, "arbitrary.module")
    job = {"id": "a" * 32, "run_id": "run", "expires_at": time.time() + 60, "kind": "shell"}
    with pytest.raises(ValueError, match="GPU job"):
        gpu_agent.Agent.run(SimpleNamespace(root=tmp_path), job)
    assert not list(tmp_path.iterdir())


def test_capability_probe_failure_and_success(monkeypatch):
    commands = []
    def probe(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({"protocol_sha256": PROTOCOL}))
    monkeypatch.setattr(gpu_agent.subprocess, "run", probe)
    assert gpu_agent.arena_capability(IMAGE) == PROTOCOL
    assert commands[0][-3:] == ["-m", "opendpd.web.gpu_container", "--arena-capability"]
    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(gpu_agent.subprocess, "run", failure)
    assert gpu_agent.arena_capability(IMAGE) is None


def test_arena_dispatch_requires_current_capability_and_fresh_heartbeat(arena_job):
    broker, job = arena_job
    assert not broker.arena_available(PROTOCOL)
    assert broker.poll("GPU") is None
    assert broker.poll("GPU", "c" * 64) is None
    assert not broker.arena_available(PROTOCOL)
    leased = broker.poll("GPU", PROTOCOL)
    assert leased["id"] == job.id and leased["kind"] == "arena"
    assert broker.arena_available(PROTOCOL)
    assert broker.poll("GPU", PROTOCOL) is None
    broker.last_seen -= 31
    assert not broker.arena_available(PROTOCOL)
    with pytest.raises(ValueError, match="capability"):
        broker.poll("GPU", "unknown")


def test_arena_input_contains_only_canonical_request(arena_job):
    broker, job = arena_job
    with zipfile.ZipFile(io.BytesIO(broker.input(job))) as data:
        assert data.namelist() == ["runs/arena-test/request.json"]
        assert data.read(data.namelist()[0]) == b'{"canonical":true}'


@pytest.mark.parametrize("filename", ["request.json", "submission.json", "evaluation/weights.npz", "../result.json", ".gpu-result"])
def test_arena_rejects_unexpected_result_files_atomically(arena_job, filename):
    broker, job = arena_job
    with pytest.raises(ValueError, match="Arena result"):
        broker.result(job, archive({"result.json": "{}", filename: "poison"}), 0)
    assert not (job.root / "result.json").exists()
    assert not job.done
    assert (job.root / "request.json").read_text() == '{"canonical":true}'


def test_arena_result_import_requires_result_on_success(arena_job):
    broker, job = arena_job
    with pytest.raises(ValueError, match="no result"):
        broker.result(job, archive({"logs/worker.log": "missing"}), 0)
    broker.result(job, archive({"result.json": "{}", "logs/worker.log": "complete"}), 0)
    assert json.loads((job.root / ".gpu-result").read_text())["exit_code"] == 0
    assert (job.root / "result.json").read_text() == "{}"


def test_arena_corrupt_archive_and_cancellation_during_import(arena_job, monkeypatch):
    broker, job = arena_job
    with pytest.raises(ValueError, match="archive"):
        broker.result(job, b"not a zip", 0)
    original = gpu_broker.unpack
    def cancel_during_unpack(data, root):
        original(data, root)
        (job.root / "CANCEL").touch()
    monkeypatch.setattr(gpu_broker, "unpack", cancel_during_unpack)
    broker.result(job, archive({"result.json": "{}"}), 0)
    assert not (job.root / "result.json").exists()
    assert json.loads((job.root / ".gpu-result").read_text())["exit_code"] == 3


@pytest.mark.parametrize("cancel", ["file", "worker", "owner"])
def test_arena_late_results_cannot_resurrect_cancelled_jobs(arena_job, cancel):
    broker, job = arena_job
    if cancel == "file":
        (job.root / "CANCEL").touch()
    elif cancel == "worker":
        job.supervisor.worker_alive = lambda identifier: False
    else:
        broker.cancel_owner(job.supervisor)
    broker.result(job, archive({"result.json": "{}"}), 0)
    assert job.done and not (job.root / "result.json").exists()
    assert json.loads((job.root / ".gpu-result").read_text())["exit_code"] == 3
    assert (job.root / "request.json").exists()


def test_arena_progress_is_bounded_schema_and_checkpoint_is_disabled(arena_job):
    broker, job = arena_job
    encode = lambda value: base64.b64encode(value).decode()
    progress = {"phase": "training", "epoch": 1, "epochs": 150}
    payload = {"arena_progress": encode(json.dumps(progress).encode()), "log_offset": 0}
    assert broker.update(job, payload)["continue"]
    assert json.loads((job.root / "result.progress.json").read_text())["epoch"] == 1
    for invalid in (b" " * 4097, b'{"phase":"training","epoch":151,"epochs":150}'):
        with pytest.raises(ValueError):
            broker.update(job, {"arena_progress": encode(invalid)})
    with pytest.raises(ValueError, match="checkpoints"):
        broker.checkpoint(job, {})
    assert not (job.root / "events.jsonl").exists()


def test_invalid_final_progress_does_not_import_result(arena_job):
    broker, job = arena_job
    with pytest.raises(ValueError):
        broker.result(job, archive({"result.json": "{}", "result.progress.json": "{}"}), 0)
    assert not (job.root / "result.json").exists()


def test_container_revalidates_template_and_protocol(monkeypatch):
    from opendpd.core import arena
    from opendpd.core.backbone_template import DEFAULT_DEFINITION, source_sha256
    monkeypatch.setattr(arena, "protocol", lambda: SimpleNamespace(protocol_sha256=PROTOCOL,
        boards=[SimpleNamespace(board_id="dpa-160mhz")], training={"max_parameters": 4096}))
    request = {"board_id": "dpa-160mhz", "backbone": "user_template", "protocol_sha256": PROTOCOL,
        "model_parameters": {"definition": DEFAULT_DEFINITION},
        "model_provenance": {"backbone_id": "ub-" + "d" * 64, "source_sha256": "e" * 64,
                             "definition_sha256": source_sha256(DEFAULT_DEFINITION.encode())}}
    assert gpu_container.validate_arena_request(request) == request
    with pytest.raises(ValueError, match="protocol"):
        gpu_container.validate_arena_request({**request, "protocol_sha256": "f" * 64})
    with pytest.raises(ValueError, match="validated template"):
        gpu_container.validate_arena_request({**request, "backbone": "gru"})
    with pytest.raises(ValueError, match="invalid Arena request"):
        gpu_container.validate_arena_request({**request, "command": "echo unsafe"})
    # The parameter sweep, its presets and the device belong to the server, never to a request.
    for field, value in (("budgets", [250]), ("budget", 2000), ("sweep", []), ("device", "cpu"), ("seeds", [0])):
        with pytest.raises(ValueError, match="invalid Arena request"):
            gpu_container.validate_arena_request({**request, field: value})
    with pytest.raises(ValueError, match="validated template"):
        gpu_container.validate_arena_request(
            {**request, "model_parameters": {"definition": DEFAULT_DEFINITION, "hidden_size": 3}})
    with pytest.raises(ValueError, match="validated template"):
        gpu_container.validate_arena_request({**request, "backbone": "gru", "model_parameters": {"hidden_size": 3},
                                              "model_provenance": {}})
    definition = json.loads(DEFAULT_DEFINITION)
    definition["nodes"][0]["features"] = 64
    encoded = json.dumps(definition)
    with pytest.raises(ValueError, match="parameter budget"):
        gpu_container.validate_arena_request({**request, "model_parameters": {"definition": encoded},
            "model_provenance": {**request["model_provenance"], "definition_sha256": source_sha256(encoded.encode())}})


@pytest.mark.parametrize("backbone,device", [("gru", "cuda"), ("mp_ls", "cuda"),
                                           ("deltagru", "cuda"), ("deltajanet", "cuda")])
def test_container_invokes_only_fixed_arena_paths_and_server_owned_device(monkeypatch, backbone, device):
    from pathlib import Path
    from opendpd.core import arena_runner
    from opendpd.web import gpu_archive
    calls = []
    monkeypatch.setattr(gpu_container.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_container.torch.cuda, "set_per_process_memory_fraction", lambda *args: None)
    def read(root, relative, **kwargs):
        assert root == Path("/workspace/runs/arena-test") and relative == "request.json"
        return json.dumps({"validated": True, "backbone": backbone}).encode()
    monkeypatch.setattr(gpu_archive, "read_regular", read)
    monkeypatch.setattr(gpu_container, "validate_arena_request", lambda request: request)
    monkeypatch.setattr(arena_runner, "evaluate_request", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert gpu_container.main(["--workspace", "/workspace", "--run-id", "arena-test", "--kind", "arena"]) == 0
    assert calls == [(({"validated": True, "backbone": backbone}, Path("/workspace/runs/arena-test/result.json")), {"device": device})]
    # The same policy the local and official runner apply on their own; no launcher passes a device any more.
    assert arena_runner.default_device(backbone) == device
    with pytest.raises(ValueError, match="fixed container workspace"):
        gpu_container.main(["--workspace", "/tmp", "--run-id", "arena-test", "--kind", "arena"])


def test_capability_requires_every_pinned_asset(monkeypatch, tmp_path):
    from opendpd.core import arena
    monkeypatch.setattr(arena, "ASSETS", tmp_path)
    with pytest.raises((KeyError, FileNotFoundError)):
        gpu_container.arena_capability()


def test_agent_streams_arena_progress_without_training_artifacts(tmp_path):
    root = tmp_path
    tmp_path = root / "runs" / "run-test"
    tmp_path.mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "worker.log").write_text("training")
    (tmp_path / "result.progress.json").write_text('{"phase":"training"}')
    (tmp_path / "events.jsonl").write_text("not an Arena output")
    payloads = []
    def request(path, payload, job):
        payloads.append(payload)
        return {"continue": True, "log_offset": 8}
    agent = SimpleNamespace(publish_resources=lambda: None, request=request)
    assert gpu_agent.Agent.update(agent, {"kind": "arena", "id": "a" * 32, "run_id": "run-test"}, root, {"log_offset": 0})
    assert "arena_progress" in payloads[0] and "events" not in payloads[0] and "live" not in payloads[0]
    (tmp_path / "result.progress.json").write_bytes(b" " * 4097)
    with pytest.raises(ValueError, match="progress exceeds"):
        gpu_agent.Agent.update(agent, {"kind": "arena", "id": "a" * 32, "run_id": "run-test"}, root, {})
