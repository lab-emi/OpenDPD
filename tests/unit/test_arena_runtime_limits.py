"""The Arena wall-clock allowance never expands ordinary-job or training budgets."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

import pytest

from opendpd.runtime.arena_limits import ARENA_MAX_RUNTIME_SECONDS
from opendpd.web import gpu_agent, gpu_broker
from opendpd.web.policy import WebConfig

SWEEP_GUARD = 7 * 24 * 60 * 60        # One entry now trains four budgets, not one model.
REAL_POPEN = subprocess.Popen    # The launcher fixture records processes instead of starting them.
LEASES = importlib.util.find_spec("fcntl") is not None       # Advisory locks: a sweep unit is a lease.


def test_arena_configuration_is_separate_and_bounded(tmp_path):
    values = dict(root=tmp_path, origin="https://example.org", api_host="api.example.org",
                  tunnel_host="a" * 48 + ".internal")
    config = WebConfig(**values)
    assert config.max_runtime_seconds == 1800
    assert config.arena_max_runtime_seconds == ARENA_MAX_RUNTIME_SECONDS == SWEEP_GUARD == 604800
    assert WebConfig(**values, arena_max_runtime_seconds=900).max_runtime_seconds == 1800
    assert WebConfig(**values, arena_max_runtime_seconds=2700).arena_max_runtime_seconds == 2700
    for invalid in (0, SWEEP_GUARD + 1, True, 900.):
        with pytest.raises(ValueError, match="Arena runtime guard must be between one second and 7 days"):
            WebConfig(**values, arena_max_runtime_seconds=invalid)


@pytest.mark.parametrize("kind,expected", [("run", 1800), ("arena", SWEEP_GUARD)])
def test_agent_container_timeout_and_command_are_bounded_per_kind(tmp_path, kind, expected):
    assert gpu_agent.container_timeout_seconds(kind, expires_at=10000000, now=1000) == expected
    assert gpu_agent.container_timeout_seconds(kind, expires_at=1123, now=1000) == 123
    command = gpu_agent.container_command("sha256:" + "a" * 64, "job", tmp_path, "test", 10000000, kind)
    assert command[command.index("--timeout") + 1] == str(expected)
    # A lease that ends earlier still ends the container earlier.
    command = gpu_agent.container_command("sha256:" + "a" * 64, "job", tmp_path, "test", 600, kind)
    assert command[command.index("--timeout") + 1] == "600"


@pytest.mark.parametrize("kind,configured,expected", [("run", SWEEP_GUARD, 2800), ("arena", SWEEP_GUARD, 1000 + SWEEP_GUARD),
                                                     ("arena", 2700, 3700), ("arena", 10 * SWEEP_GUARD, 1000 + SWEEP_GUARD)])
def test_broker_lease_uses_correct_guard_and_respects_session_expiry(tmp_path, monkeypatch, kind, configured, expected):
    monkeypatch.setattr(gpu_broker, "time", SimpleNamespace(time=lambda: 1000, monotonic=time.monotonic))
    owner = SimpleNamespace(job_kind=kind, expires_at=10000000,
        manager=SimpleNamespace(config=SimpleNamespace(max_runtime_seconds=1800, arena_max_runtime_seconds=configured)))
    broker = gpu_broker.GpuBroker()
    assert broker.enqueue(owner, SimpleNamespace(run_id="job")).expires_at == expected
    owner.expires_at = 1200
    assert broker.enqueue(owner, SimpleNamespace(run_id="other")).expires_at == 1200


def test_local_worker_survives_ordinary_limit_but_stops_at_arena_limit(tmp_path, monkeypatch):
    from opendpd.services import arena as service
    from opendpd.services.user_backbones import BackboneController
    from opendpd.services.workspace import Workspace, write_json_atomic
    ws = Workspace.open_or_create(tmp_path / "workspace")
    controller = service.ArenaController(ws, BackboneController(ws))
    directory = controller.directory("arena-" + "a" * 32)
    directory.mkdir()
    write_json_atomic(directory / "request.json", {"backbone": "gru"})
    proc = SimpleNamespace(pid=999999, returncode=None, poll=lambda: None)
    launched = []
    monkeypatch.setattr(service.subprocess, "Popen", lambda command, **kwargs: launched.append(command) or proc)
    monkeypatch.setattr(service, "process_identity", lambda pid: None)
    killed = []
    monkeypatch.setattr(service, "kill_tree", lambda pid, **kwargs: killed.append(pid))
    # An ordinary job's 30 minutes and the single-model guard of 45 minutes both pass.
    ticks = iter([0, 1801, 2701, SWEEP_GUARD - 1, SWEEP_GUARD])
    observed = []
    def now():
        value = next(ticks)
        observed.append(value)
        return value
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=now))
    monkeypatch.setattr(controller.stopping, "wait", lambda seconds: False)
    with pytest.raises(RuntimeError, match="exceeded the 7 day runtime limit"):
        controller._evaluate(directory)
    assert observed == [0, 1801, 2701, SWEEP_GUARD - 1, SWEEP_GUARD] and killed == [999999]
    assert service.MAX_RUNTIME_SECONDS == ARENA_MAX_RUNTIME_SECONDS
    assert launched == [controller.worker_command(directory)] and "--device" not in launched[0]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    """The official launcher around recorded, never executed, worker processes."""
    from benchmark import run_arena_baselines as module
    from opendpd.core import arena
    from opendpd.schemas.arena import ArenaBackbone, ArenaProtocol, ArenaRow
    from opendpd.services.workspace import write_json_atomic
    protocol = ArenaProtocol(protocol_id="test", protocol_sha256="a" * 64, training_sha256=arena.training_fingerprint(),
        title="Test", description="Test", budgets=arena.BUDGETS, seeds=[0, 1, 2], rules=[], score_formula="test",
        boards=[dict(board_id="dpa-160mhz", title="Synthetic", description="Test", dataset="synthetic",
            evidence_type="synthetic_simulation", evidence_label="Simulation", conditions=["first", "second"])],
        rankings=[dict(ranking_id="overall", title="Overall", description="Test", unit="dB")],
        training={}, scoring={}, cost_model={})
    monkeypatch.setattr(module.arena, "protocol", lambda: protocol)
    monkeypatch.setattr(module.arena, "bundled_backbones", lambda: [
        ArenaBackbone(key=key, display_name=key.upper(), family="test") for key in ("gru", "mcldnn", "deltagru")])
    monkeypatch.setattr(module, "publish", lambda *args: [])
    # Sweep units of this machine's own matrix run are none of a test's business.
    monkeypatch.setattr(module, "foreign_worker", lambda unit: False)
    state = SimpleNamespace(module=module, protocol=protocol, root=tmp_path, waits=[], commands=[], environments={},
                            hang=False, killed=[], exit_code=0, claims=[])

    class Process:
        pid = 4242

        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True        # One process group: a timeout leaves no children behind.
            state.commands.append(command)
            state.environments[tuple(command)] = kwargs.get("env")
            self.output = Path(command[command.index("--output") + 1]) if "--output" in command else None
            if "--prefetch" in command:        # Is this unit held while it starts, and which other claims exist?
                name = "--".join(command[command.index("--prefetch") + 1:][:3])
                own, holding = tmp_path / "prefetch" / f"{name}.claim", "absent"
                if own.exists():
                    probe = module.claim(own)              # Refused (None) while somebody holds the unit.
                    holding = "held" if probe is None else "not held"
                    if probe is not None:
                        probe.close()
                state.claims.append((name, holding, sorted(set(state.leftover()) - {own.name})))

        def wait(self, timeout=None):
            state.waits.append(timeout)
            if self.output is None:
                return state.exit_code
            if state.hang and timeout == ARENA_MAX_RUNTIME_SECONDS:
                raise subprocess.TimeoutExpired("worker", timeout)
            if not state.hang:
                key = self.output.parent.name.split("--")[1]
                write_json_atomic(self.output, ArenaRow(entry_id="test", board_id="dpa-160mhz", backbone=key,
                    display_name=key, origin="official", status="failed", protocol_sha256="a" * 64,
                    evidence_type="synthetic_simulation", error="fixture"))
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.os, "killpg", lambda pid, signal: state.killed.append((pid, signal)))

    def run(*arguments):
        monkeypatch.setattr(module.sys, "argv", ["launcher", "--workspace", str(tmp_path), *arguments])
        module.main()
    state.run = run
    state.units = lambda: sorted(tuple(command[command.index("--prefetch") + 1:][:3])
                                 for command in state.commands if "--prefetch" in command)
    state.leftover = lambda: sorted(claimed.name for claimed in (tmp_path / "prefetch").glob("*.claim"))
    return state


def units_of(keys, conditions=("first", "second")):
    from opendpd.core import arena
    return sorted((key, str(budget), condition) for key in keys for budget in arena.BUDGETS
                  for condition in conditions if arena.model_parameters(key, budget) is not None)


def test_official_launcher_passes_arena_limit_to_every_prefetch_unit_and_worker(launcher, tmp_path):
    from opendpd.core import arena
    launcher.run("--keys", "gru")
    workers = [command for command in launcher.commands if "--request" in command]
    # One isolated process per sweep unit (backbone, budget, condition), then one judging job per board.
    assert launcher.units() == units_of(["gru"]) and len(launcher.units()) == 4 * 2
    assert len(workers) == 1 and len(launcher.commands) == 9
    assert launcher.waits == [ARENA_MAX_RUNTIME_SECONDS] * 9 == [SWEEP_GUARD] * 9
    for command in launcher.commands:
        assert command[:3] == [sys.executable, "-m", "opendpd.core.arena_runner"]
        assert command[command.index("--cache") + 1] == str(tmp_path.resolve() / "cache")
        assert "--device" not in command            # A GRU goes to the GPU queue, where the runner decides.
        environment = launcher.environments[tuple(command)]     # Tuned or inherited, never an emptied environment.
        assert environment is None or environment["PATH"] == os.environ["PATH"]
    markers = sorted(path.name for path in (tmp_path / "prefetch").glob("*.done"))
    assert markers == ["--".join(unit) + ".done" for unit in units_of(["gru"])]
    # Each unit ran while this launcher held it, one at a time, and no claim outlives its unit.
    assert [(state, others) for _, state, others in launcher.claims] == [("held" if LEASES else "not held", [])] * 8
    assert launcher.leftover() == []
    assert {path.read_text() for path in (tmp_path / "prefetch").glob("*.done")} == {arena.training_fingerprint()}
    # A second launch neither trains a finished unit again nor repeats a finished job.
    launcher.run("--keys", "gru")
    assert len(launcher.commands) == 9
    launcher.run("--keys", "gru", "--retry-failed")
    assert len(launcher.commands) == 10 and "--request" in launcher.commands[-1]
    # Checkpoints of other training sources are not a finished unit.
    (tmp_path / "prefetch" / "gru--500--first.done").write_text("0" * 64)
    launcher.run("--keys", "gru", "--prefetch-only")
    assert launcher.commands[-1][launcher.commands[-1].index("--prefetch") + 1:] == ["gru", "500", "first"]
    assert len(launcher.commands) == 11 and launcher.waits[-1] == ARENA_MAX_RUNTIME_SECONDS


def test_prefetch_covers_each_base_unit_once_and_uses_only_supported_cpu_workers(launcher, tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_TEST_INHERITED", "kept")
    keys = ["gru_stream", "gru", "mcldnn", "mp_ls", "deltagru", "pgjanet"]
    launcher.module.prefetch(tmp_path, keys, launcher.protocol.boards, 2, 2)
    # Streaming entries reuse their base's checkpoints; a budget without a configuration has no unit.
    assert launcher.units() == units_of(["gru", "mcldnn", "mp_ls", "deltagru", "pgjanet"])
    assert len(launcher.units()) == (4 + 2 + 4 + 4 + 4) * 2
    assert not any(unit[0] == "gru_stream" or unit[:2] == ("mcldnn", "250") for unit in launcher.units())
    for command in launcher.commands:
        key, forced = command[command.index("--prefetch") + 1], command[-2:] == ["--device", "cpu"]
        assert forced == ("--device" in command)
        # One process per core: idle OpenMP helpers sleep instead of spinning. The runner itself sets the
        # protocol's intra-op thread count, so the launcher's environment decides no numerics.
        environment = launcher.environments[tuple(command)]
        assert (environment["OMP_WAIT_POLICY"], environment["GOMP_SPINCOUNT"]) == ("PASSIVE", "0")
        assert environment["ARENA_TEST_INHERITED"] == "kept" and environment["PATH"] == os.environ["PATH"]
        if key not in launcher.module.CPU_SECONDS:
            assert not forced
    assert launcher.waits == [ARENA_MAX_RUNTIME_SECONDS] * len(launcher.commands)
    # Four workers in parallel: every unit starts held, beside at most the three units of the other workers.
    assert all(state == ("held" if LEASES else "not held") and len(others) <= 3 for _, state, others in launcher.claims)
    assert len(launcher.claims) == len(launcher.commands) and launcher.leftover() == []


@pytest.mark.parametrize("key,cpu_workers,gpu_workers", [("deltagru", 0, 0), ("gru", 2, 0)])
def test_prefetch_refuses_to_leave_sweep_units_without_a_worker(launcher, tmp_path, key, cpu_workers, gpu_workers):
    with pytest.raises(RuntimeError, match="8 sweep units had no worker"):
        launcher.module.prefetch(tmp_path, [key], launcher.protocol.boards, cpu_workers, gpu_workers)
    assert launcher.commands == []


def test_failed_prefetch_unit_is_not_marked_finished(launcher, tmp_path):
    launcher.exit_code = 1
    launcher.module.prefetch(tmp_path, ["mcldnn"], launcher.protocol.boards, 1, 1)
    assert len(launcher.commands) == 4 and not list((tmp_path / "prefetch").glob("*.done"))
    assert launcher.leftover() == []                # A failed unit is free for whoever tries it next.
    launcher.exit_code = 0
    launcher.module.prefetch(tmp_path, ["mcldnn"], launcher.protocol.boards, 1, 1)
    assert len(launcher.commands) == 8 and len(list((tmp_path / "prefetch").glob("*.done"))) == 4


def _no_process(launcher, monkeypatch, name):
    import errno
    recorded = launcher.module.subprocess.Popen

    def refused(command, **kwargs):
        if "--".join(command[command.index("--prefetch") + 1:][:3]) == name:
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")        # fork() on a machine at its limit
        return recorded(command, **kwargs)
    monkeypatch.setattr(launcher.module.subprocess, "Popen", refused)


def _no_log(launcher, monkeypatch, name):
    (launcher.root / "prefetch").mkdir()
    (launcher.root / "prefetch" / f"{name}.gpu.log").mkdir()


@pytest.mark.parametrize("obstacle,error", [(_no_process, "BlockingIOError"), (_no_log, "IsADirectoryError")])
def test_prefetch_unit_that_cannot_be_launched_is_reported_failed_and_the_worker_carries_on(
        launcher, tmp_path, monkeypatch, capsys, obstacle, error):
    import json
    # One GPU worker takes the four units in turn, the larger budget first; the second one cannot start.
    order = ["mcldnn--2000--first", "mcldnn--2000--second", "mcldnn--1000--first", "mcldnn--1000--second"]
    with monkeypatch.context() as patch:
        obstacle(launcher, patch, order[1])
        launcher.module.prefetch(tmp_path, ["mcldnn"], launcher.protocol.boards, 1, 1)
        if error == "IsADirectoryError":
            (tmp_path / "prefetch" / f"{order[1]}.gpu.log").rmdir()
    reported = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(line["prefetch"], line["status"]) for line in reported] == [
        (order[0], "ok"), (order[1], f"failed ({error})"), (order[2], "ok"), (order[3], "ok")]
    assert [(line["done"], line["units"], line["worker"]) for line in reported] == [(n, 4, "gpu") for n in (1, 2, 3, 4)]
    # The unit that never ran is not a finished unit, so the next launch trains it, and only it.
    done = lambda: sorted(path.name[:-5] for path in (tmp_path / "prefetch").glob("*.done"))
    assert done() == sorted(order[:1] + order[2:])
    assert launcher.leftover() == []                # Nor does the unit that could not start stay reserved.
    assert ["--".join(unit) for unit in launcher.units()] == sorted(order[:1] + order[2:])
    launcher.module.prefetch(tmp_path, ["mcldnn"], launcher.protocol.boards, 1, 1)
    assert launcher.commands[-1][launcher.commands[-1].index("--prefetch") + 1:][:3] == order[1].split("--")
    assert len(launcher.commands) == 4 and done() == sorted(order)
    statuses = {line["prefetch"]: line["status"] for line in map(json.loads, capsys.readouterr().out.splitlines())}
    assert statuses == {**dict.fromkeys(order, "cached"), order[1]: "ok"}


def test_prefetch_never_trains_a_unit_while_another_process_is_training_it(launcher, tmp_path, monkeypatch):
    first, second = ("gmp", "500", "first"), ("gmp", "500", "second")
    foreign, polls, naps, overlapped = {first}, [], [], []          # As /proc shows another launcher's worker.

    def training_elsewhere(unit):
        polls.append(unit)
        if len(polls) > 100:                   # Never hang the suite if the launcher stops sleeping between polls.
            foreign.clear()
        return unit in foreign

    def nap(seconds):                          # The other process exits while this launcher waits for it.
        naps.append(seconds)
        foreign.clear()

    recorded = launcher.module.subprocess.Popen

    def started(command, **kwargs):
        overlapped.extend(unit for unit in foreign if list(unit) == command[command.index("--prefetch") + 1:][:3])
        return recorded(command, **kwargs)

    monkeypatch.setattr(launcher.module, "foreign_worker", training_elsewhere)
    monkeypatch.setattr(launcher.module, "time", SimpleNamespace(monotonic=time.monotonic, sleep=nap))
    monkeypatch.setattr(launcher.module.subprocess, "Popen", started)
    launcher.module.prefetch(tmp_path, ["gmp"], launcher.protocol.boards, 0, 1)
    assert overlapped == [] and naps and set(naps) == {15} and len(polls) < 100
    # Once it has exited, the unit still runs here: a verified cache makes that a reload, and it is marked finished.
    assert launcher.units() == [first, second]
    # No launcher held the unit, only an orphaned worker trained it: its claim was handed back while it
    # was postponed (absent when the other unit started) and taken again for its own run.
    held = "held" if LEASES else "not held"
    assert launcher.claims == [("--".join(second), held, []), ("--".join(first), held, [])]
    assert launcher.leftover() == []
    assert sorted(path.name for path in (tmp_path / "prefetch").glob("*.done")) == [
        "gmp--500--first.done", "gmp--500--second.done"]


def test_unit_postponed_for_another_launcher_is_revisited_while_the_other_queue_is_still_long(
        launcher, tmp_path, monkeypatch):
    import threading
    postponed = ("deltagru", "2000", "first")
    foreign, cpu_workers, revisited, naps = {postponed}, set(), threading.Event(), []

    def training_elsewhere(unit):
        if unit[0] == "deltagru":
            cpu_workers.add(threading.current_thread())
        return unit in foreign

    recorded = launcher.module.subprocess.Popen

    def started(command, **kwargs):
        unit = tuple(command[command.index("--prefetch") + 1:][:3])
        if unit == postponed:
            revisited.set()
        elif unit[0] == "gru":                 # The GPU queue stays busy until the CPU worker has settled its own.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not revisited.is_set() and (
                    not cpu_workers or any(worker.is_alive() for worker in cpu_workers)):
                time.sleep(.01)
        return recorded(command, **kwargs)

    monkeypatch.setattr(launcher.module, "foreign_worker", training_elsewhere)
    monkeypatch.setattr(launcher.module, "time", SimpleNamespace(monotonic=time.monotonic,
                                                                 sleep=lambda seconds: (naps.append(seconds), foreign.clear())))
    monkeypatch.setattr(launcher.module.subprocess, "Popen", started)
    launcher.module.prefetch(tmp_path, ["deltagru", "gru"], launcher.protocol.boards, 1, 1)
    assert naps and launcher.units() == units_of(["deltagru", "gru"])
    assert len(list((tmp_path / "prefetch").glob("*.done"))) == 16


# Another launcher as far as a lease is concerned: it holds the unit's advisory lock, gives it back the way
# the launcher does when told to, and takes it along when it dies.
HOLDER = """
import fcntl, os, sys, time
handle = open(sys.argv[1], "a")
fcntl.flock(handle, fcntl.LOCK_EX)
print("held", flush=True)
if sys.stdin.readline().strip() == "release":
    os.unlink(sys.argv[1])
    handle.close()
    print("released", flush=True)
time.sleep(300)
"""


def other_launcher_holding(path):
    process = REAL_POPEN([sys.executable, "-c", HOLDER, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if process.stdout.readline().strip() != "held":
        stop(process)
        raise AssertionError("the helper process could not take the lease")
    return process


def stop(process):
    process.kill()
    process.wait()
    process.stdin.close()
    process.stdout.close()


def test_claim_is_one_exclusive_hold_and_release_removes_it(tmp_path):
    pytest.importorskip("fcntl")
    from benchmark.run_arena_baselines import claim, release
    path = tmp_path / "gru--500--first.claim"
    hold = claim(path)
    assert hold is not None and not hold.closed and path.is_file()
    # A hold is one open file, not an owner's name: whoever asks again is refused, the holder included.
    assert claim(path) is None and claim(path) is None and path.is_file() and not hold.closed
    unrelated = claim(tmp_path / "gru--500--second.claim")                # Another unit is another hold.
    assert unrelated is not None
    release(tmp_path / "gru--500--second.claim", unrelated)
    release(path, None)                          # Nothing was obtained: nothing to give back, the holder keeps its file.
    assert path.is_file() and claim(path) is None
    release(path, hold)
    assert not path.exists() and hold.closed
    again = claim(path)                          # Free for the next launcher at once.
    assert again is not None and path.is_file()
    release(path, again)
    assert not path.exists() and list(tmp_path.iterdir()) == []


def test_claim_held_by_another_live_process_is_refused_untouched_and_ends_with_that_process(tmp_path):
    pytest.importorskip("fcntl")
    from benchmark.run_arena_baselines import claim, release
    path = tmp_path / "gru--500--first.claim"
    path.write_text("whatever the other launcher left in it")
    before = (path.stat().st_ino, path.read_text())
    other = other_launcher_holding(path)
    try:
        assert claim(path) is None and claim(path) is None
        assert (path.stat().st_ino, path.read_text()) == before          # Refused without touching its file.
    finally:
        stop(other)
    # No owner has to be guessed dead: the lock went with its process, whatever the file still says.
    hold = claim(path)
    assert hold is not None and path.stat().st_ino == before[0]
    release(path, hold)
    assert not path.exists()


@pytest.mark.parametrize("left_behind", [b"", b"4242", b"1", b"not a pid", b"\xff\xfe\x00", b"x" * 100000],
                         ids=["empty", "a-pid", "pid-of-a-live-process", "text", "bytes", "large"])
def test_leftover_claim_file_without_a_holder_is_claimable_whatever_it_contains(tmp_path, left_behind):
    pytest.importorskip("fcntl")
    from benchmark.run_arena_baselines import claim, release
    path = tmp_path / "gru--500--first.claim"
    path.write_bytes(left_behind)
    hold = claim(path)
    assert hold is not None and claim(path) is None
    release(path, hold)
    assert not path.exists()


def test_hold_on_a_file_that_was_removed_and_created_again_is_not_shared(tmp_path, monkeypatch):
    """Between opening the claim and locking it, its holder releases it and a third launcher takes the unit."""
    pytest.importorskip("fcntl")
    import builtins
    from benchmark import run_arena_baselines as module
    path = tmp_path / "gru--500--first.claim"
    first = module.claim(path)
    taken, opened = {}, []

    def interrupted_open(file, mode="r", *arguments, **options):
        handle = builtins.open(file, mode, *arguments, **options)
        opened.append(os.fstat(handle.fileno()).st_ino)
        if len(opened) == 1:                     # The second launcher has opened the claim and not locked it yet ...
            module.release(path, first)          # ... the holder gives the unit back (the file goes first) ...
            taken["third"] = module.claim(path)  # ... and a third launcher takes it: a new file of the same name.
        return handle

    monkeypatch.setattr(module, "open", interrupted_open, raising=False)
    second = module.claim(path)
    # The second launcher's lock on the removed file succeeds, and is recognized as a hold on nothing.
    assert taken["third"] is not None and second is None
    assert len(set(opened)) == 2 and opened[0] != opened[1]              # removed file, then the third launcher's file
    assert path.stat().st_ino == os.fstat(taken["third"].fileno()).st_ino == opened[1]
    module.release(path, taken["third"])
    assert not path.exists()


CONTENDER = """
import sys, time
from pathlib import Path
from benchmark.run_arena_baselines import claim
folder, units = Path(sys.argv[1]), int(sys.argv[2])
print("ready", flush=True)
start = float(sys.stdin.readline())
holds = []
for index in range(units):
    while time.time() < start + index * 0.002:          # every launcher reaches each unit at the same moment
        pass
    hold = claim(folder / f"unit-{index}.claim")
    if hold is not None:
        holds.append((index, hold))                      # and keeps what it was granted
print(" ".join(str(index) for index, _ in holds), flush=True)
sys.stdin.readline()                                     # until every launcher has reported
"""


def test_two_live_launchers_asking_for_a_unit_at_the_same_moment_are_not_both_granted_it(tmp_path):
    pytest.importorskip("fcntl")
    units, root = 300, str(Path(__file__).resolve().parents[2])
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))
    launchers = [REAL_POPEN([sys.executable, "-c", CONTENDER, str(tmp_path), str(units)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, env=environment) for _ in range(3)]
    try:
        assert [process.stdout.readline().strip() for process in launchers] == ["ready"] * 3
        start = time.time() + .5
        for process in launchers:
            process.stdin.write(f"{start}\n")
            process.stdin.flush()
        granted = [[int(index) for index in process.stdout.readline().split()] for process in launchers]
    finally:
        for process in launchers:
            stop(process)
    # Every unit went to exactly one of the three launchers.
    assert sorted(index for mine in granted for index in mine) == list(range(units)), [len(mine) for mine in granted]


@pytest.mark.parametrize("release", ["its launcher exits", "its launcher finishes the unit"])
def test_unit_held_by_another_live_launcher_is_postponed_and_trained_once_it_is_released(
        launcher, tmp_path, monkeypatch, release):
    pytest.importorskip("fcntl")
    first, second = "gmp--500--first", "gmp--500--second"
    (tmp_path / "prefetch").mkdir()
    reserved, naps, observed = tmp_path / "prefetch" / f"{first}.claim", [], []
    reserved.write_text("the other launcher's")
    untouched = (reserved.stat().st_ino, reserved.read_text())
    other = other_launcher_holding(reserved)    # Alive, and none of its workers is in the process table yet.

    def nap(seconds):
        naps.append(seconds)
        if len(naps) > 50 or release == "its launcher exits":          # (Never hang the suite on a lost release.)
            other.kill()
            other.wait()
        else:                                   # It gives the unit back as the launcher does, and lives on.
            other.stdin.write("release\n")
            other.stdin.flush()
            assert other.stdout.readline().strip() == "released"

    recorded = launcher.module.subprocess.Popen

    def started(command, **kwargs):
        if reserved.exists():
            observed.append(("--".join(command[command.index("--prefetch") + 1:][:3]),
                             (reserved.stat().st_ino, reserved.read_text()), other.poll()))
        return recorded(command, **kwargs)

    monkeypatch.setattr(launcher.module, "time", SimpleNamespace(monotonic=time.monotonic, sleep=nap))
    monkeypatch.setattr(launcher.module.subprocess, "Popen", started)
    try:
        launcher.module.prefetch(tmp_path, ["gmp"], launcher.protocol.boards, 0, 1)
        alive = other.poll() is None
    finally:
        stop(other)
    # The free unit ran first, beside the other launcher's held and untouched claim ...
    assert launcher.claims[0] == (second, "held", [first + ".claim"]) and observed[0] == (second, untouched, None)
    # ... and the held one only after one wait, under this launcher's own hold.
    assert launcher.claims[1:] == [(first, "held", [])] and naps == [15]
    assert alive == (release == "its launcher finishes the unit")
    assert sorted(path.name for path in (tmp_path / "prefetch").glob("*.done")) == [first + ".done", second + ".done"]
    assert launcher.leftover() == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_foreign_worker_recognizes_exactly_the_same_sweep_unit():
    from benchmark.run_arena_baselines import foreign_worker
    unit = ("gru", "500", "a-condition-only-this-test-uses")
    process = subprocess.Popen([sys.executable, "-c", "import sys, time; print('up', flush=True); time.sleep(60)",
                                "--cache", "unused", "--prefetch", *unit], stdout=subprocess.PIPE)
    try:
        assert process.stdout.readline().strip() == b"up"
        assert foreign_worker(unit)
        assert not foreign_worker(("gru", "1000", unit[2])) and not foreign_worker(("lstm", "500", unit[2]))
    finally:
        process.kill()
        process.wait()
    assert not foreign_worker(unit)


def test_worker_beyond_the_guard_is_stopped_and_recorded_as_a_failed_unranked_sweep(launcher, tmp_path):
    import signal
    from opendpd.services.workspace import read_json
    launcher.hang = True
    stale = tmp_path / "jobs" / "dpa-160mhz--mcldnn"
    stale.mkdir(parents=True)
    (stale / "result.progress.json").write_text('{"phase": "left by an earlier attempt"}')
    launcher.run("--keys", "mcldnn")
    assert launcher.killed == [(4242, signal.SIGTERM)]
    assert launcher.waits[-2:] == [ARENA_MAX_RUNTIME_SECONDS, 10]
    row = read_json(stale / "result.json")
    assert row["status"] == "failed" and not row["eligible"] and row["score"] is None and row["rankings"] == {}
    assert "runtime guard" in row["error"] and row["origin"] == "official"
    assert [(point["budget"], point["available"]) for point in row["budgets"]] == [
        (250, False), (500, False), (1000, True), (2000, True)]
    assert (row["available_budgets"], row["expected_cases"], row["completed_cases"]) == (2, 2 * 2 * 3, 0)
    assert not (stale / "result.progress.json").exists()
