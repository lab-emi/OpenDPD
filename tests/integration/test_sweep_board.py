"""Sweep submission, workers and durable failure history on real local runtime."""

import time
import pytest

from opendpd.runtime.db import RunStore
from opendpd.runtime.supervisor import Supervisor
from opendpd.schemas.sweep import SweepDraft
from opendpd.services.sweeps import SweepController, preview, audit_conditions
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace, WorkspaceError

pytestmark = pytest.mark.integration


@pytest.fixture
def board(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    ws.register_builtin_dataset("DPA_200MHz")
    store = RunStore(ws.root / "metadata.sqlite")
    sup = Supervisor(ws, store, poll_interval=.05)
    controller = SweepController(ws, sup)
    yield ws, sup, controller
    controller.stop()
    sup.stop(timeout=5)
    store.close()


def draft(**changes):
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    cfg.training.epochs = 1
    return SweepDraft(title="CPU smoke matrix", mode="same_condition", dataset={"id": "dpa-200mhz"},
                      methods=[{"entry_id": "pa", "config": cfg}], **changes)


def wait_for_board(controller, sweep_id, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        controller.tick()
        record = controller.get(sweep_id)
        if record.status != "running":
            return record
        time.sleep(.1)
    raise AssertionError("sweep did not finish")


def test_preview_is_read_only_and_cancel_retry_preserves_attempts(board):
    ws, sup, controller = board
    spec = draft(seeds=[0])
    p = preview(ws, spec)
    assert not p.errors and p.training_runs == 1 and p.evaluation_runs == 0
    assert p.sample_epochs == 23040 and not ws.list_run_ids()
    r = controller.create(spec)
    assert controller.create(spec).sweep_id == r.sweep_id
    controller.start_board(r.sweep_id)
    controller.tick()
    queued = controller.get(r.sweep_id).cells[0]
    assert queued.status == "queued" and len(queued.attempts) == 1
    cancelled = controller.cancel(r.sweep_id)
    assert cancelled.cells[0].status == "cancelled"
    with pytest.raises(WorkspaceError, match="resume_failed"):
        controller.start_board(r.sweep_id)
    controller.start_board(r.sweep_id, resume_failed=True)
    sup.start()
    done = wait_for_board(controller, r.sweep_id)
    assert done.status == "complete", done
    cell = done.cells[0]
    assert cell.status == "succeeded" and len(cell.attempts) == 2 and cell.attempts[0] == queued.run_id
    assert sup.store.get_run(queued.run_id).status.value == "cancelled"
    controller.start_board(r.sweep_id, resume_failed=True)
    controller.tick()
    assert len(ws.list_run_ids()) == 2
    from opendpd.services.sweeps import report
    aggregate = report(ws, done).aggregates[0]
    assert aggregate.n_succeeded == aggregate.n_requested_seeds == 1
    assert aggregate.metrics["NMSE"].std is None and aggregate.metrics["NMSE"].n == 1


def test_budget_cap_and_restart_are_visible_and_do_not_duplicate_runs(board):
    ws, sup, controller = board
    p = preview(ws, draft(seeds=[0, 1, 2], max_runs=2))
    assert p.errors and p.training_runs == 3 and not ws.list_run_ids()
    with pytest.raises(WorkspaceError, match="declared limit"):
        controller.create(p.draft)
    r = controller.create(draft(seeds=[0], max_wall_clock_seconds=1))
    controller.start_board(r.sweep_id)
    controller._last[r.sweep_id] -= 2
    controller.tick()
    expired = controller.get(r.sweep_id)
    assert expired.status == "cancelled" and "budget exhausted" in expired.reason
    assert not ws.list_run_ids()
    with pytest.raises(WorkspaceError, match="budget exhausted"):
        controller.start_board(r.sweep_id, resume_failed=True)
    r = controller.create(draft(seeds=[1]))
    controller.start_board(r.sweep_id)
    controller.start()
    assert controller.get(r.sweep_id).status == "interrupted"


def test_source_change_requires_a_new_preview(board, monkeypatch):
    ws, sup, controller = board
    r = controller.create(draft(seeds=[0]))
    original = ws.get_dataset
    monkeypatch.setattr(ws, "get_dataset", lambda id: original(id).model_copy(update={"display_name": "changed"}))
    with pytest.raises(WorkspaceError, match="source data"):
        controller.start_board(r.sweep_id)
    assert not ws.list_run_ids()


def test_condition_card_requires_independent_raw_files_and_complete_load(board):
    from opendpd.schemas import ConditionSet
    ws, _, _ = board
    ws.register_builtin_dataset("DPA_200MHz", dataset_id="same-capture")
    card = ConditionSet(set_id="bad", device="PA", dimension="vswr", conditions=[
        {"condition_id": "a", "dataset_id": "dpa-200mhz", "role": "source", "capture_batch": "a", "values": {"vswr": 1}},
        {"condition_id": "b", "dataset_id": "same-capture", "role": "target", "capture_batch": "b", "values": {"vswr": 2}}])
    with pytest.raises(WorkspaceError, match="reflection_phase"):
        audit_conditions(ws, card)
    for c in card.conditions:
        c.values["reflection_phase_deg"] = 0
    with pytest.raises(WorkspaceError, match="same raw capture"):
        audit_conditions(ws, card)


def test_api_previews_and_writes_use_existing_session_csrf_boundary(tmp_path):
    from fastapi.testclient import TestClient
    from opendpd.server.app import create_app
    ws = Workspace.create(tmp_path / "api")
    ws.register_builtin_dataset("DPA_200MHz")
    with TestClient(create_app(ws.root, bootstrap_token="sweep-test"), base_url="http://127.0.0.1") as client:
        assert client.get("/api/v1/sweeps").status_code == 401
        client.get("/bootstrap?token=sweep-test", follow_redirects=False)
        session = client.get("/api/v1/session").json()
        body = draft(seeds=[0]).model_dump(mode="json")
        assert client.post("/api/v1/sweeps/preview", json=body).status_code == 200
        assert client.post("/api/v1/sweeps", json=body).status_code == 403
        created = client.post("/api/v1/sweeps", json=body, headers={"X-OpenDPD-CSRF": session["csrf_token"]})
        assert created.status_code == 200, created.text
        assert created.json()["status"] == "ready" and not ws.list_run_ids()


def test_cross_condition_matrix_dispatches_existing_adaptation_dependencies(tmp_path):
    from tests.integration.test_adaptation import _capture, SIGNAL
    from opendpd.services.datasets import import_dataset
    from opendpd.schemas import DatasetOrigin, SignalSpec
    from opendpd.services.experiments import load_resolved
    ws = Workspace.create(tmp_path / 'cross')
    conditions = []
    for i in range(3):
        id = f'condition-{i}'
        import_dataset(ws, _capture(tmp_path / f'{id}.csv', 18+i, 1-i*.1), dataset_id=id, display_name=f'MOCK {id}',
                       signal=SignalSpec(**SIGNAL), origin=DatasetOrigin.synthetic, guard_samples=64)
        conditions.append(dict(condition_id=id, dataset_id=id, role='source' if i == 0 else 'target',
                               capture_batch=f'synthetic-{i}', values={'output_power_dbm': 30-i}))
    pa = instantiate('pa-gru-smoke-v1', 'condition-0')
    dpd = instantiate('dpd-gru-smoke-v1', 'condition-0', pa_run_id='deferred-pa')
    pa.training.epochs = dpd.training.epochs = 1
    spec = SweepDraft(title='Synthetic cross-condition smoke', mode='cross_condition', seeds=[0], tasks=['zero_update'],
                      condition_set={'set_id':'synthetic-card','device':'synthetic cubic PA','dimension':'output_power_dbm','conditions':conditions},
                      methods=[{'entry_id':'pa','config':pa}, {'entry_id':'dpd','config':dpd,'pa_entry':'pa'}])
    p = preview(ws, spec)
    assert not p.errors and (p.training_runs, p.evaluation_runs) == (4, 4)
    assert any('rehearsal' in w for w in p.warnings)
    store = RunStore(ws.root / 'metadata.sqlite')
    sup = Supervisor(ws, store, poll_interval=.05)
    controller = SweepController(ws, sup)
    try:
        sup.start()
        r = controller.create(spec)
        controller.start_board(r.sweep_id)
        done = wait_for_board(controller, r.sweep_id)
        assert done.status == 'complete', [(c.cell_id, c.reason) for c in done.cells]
        source_dpd = next(c for c in done.cells if c.entry_id == 'dpd' and c.task == 'full_retrain')
        target = next(c for c in done.cells if c.entry_id == 'dpd' and c.condition_id == 'condition-2')
        target_pa = next(c for c in done.cells if c.entry_id == 'pa' and c.condition_id == 'condition-2' and c.task == 'full_retrain')
        config = load_resolved(ws, target.run_id)
        assert config.dpd_reference.run_id == source_dpd.run_id and config.pa_reference.run_id == target_pa.run_id
        assert config.dpd_reference.transfer and len(ws.list_run_ids()) == 8
    finally:
        controller.stop()
        sup.stop(timeout=5)
        store.close()
