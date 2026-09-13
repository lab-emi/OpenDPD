"""Real compute evidence: observations preserve weights, metrics and loader order."""
import json

import pytest
import torch

from opendpd.schemas import RunStatus, ExperimentConfig, TaskType, DatasetRef, ModelSpec, PAReference
from opendpd.services import live
from opendpd.services.experiments import create_run, execute_run, load_artifacts, load_result
from opendpd.services.legacy_adapter import load_checkpoint
from opendpd.services.recipes import instantiate, run_dpd_config
from opendpd.services.workspace import Workspace
from opendpd.core.registry import list_models, validate_parameters

pytestmark = pytest.mark.integration


def run(ws, config):
    events = []
    record = execute_run(ws, create_run(ws, config).run_id, emit=lambda kind, payload: events.append((kind.value, payload)))
    assert record.status == RunStatus.succeeded, record.error
    return record, events


def test_observer_preserves_checkpoint_and_scores_and_four_tasks_produce_live_snapshots(tmp_path, monkeypatch):
    ws = Workspace.create(tmp_path / "workspace")
    ws.register_builtin_dataset("DPA_200MHz")
    config = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    pa, events = run(ws, config)
    from opendpd.services.model_download import read_model
    snapshot = read_model(ws.run_dir(pa.run_id))
    assert snapshot is not None and any(kind == 'checkpoint' for kind, _ in events)
    artifact = next(a for a in load_artifacts(ws, pa.run_id).artifacts if a.artifact_id == 'checkpoint-best')
    assert snapshot[0]['sha256'] == artifact.file.sha256
    assert snapshot[1] == (ws.run_dir(pa.run_id) / artifact.file.path).read_bytes()
    monitor = live.load_live(ws, pa.run_id)
    assert monitor["preview"]["source"] == "final_test"
    assert monitor["preview"]["metrics"] == {m.name: m.value for m in load_result(ws, pa.run_id).metrics if m.value is not None}
    assert monitor["preview"]["units"]["EVM"] == "dB"
    assert any(kind == "progress" and payload.get("sequences") == 64 and payload.get("sequence_samples") == 50 for kind, payload in events)
    assert any(kind == "metric" and payload.get("split") == "validation_probe" for kind, payload in events)
    assert "preview_error" not in monitor
    assert len(json.dumps(monitor)) < 1_000_000

    class NoObserver:
        def __init__(self, *args): pass
        def attach(self, project): pass
        def stage(self, phase): pass
        def complete(self, result): pass
        def evaluation_loader(self, loader, net, project): return loader, None

    with monkeypatch.context() as patch:
        patch.setattr(live, "LiveMonitor", NoObserver)
        baseline, _ = run(ws, config)
    def weights(record):
        artifact = next(a for a in load_artifacts(ws, record.run_id).artifacts if a.artifact_id == "checkpoint-best")
        return load_checkpoint(ws.run_dir(record.run_id) / artifact.file.path)
    observed, original = weights(pa), weights(baseline)
    assert observed.keys() == original.keys()
    assert all(torch.equal(observed[key], original[key]) for key in observed)
    assert load_result(ws, pa.run_id).metrics == load_result(ws, baseline.run_id).metrics

    pa_test, test_events = run(ws, ExperimentConfig(task=TaskType.evaluate_pa, dataset=DatasetRef(id="dpa-200mhz"),
                               model=ModelSpec(key="gru"), pa_reference=PAReference(run_id=pa.run_id)))
    assert any(p.get("split") == "test_probe" for _, p in test_events)
    dpd, dpd_events = run(ws, instantiate("dpd-gru-smoke-v1", "dpa-200mhz", pa_run_id=pa.run_id))
    assert any("ACLR_AVG" in p.get("values", {}) and p.get("split") == "validation_probe" for _, p in dpd_events)
    applied, _ = run(ws, run_dpd_config("dpa-200mhz", dpd.run_id))
    for record in (pa_test, dpd, applied):
        snapshot = live.load_live(ws, record.run_id)
        assert snapshot["preview"]["source"] == "final_test"
        assert snapshot["preview"]["plots"]["spectrum"]["traces"]
        assert "preview_error" not in snapshot


def test_recipe_catalog_covers_every_trainable_registry_model():
    from opendpd.services.recipes import list_recipes
    for task, role in ((TaskType.train_pa, "pa"), (TaskType.train_dpd, "dpd")):
        expected = {m.key for m in list_models() if role in m.roles and not m.weights_from}
        offered = {r.model.key for r in list_recipes() if r.task == task}
        assert offered == expected
        for recipe in (r for r in list_recipes() if r.task == task):
            config = instantiate(recipe.recipe_id, "example", pa_run_id="example-pa" if role == "dpd" else None)
            # Workspace bindings are checked at submission; registry parameter validation is shared.
            validate_parameters(config.model.key, config.model.parameters, role)


@pytest.mark.parametrize("key", [m.key for m in list_models() if m.training_method == "gradient" and not m.weights_from])
def test_native_backbone_recipe_trains_and_previews_through_original_pipeline(tmp_path, key):
    ws = Workspace.create(tmp_path / key)
    ws.register_builtin_dataset("DPA_200MHz")
    config = instantiate(f"pa-{key}-smoke-v1", "dpa-200mhz")
    config.training.epochs = 1
    config.training.batch_size = 2048
    record, events = run(ws, config)
    snapshot = live.load_live(ws, record.run_id)
    assert "preview_error" not in snapshot
    assert any(p.get("split") == "validation_probe" for _, p in events)
    assert snapshot["preview"]["source"] == "final_test"
    assert snapshot["preview"]["plots"]["time"]["traces"]
