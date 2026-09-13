"""Actual CPU results -> review -> saved figure -> exported replay, with stale-source refusal."""

import csv
import io
import subprocess
import sys
import zipfile

import pytest

from opendpd.schemas import RunStatus
from opendpd.schemas.review import FigureSpec
from opendpd.schemas.rf import RFConditions
from opendpd.services import experiments, figures, review
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, sha256_file, write_json_atomic

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    ws = Workspace.create(tmp_path_factory.mktemp("review-ws"))
    dataset = ws.register_builtin_dataset("DPA_200MHz")
    config = instantiate("pa-gru-smoke-v1", dataset.dataset_id)
    config.training.epochs = 1
    run = experiments.execute_run(ws, experiments.create_run(ws, config).run_id)
    assert run.status == RunStatus.succeeded, run.error
    return ws, run.run_id


def make_spec(ws, run_id):
    data, _, _ = review.plot_artifact(ws, run_id, "spectrum")
    primary = next(t for t in data["traces"] if t["role"] == "primary")
    return FigureSpec(title="RF review", reference_run_id=run_id,
                      profiles={run_id: "general-spectral-v1"},
                      panels=[dict(traces=[dict(run_id=run_id, trace_name=primary["name"], color="#2563EB", dash="dash")],
                                   x_range=[-300, 300], y_range=[-130, -50], cursor_x=120)])


def test_review_and_conditions_preserve_original_metrics_and_history(completed):
    ws, run = completed
    original = (ws.run_dir(run) / "result.json").read_bytes()
    initial = review.review_result(ws, run)
    assert initial.signal.sample_rate_hz == 800e6
    assert initial.signal_source == "evaluation snapshot"
    assert next(f for f in initial.facts if f.key == "average_output_power_dbm").value is None
    review.save_conditions(ws, run, RFConditions(note="declared bench reading", dut="Example DUT", average_output_power_dbm=30,
                                                input_power_dbm=20, pa_dc_power_w=4, dc_rails_w={"PA": 4, "DSP": 1}, included_rails=["PA", "DSP"]))
    updated = review.review_result(ws, run)
    facts = {f.key: f for f in updated.facts}
    assert float(facts["de"].value) == pytest.approx(25)
    assert float(facts["pae"].value) == pytest.approx(22.5)
    assert float(facts["tx_efficiency"].value) == pytest.approx(20)
    assert updated.result.metrics == initial.result.metrics
    assert (ws.run_dir(run) / "result.json").read_bytes() == original
    assert len(list((ws.run_dir(run) / "rf-conditions").glob("*.json"))) == 1


def test_saved_figure_roundtrip_and_exported_replay(completed, tmp_path):
    ws, run = completed
    spec = make_spec(ws, run)
    saved = figures.save_figure(ws, spec)
    restored = figures.load_figure(ws, saved.figure_id)
    assert restored.spec == spec
    assert figures.list_figures(ws, [run])[0].figure_id == saved.figure_id
    assert len(restored.bindings[0].result_sha256) == 64
    bundle = figures.export_figure(ws, saved.figure_id)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert {"figure.png", "figure.svg", "figure.pdf", "chart-data.csv", "metrics.csv", "profiles.json", "replay.py"} <= set(archive.namelist())
        archive.extractall(tmp_path)
    csv_metrics = {r["metric"]: r for r in csv.DictReader((tmp_path / "metrics.csv").open())}
    result = experiments.load_result(ws, run, "general-spectral-v1")
    for metric in result.metrics:
        assert csv_metrics[metric.name]["unit"] == metric.unit
        assert csv_metrics[metric.name]["value"] == str(metric.value) if metric.value is not None else csv_metrics[metric.name]["value"] == ""
    rows = list(csv.DictReader((tmp_path / "chart-data.csv").open()))
    data, _, _ = review.plot_artifact(ws, run, "spectrum")
    primary = next(t for t in data["traces"] if t["role"] == "primary")
    assert [float(r["x"]) for r in rows] == data["frequency"]
    assert [float(r["y"]) for r in rows] == primary["psd_db"]
    proc = subprocess.run([sys.executable, str(tmp_path / "replay.py"), str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "replayed" / "figure.png").read_bytes() == (tmp_path / "figure.png").read_bytes()
    (tmp_path / "plot-data.json").write_text("{}")
    proc = subprocess.run([sys.executable, str(tmp_path / "replay.py"), str(tmp_path)], capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0 and "hash mismatch" in proc.stderr


def test_saved_figure_rejects_updated_data_even_with_an_updated_artifact_manifest(completed):
    ws, run = completed
    saved = figures.save_figure(ws, make_spec(ws, run))
    data, path, _ = review.plot_artifact(ws, run, "spectrum")
    artifact_path = ws.run_dir(run) / path
    manifest_path = ws.run_dir(run) / "artifacts.json"
    original, manifest = artifact_path.read_bytes(), manifest_path.read_bytes()
    try:
        data["traces"][0]["psd_db"][0] += 1
        write_json_atomic(artifact_path, data)
        with pytest.raises(WorkspaceError, match="hash mismatch"):
            figures.validate_sources(ws, saved)
        updated = read_json(manifest_path)
        next(a for a in updated["artifacts"] if a["artifact_id"] == "plot-spectrum")["file"]["sha256"] = sha256_file(artifact_path)
        write_json_atomic(manifest_path, updated)
        with pytest.raises(WorkspaceError, match="sources changed"):
            figures.export_figure(ws, saved.figure_id)
    finally:
        artifact_path.write_bytes(original)
        manifest_path.write_bytes(manifest)


def test_rf_condition_revision_invalidates_a_saved_view(completed):
    ws, run = completed
    saved = figures.save_figure(ws, make_spec(ws, run))
    review.save_conditions(ws, run, RFConditions(note="new measurement declaration", average_output_power_dbm=29))
    with pytest.raises(WorkspaceError, match="sources changed"):
        figures.validate_sources(ws, saved)


def test_api_review_save_restore_and_csrf_boundary(completed):
    from fastapi.testclient import TestClient
    from opendpd.server.app import create_app
    ws, run = completed
    app = create_app(ws.root, bootstrap_token="review-test-token")
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get(f"/api/v1/results/{run}/review").status_code == 401
        token = client.post("/api/v1/session/bootstrap", json={"token": "review-test-token"}).json()["csrf_token"]
        assert client.post("/api/v1/figures", json=make_spec(ws, run).model_dump(mode="json")).status_code == 403
        client.headers["X-OpenDPD-CSRF"] = token
        context = client.get(f"/api/v1/results/{run}/review?profile=general-spectral-v1")
        assert context.status_code == 200, context.text
        assert context.json()["result"]["metrics"] == experiments.load_result(ws, run, "general-spectral-v1").model_dump(mode="json")["metrics"]
        response = client.post("/api/v1/figures", json=make_spec(ws, run).model_dump(mode="json"))
        assert response.status_code == 200, response.text
        saved = response.json()
        assert client.get(f'/api/v1/figures/{saved["figure_id"]}').json() == saved
        assert client.get(f'/api/v1/figures/{saved["figure_id"]}/export').content.startswith(b"PK")
        assert client.get(f"/api/v1/results/{run}/review?profile=../../result").status_code == 422
        invalid = client.post(f"/api/v1/results/{run}/rf-conditions", json={"note": "incomplete load", "vswr": 2})
        assert invalid.status_code == 422


def test_multiplot_sources_declarations_and_full_metric_view_reproduction(completed, tmp_path):
    import json
    from opendpd.services.figure_reproduction import export_reproduction
    ws, run = completed
    profiles = {run: 'general-spectral-v1'}
    error, _, _ = review.plot_artifact(ws, run, 'error_distribution')
    primary = next(t['name'] for t in error['traces'] if t['role'] == 'primary')
    assert sum(next(t for t in error['traces'] if t['name'] == primary)['counts']) == error['n_samples']
    condition_path = ws.run_dir(run) / 'rf-conditions.json'
    original = condition_path.read_bytes() if condition_path.exists() else None
    try:
        condition_path.unlink(missing_ok=True)
        source = figures.sources(ws, profiles)
        assert not any(s.kind == 'power_scan' for s in source.sources)
        assert any('normalized IQ cannot supply' in text for text in source.missing)
        review.save_conditions(ws, run, RFConditions(note='Test fixture declaration only', average_output_power_dbm=27))
        spec = make_spec(ws, run)
        spec.version = 'figure-v2'
        spec.panels += [dict(kind='amam', traces=[dict(run_id=run, trace_name=primary)]),
                        dict(kind='error_distribution', traces=[dict(run_id=run, trace_name=primary)]),
                        dict(kind='power_scan', traces=[dict(run_id=run, trace_name='ACPR_L')])]
        # model_copy is not a nested-dict validator; cross the real API schema boundary.
        spec = FigureSpec.model_validate(spec.model_dump(mode='json'))
        preview = figures.preview_figure(ws, spec)
        assert preview.plots[f'{run}/power_scan']['x'] == [27]
        metric = experiments.load_result(ws, run, 'general-spectral-v1').metric('ACPR_L')
        assert next(t for t in preview.plots[f'{run}/power_scan']['traces'] if t['name'] == 'ACPR_L')['y'] == [metric.value]
        saved = figures.save_figure(ws, spec)
        bundle = export_reproduction(ws, saved.figure_id, tmp_path / 'complete.zip')
        root = tmp_path / 'bundle'
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(root)
        with zipfile.ZipFile(root / 'run-packages' / f'{run}.zip') as zf:
            assert json.loads(zf.read('package.json'))['dataset']['included']
            assert any(name.startswith('dataset/raw/') for name in zf.namelist())
        command = [sys.executable, str(root / 'reproduce.py'), str(root), '--use-bundled-source', '--workspace', str(tmp_path / 'fresh')]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        checks = json.loads((root / 'reproduction-check.json').read_text())
        assert checks['passed'] and checks['png_exact']
        assert checks['implementation'] == 'bundled'
        assert all(p['coordinates_exact'] for r in checks['runs'] for p in r['plots'])
        assert {p['kind'] for p in checks['runs'][0]['plots']} == {'spectrum', 'amam', 'error_distribution', 'power_scan'}
        second = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert second.returncode != 0 and 'nonexistent workspace' in second.stderr
        (root / 'run-packages' / f'{run}.zip').write_bytes(b'tamper')
        proc = subprocess.run([*command[:-1], str(tmp_path / 'tampered')], capture_output=True, text=True, timeout=10)
        assert proc.returncode != 0 and 'Bundle hash mismatch' in proc.stderr
        assert not (tmp_path / 'tampered').exists()
    finally:
        if original is not None:
            condition_path.write_bytes(original)
        else:
            condition_path.unlink(missing_ok=True)
