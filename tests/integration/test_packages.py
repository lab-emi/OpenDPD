"""Reproducible experiment packages (plan S11): export, redaction, cross-workspace import and re-evaluation,
refusal of damaged packages, reports."""

import json
import shlex
import socket
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from opendpd.cli import studio_main
from opendpd.schemas import RunStatus
from opendpd.services import datasets as ds
from opendpd.services.evaluation import evaluate_run
from opendpd.services.experiments import create_run, execute_run, load_artifacts, load_result, load_run
from opendpd.services.packages import PackageError, export_run, import_package, inspect_package
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace
from tests.fixtures.synthetic import Impairments, synthesize

pytestmark = pytest.mark.integration

SIGNAL = dict(sample_rate_hz=800e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560, amplitude_units="normalized")
REL, ABS = 1e-4, 1e-5          # frozen checkpoint re-evaluation tolerance (docs/protocols/acceptance-thresholds.md)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    x, y = synthesize(24000, 7, impairments=Impairments(delay_samples=0, n_outliers=0))
    path = tmp_path_factory.mktemp("src") / "capture.csv"
    pd.DataFrame({"I_in": x[:, 0], "Q_in": x[:, 1], "I_out": y[:, 0], "Q_out": y[:, 1]}).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def ws(tmp_path_factory, source):
    from opendpd.schemas import DatasetOrigin, SignalSpec

    ws = Workspace.create(tmp_path_factory.mktemp("ws1"))
    ds.import_dataset(ws, source, dataset_id="capture", display_name="capture (synthetic)", signal=SignalSpec(**SIGNAL),
                      origin=DatasetOrigin.synthetic, guard_samples=64)
    return ws


@pytest.fixture(scope="module")
def pa_run(ws):
    record = execute_run(ws, create_run(ws, instantiate("pa-gru-smoke-v1", "capture", name="PA on capture")).run_id)
    assert record.status == RunStatus.succeeded, record.error
    return record


@pytest.fixture(scope="module")
def dpd_run(ws, pa_run):
    record = execute_run(ws, create_run(ws, instantiate("dpd-gru-smoke-v1", "capture", pa_run_id=pa_run.run_id)).run_id)
    assert record.status == RunStatus.succeeded, record.error
    return record


def _texts(path: Path):
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith((".json", ".md", ".html", ".csv")):
                yield name, zf.read(name).decode("utf-8", errors="replace")


def _assert_artifacts_present(ws2, run_id):
    """An imported run lists only files it has; required ones keep their hashes."""
    manifest = load_artifacts(ws2, run_id)
    assert manifest is not None and manifest.artifacts
    for a in manifest.artifacts:
        assert (ws2.run_dir(run_id) / a.file.path).is_file(), f"{run_id} lists {a.artifact_id} without its file"
    return manifest


def _assert_close(a, b):
    for m in a.metrics:
        other = b.metric(m.name)
        assert m.status == other.status, m.name
        if m.value is not None:
            assert other.value == pytest.approx(m.value, rel=REL, abs=ABS), m.name


def test_full_package_round_trips_into_a_new_workspace_and_re_evaluates(ws, pa_run, dpd_run, tmp_path):
    out = tmp_path / "dpd-full.zip"
    manifest = export_run(ws, dpd_run.run_id, out, kind="full")
    assert manifest.kind == "full" and manifest.dataset.included and manifest.missing == []
    names = {f.path for f in manifest.files}
    assert f"run/{dpd_run.run_id}/config.resolved.json" in names and f"run/{dpd_run.run_id}/result.json" in names
    assert f"refs/{pa_run.run_id}/artifacts.json" in names and any(n.startswith("dataset/raw/") for n in names)
    assert {"report.html", "report.md"} <= names
    assert [r.role for r in manifest.references] == ["pa_surrogate"]
    assert manifest.reproduction["evaluate"].startswith(f"opendpd evaluate {dpd_run.run_id}")
    assert "reproducibility mode" in manifest.retraining_note
    assert inspect_package(out).run_id == dpd_run.run_id

    ws2 = Workspace.create(tmp_path / "imported 工作区 with spaces")
    report = import_package(ws2, out)
    assert report.dataset_status == "imported" and set(report.imported_runs) == {dpd_run.run_id, pa_run.run_id}
    assert shlex.split(report.evaluate_command) == ["opendpd", "evaluate", dpd_run.run_id, "--workspace", str(ws2.root),
                                                    "--profile", manifest.metric_profile_id]
    assert ws2.get_dataset("capture").raw_sha256 == ws.get_dataset("capture").raw_sha256
    assert load_run(ws2, dpd_run.run_id).status == RunStatus.succeeded
    assert _assert_artifacts_present(ws2, dpd_run.run_id).complete
    ref_manifest = _assert_artifacts_present(ws2, pa_run.run_id)
    assert ref_manifest.complete and {a.kind.value for a in ref_manifest.artifacts} >= {"checkpoint", "result", "log_history"}
    assert not any(a.kind.value == "worker_log" for a in ref_manifest.artifacts)
    assert load_result(ws2, pa_run.run_id) is not None, "the surrogate's own result travels with the reference"
    packaged = load_result(ws2, dpd_run.run_id)
    assert packaged == load_result(ws, dpd_run.run_id)
    recomputed = evaluate_run(ws2, dpd_run.run_id, packaged.metric_profile_id)
    _assert_close(packaged, recomputed)
    assert {m.role: m.weights_sha256 for m in recomputed.models} == {m.role: m.weights_sha256 for m in packaged.models}
    with pytest.raises(PackageError) as info:
        import_package(ws2, out)
    assert info.value.code == "run_exists"


def test_share_package_is_redacted_and_says_what_is_missing(ws, pa_run, tmp_path):
    out = tmp_path / "pa-share.zip"
    manifest = export_run(ws, pa_run.run_id, out, kind="share")
    assert not manifest.dataset.included and manifest.missing and "raw sha256" in manifest.missing[0]
    assert any("PA data" in line for line in manifest.redaction)
    names = {f.path for f in manifest.files}
    assert not any(n.startswith("dataset/raw") or n.startswith("dataset/versions") for n in names)
    assert not any("/logs/" in n for n in names)
    forbidden = [str(ws.root), str(Path.home())]
    host = socket.gethostname()
    if host:
        forbidden.append(host)
    for name, text in _texts(out):
        for secret in forbidden:
            assert secret not in text, f"{secret!r} leaked into {name}"
    with zipfile.ZipFile(out) as zf:
        run = json.loads(zf.read(f"run/{pa_run.run_id}/run.json"))
        assert run["worker"] is None
        dataset = json.loads(zf.read("dataset/manifest.json"))
        assert dataset["source"]["original_path"] is None
        provenance = json.loads(zf.read(f"run/{pa_run.run_id}/provenance.json"))
        assert "<workspace>" in provenance["legacy_equivalent_command"]

    ws3 = Workspace.create(tmp_path / "ws3")
    report = import_package(ws3, out)
    assert report.dataset_status == "missing" and report.missing and "capture" in report.missing[0]
    assert load_result(ws3, pa_run.run_id) is not None, "results are readable without the data"
    assert _assert_artifacts_present(ws3, pa_run.run_id).complete      # in-process runs have no worker log to drop
    with pytest.raises(Exception, match="not registered"):
        evaluate_run(ws3, pa_run.run_id, "legacy-opendpd-v1")


def test_share_package_re_evaluates_once_the_data_is_imported_with_the_same_hash(ws, pa_run, source, tmp_path):
    from opendpd.schemas import DatasetOrigin, SignalSpec

    out = tmp_path / "pa-share.zip"
    export_run(ws, pa_run.run_id, out, kind="share")
    ws4 = Workspace.create(tmp_path / "ws4")
    import_package(ws4, out)
    ds.import_dataset(ws4, source, dataset_id="capture", display_name="capture (synthetic)", signal=SignalSpec(**SIGNAL),
                      origin=DatasetOrigin.synthetic, guard_samples=64)
    assert ws4.get_dataset("capture").raw_sha256 == ws.get_dataset("capture").raw_sha256
    packaged = load_result(ws4, pa_run.run_id)
    _assert_close(packaged, evaluate_run(ws4, pa_run.run_id, packaged.metric_profile_id))


def test_dataset_conflicts_are_refused_not_merged(ws, pa_run, tmp_path):
    from opendpd.schemas import DatasetOrigin, SignalSpec

    out = tmp_path / "pa-share.zip"
    export_run(ws, pa_run.run_id, out, kind="share")
    ws5 = Workspace.create(tmp_path / "ws5")
    x, y = synthesize(3000, 11)
    other = tmp_path / "other.csv"
    pd.DataFrame({"I_in": x[:, 0], "Q_in": x[:, 1], "I_out": y[:, 0], "Q_out": y[:, 1]}).to_csv(other, index=False)
    ds.import_dataset(ws5, other, dataset_id="capture", display_name="capture (synthetic)", signal=SignalSpec(**SIGNAL),
                      origin=DatasetOrigin.synthetic)
    with pytest.raises(PackageError) as info:
        import_package(ws5, out)
    assert info.value.code == "dataset_conflict" and not ws5.run_dir(pa_run.run_id).exists()


@pytest.mark.parametrize("damage", ["tamper", "delete", "version", "not_zip", "unlisted"])
def test_damaged_packages_are_refused_with_a_specific_diagnostic(ws, pa_run, tmp_path, damage):
    out = tmp_path / "pa-share.zip"
    export_run(ws, pa_run.run_id, out, kind="share")
    bad = tmp_path / f"bad-{damage}.zip"
    if damage == "not_zip":
        bad.write_bytes(b"this is not a zip file")
    else:
        with zipfile.ZipFile(out) as src, zipfile.ZipFile(bad, "w") as dst:
            for name in src.namelist():
                data = src.read(name)
                if damage == "tamper" and name.endswith("/result.json"):
                    data = data.replace(b'"value": ', b'"value": 1', 1)
                if damage == "delete" and name.endswith("/config.resolved.json"):
                    continue
                if damage == "version" and name == "package.json":
                    data = json.dumps({**json.loads(data), "package_version": 99}).encode()
                dst.writestr(name, data)
            if damage == "unlisted":
                dst.writestr("run/extra.txt", b"surprise")
    expected = {"tamper": "hash_mismatch", "delete": "missing_file", "version": "unsupported_version",
                "not_zip": "not_a_package", "unlisted": "unlisted_file"}[damage]
    with pytest.raises(PackageError) as info:
        inspect_package(bad)
    assert info.value.code == expected
    ws6 = Workspace.create(tmp_path / f"ws-{damage}")
    with pytest.raises(PackageError):
        import_package(ws6, bad)
    assert ws6.list_run_ids() == [], "a refused package writes nothing"


def test_reports_are_bound_to_the_stored_result(ws, dpd_run, tmp_path):
    from opendpd.services.reports import report_html, report_markdown

    result = load_result(ws, dpd_run.run_id)
    md = report_markdown(ws, dpd_run.run_id)
    html = report_html(ws, dpd_run.run_id)
    for text in (md, html):
        assert dpd_run.run_id in text and result.metric_profile_id in text
        assert f"{result.metric('NMSE').value:.4f}" in text
        assert "surrogate_without_dpd" in text and "measured_without_dpd" in text
        assert "Nothing is recomputed" in text and "opendpd evaluate" in text
        assert "simulated" in text
    assert "data:image/png;base64," in html
    assert "config_sha256" in md and result.models[0].weights_sha256 in md


def test_all_nine_report_languages_through_api_preserve_scientific_records(ws, dpd_run, tmp_path):
    import re
    from fastapi.testclient import TestClient
    from opendpd.schemas.settings import UI_LANGUAGES
    from opendpd.server.app import create_app
    from opendpd.server.security import CSRF_HEADER
    from opendpd.studio.localization import language_tag, localize

    result_path = ws.run_dir(dpd_run.run_id) / "result.json"
    original = result_path.read_bytes()
    result = load_result(ws, dpd_run.run_id)
    token = "report-language-test"
    app = create_app(ws.root, bootstrap_token=token)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        session = client.post("/api/v1/session/bootstrap", json={"token": token})
        client.headers[CSRF_HEADER] = session.json()["csrf_token"]
        for language in UI_LANGUAGES:
            assert client.put("/api/v1/settings", json={"language": language}).status_code == 200
            url = f"/api/v1/results/{dpd_run.run_id}/report"
            html = client.get(url)
            md = client.get(url + "?format=md")
            assert html.status_code == md.status_code == 200
            assert f"lang='{language_tag(language)}'" in html.text
            assert "data:image/png;base64," in html.text
            for body in (html.text, md.text):
                assert localize("Facts", language) in body
                assert dpd_run.run_id in body
                assert f"{result.metric('NMSE').value:.4f}" in body
                assert result.models[0].weights_sha256 in body
                assert f"opendpd evaluate {dpd_run.run_id}" in body
                if language != "en":
                    # Source configuration deliberately retains its original text.
                    prose = re.sub(r"<pre>.*?</pre>|```json.*?```", "", body, flags=re.S)
                    assert "not a benchmark result" not in prose
                    assert "no physical calibration, no absolute power derived" not in prose
                    assert "target input: the PA output should equal" not in prose
            if language == "nl":
                assert "Feiten" in md.text and "Signaalketen" in md.text
            if language == "it":
                assert "Dati essenziali" in md.text and "Catena del segnale" in md.text
        # An explicit download language wins over the workspace preference.
        assert "Nothing is recomputed" in client.get(url + "?language=en").text
        assert client.get(url + "?language=xx").status_code == 422
        assert client.put("/api/v1/settings", json={"language": None}).status_code == 200
    # Exported presentation follows its requested language; stored evidence stays identical.
    out = tmp_path / "italian.zip"
    export_run(ws, dpd_run.run_id, out, kind="share", language="it")
    with zipfile.ZipFile(out) as package:
        assert "Catena del segnale" in package.read("report.md").decode()
        assert "lang='it'" in package.read("report.html").decode()
    assert result_path.read_bytes() == original


def test_export_import_and_report_through_the_cli(ws, pa_run, tmp_path, capsys):
    out = tmp_path / "cli.zip"
    capsys.readouterr()
    rc = studio_main(["export", pa_run.run_id, "--workspace", str(ws.root), "--out", str(out), "--kind", "full", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0 and payload["manifest"]["kind"] == "full" and out.exists()
    rc = studio_main(["import", str(out), "--workspace", str(tmp_path / "ws-cli"), "--inspect"])
    assert rc == 0 and "valid full package" in capsys.readouterr().out
    rc = studio_main(["import", str(out), "--workspace", str(tmp_path / "ws-cli"), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0 and report["dataset_status"] == "imported" and report["run_id"] == pa_run.run_id
    rc = studio_main(["evaluate", pa_run.run_id, "--workspace", str(tmp_path / "ws-cli"), "--profile", "legacy-opendpd-v1", "--json"])
    recomputed = json.loads(capsys.readouterr().out)
    assert rc == 0 and recomputed["metrics"][0]["name"] == "NMSE"
    assert recomputed["metrics"][0]["value"] == pytest.approx(load_result(ws, pa_run.run_id).metric("NMSE").value, rel=REL, abs=ABS)
    rc = studio_main(["report", pa_run.run_id, "--workspace", str(ws.root), "--format", "md", "--out", str(tmp_path / "r.md")])
    assert rc == 0 and (tmp_path / "r.md").read_text().startswith("# OpenDPD Studio report")
    rc = studio_main(["import", str(tmp_path / "missing.zip"), "--workspace", str(tmp_path / "ws-cli2")])
    assert rc == 2 and "not_a_package" in capsys.readouterr().err
