"""Public consent, byte provenance and real isolated Git branch/push recovery.

GitHub calls are replaced by a finite fixture; Git uses local bare repositories.
These tests never publish a real repository, create an account or send a message.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendpd.schemas.dataset_catalog import DatasetPublicationDraft, PublicationConsent, SyntheticSuiteRequest
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services import dataset_catalog
from opendpd.services.dataset_publication import GitHubPublisher, PublicationController
from opendpd.services.datasets import load_version_arrays, update_manifest
from opendpd.services.synthetic_datasets import capture, generate_suite
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, sha256_file

pytestmark = pytest.mark.integration


@pytest.fixture
def data(tmp_path):
    ws = Workspace.create(tmp_path / "workspace")
    suite = generate_suite(ws, SyntheticSuiteRequest(samples_per_capture=8192))
    return ws, suite


def draft(dataset_id):
    return DatasetPublicationDraft(dataset_id=dataset_id, description="Public synthetic fixture", license="CC0-1.0", attribution="Test contributor")


def consent(record):
    return PublicationConsent(package_sha256=record.package_sha256, publish_publicly=True, rights_confirmed=True)


def wait(controller, record):
    deadline = time.monotonic() + 10
    while record.publication_id in controller.threads and time.monotonic() < deadline:
        time.sleep(.02)
    assert record.publication_id not in controller.threads
    return controller.get(record.publication_id)


def test_synthetic_truth_independent_streams_recreation_and_origin_lock(data, tmp_path):
    ws, suite = data
    assert len(suite.datasets) == 6
    assert len({d.raw_sha256 for d in suite.datasets}) == 6
    assert all(d.origin.value == "synthetic" and d.signal.amplitude_units == "normalized" for d in suite.datasets)
    assert all(d.simulation["physical_measurement"] is False for d in suite.datasets)
    assert suite.condition_set.dimension == "mode" and suite.condition_set.card_sha256 == suite.condition_set.compute_sha256()
    assert "not independent hardware captures" in " ".join(suite.limitations)
    generated = generate_suite(Workspace.create(tmp_path / "second"), suite.request)
    assert [d.raw_sha256 for d in generated.datasets] == [d.raw_sha256 for d in suite.datasets]
    original = generate_suite(ws, suite.request)
    assert [d.raw_sha256 for d in original.datasets] == [d.raw_sha256 for d in suite.datasets]
    x, y, _ = load_version_arrays(ws, suite.datasets[0].dataset_id)
    truth, _ = capture(suite.request, 0, 0)
    np.testing.assert_array_equal(np.column_stack((x, y)), truth)
    with pytest.raises(WorkspaceError, match="cannot be relabelled"):
        update_manifest(ws, suite.datasets[0].dataset_id, origin="measured")
    with pytest.raises(WorkspaceError, match="different data"):
        generate_suite(ws, suite.request.model_copy(update={"seed": 1}))


def test_catalog_preview_removes_private_fields_and_roundtrips(data, tmp_path, monkeypatch):
    ws, suite = data
    ds = suite.datasets[0]
    ws.save_dataset(ds.model_copy(update={"notes": "do not publish secret-note", "source": ds.source.model_copy(update={"original_path": "/private/operator/source.csv"})}))
    destination = tmp_path / "catalog" / "synthetic" / "fixture"
    metadata, files, digest = dataset_catalog.write_package(ws, draft(ds.dataset_id), destination)
    text = (destination / "dataset.json").read_text() + (destination / "README.md").read_text()
    assert "/private" not in text and "secret-note" not in text
    assert {f.path for f in files} == {"data.csv", "dataset.json", "README.md"} and len(digest) == 64
    assert metadata.simulation == ds.simulation
    monkeypatch.setattr(dataset_catalog, "CATALOG_ROOT", tmp_path / "catalog")
    items = dataset_catalog.list_catalog()
    assert len(items) == 1 and items[0].origin.value == "synthetic" and not items[0].problem
    other = Workspace.create(tmp_path / "imported")
    registered = other.register_builtin_dataset(items[0].name)
    assert other.register_builtin_dataset(items[0].name).dataset_id == registered.dataset_id
    a, b, split = load_version_arrays(other, registered.dataset_id)
    x, y, expected = load_version_arrays(ws, ds.dataset_id)
    np.testing.assert_array_equal(a, x)
    np.testing.assert_array_equal(b, y)
    assert split == expected
    (destination / "data.csv").write_text("changed\n")
    assert dataset_catalog.list_catalog()[0].problem
    with pytest.raises(WorkspaceError, match="bytes changed"):
        dataset_catalog.register_catalog(Workspace.create(tmp_path / "bad"), items[0].name)


def test_changed_raw_or_preview_cannot_be_submitted(data):
    ws, suite = data
    controller = PublicationController(ws)
    record = controller.prepare(draft(suite.datasets[0].dataset_id))
    with pytest.raises(WorkspaceError, match="exact reviewed"):
        controller.start(record.publication_id, consent(record).model_copy(update={"package_sha256": "f" * 64}))
    source = ws.dataset_dir(record.dataset_id) / suite.datasets[0].files[0].path
    source.write_text(source.read_text() + "0,0,0,0\n")
    with pytest.raises(WorkspaceError, match="bytes changed"):
        controller.start(record.publication_id, consent(record))
    assert not controller.threads and controller.get(record.publication_id).status == "prepared"


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


class LocalGitHub(GitHubPublisher):
    def __init__(self, root, *, fail_after_push=False, fork=False):
        self.upstream = root / "upstream.git"
        git("init", "--bare", "--initial-branch=main", str(self.upstream))
        source = root / "seed"
        git("clone", str(self.upstream), str(source))
        (source / "README.md").write_text("upstream only\n")
        git("add", "README.md", cwd=source)
        git("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "base", cwd=source)
        git("push", "origin", "main", cwd=source)
        self.base = git("rev-parse", "HEAD", cwd=source)
        self.fork_repo = root / "fork.git"
        self.needs_fork = fork
        self.fail_after_push = fail_after_push
        self.prs, self.calls = [], []

    def url(self, repo):
        return str(self.upstream if repo == self.repository else self.fork_repo)

    def gh(self, *args, cwd=None, optional=False):
        self.calls.append(args)
        if args[:2] == ("api", "user"):
            return json.dumps({"login": "fixture", "id": 123})
        if args[:3] == ("repo", "view", self.repository):
            return json.dumps({"viewerPermission": "READ" if self.needs_fork else "WRITE", "defaultBranchRef": {"name": "main"}})
        if args[:2] == ("repo", "fork"):
            git("clone", "--bare", str(self.upstream), str(self.fork_repo))
            return ""
        if args[:3] == ("repo", "view", "fixture/OpenDPD"):
            return json.dumps({"isFork": True, "parent": {"nameWithOwner": self.repository}})
        if args[:2] == ("pr", "list"):
            return json.dumps(self.prs)
        if args[:2] == ("pr", "create"):
            body = Path(args[args.index("--body-file") + 1]).read_text()
            assert "Human review" in body and "emi.lab@outlook.com" in body
            if self.fail_after_push:
                self.fail_after_push = False
                raise WorkspaceError("Simulated network failure after push")
            self.prs.append({"url": "https://github.com/lab-emi/OpenDPD/pull/123", "state": "OPEN"})
            return self.prs[0]["url"]
        raise AssertionError(f"unexpected GitHub call: {args}")


@pytest.mark.parametrize("fork", [False, True])
def test_real_isolated_git_branch_push_pr_and_retry(data, tmp_path, fork):
    ws, suite = data
    transport = LocalGitHub(tmp_path, fail_after_push=True, fork=fork)
    controller = PublicationController(ws, transport)
    record = controller.prepare(draft(suite.datasets[0].dataset_id))
    assert not transport.calls, "previews must stay private and offline"
    assert controller.prepare(draft(record.dataset_id)).publication_id == record.publication_id
    controller.start(record.publication_id, consent(record))
    failed = wait(controller, record)
    assert failed.status == "failed" and failed.commit_sha
    target = transport.fork_repo if fork else transport.upstream
    assert git("--git-dir", str(transport.upstream), "rev-parse", "main") == transport.base
    assert git("--git-dir", str(target), "rev-parse", record.branch) == failed.commit_sha
    changed = git("--git-dir", str(target), "diff-tree", "--no-commit-id", "--name-only", "-r", record.branch).splitlines()
    assert set(changed) == {f"{record.directory}/{f.path}" for f in record.files}
    # Restart recovers the existing branch, does not force push or create an
    # extra commit, then creates the one PR that was missing.
    if fork:
        original = transport.gh
        def existing_fork(*args, **kwargs):
            return "" if args[:2] == ("repo", "fork") else original(*args, **kwargs)
        transport.gh = existing_fork
    resumed = PublicationController(ws, transport)
    resumed.start(record.publication_id, consent(record))
    complete = wait(resumed, record)
    assert complete.status == "submitted" and complete.pull_request_url.endswith("/123")
    assert complete.commit_sha == failed.commit_sha
    calls = len(transport.calls)
    assert resumed.start(record.publication_id, consent(record)).status == "submitted"
    assert len(transport.calls) == calls and len(transport.prs) == 1
    assert not any("merge" in call for call in transport.calls)


def test_api_auth_consent_and_private_preview(data):
    ws, suite = data
    app = create_app(ws.root, bootstrap_token="datasets", shutdown_timeout=2)
    with TestClient(app, base_url="http://127.0.0.1:8877") as client:
        endpoint = "/api/v1/dataset-publications/prepare"
        body = draft(suite.datasets[0].dataset_id).model_dump(mode="json")
        assert client.post(endpoint, json=body).status_code == 401
        auth = client.post("/api/v1/session/bootstrap", json={"token": "datasets"}).json()
        assert client.post(endpoint, json=body).status_code == 403
        client.headers[CSRF_HEADER] = auth["csrf_token"]
        response = client.post(endpoint, json=body)
        assert response.status_code == 201, response.text
        record = response.json()
        assert record["status"] == "prepared" and record["contact_email"] == "emi.lab@outlook.com"
        assert client.get(f"/api/v1/dataset-publications/{record['publication_id']}/download").status_code == 200
        submit = f"/api/v1/dataset-publications/{record['publication_id']}/submit"
        assert client.post(submit, json={"package_sha256": record["package_sha256"], "publish_publicly": False, "rights_confirmed": True}).status_code == 422
        assert client.post(submit, json={"package_sha256": record["package_sha256"], "publish_publicly": True}).status_code == 422
        assert not app.state.dataset_publications.threads
        generated = client.post("/api/v1/datasets/synthetic", json=suite.request.model_dump(mode="json"))
        assert generated.status_code == 201 and len(generated.json()["datasets"]) == 6
