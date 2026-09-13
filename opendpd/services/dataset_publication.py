"""Explicit public dataset contributions through isolated Git branches and PRs.

The current source checkout is never staged, committed or pushed. GitHub CLI
uses the local operator's authenticated identity; credentials never enter the
browser, saved publication or command output. Retries recover an existing branch
and PR. There is deliberately no merge operation.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from opendpd.schemas.benchmark import canonical_sha256
from opendpd.schemas.common import utcnow
from opendpd.schemas.dataset_catalog import (DatasetPublication, DatasetPublicationDraft,
    PublicationCapability, PublicationConsent)
from opendpd.services.dataset_catalog import checked_file, write_package
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, sha256_file, write_json_atomic

REPOSITORY = "lab-emi/OpenDPD"
ACTIVE = {"queued", "branch", "push", "pull_request"}
PUBLICATION_ID = re.compile(r"dspr-[a-f0-9]{64}\Z")
PUBLISH_LOCK = threading.Lock()


class GitHubPublisher:
    repository = REPOSITORY

    def command(self, args: list[str], cwd: Path | None = None, *, optional=False):
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GH_PROMPT_DISABLED": "1", "GIT_ASKPASS": "", "GH_HOST": "github.com"}
        # Do not pass shell source or echo stderr: Git/gh diagnostics can contain
        # credential-helper URLs. The user sees a stage-specific recovery hint.
        try:
            result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            if optional:
                return None
            raise WorkspaceError("GitHub operation did not complete. Check connectivity and gh authentication, then retry this submission.") from None
        if result.returncode:
            if optional:
                return None
            raise WorkspaceError("GitHub operation failed. Check repository access and gh authentication, then retry this submission.")
        return result.stdout.strip()

    def gh(self, *args, cwd=None, optional=False):
        return self.command(["gh", *args], cwd, optional=optional)

    def git(self, directory: Path, *args, optional=False):
        return self.command(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false",
            "-c", "core.autocrlf=false", "-c", "credential.helper=", "-c", "credential.https://github.com.helper=!gh auth git-credential",
            "-C", str(directory), *args], optional=optional)

    def url(self, repository):
        return f"https://github.com/{repository}.git"

    def capability(self):
        available = bool(shutil.which("git") and shutil.which("gh"))
        if not available:
            return PublicationCapability(available=False, reason="Install Git and GitHub CLI (gh) on the Studio host to submit a dataset PR.")
        authenticated = self.gh("auth", "status", "--hostname", "github.com", optional=True) is not None
        return PublicationCapability(available=authenticated, reason=None if authenticated else
            "Sign in on the Studio host with gh auth login --hostname github.com, then refresh. Data remains private until you submit.")

    def _existing_pr(self, head):
        prs = json.loads(self.gh("pr", "list", "--repo", self.repository, "--head", head, "--state", "all",
            "--json", "url,state") or "[]")
        return prs[0] if prs else None

    def publish(self, record: DatasetPublication, package: Path, progress):
        user = json.loads(self.gh("api", "user"))
        login = user.get("login", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", login) or not isinstance(user.get("id"), int):
            raise WorkspaceError("GitHub returned an invalid account identity.")
        repo = json.loads(self.gh("repo", "view", self.repository, "--json", "nameWithOwner,viewerPermission,defaultBranchRef"))
        base = repo["defaultBranchRef"]["name"]
        if self.command(["git", "check-ref-format", "--branch", base], optional=True) is None:
            raise WorkspaceError("The repository's default branch is not valid.")
        push_repo = self.repository
        if repo.get("viewerPermission") not in {"WRITE", "MAINTAIN", "ADMIN"}:
            progress("branch")
            # An isolated temp cwd means --remote=false cannot alter the user's
            # project, even when a fork already exists.
            with tempfile.TemporaryDirectory(prefix="opendpd-fork-") as location:
                self.gh("repo", "fork", self.repository, "--clone=false", "--remote=false", cwd=Path(location))
            push_repo = f"{login}/{self.repository.split('/')[1]}"
            fork = json.loads(self.gh("repo", "view", push_repo, "--json", "isFork,parent"))
            if not fork.get("isFork") or (fork.get("parent") or {}).get("nameWithOwner", "").lower() != self.repository.lower():
                raise WorkspaceError("The account's fork does not match the central dataset repository.")
        head = record.branch if push_repo == self.repository else f"{login}:{record.branch}"
        with tempfile.TemporaryDirectory(prefix="opendpd-dataset-pr-") as temporary:
            checkout = Path(temporary) / "checkout"
            progress("branch")
            self.command(["git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=",
                "-c", "credential.https://github.com.helper=!gh auth git-credential", "clone", "--no-checkout",
                "--filter=blob:none", "--depth=1", "--single-branch", "--branch", base, self.url(self.repository), str(checkout)])
            self.git(checkout, "sparse-checkout", "init", "--cone")
            self.git(checkout, "sparse-checkout", "set", str(Path(record.directory).parent))
            self.git(checkout, "remote", "add", "publication", self.url(push_repo))
            remote = self.git(checkout, "ls-remote", "--heads", "publication", f"refs/heads/{record.branch}")
            if remote:
                self.git(checkout, "fetch", "--depth=2", "--filter=blob:none", "publication", f"refs/heads/{record.branch}")
                self.git(checkout, "checkout", "--detach", "FETCH_HEAD")
                found = set((self.git(checkout, "ls-tree", "-r", "--name-only", "HEAD", "--", record.directory) or "").splitlines())
                expected = {f"{record.directory}/{f.path}" for f in record.files}
                if found != expected:
                    raise WorkspaceError("The submission branch contains a different dataset; automatic overwrite is refused.")
                for ref in record.files:
                    checked_file(checkout / record.directory, ref)
                commit = self.git(checkout, "rev-parse", "HEAD")
                if record.commit_sha and record.commit_sha != commit:
                    raise WorkspaceError("The submission branch changed after publication. Review it on GitHub before retrying.")
            else:
                self.git(checkout, "checkout", "-b", record.branch, f"origin/{base}")
                destination = checkout / record.directory
                if destination.exists():
                    raise WorkspaceError("This dataset directory already exists upstream. Review the existing catalog entry.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(package, destination)
                self.git(checkout, "add", "--", record.directory)
                staged = set((self.git(checkout, "diff", "--cached", "--name-only") or "").splitlines())
                if staged != {f"{record.directory}/{f.path}" for f in record.files}:
                    raise WorkspaceError("The staged files differ from the reviewed data package.")
                for ref in record.files:
                    filename = f"{record.directory}/{ref.path}"
                    if self.git(checkout, "hash-object", "--no-filters", "--", filename) != self.git(checkout, "rev-parse", f":{filename}"):
                        raise WorkspaceError("A Git filter changed the staged data. Review Git attributes before retrying publication.")
                self.git(checkout, "-c", f"user.name={login}", "-c", f"user.email={user['id']}+{login}@users.noreply.github.com",
                    "commit", "-m", f"Add dataset {record.dataset_id}\n\nDataset package SHA256: {record.package_sha256}")
                commit = self.git(checkout, "rev-parse", "HEAD")
                progress("push", commit_sha=commit)
                self.git(checkout, "push", "publication", f"HEAD:refs/heads/{record.branch}")
            progress("pull_request", commit_sha=commit)
            existing = self._existing_pr(head)
            if existing:
                return existing
            body = Path(temporary) / "body.md"
            body.write_text(
                f"Adds a {record.catalog.origin.value} dataset with {record.catalog.n_samples} paired IQ samples "
                f"under `{record.directory}`.\n\n"
                "The data contributor confirmed public disclosure and the license/attribution stored in `dataset.json`. "
                "The package includes canonical finite numeric CSV, declared signal/split metadata and full hashes. "
                "Local paths, private workspace notes and executables are excluded.\n\n"
                f"Package SHA256: `{record.package_sha256}`.\n\n"
                "Human review of data provenance, rights, format and scientific claims is required before merge. "
                "Synthetic datasets demonstrate software behavior; measured labels are contributor declarations. "
                "No automatic merge is requested. Questions: emi.lab@outlook.com.\n", encoding="utf-8")
            url = self.gh("pr", "create", "--repo", self.repository, "--head", head, "--base", base,
                "--title", f"Dataset: {record.dataset_id} ({record.catalog.origin.value})", "--body-file", str(body), cwd=checkout)
            return {"url": url, "state": "OPEN"}


class PublicationController:
    def __init__(self, ws: Workspace, publisher=None):
        self.ws, self.publisher = ws, publisher or GitHubPublisher()
        self.root = ws.root / "dataset-publications"
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.threads: dict[str, threading.Thread] = {}
        self.stopping = threading.Event()
        for record in self.list():
            if record.status in ACTIVE:
                self._update(record.publication_id, status="interrupted", error="Studio restarted during submission. Retry to recover the same branch and PR.")

    def directory(self, identifier):
        if not isinstance(identifier, str) or not PUBLICATION_ID.fullmatch(identifier):
            raise WorkspaceError("Invalid dataset publication identifier.")
        return self.root / identifier

    def get(self, identifier):
        path = self.directory(identifier) / "publication.json"
        if not path.is_file():
            raise WorkspaceError("Dataset publication does not exist in this workspace.")
        return DatasetPublication.model_validate(read_json(path))

    def list(self, dataset_id=None):
        records = [DatasetPublication.model_validate(read_json(p)) for p in self.root.glob("dspr-*/publication.json")]
        return sorted([r for r in records if dataset_id is None or r.dataset_id == dataset_id], key=lambda r: r.created_at, reverse=True)

    def _update(self, identifier, **updates):
        with self.lock:
            record = self.get(identifier).model_copy(update={**updates, "updated_at": utcnow()})
            write_json_atomic(self.directory(identifier) / "publication.json", record)
            return record

    def prepare(self, draft: DatasetPublicationDraft):
        with self.lock, tempfile.TemporaryDirectory(prefix="prepare-", dir=self.root) as temporary:
            source_digest = sha256_file(self.ws.dataset_dir(draft.dataset_id) / "manifest.json")
            package = Path(temporary) / "package"
            catalog, files, digest = write_package(self.ws, draft, package)
            if sha256_file(self.ws.dataset_dir(draft.dataset_id) / "manifest.json") != source_digest:
                raise WorkspaceError("Dataset metadata changed while preparing the preview. Prepare it again.")
            identifier = f"dspr-{digest}"
            directory = self.directory(identifier)
            if (directory / "publication.json").exists():
                old = self.get(identifier)
                if old.status in {"prepared", "failed", "interrupted"} and old.source_manifest_sha256 != source_digest:
                    return self._update(identifier, source_manifest_sha256=source_digest, status="prepared", consent_at=None, error=None)
                return old
            record = DatasetPublication(publication_id=identifier, dataset_id=draft.dataset_id,
                package_sha256=digest, source_manifest_sha256=source_digest,
                branch=f"codex/dataset-{draft.dataset_id}-{digest}",
                directory=f"dataset/community/{catalog.origin.value}/{draft.dataset_id}/{digest}",
                files=files, catalog=catalog)
            directory.mkdir(exist_ok=True)
            shutil.move(str(package), directory / "package")
            write_json_atomic(directory / "publication.json", record)
            return record

    def start(self, identifier: str, consent: PublicationConsent):
        with self.lock:
            record = self.get(identifier)
            if consent.package_sha256 != record.package_sha256:
                raise WorkspaceError("Publication consent must match the exact reviewed package.")
            if record.status == "submitted" or identifier in self.threads:
                return record
            if self.threads:
                raise WorkspaceError("Another dataset publication is running. Wait for it to finish before submitting this package.")
            if sha256_file(self.ws.dataset_dir(record.dataset_id) / "manifest.json") != record.source_manifest_sha256:
                raise WorkspaceError("Dataset metadata changed after preview. Prepare a new publication before submitting.")
            package = self.directory(identifier) / "package"
            for ref in self.ws.get_dataset(record.dataset_id).files:
                checked_file(self.ws.dataset_dir(record.dataset_id), ref)
            for ref in record.files:
                checked_file(package, ref)
            if {p.name for p in package.iterdir()} != {f.path for f in record.files}:
                raise WorkspaceError("The preview package contains unexpected files.")
            if canonical_sha256([f.model_dump(mode="json") for f in record.files]) != record.package_sha256:
                raise WorkspaceError("The publication package record changed.")
            record = self._update(identifier, status="queued", consent_at=utcnow(), error=None)
            thread = threading.Thread(target=self._run, args=(identifier,), daemon=True, name="dataset-publication")
            self.threads[identifier] = thread
            thread.start()
            return record

    def _run(self, identifier):
        try:
            def progress(status, **fields):
                if self.stopping.is_set():
                    raise WorkspaceError("Studio stopped during submission. Retry to recover the same branch and PR.")
                self._update(identifier, status=status, **fields)
            with PUBLISH_LOCK:
                if self.stopping.is_set():
                    return
                record = self.get(identifier)
                result = self.publisher.publish(record, self.directory(identifier) / "package", progress)
            url = result.get("url", "")
            if not re.fullmatch(r"https://github\.com/lab-emi/OpenDPD/pull/[0-9]+", url):
                raise WorkspaceError("The GitHub response did not contain a verified dataset PR URL. Retry to recover its status.")
            self._update(identifier, status="submitted", pull_request_url=url,
                pull_request_state=result.get("state"), error=None)
        except Exception as exc:
            self._update(identifier, status="interrupted" if self.stopping.is_set() else "failed", error=str(exc) if isinstance(exc, WorkspaceError) else
                "Submission failed. Check the Studio host's GitHub connection and retry the same publication.")
        finally:
            with self.lock:
                self.threads.pop(identifier, None)

    def stop(self):
        # Network subprocesses are bounded. Do not lose persisted state or block
        # Studio shutdown; recovery reconciles the same branch/PR on next start.
        self.stopping.set()
        with self.lock:
            for identifier in self.threads:
                if self.get(identifier).status in ACTIVE:
                    self._update(identifier, status="interrupted", error="Studio stopped during submission. Retry to recover the same branch and PR.")
