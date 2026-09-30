"""Private template storage, reviewed catalog and opt-in GitHub contributions.

Source files live outside Python's import path. Catalog entries and API configs
carry bounded JSON graphs, not filenames or executable source. No import/exec
or model deserialization takes place during scanning or catalog refresh.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from opendpd.core.backbone_template import (MAX_SOURCE_BYTES, TemplateError, canonical_definition,
    scan_source, source_sha256, validate_definition)
from opendpd.schemas.benchmark import canonical_sha256
from opendpd.schemas.common import FileRef, utcnow
from opendpd.schemas.experiment import ModelSpec
from opendpd.schemas.user_backbones import (BackboneCatalog, BackboneConsent, BackboneEntry, BackboneUpload)
from opendpd.services.dataset_catalog import checked_file
from opendpd.services.dataset_publication import ACTIVE, GitHubPublisher, PublicationController
from opendpd.services.workspace import (InvalidInput, Workspace, WorkspaceError, read_json, sha256_file,
    write_atomic, write_json_atomic)

REPOSITORY = "lab-emi/OpenDPD"
CATALOG_DIRECTORY = "backbones/user_uploaded"
BUNDLED_CATALOG = Path(__file__).resolve().parents[2] / CATALOG_DIRECTORY
REVIEW_CHECK = "user-backbone-review"
GITHUB_ACTIONS_APP_ID = 15368
ENTRY_DIRECTORY = re.compile(r"[a-z][a-z0-9_]{0,39}_[a-f0-9]{16}\Z")


def entry_from_source(source: bytes, *, origin="private", commit=None) -> BackboneEntry:
    definition = scan_source(source)
    if origin == "community" and not definition["author"].strip():
        raise TemplateError("Community backbones need public author attribution.")
    encoded = canonical_definition(definition)
    report = validate_definition(definition)
    digest = source_sha256(source)
    return BackboneEntry(backbone_id=f"ub-{digest}", name=definition["name"], description=definition["description"],
        author=definition["author"], source_sha256=digest, definition_sha256=source_sha256(encoded.encode()),
        parameter_count=report["parameters"], node_count=report["nodes"],
        model=ModelSpec(key="user_template", parameters={"definition": encoded}), origin=origin, source_commit=commit)


def package_manifest(source: bytes):
    return {"format": "opendpd-backbone-template-v1", "source_sha256": source_sha256(source)}


class BackbonePublisher(GitHubPublisher):
    """Only publishes a new contribution branch; no merge or rules mutation API."""

    def review_gate_available(self):
        # Fail closed until a repository administrator deploys the trusted
        # workflow AND an active no-bypass ruleset for main. Do not silently
        # weaken existing rules or assume that CODEOWNERS enforces review.
        raw = self.gh("api", f"repos/{self.repository}/rules/branches/main", optional=True)
        if raw is None:
            return False
        try:
            rules = json.loads(raw)
            ids = {rule["ruleset_id"] for rule in rules if rule.get("ruleset_source_type") == "Repository"}
            for identifier in ids:
                if type(identifier) is not int:
                    continue
                raw_rule = self.gh("api", f"repos/{self.repository}/rulesets/{identifier}", optional=True)
                if raw_rule is None:
                    continue
                rule = json.loads(raw_rule)
                if rule.get("enforcement") != "active" or rule.get("bypass_actors") != []:
                    continue
                policies = {r["type"]: r.get("parameters", {}) for r in rule.get("rules", [])}
                reviews, checks = policies.get("pull_request", {}), policies.get("required_status_checks", {})
                if (reviews.get("required_approving_review_count", 0) >= 1
                        and reviews.get("dismiss_stale_reviews_on_push")
                        and reviews.get("require_last_push_approval") and reviews.get("require_code_owner_review")
                        and checks.get("strict_required_status_checks_policy")
                        and any(c.get("context") == REVIEW_CHECK and c.get("integration_id") == GITHUB_ACTIONS_APP_ID
                                for c in checks.get("required_status_checks", []))):
                    return True
        except (ValueError, KeyError, TypeError):
            pass
        return False

    def capability(self):
        from opendpd.schemas.user_backbones import BackboneCapability
        base = super().capability()
        if not base.available:
            return BackboneCapability(reason="Git and an authenticated GitHub CLI are required on the Studio host for contributions. Private uploads remain available.")
        if not self.review_gate_available():
            return BackboneCapability(reason="Repository setup required: enable the no-bypass human review ruleset and user-backbone-review check on main before accepting contributions. Private uploads remain available.")
        return BackboneCapability(publication_available=True)

    def publish(self, record, package, progress):
        if not self.review_gate_available():
            raise WorkspaceError("The required human review gate is unavailable. The upload remains private; ask a repository administrator to enable it.")
        verify_package(record, package)
        return super().publish(record, package, progress)

    def commit_message(self, record):
        return f"Add backbone template {record.backbone_id}\n\nSource SHA256: {record.source_sha256}"

    def validate_base(self, base):
        if base != "main":
            raise WorkspaceError("Backbone contributions must target the protected main branch.")

    def validate_checkout(self, record, checkout, base):
        expected = {f"{record.directory}/{ref.path}" for ref in record.files}
        changed = set(self.git(checkout, "diff", "--name-only", f"origin/{base}", "HEAD").splitlines())
        if changed != expected:
            raise WorkspaceError("The contribution branch differs from the validated package or main has advanced. Automatic publication is refused; review the branch on GitHub.")

    def pr_title(self, record):
        return f"Backbone template: {record.name}"

    def pr_body(self, record):
        return (f"Adds a template v1 backbone under `{record.directory}`.\n\n"
            f"Source SHA256: `{record.source_sha256}`\n\nPackage SHA256: `{record.package_sha256}`\n\n"
            f"Graph: {record.node_count} nodes, {record.parameter_count} trainable parameters; IQ input/output, causal, frame-local state.\n\n"
            "The contributor explicitly authorized publishing the entire source and public attribution under Apache-2.0. "
            "The template is parsed as bounded data; uploaded Python is never executed. The scan is a format and resource check, "
            "not a malware-free certification or a claim of model accuracy.\n\n"
            "Human review is required: an independent maintainer must approve the current head, including architecture, "
            "licensing, attribution and scientific claims. Bots, comments, self-approval and stale approvals do not count. "
            "Studio never merges PRs or enables auto-merge. After merge to main, refresh User Uploaded Backbones in Studio.\n")


def verify_package(record: BackboneUpload, package: Path):
    if package.is_symlink() or not package.is_dir():
        raise WorkspaceError("Backbone package storage must be a regular directory, not a symbolic link.")
    if set(p.name for p in package.iterdir()) != {"backbone.py", "manifest.json"}:
        raise WorkspaceError("The backbone package contains unexpected files.")
    if {f.path for f in record.files} != {"backbone.py", "manifest.json"} or len(record.files) != 2:
        raise WorkspaceError("Invalid backbone file list.")
    for ref in record.files:
        checked_file(package, ref)
    if (package / "backbone.py").stat().st_size > MAX_SOURCE_BYTES:
        raise WorkspaceError("The source exceeds the template size limit.")
    source = (package / "backbone.py").read_bytes()
    try:
        fresh = entry_from_source(source)
    except TemplateError as exc:
        raise InvalidInput(str(exc)) from None
    if (fresh.model_dump() != record.model_dump(include=set(BackboneEntry.model_fields))
            or read_json(package / "manifest.json") != package_manifest(source)
            or canonical_sha256([f.model_dump(mode="json") for f in record.files]) != record.package_sha256):
        raise WorkspaceError("The backbone changed after validation. Upload it again; old consent cannot be reused.")
    slug = re.sub(r"[^a-z0-9]+", "_", fresh.name.lower()).strip("_")[:36] or "backbone"
    if not slug[0].isalpha():
        slug = "b_" + slug
    expected_dir = f"{CATALOG_DIRECTORY}/{slug}_{fresh.source_sha256[:16]}"
    if (record.directory != expected_dir or record.branch != f"codex/backbone-{fresh.source_sha256}"
            or record.publication_id != f"bbpr-{fresh.source_sha256}"):
        raise WorkspaceError("Invalid backbone contribution destination.")


class BackboneController(PublicationController):
    """Reuse persisted publication progress/recovery, with backbone-only inputs."""
    def __init__(self, ws: Workspace, publisher=None):
        self.ws, self.publisher = ws, publisher or BackbonePublisher()
        self.root = ws.root / "user-backbones"
        if self.root.is_symlink():
            raise WorkspaceError("Backbone storage cannot be a symbolic link.")
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.threads = {}
        self.stopping = threading.Event()
        self.catalog_lock = threading.Lock()
        self.last_catalog_refresh = float("-inf")
        for record in self.list():
            if record.status in ACTIVE:
                self._update(record.publication_id, status="interrupted", error="Studio restarted during submission. Retry to recover the same branch and PR.")

    def directory(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"bbpr-[a-f0-9]{64}", identifier):
            raise InvalidInput("Invalid backbone identifier.")
        return self.ws.hashed_store("user-backbones", "bbpr").directory(identifier)

    def get(self, identifier):
        path = self.directory(identifier) / "publication.json"
        if path.is_symlink() or not path.is_file():
            raise WorkspaceError("Backbone upload does not exist in this workspace.")
        return BackboneUpload.model_validate(read_json(path))

    def list(self):
        return sorted((self.get(p.parent.name) for p in self.root.glob("bbpr-*/publication.json")),
                      key=lambda item: item.created_at, reverse=True)

    def upload(self, filename: str, source: bytes):
        try:
            scan_source(source, filename)
            entry = entry_from_source(source)
        except TemplateError as exc:
            raise InvalidInput(str(exc)) from None
        with self.lock:
            identifier = f"bbpr-{entry.source_sha256}"
            directory = self.directory(identifier)
            if (directory / "publication.json").exists():
                record = self.get(identifier)
                verify_package(record, directory / "package")
                return record
            if len(self.list()) >= 64:
                raise InvalidInput("This workspace already has 64 uploaded backbones.")
            package = directory / "package"
            if package.is_symlink():
                raise WorkspaceError("Backbone package storage cannot be a symbolic link.")
            package.mkdir(parents=True, exist_ok=True)
            write_atomic(package / "backbone.py", lambda p: p.write_bytes(source))
            write_json_atomic(package / "manifest.json", package_manifest(source))
            files = [FileRef(path=name, sha256=sha256_file(package / name), size_bytes=(package / name).stat().st_size)
                     for name in ("backbone.py", "manifest.json")]
            digest = canonical_sha256([ref.model_dump(mode="json") for ref in files])
            slug = re.sub(r"[^a-z0-9]+", "_", entry.name.lower()).strip("_")[:36] or "backbone"
            if not slug[0].isalpha():
                slug = "b_" + slug
            record = BackboneUpload(**entry.model_dump(), publication_id=identifier, package_sha256=digest,
                directory=f"{CATALOG_DIRECTORY}/{slug}_{entry.source_sha256[:16]}",
                branch=f"codex/backbone-{entry.source_sha256}", files=files)
            write_json_atomic(directory / "publication.json", record)
            return record

    def start(self, identifier: str, consent: BackboneConsent):
        with self.lock:
            record = self.get(identifier)
            if (consent.source_sha256 != record.source_sha256 or consent.package_sha256 != record.package_sha256
                    or consent.publish_publicly is not True or consent.rights_confirmed is not True):
                raise InvalidInput("Consent must match the exact validated source and package hashes.")
            verify_package(record, self.directory(identifier) / "package")
            if not record.author.strip():
                raise InvalidInput("Add your public author attribution to the template before contributing.")
            if record.status == "submitted" or identifier in self.threads:
                return record
            if self.threads or self.stopping.is_set():
                raise WorkspaceError("Another submission is in progress or Studio is stopping. Retry later.")
            capability = self.publisher.capability()
            if not capability.publication_available:
                raise WorkspaceError(capability.reason or "Backbone contributions are unavailable.")
            record = self._update(identifier, status="queued", consent_at=utcnow(), error=None)
            thread = threading.Thread(target=self._run, args=(identifier,), daemon=True, name="backbone-publication")
            self.threads[identifier] = thread
            thread.start()
            return record

    def catalog(self):
        cached = self.root / "community.json"
        if self.root.is_symlink() or cached.is_symlink():
            raise WorkspaceError("Backbone catalog storage cannot be a symbolic link.")
        if cached.is_file():
            saved = BackboneCatalog.model_validate(read_json(cached))
            # Cached graphs remain subject to the current scanner's limits.
            from opendpd.core.registry import validate_parameters
            for entry in saved.entries:
                validate_parameters(entry.model.key, entry.model.parameters, "pa")
            return saved
        entries = []
        for folder in sorted(BUNDLED_CATALOG.glob("*")):
            if not folder.is_dir() or not ENTRY_DIRECTORY.fullmatch(folder.name):
                continue
            if (folder.is_symlink() or (folder / "backbone.py").is_symlink()
                    or (folder / "manifest.json").is_symlink() or (folder / "backbone.py").stat().st_size > MAX_SOURCE_BYTES):
                raise WorkspaceError("A bundled backbone is not a bounded regular file.")
            source = (folder / "backbone.py").read_bytes()
            if read_json(folder / "manifest.json") != package_manifest(source):
                raise WorkspaceError("A bundled backbone failed its integrity check.")
            entries.append(entry_from_source(source, origin="community"))
        return BackboneCatalog(entries=entries)

    def refresh_catalog(self, get=None):
        """Fetch only main at an immutable commit, never PR branches or code URLs."""
        with self.catalog_lock:
            if time.monotonic() - self.last_catalog_refresh < 60:
                return self.catalog()
            self.last_catalog_refresh = time.monotonic()
            return self._refresh_catalog(get)

    def _refresh_catalog(self, get=None):
        get = get or public_github_json
        deadline = time.monotonic() + 40
        try:
            ref = get("git/ref/heads/main")
            commit = ref["object"]["sha"]
            if not re.fullmatch(r"[a-f0-9]{40}", commit):
                raise ValueError("invalid commit")
            try:
                folders = get(f"contents/{CATALOG_DIRECTORY}?ref={commit}")
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    folders = []
                else:
                    raise
            if not isinstance(folders, list) or len(folders) > 128:
                raise ValueError("catalog size")
            entries = []
            for folder in folders:
                if time.monotonic() > deadline:
                    raise ValueError("catalog refresh deadline exceeded")
                name = folder.get("name", "")
                if folder.get("type") != "dir" or not ENTRY_DIRECTORY.fullmatch(name):
                    continue
                tree_sha = folder.get("sha", "")
                if not re.fullmatch(r"[a-f0-9]{40}", tree_sha):
                    raise ValueError("invalid catalog tree")
                tree = get(f"git/trees/{tree_sha}")
                nodes = tree.get("tree", [])
                if (tree.get("truncated") or len(nodes) != 2
                        or {n.get("path") for n in nodes} != {"backbone.py", "manifest.json"}
                        or any(n.get("type") != "blob" or n.get("mode") != "100644" for n in nodes)):
                    raise ValueError("catalog packages must contain exactly two regular, non-executable files")
                files = {}
                for node in nodes:
                    filename, blob_sha = node["path"], node.get("sha", "")
                    if not re.fullmatch(r"[a-f0-9]{40}", blob_sha):
                        raise ValueError("invalid catalog blob")
                    item = get(f"git/blobs/{blob_sha}")
                    if (item.get("sha") != blob_sha or item.get("encoding") != "base64"
                            or not isinstance(item.get("size"), int) or not 1 <= item["size"] <= MAX_SOURCE_BYTES
                            or len(item.get("content", "")) > 48_000):
                        raise ValueError("invalid catalog file")
                    content = base64.b64decode(item["content"].replace("\n", ""), validate=True)
                    if len(content) != item["size"]:
                        raise ValueError("file size mismatch")
                    files[filename] = content
                if json.loads(files["manifest.json"]) != package_manifest(files["backbone.py"]):
                    raise ValueError("source hash mismatch")
                entry = entry_from_source(files["backbone.py"], origin="community", commit=commit)
                if name.split("_")[-1] != entry.source_sha256[:16]:
                    raise ValueError("directory digest mismatch")
                entries.append(entry)
            if len({e.backbone_id for e in entries}) != len(entries):
                raise ValueError("duplicate catalog backbone")
            result = BackboneCatalog(entries=entries, commit=commit)
            with self.lock:
                write_json_atomic(self.root / "community.json", result)
            return result
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            # Atomic all-or-nothing refresh: keep last usable catalog on outage,
            # incompatible format or an invalid upstream entry.
            return self.catalog().model_copy(update={"warning": "Could not verify the community catalog from GitHub main. The previous catalog is still available; try Refresh later."})


def public_github_json(path):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    request = urllib.request.Request(f"https://api.github.com/repos/{REPOSITORY}/{path}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "OpenDPD-Studio-backbone-catalog"})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=10) as response:
        body = response.read(256 * 1024 + 1)
    if len(body) > 256 * 1024:
        raise ValueError("GitHub response exceeds catalog limit")
    return json.loads(body)
