"""Reproducible experiment packages (plan S11): export a run, inspect a package, import it elsewhere.

A *full* package is private: it carries the run directory, the checkpoints of
every referenced run and, for imported data, the raw copy and the data
version the run used. A *share* package leaves the user's PA data, worker
logs and every machine path out and says so. Both carry the resolved
configuration, the result under every profile, the derived plot data, the
reports and the commands to reproduce. Every member is hashed; nothing is
imported before every hash matches.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import stat
import socket
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

from opendpd import __version__
from opendpd.schemas import (
    ArtifactKind,
    ArtifactManifest,
    DatasetSourceKind,
    ImportReport,
    PACKAGE_VERSION,
    PackageDataset,
    PackageFile,
    PackageManifest,
    PackageReference,
    RunRecord,
    TaskType,
)
from opendpd.services import experiments
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, sha256_file, software_provenance

MANIFEST_NAME = "package.json"
MAX_PACKAGE_MEMBERS = 10_000            # a package is one run, its references and one dataset
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
REPORT_HTML = "report.html"
REPORT_MD = "report.md"
ARTIFACTS_FILE = "artifacts.json"
_RUN_JSON = ("run.json", "config.user.json", "config.resolved.json", "provenance.json", ARTIFACTS_FILE)
_SHARE_EXCLUDED_DIRS = ("logs",)          # worker / stdout logs may quote machine paths
_SHARE_EXCLUDED_FILES = ("events.jsonl",)
RETRAINING_NOTE = ("Re-evaluating the packaged checkpoint reproduces the stored metrics within the frozen tolerance "
                   "(docs/protocols/acceptance-thresholds.md). Re-training from config.user.json is a new experiment: "
                   "its agreement with the packaged numbers depends on the reproducibility mode, the device and the "
                   "software versions, and is reported separately; it is never implied by a successful re-evaluation.")


class PackageError(WorkspaceError):
    """A package that must not be used, with a specific diagnostic."""

    def __init__(self, code: str, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.hint = hint


# --- export -----------------------------------------------------------------------------

def _secrets(ws: Workspace) -> List[Tuple[str, str]]:
    """Strings a share package must not contain, with their replacements."""
    out = [(str(ws.root.resolve()), "<workspace>"), (str(ws.root), "<workspace>")]
    home = str(Path.home())
    if home and home != "/":
        out.append((home, "<home>"))
    host = socket.gethostname()
    if host:
        out.append((host, "<host>"))
    return out


def _redact_text(text: str, secrets: List[Tuple[str, str]]) -> str:
    for secret, replacement in sorted(secrets, key=lambda s: -len(s[0])):
        text = text.replace(secret, replacement)
    return text


def _redacted_json(path: Path, secrets: List[Tuple[str, str]], drop_worker: bool = False,
                   drop_original_path: bool = False) -> bytes:
    data = read_json(path)
    if drop_worker and isinstance(data, dict):
        data["worker"] = None
    if drop_original_path and isinstance(data, dict) and isinstance(data.get("source"), dict):
        data["source"]["original_path"] = None
    return _redact_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), secrets).encode("utf-8")


def _walk(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


_REFERENCE_KINDS = (ArtifactKind.checkpoint, ArtifactKind.result, ArtifactKind.log_history, ArtifactKind.log_best)


def _reference_members(ws: Workspace, run_id: str) -> Iterable[Tuple[str, Path]]:
    """The JSON files, results, metric logs and checkpoint of a referenced run: enough to evaluate through it and to
    read what it scored; worker logs and plot data stay behind."""
    run_dir = ws.run_dir(run_id)
    for name in _RUN_JSON:
        if (run_dir / name).exists():
            yield f"refs/{run_id}/{name}", run_dir / name
    manifest = experiments.load_artifacts(ws, run_id)
    for artifact in manifest.artifacts if manifest else []:
        if artifact.kind in _REFERENCE_KINDS and (run_dir / artifact.file.path).is_file():
            yield f"refs/{run_id}/{artifact.file.path}", run_dir / artifact.file.path


def _artifacts_in_package(path: Path, prefix: str, present: set) -> Tuple[bytes, List[str]]:
    """artifacts.json restricted to the members that travel with the package, so an imported run never lists a file
    it does not have. Returns the bytes and the ids that were dropped; ``complete`` is cleared if a required one was."""
    manifest = ArtifactManifest.model_validate(read_json(path))
    kept = [a for a in manifest.artifacts if f"{prefix}/{a.file.path}" in present]
    dropped = [a for a in manifest.artifacts if f"{prefix}/{a.file.path}" not in present]
    complete = manifest.complete and not any(a.required for a in dropped)
    payload = manifest.model_copy(update={"artifacts": kept, "complete": complete}).model_dump_json(indent=2)
    return payload.encode("utf-8"), [a.artifact_id for a in dropped]


def export_run(ws: Workspace, run_id: str, out: Path, *, kind: str = "share", language: str = "en") -> PackageManifest:
    """Write ``out`` (a zip) for ``run_id``. ``kind`` is ``full`` (private, complete) or ``share`` (redacted)."""
    if kind not in ("full", "share"):
        raise PackageError("invalid_kind", f"package kind must be full or share, not '{kind}'")
    record = experiments.load_run(ws, run_id)
    resolved = experiments.load_resolved(ws, run_id)
    result = experiments.load_result(ws, run_id)
    dataset = ws.get_dataset(resolved.dataset.id)
    share = kind == "share"
    secrets = _secrets(ws)
    run_dir = ws.run_dir(run_id)
    redaction: List[str] = []
    missing: List[str] = []

    members: List[Tuple[str, Path]] = []          # (archive name, source file) copied verbatim
    rewritten: Dict[str, bytes] = {}              # archive name -> redacted bytes
    for path in _walk(run_dir):
        rel = path.relative_to(run_dir).as_posix()
        if share and (rel.split("/")[0] in _SHARE_EXCLUDED_DIRS or rel in _SHARE_EXCLUDED_FILES):
            continue
        name = f"run/{run_id}/{rel}"
        if share and rel in ("provenance.json", "run.json"):
            rewritten[name] = _redacted_json(path, secrets, drop_worker=(rel == "run.json"))
        else:
            members.append((name, path))
    if share:
        redaction += ["worker logs (logs/) and the event journal are not included",
                      "provenance.json: machine paths replaced by <workspace> / <home> / <host>",
                      "run.json: worker identity (host, pid) removed"]
        present = {name for name, _ in members} | set(rewritten)
        payload, dropped = _artifacts_in_package(run_dir / ARTIFACTS_FILE, f"run/{run_id}", present)
        if dropped:
            members = [(name, path) for name, path in members if name != f"run/{run_id}/{ARTIFACTS_FILE}"]
            rewritten[f"run/{run_id}/{ARTIFACTS_FILE}"] = payload
            redaction.append(f"artifacts.json: entries for files not included were removed ({', '.join(dropped)})")

    references: List[PackageReference] = []
    ref_ids: List[Tuple[str, str]] = []
    if resolved.pa_reference is not None and resolved.task != TaskType.train_pa:
        ref_ids.append((resolved.pa_reference.run_id, "pa_surrogate"))
    if resolved.dpd_reference is not None:
        ref_ids.append((resolved.dpd_reference.run_id, "dpd_model"))
    for ref_id, role in ref_ids:
        ref_manifest = experiments.load_artifacts(ws, ref_id)
        ckpt = ref_manifest.by_kind(ArtifactKind.checkpoint)[0] if ref_manifest else None
        if ckpt is None or ckpt.file.sha256 is None:
            raise PackageError("reference_incomplete", f"referenced run {ref_id} has no verified checkpoint")
        ref_members = list(_reference_members(ws, ref_id))
        present = {name for name, _ in ref_members}
        for name, path in ref_members:
            if name.endswith(f"/{ARTIFACTS_FILE}"):
                rewritten[name] = _artifacts_in_package(path, f"refs/{ref_id}", present)[0]
            elif share and name.endswith(("/provenance.json", "/run.json")):
                rewritten[name] = _redacted_json(path, secrets, drop_worker=name.endswith("/run.json"))
            else:
                members.append((name, path))
        references.append(PackageReference(run_id=ref_id, role=role, checkpoint_sha256=ckpt.file.sha256))

    builtin = dataset.source.kind == DatasetSourceKind.builtin
    include_data = not share and not builtin
    rewritten["dataset/manifest.json"] = _redacted_json(ws.dataset_dir(dataset.dataset_id) / "manifest.json", secrets,
                                                        drop_original_path=share)
    if share:
        redaction.append("dataset/manifest.json: the original import path is removed")
    if include_data:
        data_dir = ws.dataset_dir(dataset.dataset_id)
        wanted = [data_dir / "raw"]
        for version in {resolved.dataset.preprocessing_version, "raw-v1"}:
            if (data_dir / "versions" / version).is_dir():
                wanted.append(data_dir / "versions" / version)
        for base in wanted:
            for path in _walk(base):
                members.append((f"dataset/{path.relative_to(data_dir).as_posix()}", path))
    if builtin:
        how = (f"built-in dataset '{dataset.source.name}' ships with every OpenDPD install; the importer registers it "
               f"and checks its raw sha256 {dataset.raw_sha256}")
    elif include_data:
        how = "included: dataset/raw and the used data version"
    else:
        how = (f"ask the author for dataset '{dataset.dataset_id}' (raw sha256 {dataset.raw_sha256}) and import it "
               f"with `opendpd datasets import` under the same id; the hash check verifies consistency only, it "
               "cannot replace the data")
        missing.append(f"dataset '{dataset.dataset_id}' raw data and version '{resolved.dataset.preprocessing_version}' "
                       f"(raw sha256 {dataset.raw_sha256}): {how}")
        redaction.append("the user's PA data (raw copy and data versions) is not included")

    from opendpd.services.reports import report_html, report_markdown
    rewritten[REPORT_HTML] = _redact_text(report_html(ws, run_id, language=language), secrets).encode("utf-8")
    rewritten[REPORT_MD] = _redact_text(report_markdown(ws, run_id, language=language), secrets).encode("utf-8")

    reproduction = {
        "import": f"opendpd import {out.name} --workspace <workspace>",
        "evaluate": f"opendpd evaluate {run_id} --workspace <workspace> --profile "
                    f"{resolved.evaluation.profile_id}",
        "rerun": f"opendpd run --config run/{run_id}/config.user.json --workspace <workspace>",
    }
    legacy = read_json(run_dir / "provenance.json").get("legacy_equivalent_command")
    if legacy:
        reproduction["legacy"] = _redact_text(str(legacy), secrets)

    files: List[PackageFile] = []
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, path in members:
            zf.write(path, name)
            files.append(PackageFile(path=name, sha256=sha256_file(path), size_bytes=path.stat().st_size))
        for name, payload in rewritten.items():
            zf.writestr(name, payload)
            files.append(PackageFile(path=name, sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)))
        manifest = PackageManifest(
            kind=kind, opendpd_version=__version__, software=software_provenance(), run_id=run_id, task=record.task,
            config_sha256=resolved.resolution.config_sha256, seed=resolved.training.seed,
            result_id=result.result_id if result else None,
            metric_profile_id=result.metric_profile_id if result else None,
            dataset=PackageDataset(dataset_id=dataset.dataset_id, raw_sha256=dataset.raw_sha256,
                                   preprocessing_version=resolved.dataset.preprocessing_version,
                                   split_version=resolved.dataset.split_version, source_kind=dataset.source.kind,
                                   builtin_name=dataset.source.name if builtin else None, included=include_data,
                                   how_to_obtain=how),
            references=references, files=sorted(files, key=lambda f: f.path), reproduction=reproduction,
            redaction=redaction, missing=missing, retraining_note=RETRAINING_NOTE,
        )
        zf.writestr(MANIFEST_NAME, manifest.model_dump_json(indent=2))
    tmp.replace(out)
    return manifest


# --- inspect ----------------------------------------------------------------------------

def _safe_member(name: str, info: Optional[zipfile.ZipInfo] = None) -> None:
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or ".." in parts or ":" in parts[0] or "\\" in name:
        raise PackageError("unsafe_path", f"package member '{name}' would escape the workspace")
    if info is not None:
        # Unix type bits live in the high 16 bits of external_attr; entries written without them
        # (plain writestr, Windows archivers) are ordinary files. Anything typed as a symlink,
        # device or pipe is refused before a single byte is read.
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind and kind not in (stat.S_IFREG, stat.S_IFDIR):
            raise PackageError("unsafe_member", f"package member '{name}' is not a regular file (symlinks and "
                                                "special files are never extracted)")


def _member_sha256(zf: zipfile.ZipFile, name: str, expected_size: int) -> Tuple[str, int]:
    """Hash a member, reading never more than one byte past its recorded size."""
    digest, total = hashlib.sha256(), 0
    with zf.open(name) as f:
        while True:
            chunk = f.read(min(1 << 20, expected_size - total + 1))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total > expected_size:
                raise PackageError("hash_mismatch", f"package member '{name}' is larger than its recorded size; "
                                                    "the package was modified or damaged")
    return digest.hexdigest(), total


def inspect_package(path: Path) -> PackageManifest:
    """Validate a package without importing anything: version, member list, every hash."""
    path = Path(path)
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise PackageError("not_a_package", f"{path} is not an OpenDPD experiment package (zip)")
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_PACKAGE_MEMBERS:
            raise PackageError("too_many_members", f"package lists {len(infos)} members; at most {MAX_PACKAGE_MEMBERS} "
                                                   "are accepted")
        for info in infos:
            _safe_member(info.filename, info)
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            raise PackageError("manifest_missing", f"{path.name} has no {MANIFEST_NAME}")
        if zf.getinfo(MANIFEST_NAME).file_size > MAX_MANIFEST_BYTES:
            raise PackageError("manifest_invalid", f"{MANIFEST_NAME} is larger than {MAX_MANIFEST_BYTES} bytes")
        try:
            raw = json.loads(zf.read(MANIFEST_NAME))
        except ValueError as err:
            raise PackageError("manifest_invalid", f"{MANIFEST_NAME} is not valid JSON: {err}") from None
        version = raw.get("package_version") if isinstance(raw, dict) else None
        if version != PACKAGE_VERSION:
            raise PackageError("unsupported_version",
                               f"package version {version!r} is not supported (this OpenDPD reads version {PACKAGE_VERSION})",
                               hint="export the run again with a matching OpenDPD version")
        try:
            manifest = PackageManifest.model_validate(raw)
        except Exception as err:  # noqa: BLE001 - pydantic detail is the diagnostic
            raise PackageError("manifest_invalid", f"{MANIFEST_NAME} does not match the contract: {err}") from None
        listed = {f.path for f in manifest.files}
        extra = names - listed - {MANIFEST_NAME}
        if extra:
            raise PackageError("unlisted_file", f"package contains files not covered by the manifest: {sorted(extra)[:5]}")
        for entry in manifest.files:
            _safe_member(entry.path)
            if entry.path not in names:
                raise PackageError("missing_file", f"package member '{entry.path}' listed in the manifest is missing")
            digest, size = _member_sha256(zf, entry.path, entry.size_bytes)
            if digest != entry.sha256 or size != entry.size_bytes:
                raise PackageError("hash_mismatch", f"package member '{entry.path}' does not match its recorded sha256; "
                                                    "the package was modified or damaged")
        needed = f"run/{manifest.run_id}/config.resolved.json"
        if needed not in names:
            raise PackageError("missing_file", f"package member '{needed}' is missing")
    return manifest


# --- import -----------------------------------------------------------------------------

def _extract(zf: zipfile.ZipFile, prefix: str, target: Path, sizes: Dict[str, int]) -> List[str]:
    """Write the members under ``prefix`` into ``target``; a member may never exceed its recorded size."""
    written = []
    root = target.resolve()
    for name in zf.namelist():
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        rel = name[len(prefix):]
        dest = target / rel
        if root not in dest.resolve().parents:
            raise PackageError("unsafe_path", f"package member '{name}' would escape the workspace")
        dest.parent.mkdir(parents=True, exist_ok=True)
        remaining = sizes.get(name, 0)
        with zf.open(name) as src, open(dest, "wb") as dst:
            while True:
                chunk = src.read(min(1 << 20, remaining + 1))
                if not chunk:
                    break
                remaining -= len(chunk)
                if remaining < 0:
                    dst.close()
                    dest.unlink(missing_ok=True)
                    raise PackageError("hash_mismatch", f"package member '{name}' is larger than its recorded size")
                dst.write(chunk)
        written.append(rel)
    return written


def import_package(ws: Workspace, path: Path) -> ImportReport:
    """Import a validated package into ``ws``. Nothing is written before every check passed."""
    manifest = inspect_package(path)
    run_id = manifest.run_id
    sizes = {f.path: f.size_bytes for f in manifest.files}
    free = shutil.disk_usage(ws.root).free
    if sum(sizes.values()) > free:
        raise PackageError("insufficient_space", f"the package unpacks to {sum(sizes.values())} bytes but the workspace "
                                                 f"volume has {free} bytes free")
    if ws.run_dir(run_id).exists():
        raise PackageError("run_exists", f"run '{run_id}' already exists in {ws.root}",
                           hint="import into another workspace, or remove the existing run directory first")
    with zipfile.ZipFile(path) as zf:
        packaged_dataset = json.loads(zf.read("dataset/manifest.json"))
        ds = manifest.dataset
        existing = None
        try:
            existing = ws.get_dataset(ds.dataset_id)
        except WorkspaceError:
            pass
        if existing is not None and existing.raw_sha256 != ds.raw_sha256:
            raise PackageError("dataset_conflict",
                               f"dataset '{ds.dataset_id}' exists in this workspace with different raw data "
                               f"(sha256 {existing.raw_sha256} vs packaged {ds.raw_sha256})",
                               hint="import into another workspace or use a different dataset id there")
        for ref in manifest.references:
            if ws.run_dir(ref.run_id).exists():
                ref_manifest = experiments.load_artifacts(ws, ref.run_id)
                ckpt = ref_manifest.by_kind(ArtifactKind.checkpoint) if ref_manifest else []
                if not ckpt or ckpt[0].file.sha256 != ref.checkpoint_sha256:
                    raise PackageError("reference_conflict",
                                       f"run '{ref.run_id}' exists here with different weights than the package's "
                                       f"{ref.role} ({ref.checkpoint_sha256[:12]})")

        # every check passed: write
        imported = [run_id]
        missing: List[str] = list(manifest.missing)
        if existing is not None:
            dataset_status = "existing"
        elif ds.source_kind == DatasetSourceKind.builtin and ds.builtin_name:
            registered = ws.register_builtin_dataset(ds.builtin_name, ds.dataset_id)
            if registered.raw_sha256 != ds.raw_sha256:
                shutil.rmtree(ws.dataset_dir(ds.dataset_id), ignore_errors=True)
                raise PackageError("dataset_conflict",
                                   f"built-in dataset '{ds.builtin_name}' of this install has raw sha256 "
                                   f"{registered.raw_sha256}, the package was made with {ds.raw_sha256}")
            dataset_status = "registered_builtin"
        elif ds.included:
            target = ws.dataset_dir(ds.dataset_id)
            _extract(zf, "dataset/", target, sizes)
            (target / "manifest.json").write_text(json.dumps(packaged_dataset, indent=2, sort_keys=True), encoding="utf-8")
            dataset_status = "imported"
        else:
            dataset_status = "missing"
            if not any(ds.dataset_id in m for m in missing):
                missing.append(f"dataset '{ds.dataset_id}' (raw sha256 {ds.raw_sha256}): {ds.how_to_obtain}")
        for ref in manifest.references:
            if not ws.run_dir(ref.run_id).exists():
                _extract(zf, f"refs/{ref.run_id}/", ws.run_dir(ref.run_id), sizes)
                imported.append(ref.run_id)
        _extract(zf, f"run/{run_id}/", ws.run_dir(run_id), sizes)
    RunRecord.model_validate(read_json(ws.run_dir(run_id) / "run.json"))    # what was imported is a valid record
    evaluate = shlex.join(["opendpd", "evaluate", run_id, "--workspace", str(ws.root),
                           "--profile", manifest.metric_profile_id or "legacy-opendpd-v1"])
    if dataset_status == "missing":
        note = ("imported without the data: results and configuration are readable, re-evaluation needs the dataset "
                "listed under missing")
    else:
        note = "re-evaluate the packaged checkpoint to check the stored metrics; " + RETRAINING_NOTE
    return ImportReport(package_version=manifest.package_version, kind=manifest.kind, run_id=run_id,
                        imported_runs=imported, dataset_status=dataset_status, dataset_id=ds.dataset_id,
                        missing=missing, evaluate_command=evaluate, note=note)


def receive_package(ws: Workspace, filename: str, chunks: Iterable[bytes], max_bytes: int) -> Path:
    """Stream an uploaded package into imports/packages/ (never into a user-chosen path)."""
    safe = Path(filename).name
    if not safe.lower().endswith(".zip"):
        raise PackageError("not_a_package", "only .zip experiment packages are accepted")
    target_dir = ws.imports_dir / "packages"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = target_dir / f"{stamp}-{re.sub(r'[^A-Za-z0-9._-]', '_', safe)}"
    total = 0
    with open(target, "wb") as f:
        for chunk in chunks:
            total += len(chunk)
            if total > max_bytes:
                f.close()
                target.unlink(missing_ok=True)
                raise PackageError("too_large", f"package exceeds {max_bytes} bytes")
            f.write(chunk)
    return target
