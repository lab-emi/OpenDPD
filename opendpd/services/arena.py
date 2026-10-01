"""Workspace Arena evaluation with immutable requests and server-owned scores."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time

from opendpd.runtime.procs import is_same_process, kill_tree, process_identity
from opendpd.runtime.arena_limits import ARENA_MAX_RUNTIME_SECONDS
from opendpd.schemas.arena import (ArenaCatalog, ArenaCoverage, ArenaLeaderboard,
    ArenaProgress, ArenaRankEntry, ArenaRow, ArenaSubmission, ArenaSubmissionRequest)
from opendpd.schemas.common import utcnow
from opendpd.services.workspace import (Conflict, InvalidInput, NotFound, WorkspaceError,
    read_json, write_json_atomic)

log = logging.getLogger(__name__)
SCOPE_NOTE = ("Official baselines ship with this protocol. Your submissions are evaluated by this "
              "Studio service and remain in this workspace; submitting does not publish a global record. "
              "Arena uses only APA_200MHz_b and one frozen TRes-GRU PA. PA and DPD inputs come only "
              "from that original measured capture. Cascade outputs are model predictions, not new predistorted hardware PA measurements.")
MAX_SUBMISSIONS = 64
MAX_RUNTIME_SECONDS = ARENA_MAX_RUNTIME_SECONDS
_OFFICIAL, _OFFICIAL_LOCK = {}, threading.Lock()


class ArenaController:
    """One bounded worker per workspace; public hosts explicitly opt into compute."""

    def __init__(self, ws, backbones, *, enabled=True, evaluator=None, protocol_provider=None,
                 backbone_provider=None, official_provider=None, summary_provider=None):
        self.ws, self.backbones, self.enabled = ws, backbones, enabled
        self.root = ws.root / "arena"
        if self.root.is_symlink():
            raise WorkspaceError("Arena storage cannot be a symbolic link.")
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.thread = None
        self.process = None
        self.evaluator = evaluator or self._evaluate
        self.protocol_provider = protocol_provider
        self.backbone_provider = backbone_provider
        self.official_provider = official_provider
        self.summary_provider = summary_provider
        for record in self.list():
            if record.status in {"queued", "running"}:
                worker = self.directory(record.submission_id) / "worker.json"
                if worker.is_file() and not worker.is_symlink():
                    try:
                        identity = read_json(worker)
                    except (ValueError, OSError):
                        identity = {}
                    if not isinstance(identity, dict):
                        identity = {}
                    if (isinstance(identity.get("pid"), int) and identity["pid"] > 0
                            and isinstance(identity.get("create_time"), (int, float))
                            and identity["create_time"] > 0
                            and is_same_process(identity["pid"], identity["create_time"])):
                        kill_tree(identity["pid"], grace_seconds=0.5)
                self._update(record.submission_id, status="interrupted",
                             error="Studio stopped before this evaluation completed. Submit again to retry.")

    def protocol(self):
        if self.protocol_provider:
            return self.protocol_provider()
        from opendpd.core.arena import protocol
        return protocol()

    def bundled_backbones(self):
        if self.backbone_provider:
            return self.backbone_provider()
        from opendpd.core.arena import bundled_backbones
        return bundled_backbones()

    def official_rows(self):
        if self.official_provider:
            return self.official_provider()
        # Verifying and rescoring the sealed bundle is the same work for every
        # request and workspace until the file or the protocol changes. Callers
        # copy a row before changing it.
        from opendpd.core import arena
        path = arena.ASSETS / arena.RESULTS_FILE
        stamp = path.stat() if path.is_file() else None
        key = (self.protocol().protocol_sha256, str(path), stamp and (stamp.st_mtime_ns, stamp.st_size))
        with _OFFICIAL_LOCK:
            if _OFFICIAL.get("key") != key:
                _OFFICIAL.update(key=key, rows=arena.load_official_rows())
            return list(_OFFICIAL["rows"])

    def catalog(self):
        return ArenaCatalog(protocol=self.protocol(), backbones=self.bundled_backbones(),
            submissions_available=self.enabled and not self.stopping.is_set(),
            submission_unavailable_reason=None if self.enabled else
                "Arena evaluation is available in local Studio. This hosted workspace provides the official rankings.",
            scope_note=SCOPE_NOTE)

    def _board(self, board_id, protocol=None):
        # Deriving the protocol hashes every frozen source file; a request derives it once and passes it on.
        board = next((board for board in (protocol or self.protocol()).boards if board.board_id == board_id), None)
        if board is None:
            raise NotFound("Unknown Arena leaderboard.")
        return board

    def directory(self, submission_id):
        if not isinstance(submission_id, str) or not re.fullmatch(r"arena-[a-f0-9]{32}", submission_id):
            raise InvalidInput("Invalid Arena submission identifier.")
        path = self.root / submission_id
        if self.root.is_symlink() or path.is_symlink():
            raise WorkspaceError("Arena storage cannot be a symbolic link.")
        return path

    def get(self, submission_id):
        directory = self.directory(submission_id)
        path = directory / "submission.json"
        try:
            present = path.is_file() and not path.is_symlink()
        except OSError:
            raise WorkspaceError("A stored Arena submission is damaged.") from None
        if not present:
            raise NotFound("Arena submission does not exist in this workspace.")
        raw = None
        try:
            raw = read_json(path)
            record = ArenaSubmission.model_validate(raw)
        except (ValueError, OSError):
            # The result may be unreadable or written by an earlier Arena version.
            # What identifies the submission is kept visible without any evidence.
            try:
                record = ArenaSubmission.model_validate({**raw, "status": "failed", "result": None, "progress": None,
                    "error": "The stored Arena evaluation is damaged or was written by an earlier Arena version."})
            except (ValueError, TypeError):
                raise WorkspaceError("A stored Arena submission is damaged.") from None
        if record.submission_id != submission_id:
            raise WorkspaceError("Arena submission identity does not match its storage directory.")
        if record.result is not None:
            try:
                if record.status != record.result.status:
                    raise ValueError("Submission status does not match its evaluation result.")
                record.result = self._recompute_result(record, record.result)
            except (ValueError, TypeError, KeyError, ArithmeticError, OSError) as exc:
                # A damaged or hand-edited workspace record remains visible, but
                # never obtains a rank merely by claiming success/eligibility.
                reason = "its stored evidence could not be read" if isinstance(exc, OSError) else str(exc)[:500]
                record = record.model_copy(update={"status": "failed", "result": None,
                    "error": f"Stored Arena result failed validation: {reason}"})
        progress = directory / "result.progress.json"
        if record.status in {"queued", "running"} and not progress.is_symlink() and progress.is_file():
            try:
                if progress.stat().st_size <= 4096:
                    record.progress = ArenaProgress.model_validate(read_json(progress))
            except (ValueError, OSError):
                # A progress update is advisory; an incomplete or corrupt update
                # must not hide the durable evaluation status.
                pass
        return record

    def list(self):
        with self.lock:
            records = []
            for path in self.root.glob("arena-*/submission.json"):
                try:
                    records.append(self.get(path.parent.name))
                except WorkspaceError as exc:  # one unreadable record must not close the workspace
                    log.warning("Skipping Arena submission %s: %s", path.parent.name, exc)
            return sorted(records, key=lambda record: record.created_at, reverse=True)

    def _update(self, submission_id, **changes):
        with self.lock:
            record = self.get(submission_id)
            record = ArenaSubmission.model_validate({**record.model_dump(), **changes, "updated_at": utcnow()})
            write_json_atomic(self.directory(submission_id) / "submission.json", record)
            return record

    def _custom_model(self, backbone_id):
        from opendpd.core.backbone_template import parse_definition, source_sha256, validate_definition
        from opendpd.services.user_backbones import verify_package
        entry = next((item for item in self.backbones.list() if item.backbone_id == backbone_id), None)
        if entry is not None:
            verify_package(entry, self.backbones.directory(entry.publication_id) / "package")
        else:
            entry = next((item for item in self.backbones.catalog().entries if item.backbone_id == backbone_id), None)
        if entry is None:
            raise InvalidInput("Choose an uploaded or community backbone available in this workspace.")
        encoded = entry.model.parameters.get("definition")
        definition = parse_definition(encoded)
        if entry.model.key != "user_template" or source_sha256(encoded.encode()) != entry.definition_sha256:
            raise InvalidInput("The selected backbone definition failed its integrity check.")
        maximum = self.protocol().training.get("max_parameters")
        count = validate_definition(definition)["parameters"]
        if type(maximum) is int and count > maximum:
            raise InvalidInput(f"Arena accepts at most {maximum:,} trainable parameters; this template has {count:,}.")
        return entry.model.parameters, {"backbone_id": entry.backbone_id,
            "source_sha256": entry.source_sha256, "definition_sha256": entry.definition_sha256}

    def submit(self, request: ArenaSubmissionRequest):
        with self.lock:
            if not self.enabled:
                raise WorkspaceError("Arena evaluation is disabled on this host; run it in local Studio.")
            if self.stopping.is_set():
                raise Conflict("Studio is stopping; start it again before submitting.")
            protocol = self.protocol()
            self._board(request.board_id, protocol)
            if request.accepted_protocol_sha256 != protocol.protocol_sha256:
                raise Conflict("The Arena protocol changed. Review the current rules before submitting.")
            parameters, model_provenance = {}, {}
            if request.backbone == "user_template" and request.backbone_id is not None:
                parameters, model_provenance = self._custom_model(request.backbone_id)
            elif request.backbone not in {item.key for item in self.bundled_backbones()}:
                raise InvalidInput("Choose a bundled Arena backbone or a validated user template.")
            records = self.list()
            if len(records) >= MAX_SUBMISSIONS:
                raise Conflict("This workspace has reached its 64 Arena evaluation limit.")
            if ((self.thread is not None and self.thread.is_alive())
                    or any(record.status in {"queued", "running"} for record in records)):
                raise Conflict("An Arena evaluation is already active. Wait until it finishes.")
            identifier = "arena-" + secrets.token_hex(16)
            directory = self.directory(identifier)
            directory.mkdir()
            record = ArenaSubmission(submission_id=identifier, request=request,
                                     protocol_sha256=protocol.protocol_sha256)
            write_json_atomic(directory / "request.json", {
                "board_id": request.board_id, "backbone": request.backbone,
                "protocol_sha256": protocol.protocol_sha256, "model_parameters": parameters,
                "model_provenance": model_provenance})
            write_json_atomic(directory / "submission.json", record)
            self.thread = threading.Thread(target=self._run, args=(identifier,), daemon=True,
                                           name="arena-evaluation")
            self.thread.start()
            return record

    def _run(self, identifier):
        try:
            self._wait_for_dispatch(identifier)
            record = self._update(identifier, status="running")
            result = ArenaRow.model_validate(self.evaluator(self.directory(identifier)))
            if self.stopping.is_set():
                self._update(identifier, status="interrupted", error="Studio stopped during Arena evaluation.")
                return
            if (result.board_id != record.request.board_id or result.backbone != record.request.backbone
                    or result.protocol_sha256 != record.protocol_sha256):
                raise ValueError("Evaluator returned results for a different protocol, board or backbone.")
            result = self._recompute_result(record, result)
            self._update(identifier, status=result.status, result=result, error=result.error)
        except Exception as exc:
            log.exception("Arena evaluation %s failed", identifier)
            self._update(identifier, status="interrupted" if self.stopping.is_set() else "failed", error=str(exc)[:1000])

    def _wait_for_dispatch(self, identifier):
        """Local evaluations start immediately; hosted adapters await the shared queue."""

    def _recompute_result(self, record, row):
        """Raw per-case observations are authoritative; cached summary claims are not."""
        if (row.board_id != record.request.board_id or row.backbone != record.request.backbone
                or row.protocol_sha256 != record.protocol_sha256
                or record.request.accepted_protocol_sha256 != record.protocol_sha256):
            raise ValueError("Arena result identity does not match the accepted request.")
        changes = {"entry_id": record.submission_id, "origin": "workspace",
                   "display_name": record.request.display_name, "created_at": record.created_at, "rank": None}
        if record.protocol_sha256 != self.protocol().protocol_sha256:
            changes.update(eligible=False, score=None, rankings={},
                           eligibility_reasons=["This evaluation belongs to an earlier Arena protocol."])
            return ArenaRow.model_validate({**row.model_dump(), **changes})
        if row.status == "succeeded":
            points = [point.model_dump(mode="json") for point in row.budgets]
            if self.summary_provider:
                derived = self.summary_provider(row)
            else:
                from opendpd.core.arena import summarize_cases, sweep
                # The sweep is derived again from the accepted request: a result
                # cannot choose its own budgets, presets or template widths.
                accepted = self.directory(record.submission_id) / "request.json"
                request = read_json(accepted) if accepted.is_file() and not accepted.is_symlink() else None
                if not isinstance(request, dict):
                    raise ValueError("The accepted Arena request is missing or damaged.")
                registered = sweep(row.backbone, request.get("model_parameters") or None)
                if [point["model_parameters"] for point in points] != [point["model_parameters"] for point in registered]:
                    raise ValueError("Arena result does not use the sweep registered for this request.")
                derived = summarize_cases(row.backbone, row.board_id, row.cases, points)
            changes.update(derived)
        else:
            changes.update(eligible=False, score=None, rankings={})
        verified = ArenaRow.model_validate({**row.model_dump(), **changes})
        self._validate_complete(verified)
        return verified

    def _validate_complete(self, row, protocol=None):
        protocol = protocol or self.protocol()
        board = self._board(row.board_id, protocol)
        if row.evidence_type != board.evidence_type:
            raise ValueError("Evaluator evidence label does not match the board.")
        if row.status not in {"succeeded", "failed"}:
            raise ValueError("Evaluator did not return a terminal result.")
        if row.status == "succeeded":
            descriptor = next((item for item in self.bundled_backbones() if item.key == row.backbone), None)
            from opendpd.core.arena import EXCLUDED_BACKBONES
            if row.backbone in EXCLUDED_BACKBONES:
                raise ValueError("Backbone is excluded from the Arena catalogue.")
            seeds = protocol.seeds[:1] if descriptor and descriptor.deterministic else protocol.seeds
            if [point.budget for point in row.budgets] != protocol.budgets:
                raise ValueError("Arena results must cover every protocol budget.")
            available = [point.budget for point in row.budgets if point.available]
            expected_cases = {(budget, condition, seed) for budget in available
                              for condition in board.conditions for seed in seeds}
            expected = len(expected_cases)
            identities = [(case.get("budget"), case.get("condition_id"), case.get("seed")) for case in row.cases]
            if any(type(budget) is not int or not isinstance(condition, str) or type(seed) is not int
                   for budget, condition, seed in identities):
                raise ValueError("Arena cases must identify their canonical budget, condition and seed.")
            if (row.metrics is None or row.parameters is None or not available
                    or sorted(row.seeds) != sorted(seeds)
                    or row.expected_cases != expected or row.completed_cases != expected
                    or len(row.cases) != expected or set(identities) != expected_cases):
                raise ValueError("Incomplete Arena evaluation cannot receive a rank.")
            if row.eligible and (row.score is None or not row.qualified_budgets
                                 or any(point.available and point.ops is None for point in row.budgets)):
                raise ValueError("Ranked evaluations require an audited score and operation count.")
        elif row.score is not None:
            raise ValueError("Failed Arena evaluations cannot carry a score.")

    def leaderboard(self, board_id):
        with self.lock:
            protocol = self.protocol()
            board = self._board(board_id, protocol)
            from opendpd.core.arena import EXCLUDED_BACKBONES
            official = []
            for value in self.official_rows():
                if (value.get("board_id") if isinstance(value, dict) else value.board_id) != board_id:
                    continue
                row = ArenaRow.model_validate(value)
                if row.backbone in EXCLUDED_BACKBONES:
                    raise WorkspaceError("Official Arena result uses an excluded backbone.")
                if row.protocol_sha256 != protocol.protocol_sha256 or row.origin != "official":
                    raise WorkspaceError("Official Arena results do not match this frozen protocol.")
                self._validate_complete(row, protocol)
                official.append(row)
            # The shipped evidence is shared, never changed: only the rank fields below are rebuilt per request.
            rows = [row.model_copy() for row in official]
            for record in self.list():
                if (record.request.board_id != board_id or record.protocol_sha256 != protocol.protocol_sha256
                        or record.request.backbone in EXCLUDED_BACKBONES):
                    continue
                if record.result:
                    rows.append(record.result.model_copy(deep=True))
                else:
                    rows.append(ArenaRow(entry_id=record.submission_id, board_id=board_id,
                        backbone=record.request.backbone, display_name=record.request.display_name,
                        origin="workspace", status=record.status, protocol_sha256=protocol.protocol_sha256,
                        evidence_type=board.evidence_type, error=record.error, created_at=record.created_at))
            # Every ranking is ordered inside its cohort: shipped or workspace
            # evidence, offline or streaming execution. Operation counts do not
            # depend on the host, so no hardware cohort is needed.
            published = {ranking.ranking_id for ranking in protocol.rankings}
            for row in rows:  # a stored result cannot introduce a ranking of its own
                row.rankings = {key: entry for key, entry in row.rankings.items() if key in published}
            for ranking in protocol.rankings:
                entries = {}
                for row in rows:
                    entry = row.rankings.get(ranking.ranking_id)
                    ranked = row.status == "succeeded" and row.eligible and entry is not None and entry.score is not None
                    row.rankings[ranking.ranking_id] = ArenaRankEntry(score=entry.score if ranked else None)
                    if ranked:
                        entries.setdefault((row.origin, row.execution_semantics), []).append(row)
                for group in entries.values():
                    group.sort(key=lambda row: (-row.rankings[ranking.ranking_id].score,
                                                row.parameters or 0, row.backbone, row.entry_id))
                    for position, row in enumerate(group, 1):
                        row.rankings[ranking.ranking_id].rank = position
            for row in rows:
                overall = row.rankings.get("overall")
                row.rank = overall.rank if overall else None
            rows.sort(key=lambda row: (row.rank is None, row.rank or 0, row.backbone, row.entry_id))
            expected = {item.key for item in self.bundled_backbones()}
            evaluated = {row.backbone for row in official}
            return ArenaLeaderboard(board=board, protocol_sha256=protocol.protocol_sha256, rows=rows,
                coverage=ArenaCoverage(expected=len(expected), evaluated=len(evaluated),
                    succeeded=sum(row.status == "succeeded" for row in official),
                    failed=sum(row.status == "failed" for row in official), missing=sorted(expected - evaluated)),
                scope_note=SCOPE_NOTE)

    def worker_command(self, directory: Path):
        # The runner itself sends the Python Delta cells to the CPU.
        return [sys.executable, "-m", "opendpd.core.arena_runner",
            "--request", str(directory / "request.json"), "--output", str(directory / "result.json")]

    def _evaluate(self, directory: Path):
        output = directory / "result.json"
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1"}
        with (directory / "worker.log").open("w") as log_file:
            with self.lock:
                if self.stopping.is_set():
                    raise RuntimeError("Studio is stopping.")
                self.process = subprocess.Popen(self.worker_command(directory),
                    stdout=log_file, stderr=subprocess.STDOUT, env=environment,
                    cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL)
                process = self.process
                identity = process_identity(process.pid)
                if identity is not None:
                    write_json_atomic(directory / "worker.json", {"pid": identity[0], "create_time": identity[1]})
            try:
                deadline = time.monotonic() + MAX_RUNTIME_SECONDS
                while process.poll() is None:
                    if self.stopping.wait(0.2):
                        raise RuntimeError("Studio stopped during evaluation.")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Arena evaluation exceeded the 7 day runtime limit.")
                if process.returncode:
                    raise RuntimeError("Arena evaluation failed. See the submission worker.log for details.")
                if output.is_symlink() or not output.is_file():
                    raise RuntimeError("Arena evaluator produced no valid result.")
                return read_json(output)
            finally:
                if process.poll() is None:
                    kill_tree(process.pid, grace_seconds=0.5)
                with self.lock:
                    if self.process is process:
                        self.process = None

    def stop(self):
        self.stopping.set()
        thread = self.thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=5)
