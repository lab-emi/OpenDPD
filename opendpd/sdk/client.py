"""Local SDK: authenticated Studio client, persistent jobs and CPU inference."""

from __future__ import annotations

import http.cookiejar
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class SDKError(RuntimeError):
    """A service error with the original machine-readable code and details."""

    def __init__(self, code, message, *, details=None):
        self.code, self.details = code, details or []
        super().__init__(f"{code}: {message}")


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _python_environment():
    # Also work from a source checkout when MATLAB's current folder is elsewhere.
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    env["MPLBACKEND"] = "Agg"
    return env


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Project:
    """A connection to one local Studio workspace.

    ``close()`` disconnects; jobs keep running. ``close(stop_service=True)``
    gracefully stops a service launched by this SDK, including its workers.
    A Studio service started outside the SDK is never stopped by this method.
    """

    def __init__(self, workspace, *, start=True, timeout=30.0):
        from opendpd.studio.launcher import Lock, LOCK_FILE, probe, workspace_guard, WorkspaceBusy

        timeout = _positive(timeout, "timeout")
        self.workspace = Path(workspace).expanduser().resolve()
        self._process = None
        self._closed = False
        self._csrf = ""
        self._cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(self._cookies), _NoRedirect())
        self._base_url = ""
        deadline = time.monotonic() + timeout
        try:
            while True:
                lock = Lock.read(self.workspace / LOCK_FILE)
                if lock is not None and lock.alive() and probe(f"http://127.0.0.1:{lock.port}/healthz", timeout=0.5):
                    self._attach(lock)
                    break
                if not start:
                    raise SDKError("service_unavailable", "Start Studio for this workspace or use start=True")
                if self._process is None:
                    self.workspace.mkdir(parents=True, exist_ok=True)
                    fd, self._log_path = tempfile.mkstemp(prefix=".sdk-service-", suffix=".log", dir=self.workspace)
                    with os.fdopen(fd, "wb") as log:
                        options = {"start_new_session": True} if os.name != "nt" else {
                            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                        self._process = subprocess.Popen(
                            [sys.executable, "-m", "opendpd.sdk._server", "--workspace", str(self.workspace)],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=_python_environment(),
                            cwd=Path(__file__).resolve().parents[2], **options)
                if time.monotonic() >= deadline:
                    raise SDKError("startup_timeout", f"Service did not start within {timeout:g}s; see {self._log_path}")
                if self._process.poll() not in (None, 0):
                    # Another launcher may be taking the workspace guard. Give
                    # it the same startup deadline; never replace its lock.
                    if lock is None or not lock.alive():
                        try:
                            with workspace_guard(self.workspace):
                                raise SDKError("startup_failed", f"Service failed to start; see {self._log_path}")
                        except WorkspaceBusy:
                            pass
                time.sleep(0.1)
        except BaseException:
            self._stop_failed_start()
            raise

    def __repr__(self):
        return f"Project({str(self.workspace)!r})"

    def _mint_token(self, lock):
        """A fresh single-use bootstrap token. The lock holds the launcher secret, never a usable token."""
        url = urllib.parse.urlsplit(lock.url)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.port != lock.port
                or url.path not in ("/", "/bootstrap") or url.username is not None):
            raise SDKError("invalid_service", "Workspace lock does not describe a local Studio service")
        if not lock.launcher_secret:
            raise SDKError("launcher_auth_failed", "This Studio service cannot authorise SDK clients; "
                           "restart it with this OpenDPD installation")
        from opendpd.server.security import LAUNCHER_HEADER

        request = urllib.request.Request(f"http://127.0.0.1:{lock.port}/bootstrap/mint", data=b"",
                                         headers={LAUNCHER_HEADER: lock.launcher_secret}, method="POST")
        try:
            with self._opener.open(request, timeout=5) as response:
                return str(json.loads(response.read(4096))["token"])
        except (OSError, ValueError, KeyError, TypeError) as err:
            raise SDKError("launcher_auth_failed", "Cannot authenticate with the Studio launcher; "
                           "restart the workspace service") from err

    def _attach(self, lock):
        self._base_url = f"http://127.0.0.1:{lock.port}"
        self._csrf = self._request("POST", "/session/bootstrap", {"token": self._mint_token(lock)})["csrf_token"]
        caps = self._request("GET", "/system/capabilities")
        if Path(caps["workspace"]).resolve() != self.workspace:
            raise SDKError("workspace_mismatch", "Service belongs to a different workspace")
        from opendpd import __version__
        if caps["version"] != __version__:
            raise SDKError("version_mismatch", "Restart the workspace service using this OpenDPD installation")

    def _request(self, method, path, data=None, *, api=True, accepted_statuses=(), extra_headers=None):
        if self._closed:
            raise SDKError("project_closed", "Reconnect using open_project(workspace)")
        headers = {"Accept": "application/json", "X-OpenDPD-CSRF": self._csrf}
        if extra_headers:
            headers.update(extra_headers)
        payload = None
        if data is not None:
            payload = json.dumps(data, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self._base_url + ("/api/v1" if api else "") + path,
                                         data=payload, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as err:
            body = err.read()
            if err.code in accepted_statuses:
                return json.loads(body)
            try:
                detail = json.loads(body)["error"]
            except (ValueError, KeyError, TypeError):
                detail = {"code": "http_error", "message": f"HTTP {err.code}"}
            message = detail.get("message", "Request failed")
            fields = "; ".join(f"{d.get('field', '')}: {d.get('message', '')}" for d in detail.get("details", []))
            if fields:
                message += "; " + fields
            raise SDKError(detail.get("code", "http_error"), message, details=detail.get("details")) from None
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise SDKError("connection_lost", "Cannot reach Studio; reconnect to the workspace and use the run ID") from err

    def _stop_failed_start(self):
        if self._process is not None and self._process.poll() is None:
            from opendpd.studio.launcher import Lock, LOCK_FILE

            try:
                lock = Lock.read(self.workspace / LOCK_FILE)
                if lock is None or lock.pid != self._process.pid:
                    raise SDKError("not_owner", "This launcher does not own the workspace")
                self.close(stop_service=True)
            except (SDKError, OSError):
                self._process.terminate()
                try:
                    self._process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

    def close(self, *, stop_service=False):
        if stop_service:
            from opendpd.studio.launcher import Lock, LOCK_FILE
            from ._server import SERVICE_FILE

            lock = Lock.read(self.workspace / LOCK_FILE)
            if lock is not None and lock.alive():
                try:
                    meta = json.loads((self.workspace / SERVICE_FILE).read_text())
                    valid = (meta["pid"] == lock.pid and meta["create_time"] == lock.create_time
                             and re.fullmatch(r"\.sdk-stop-[0-9a-f]{32}", meta["stop_file"]))
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
                if not valid:
                    raise SDKError("external_service", "Close this Studio service through its original launcher")
                (self.workspace / meta["stop_file"]).touch(mode=0o600)
                deadline = time.monotonic() + 25
                while lock.alive():
                    import psutil

                    if self._process is not None:
                        self._process.poll()  # reap before checking process identity again
                    try:
                        if psutil.Process(lock.pid).status() == psutil.STATUS_ZOMBIE:
                            break  # a different SDK connection owns the Popen handle
                    except psutil.NoSuchProcess:
                        break
                    if time.monotonic() >= deadline:
                        raise SDKError("shutdown_timeout", "Service is still stopping; retain the workspace and retry")
                    time.sleep(0.1)
        if self._process is not None:
            self._process.poll()
        self._cookies.clear()
        self._closed = True

    def studio_url(self, page="home", run_id=None):
        """Private local bootstrap URL. Do not log or share it."""
        from opendpd.studio.launcher import Lock, LOCK_FILE

        lock = Lock.read(self.workspace / LOCK_FILE)
        if self._closed or lock is None or not lock.alive():
            raise SDKError("service_unavailable", "Reconnect to the workspace first")
        from opendpd.studio.navigation import destination

        target = destination(page, run_id)
        if target != "/" and self.studio_info().get("studio_navigation_version", 0) < 1:
            raise SDKError("service_update_needed", "This service supports the Studio home page only. "
                           "Finish its jobs, then restart it to enable page shortcuts")
        url = f"http://127.0.0.1:{lock.port}/bootstrap?" + urllib.parse.urlencode({"token": self._mint_token(lock)})
        return url if target == "/" else url + "&" + urllib.parse.urlencode({"next": target})

    def studio_info(self):
        """Read browser readiness without exposing the bootstrap secret."""
        info = self._request("GET", "/readyz", api=False, accepted_statuses=(503,))
        return {**info, "workspace": str(self.workspace)}

    def runs(self, limit=25):
        if isinstance(limit, bool) or int(limit) != limit or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        return self._request("GET", f"/runs?limit={int(limit)}")

    def models(self):
        return self._request("GET", "/models")

    def datasets(self):
        return self._request("GET", "/datasets")

    def import_iq(self, x, y, *, dataset_id=None, sample_rate_hz, bandwidth_hz,
                  nperseg=256, n_sub_ch=1, guard_samples=256, origin="unknown",
                  amplitude_units="unknown", display_name=None, source=None):
        """Import paired I/Q without normalization. Source precision is recorded."""
        import numpy as np
        from opendpd.core.splits import contiguous_boundaries
        from opendpd.schemas import SignalSpec, DatasetOrigin
        from opendpd.schemas.common import Slug
        from pydantic import TypeAdapter
        from .iq import as_iq

        original = {name: {"dtype": str(np.asarray(v).dtype), "shape": list(np.shape(v))}
                    for name, v in (("input", x), ("output", y))}
        x, y = as_iq(x, "input"), as_iq(y, "output")
        if x.shape != y.shape:
            raise ValueError("Input and output must have the same number of samples")
        dataset_id = dataset_id or f"matlab-{uuid.uuid4().hex[:12]}"
        TypeAdapter(Slug).validate_python(dataset_id)
        signal = SignalSpec(sample_rate_hz=sample_rate_hz, bandwidth_hz=bandwidth_hz,
                            nperseg=nperseg, n_sub_ch=n_sub_ch, amplitude_units=amplitude_units)
        if not all(math.isfinite(v) for v in (signal.sample_rate_hz, signal.bandwidth_hz)):
            raise ValueError("Sample rate and bandwidth must be finite")
        origin = DatasetOrigin(origin).value
        if isinstance(guard_samples, bool) or int(guard_samples) != guard_samples:
            raise ValueError("guard_samples must be an integer")
        guard_samples = int(guard_samples)
        contiguous_boundaries(len(x), guard_samples=guard_samples)
        if self._closed:
            raise SDKError("project_closed", "Reconnect to the workspace first")
        # Local SDK clients own their source files. Stage under the existing
        # imports root and use the same bounded import endpoint as Studio.
        directory = self.workspace / "imports"
        directory.mkdir(exist_ok=True)
        fd, filename = tempfile.mkstemp(prefix="sdk-", suffix=".npz", dir=directory)
        notes = {"sdk_iq_version": 1, "source_arrays": original, "stored_dtype": "float32",
                 "scaling": "none", "source": source}
        remove_source = True
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, input=x, output=y)
            return self._request("POST", "/datasets/import", {
                "source": {"root_id": "imports", "path": Path(filename).name},
                "dataset_id": dataset_id, "display_name": display_name or dataset_id,
                "signal": signal.model_dump(mode="json"), "origin": origin,
                "guard_samples": guard_samples, "notes": json.dumps(notes)})
        except SDKError as err:
            if err.code == "connection_lost":
                remove_source = False  # the server may still be reading it
                raise SDKError("import_uncertain", f"Check dataset '{dataset_id}' before retrying; "
                               f"the import may still finish. Staged data remains at {filename}") from err
            raise
        finally:
            if remove_source:
                Path(filename).unlink(missing_ok=True)

    def import_mat(self, path, *, input_variable="x", output_variable="y", **options):
        from opendpd.services.workspace import sha256_file
        from .iq import read_mat

        path, x, y = read_mat(path, input_variable, output_variable)
        return self.import_iq(x, y, source={"format": "mat", "name": path.name,
            "sha256": sha256_file(path), "input_variable": input_variable,
            "output_variable": output_variable}, **options)

    def submit(self, config, *, idempotency_key=None):
        from opendpd.schemas import ExperimentConfig

        if not isinstance(config, ExperimentConfig):
            config = ExperimentConfig.model_validate(config)
        record = self._request("POST", "/runs", {"config": config.model_dump(mode="json"),
            "idempotency_key": idempotency_key or uuid.uuid4().hex})
        return Job(self, record["run_id"])

    def train_pa(self, dataset_id, *, model="gru", parameters=None, training=None,
                 device="cpu", num_threads=1, profile="opendpd-spectral-v2"):
        return self.submit({"task": "train_pa", "dataset": {"id": dataset_id},
            "model": {"key": model, "parameters": parameters or {}}, "training": training or {},
            "execution": {"device": device, "num_threads": num_threads},
            "evaluation": {"profile_id": profile}})

    def train_dpd(self, dataset_id, pa, *, model="gru", parameters=None, training=None,
                  device="cpu", num_threads=1, profile="opendpd-spectral-v2"):
        self._same_project(pa)
        return self.submit({"task": "train_dpd", "dataset": {"id": dataset_id},
            "model": {"key": model, "parameters": parameters or {}}, "training": training or {},
            "execution": {"device": device, "num_threads": num_threads},
            "evaluation": {"profile_id": profile}, "pa_reference": {"run_id": pa.run_id}})

    def run_dpd(self, dpd):
        self._same_project(dpd)
        cfg = dpd.config()
        evaluation = dict(cfg["evaluation"])
        evaluation.pop("checkpoint_selection_metric", None)
        return self.submit({"task": "run_dpd", "dataset": cfg["dataset"], "model": cfg["model"],
            "training": cfg["training"], "execution": cfg["execution"], "evaluation": evaluation,
            "dpd_reference": {"run_id": dpd.run_id}})

    def _same_project(self, job):
        if not isinstance(job, Job) or job.project.workspace != self.workspace:
            raise ValueError("The referenced job must belong to this workspace")

    def job(self, run_id):
        job = Job(self, run_id)
        job.status()
        return job


class Job:
    """Durable run ID. Reconnect with ``project.job(run_id)``."""

    def __init__(self, project, run_id):
        from opendpd.services.workspace import checked_identifier

        self.project = project
        self.run_id = checked_identifier(run_id, "run id")

    def __repr__(self):
        return f"Job({self.run_id!r})"

    def status(self):
        return self.project._request("GET", f"/runs/{self.run_id}")

    def config(self):
        return self.project._request("GET", f"/runs/{self.run_id}/config")

    def result(self):
        return self.project._request("GET", f"/results/{self.run_id}")

    def artifacts(self):
        return self.project._request("GET", f"/runs/{self.run_id}/artifacts")

    def cancel(self):
        return self.project._request("POST", f"/runs/{self.run_id}/cancel", {})

    def wait(self, *, timeout=600.0, poll_interval=0.25):
        deadline = time.monotonic() + _positive(timeout, "timeout")
        interval = _positive(poll_interval, "poll_interval")
        while True:
            record = self.status()
            if record["status"] in TERMINAL:
                if record["status"] != "succeeded":
                    error = record.get("error") or {}
                    raise SDKError(record["status"], f"Run {self.run_id}: {error.get('message', record['status'])}")
                return self
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Run {self.run_id} is still {record['status']}; it was not cancelled")
            time.sleep(min(interval, remaining))

    def apply(self, x, *, execution="offline_segmented", timeout=120.0):
        """Apply a succeeded, unquantized GRU on CPU; return (Nx2 float32, metadata).

        Each call resets state at the stored dataset's segment boundaries and
        zero-pads its final segment. This is offline inference, not a stream.
        """
        import numpy as np
        from .iq import as_iq

        timeout = _positive(timeout, "timeout")
        if execution != "offline_segmented":
            raise ValueError("This preview supports only offline_segmented apply")
        if self.status()["status"] != "succeeded":
            raise SDKError("run_not_finished", f"Run {self.run_id} must succeed before apply")
        x = as_iq(x)
        with tempfile.TemporaryDirectory(prefix="opendpd-apply-") as directory:
            directory = Path(directory)
            np.save(directory / "input.npy", x, allow_pickle=False)
            completed = subprocess.run([sys.executable, "-m", "opendpd.sdk._infer",
                "--workspace", str(self.project.workspace), "--run-id", self.run_id,
                "--directory", str(directory)], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_python_environment(),
                cwd=Path(__file__).resolve().parents[2], timeout=timeout)
            if completed.returncode:
                error_path = directory / "error.json"
                if error_path.is_file():
                    error = json.loads(error_path.read_text())
                    raise SDKError(error["code"], error["message"])
                raise SDKError("inference_failed", completed.stdout.decode("utf-8", errors="replace")[-2000:])
            return (np.load(directory / "output.npy", allow_pickle=False),
                    json.loads((directory / "metadata.json").read_text()))


def open_project(workspace, *, start=True, timeout=30.0) -> Project:
    return Project(workspace, start=start, timeout=timeout)
