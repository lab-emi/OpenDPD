"""MATLAB-side role for Studio's MATLINK command broker; never exposes its secret."""
from __future__ import annotations

import json

from .client import SDKError


class MatlinkClient:
    def __init__(self, project, label, release, variables_json, capabilities_json="[]"):
        self.project = project
        self._closed = False
        info = project.studio_info()
        if info.get("matlink_protocol_version") != 1:
            raise SDKError("matlink_unavailable", "The running Studio service does not support MATLINK. "
                           "Finish its jobs, then restart it using this OpenDPD checkout")
        session = project._request("POST", "/matlink/connect", {
            "label": str(label), "release": str(release), "variables": json.loads(variables_json),
            "capabilities": json.loads(capabilities_json)})
        self.client_id = session["client_id"]
        self._bridge_token = session["bridge_token"]
        self._generation = self._server_identity()

    def __repr__(self):
        return f"MatlinkClient({self.client_id!r})"

    def _server_identity(self):
        from opendpd.studio.launcher import Lock, LOCK_FILE

        lock = Lock.read(self.project.workspace / LOCK_FILE)
        return (lock.pid, lock.create_time, lock.port) if lock is not None and lock.alive() else None

    def is_current(self):
        return (not self._closed and not self.project._closed and self._generation is not None
                and self._generation == self._server_identity())

    def _request(self, suffix, body):
        if self._closed:
            raise SDKError("matlink_closed", "Reconnect with opendpd.studio()")
        return self.project._request("POST", f"/matlink/{self.client_id}{suffix}", body,
                                    extra_headers={"X-OpenDPD-Matlink": self._bridge_token})

    def heartbeat(self, variables_json):
        try:
            result = self._request("/heartbeat", {"variables": json.loads(variables_json)})
        except SDKError as error:
            if error.code in ("matlink_session_not_found", "matlink_disconnected"):
                self._closed = True
            raise
        return json.dumps(result, ensure_ascii=False, allow_nan=False)

    def complete(self, request_id, outcome_json):
        self._request(f"/requests/{request_id}/complete", json.loads(outcome_json))

    def dataset(self, dataset_id):
        """Read every capture of a generated pair, retaining guard samples and order."""
        import numpy as np
        from opendpd.services.datasets import load_version_arrays
        from opendpd.services.workspace import Workspace, sha256_file

        ws = Workspace.open(self.project.workspace)
        parent = ws.get_dataset(str(dataset_id))
        ids = [c.dataset_id for c in parent.captures] or [parent.dataset_id]
        if len(ids) > 16 or len(set(ids)) != len(ids):
            raise SDKError("dataset_invalid", "Invalid generated capture collection")
        captures, total = [], 0
        for cid in ids:
            manifest = ws.get_dataset(cid)
            if not manifest.simulation or (cid != parent.dataset_id and manifest.parent_dataset_id != parent.dataset_id):
                raise SDKError("dataset_invalid", "MATLINK requires a generated paired dataset")
            version = manifest.version("raw-v1")
            directory = ws.dataset_version_dir(cid, "raw-v1")
            refs = {f.path: f for f in version.files} if version else {}
            for filename in ("input_iq.npy", "output_iq.npy"):
                path = directory / filename
                if (filename not in refs or path.is_symlink() or not path.resolve().is_relative_to(ws.root.resolve())
                        or not path.is_file() or sha256_file(path) != refs[filename].sha256):
                    raise SDKError("dataset_changed", "Generated I/Q is missing or its recorded hash changed")
            x, y, _ = load_version_arrays(ws, cid, "raw-v1")
            total += len(x)
            if (total > 4_000_000 or x.shape != y.shape or x.shape != (manifest.n_samples, 2)
                    or not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise SDKError("dataset_invalid", "Invalid generated I/Q arrays")
            captures.append((json.dumps(manifest.model_dump(mode="json")), np.array(x), np.array(y)))
        return json.dumps(parent.model_dump(mode="json")), captures

    def result_bundle(self, run_id):
        """Export the primary report, resolved settings and registered plot data."""
        from opendpd.services.experiments import load_artifacts
        from opendpd.services.workspace import Workspace, sha256_file

        job = self.project.job(str(run_id))
        report = job.result()
        report["configuration"] = job.config()
        report["plots"] = {}
        ws = Workspace.open(self.project.workspace)
        manifest = load_artifacts(ws, str(run_id))
        allowed = {"plot-spectrum": "spectrum", "plot-time": "time", "plot-amam": "am_am_pm"}
        for artifact in manifest.artifacts if manifest else []:
            if artifact.artifact_id not in allowed:
                continue
            path = (ws.run_dir(str(run_id)) / artifact.file.path).resolve()
            if (not path.is_relative_to(ws.run_dir(str(run_id)).resolve()) or not path.is_file()
                    or path.stat().st_size > 32 * 1024 * 1024 or sha256_file(path) != artifact.file.sha256):
                raise SDKError("artifact_changed", "A result plot is missing or its recorded hash changed")
            report["plots"][allowed[artifact.artifact_id]] = json.loads(path.read_text())
        return json.dumps(report, allow_nan=False)

    def disconnect(self):
        if not self._closed:
            try:
                self._request("/disconnect", {})
            finally:
                self._closed = True
