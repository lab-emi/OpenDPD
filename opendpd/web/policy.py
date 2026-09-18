"""Explicit public surface and resource policy; local Studio remains unrestricted."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import HTTPException

from opendpd.core.registry import get_model
from opendpd.schemas import ExperimentConfig

from opendpd.schemas.common import SLUG_PATTERN as SLUG, FILE_ID_PATTERN as FILE_ID
# New local routes are NOT automatically published. In particular: no path imports,
# arbitrary path imports, RF control, package imports, or executable exports.
ROUTES = {
    "GET": [
        r"/system/(capabilities|about|status)", r"/settings", r"/models", r"/recipes",
        r"/datasets", r"/datasets/builtin", r"/datasets/import-defaults", rf"/datasets/{SLUG}(/(analysis|diagnostics))?",
        r"/runs", r"/runs/count", rf"/runs/{SLUG}(/(config|artifacts|history|live|lineage|logs|events/list))?",
        rf"/runs/{SLUG}/checkpoint(/download)?",
        r"/metrics/profiles", rf"/metrics/profiles/{SLUG}", r"/results/compare",
        rf"/results/{SLUG}(/(profiles|report|review))?", rf"/artifacts/{SLUG}/{FILE_ID}",
        rf"/exports/{FILE_ID}", r"/adaptation/reports", rf"/adaptation/reports/{SLUG}",
        r"/dataset-publications", r"/dataset-publications/capability", r"/dataset-publications/dspr-[a-f0-9]{64}(/download)?",
    ],
    "POST": [r"/datasets/import-builtin", r"/datasets/upload", r"/datasets/csv(/preview)?", rf"/datasets/{SLUG}/diagnostics",
             rf"/datasets/{SLUG}/manifest",
             rf"/datasets/{SLUG}/preprocess(/preview)?", r"/experiments/validate",
             r"/runs", rf"/runs/{SLUG}/(cancel|retry)", r"/exports"],
    "PUT": [r"/settings"],
}
ROUTES["POST"] += [r"/datasets/synthetic", r"/dataset-publications/prepare", r"/dataset-publications/dspr-[a-f0-9]{64}/submit"]
ROUTES["GET"] += [r"/signal-generator/presets", r"/signal-generator/signals/sg-[a-f0-9]{64}(/download)?", rf"/datasets/{SLUG}/(sample-counts|download)"]
ROUTES["POST"] += [r"/signal-generator/batches", r"/pa-library/datasets", r"/signal-generator/validate", r"/signal-generator/signals", r"/signal-generator/signals/sg-[a-f0-9]{64}/dataset"]
ROUTES["GET"] += [r"/signal-generator/signals", r"/signal-generator/signals/sg-[a-f0-9]{64}/(input\.csv|metadata\.json)",
    r"/pa-library/models", r"/pa-library/simulations/vpa-[a-f0-9]{64}(/(output\.csv|paired\.csv|metadata\.json))?"]
ROUTES["POST"] += [r"/signal-generator/signals/sg-[a-f0-9]{64}/(archive|restore)", r"/pa-library/simulations", r"/pa-library/simulations/vpa-[a-f0-9]{64}/dataset"]
ROUTES["GET"] += [r"/signal-analyzer/(sources|datasets)", r"/signal-generator/datasets/sds-[a-f0-9]{64}(/download)?"]
ROUTES["POST"] += [r"/signal-analyzer/(analyze|upload)"]


def reject(status: int, code: str, message: str):
    from opendpd.server.errors import api_error
    raise api_error(status, code, message)


@dataclass(frozen=True)
class WebConfig:
    root: Path
    origin: str
    api_host: str
    # Host header rewritten by the host-local cloudflared connector. This is an
    # origin credential, NOT a frontend secret. Never trust arbitrary proxy headers.
    tunnel_host: str
    gpu_token: str | None = field(default=None, repr=False)
    max_sessions: int = 256
    sessions_per_ip: int = 64
    max_waiting: int = 1024
    waiting_per_ip: int = 16
    queue_lease_seconds: int = 180
    requests_per_minute: int = 3600
    requests_per_session_minute: int = 120
    runs_per_ip: int = 96
    runs_per_session: int = 8
    runs_per_day: int = 512
    max_pending_global: int = 128
    max_pending: int = 2
    max_parallel: int = 1
    max_runtime_seconds: int = 1800
    max_workspace_bytes: int = 256 * 1024 * 1024
    max_body: int = 64 * 1024
    max_requests: int = 32
    max_expensive_requests: int = 2
    # Hosted publication uses the operator's dedicated GitHub CLI identity and
    # must be explicitly enabled. Browser users never supply a GitHub token.
    dataset_publications: bool = False
    publications_per_ip: int = 2
    publications_per_day: int = 20
    sweep_seconds: float = 5.0
    quota_checks_per_sweep: int = 16
    cleanup_interval_seconds: int = 12 * 3600
    inactivity_seconds: int = 2 * 3600
    # Drain before each independent reset at 11:59 / 23:59 UTC.
    drain_seconds: int = 300

    def __post_init__(self):
        for name in ('max_sessions', 'sessions_per_ip', 'max_waiting', 'waiting_per_ip',
                     'queue_lease_seconds', 'max_pending_global', 'max_requests', 'quota_checks_per_sweep'):
            if not 1 <= getattr(self, name) <= 65536:
                raise ValueError(f'{name} must be a bounded positive integer')
        if self.cleanup_interval_seconds != 12 * 3600 or not 60 <= self.drain_seconds < 3600:
            raise ValueError('public cleanup must run every 12 hours with a bounded drain window')
        if not 60 <= self.inactivity_seconds <= 2 * 3600:
            raise ValueError('public inactivity must be between one minute and two hours')
        if self.gpu_token is not None and len(self.gpu_token) < 48:
            raise ValueError("GPU broker requires a private 48+ character token")
        url = urlsplit(self.origin)
        if (url.scheme != "https" or url.path or url.query or url.fragment or url.username
                or url.password or url.port not in (None, 443) or not url.hostname):
            raise ValueError("OpenDPD web origin must be an exact HTTPS origin, without a path")
        for host in (self.api_host, self.tunnel_host):
            if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,252}", host):
                raise ValueError("API and tunnel hosts must be lowercase DNS hostnames")
        if len(self.tunnel_host.split(".")[0]) < 32 or self.tunnel_host == self.api_host:
            raise ValueError("tunnel host must begin with a random, private 32+ character label")


# A dataset id contains no separator, so `/datasets/<id>` also matches static
# sibling routes registered under the same prefix. FastAPI dispatches those to
# the static endpoint, never to `dataset_get`, so an id pattern silently
# publishes a route nobody listed. `import-roots` answers with absolute host
# paths, which is exactly what `app.state.workspace_label` exists to keep out
# of `/system/capabilities`. Keep such routes out by name.
# `tests/unit/test_public_policy_surface.py` fails if a new one appears.
NEVER_PUBLIC = ("/datasets/import-roots",)


def allowed(method: str, path: str) -> bool:
    if ".." in path or any(path == p or path.startswith(p + "/") for p in NEVER_PUBLIC):
        return False
    return any(re.fullmatch(pattern, path) for pattern in ROUTES.get(method, []))


def expensive_request(method: str, path: str) -> bool:
    """Bound in-process numeric/file work separately from lightweight status and cancellation."""
    if method == 'GET':
        return bool(re.fullmatch(r'/datasets/builtin|/pa-library/models|/signal-analyzer/(sources|datasets)|/signal-generator/datasets/[^/]+/download|/datasets/[^/]+/(analysis|download)|/results/compare|/results/[^/]+(/(report|review))?', path))
    return method == 'POST' and (path == '/exports' or path.startswith(('/datasets/', '/signal-generator/', '/signal-analyzer/', '/pa-library/', '/dataset-publications/')))


def check_slug(value, field: str):
    if not isinstance(value, str) or not re.fullmatch(SLUG, value) or ".." in value:
        reject(422, "invalid_identifier", f"{field} must be an identifier, never a filesystem path")


def check_query(query):
    for name, value in query.multi_items():
        if len(name) > 64 or len(value) > 1024:
            reject(422, "invalid_query", "query is too large")
        if name in {"version", "profile", "profile_id", "dataset_id"}:
            check_slug(value, name)
        if name == "runs":
            if len(value.split(",")) > 8:
                reject(422, "comparison_limit", "compare at most eight runs")
            for run_id in value.split(","):
                check_slug(run_id, "runs")


def check_config(config: ExperimentConfig):
    if config.task.value == "evaluate_measured" or config.measurement is not None:
        reject(403, "feature_unavailable", "RF measurement is unavailable in the public demo")
    check_slug(config.dataset.preprocessing_version, "preprocessing_version")
    check_slug(config.dataset.split_version, "split_version")
    t = config.training
    limits = {"epochs": 300, "batch_size": 256, "batch_size_eval": 256, "frame_length": 400,
              "frame_stride": 400, "train_samples": 1000000, "seed": 2**32 - 1}
    for key, limit in limits.items():
        value = getattr(t, key)
        if value is not None and value > limit:
            reject(422, "compute_limit", f"training.{key} must be at most {limit} in the public demo")
    if (config.execution.device not in {"cpu", "cuda"} or config.execution.device_index != 0
            or (config.execution.num_threads or 1) > 4 or config.execution.cuda_graph_training):
        reject(422, "compute_limit", "use CPU or CUDA device 0, at most four threads, without CUDA graphs")
    if config.evaluation.chunk_samples and config.evaluation.chunk_samples > 65536:
        reject(422, "compute_limit", "chunk_samples must be at most 65536")
    model = get_model(config.model.key)
    if model.key == "ilc_dpd":
        from opendpd.core.registry import validate_parameters
        params = validate_parameters(model.key, config.model.parameters, "dpd")
        for name, maximum in {"K": 9, "Q": 16, "iterations": 60, "fit_samples": 32768, "backtracking_steps": 6}.items():
            if params[name] > maximum:
                reject(422, "compute_limit", f"ILC {name} must be at most {maximum} in the public app")
        return
    if model.status != "supported" or model.training_method != "gradient":
        reject(422, "compute_limit", "the public demo currently supports reviewed neural models only")
    params = {**model.defaults(), **config.model.parameters}
    for name, value in params.items():
        if isinstance(value, (float, int)) and (not math.isfinite(value) or abs(value) > 128):
            reject(422, "compute_limit", f"model parameter {name} must be finite and at most 128")
    if params.get("num_layers", 1) > 2:
        reject(422, "compute_limit", "at most two neural network layers are supported")


def check_body(path: str, body: dict):
    if not isinstance(body, dict):
        reject(422, "invalid_request", "expected a JSON object")
    for name in ("dataset_id", "run_id", "version", "base_version"):
        if body.get(name) is not None:
            check_slug(body[name], name)
    if path == "/datasets/import-builtin":
        # One canonical copy of each built-in per session, no duplicate disk filling.
        if body.get("dataset_id") is not None:
            reject(422, "invalid_request", "use the built-in dataset's default identifier")
    if path in {"/datasets/csv", "/datasets/csv/preview"} or re.fullmatch(rf"/datasets/{SLUG}/manifest", path):
        signal = body.get("signal", {})
        if signal is None and path.endswith('/manifest'):
            signal = {}
        if not isinstance(signal, dict):
            reject(422, "invalid_request", "expected signal metadata")
        if signal.get("waveform") is not None:
            reject(422, "feature_unavailable", "reference waveform binding is unavailable for public uploads")
        # Uploaded metadata controls FFT allocations and channel loops downstream.
        for name, limit in (("nperseg", 65536), ("n_sub_ch", 32),
                            ("sample_rate_hz", 1e12), ("bandwidth_hz", 1e12),
                            ("sub_channel_bandwidth_hz", 1e12)):
            value = signal.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0 or value > limit):
                reject(422, "compute_limit", f"signal.{name} must be positive, finite and at most {limit:g}")
        for name in ("standard", "modulation"):
            value = signal.get(name)
            if value is not None and (not isinstance(value, str) or len(value) > 128):
                reject(422, "invalid_request", f"signal.{name} must be at most 128 characters")
    if "/preprocess" in path:
        params = body.get("params", {})
        if not isinstance(params, dict):
            reject(422, "invalid_request", "expected preprocessing parameters")
        for key, bound in (("delay_samples", 4096), ("gain_db", 100)):
            value = params.get(key, 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > bound:
                reject(422, "compute_limit", f"{key} must be finite and between {-bound} and {bound}")
    if path in {"/runs", "/experiments/validate"}:
        try:
            check_config(ExperimentConfig.model_validate(body.get("config", {})))
        except ValueError:
            reject(422, "invalid_config", "invalid experiment configuration")
