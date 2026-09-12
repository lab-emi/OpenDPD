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

SLUG = r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}"
FILE_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,200}"
# New local routes are NOT automatically published. In particular: no path imports,
# uploads, manifest/signal mutation, RF control, package imports, or executable exports.
ROUTES = {
    "GET": [
        r"/system/(capabilities|about)", r"/settings", r"/models", r"/recipes",
        r"/datasets", r"/datasets/builtin", rf"/datasets/{SLUG}(/(analysis|diagnostics))?",
        r"/runs", r"/runs/count", rf"/runs/{SLUG}(/(config|artifacts|history|live|lineage|logs|events/list))?",
        r"/metrics/profiles", rf"/metrics/profiles/{SLUG}", r"/results/compare",
        rf"/results/{SLUG}(/(profiles|report))?", rf"/artifacts/{SLUG}/{FILE_ID}",
        rf"/exports/{FILE_ID}", r"/adaptation/reports", rf"/adaptation/reports/{SLUG}",
    ],
    "POST": [r"/datasets/import-builtin", rf"/datasets/{SLUG}/diagnostics",
             rf"/datasets/{SLUG}/preprocess(/preview)?", r"/experiments/validate",
             r"/runs", rf"/runs/{SLUG}/(cancel|retry)", r"/exports"],
    "PUT": [r"/settings"],
}


def reject(status: int, code: str, message: str):
    raise HTTPException(status, {"error": {"code": code, "message": message, "details": [], "hint": None}})


@dataclass(frozen=True)
class WebConfig:
    root: Path
    origin: str
    api_host: str
    # Host header rewritten by the host-local cloudflared connector. This is an
    # origin credential, NOT a frontend secret. Never trust arbitrary proxy headers.
    tunnel_host: str
    gpu_token: str | None = field(default=None, repr=False)
    max_sessions: int = 16
    sessions_per_ip: int = 8
    requests_per_minute: int = 600
    requests_per_session_minute: int = 120
    runs_per_ip: int = 12
    runs_per_session: int = 8
    runs_per_day: int = 60
    max_pending: int = 2
    max_parallel: int = 1
    max_runtime_seconds: int = 1800
    max_workspace_bytes: int = 256 * 1024 * 1024
    max_body: int = 64 * 1024
    max_requests: int = 8
    sweep_seconds: float = 15.0
    # Stop new requests five minutes before the daily reset. The independent
    # systemd reset kills the whole service cgroup and purges at 23:59 UTC.
    drain_seconds: int = 300

    def __post_init__(self):
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


def allowed(method: str, path: str) -> bool:
    return ".." not in path and any(re.fullmatch(pattern, path) for pattern in ROUTES.get(method, []))


def check_slug(value, field: str):
    if not isinstance(value, str) or not re.fullmatch(FILE_ID, value) or ".." in value:
        reject(422, "invalid_identifier", f"{field} must be an identifier, never a filesystem path")


def check_query(query):
    for name, value in query.multi_items():
        if len(name) > 64 or len(value) > 1024:
            reject(422, "invalid_query", "query is too large")
        if name in {"version", "profile", "profile_id"}:
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
