"""Public project information, fetched only when the About page is requested.

Fixed public GitHub endpoints receive no workspace, experiment or account data.
One bounded cache protects GitHub's anonymous rate budget; a network failure
keeps the last successful snapshot with an explicit stale status.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from opendpd import __version__

REPOSITORY = "https://github.com/lab-emi/OpenDPD"
API = "https://api.github.com/repos/lab-emi/OpenDPD"
_lock = threading.Lock()
_cache = None
_checked = -float("inf")


def _get(endpoint):
    request = urllib.request.Request(API + endpoint, headers={"User-Agent": "OpenDPD-Studio",
                                                              "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=6) as response:
        return json.loads(response.read(2_000_000))


def project_info():
    global _cache, _checked
    with _lock:
        if _cache is not None and time.monotonic() - _checked < 300:
            return _cache
        local_commit = None
        try:
            local_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            pass
        base = {"version": __version__, "repository": REPOSITORY, "lab_url": "https://www.tudemi.com/",
                "local_commit": local_commit, "refresh_seconds": 300, "contributors": [], "commits": [],
                "updated_at": None, "status": "unavailable"}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                contributors_future = pool.submit(_get, "/contributors?per_page=100")
                commits_future = pool.submit(_get, "/commits?per_page=5")
                contributors, commits = contributors_future.result(), commits_future.result()
            base.update(contributors=[{"login": item["login"], "contributions": item["contributions"],
                                       "url": item["html_url"]} for item in contributors],
                        commits=[{"sha": item["sha"], "url": item["html_url"],
                                  "message": item["commit"]["message"].splitlines()[0],
                                  "author": item["commit"]["author"]["name"],
                                  "date": item["commit"]["author"]["date"]} for item in commits],
                        updated_at=datetime.now(timezone.utc).isoformat(), status="current",
                        contributors_truncated=len(contributors) == 100)
            _cache = base
        except (OSError, ValueError, KeyError, TypeError) as error:
            _cache = {**(_cache or base), "status": "stale" if _cache and _cache.get("updated_at") else "unavailable",
                      "error": f"GitHub is unavailable ({type(error).__name__}); use the repository links or retry later."}
        _checked = time.monotonic()
        return _cache
