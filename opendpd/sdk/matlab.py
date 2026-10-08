"""Small JSON boundary for MATLAB; all behavior lives in the public SDK."""

from __future__ import annotations

import json

from . import API_VERSION, doctor


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def diagnostics():
    return encode(doctor())


def import_iq(project, x, y, options):
    return encode(project.import_iq(x, y, **json.loads(options)))


def import_mat(project, path, options):
    return encode(project.import_mat(path, **json.loads(options)))


def submit(project, config):
    return project.submit(json.loads(config))


def apply(job, x, execution="offline_segmented", timeout=120.0):
    y, metadata = job.apply(x, execution=execution, timeout=float(timeout))
    return y, encode(metadata)
