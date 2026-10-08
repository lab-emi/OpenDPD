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


def default_device(project, model):
    return str(project.default_device(str(model)))


def file_sha256(path):
    from opendpd.services.workspace import sha256_file

    return sha256_file(path)


def apply(job, x, execution="offline_segmented", timeout=120.0, chunk_samples=0):
    y, metadata = job.apply(x, execution=execution, chunk_samples=int(chunk_samples) or None, timeout=float(timeout))
    return y, encode(metadata)


def export_model(job, destination, timeout=300.0):
    return encode(job.export(destination, timeout=float(timeout)))


def lte_waveform(seed, n_subframes):
    from .metrics import lte_waveform as build

    waveform = build(int(seed), int(n_subframes))
    return waveform["iq"], waveform["symbols_iq"], encode(waveform["metadata"])


def evaluate_metrics(y, reference, options):
    from .metrics import evaluate

    return encode(evaluate(y, reference=reference, **json.loads(options)))
