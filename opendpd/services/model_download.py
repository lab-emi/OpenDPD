"""Atomic snapshots of generated model weights, available while training runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODEL_FILE = 'model-download.pt'
MODEL_META = 'model-download.json'
MAX_MODEL_BYTES = 16 * 1024 * 1024


def publish_model(root: Path, data: bytes, *, epoch: int, sha256: str | None = None):
    if not data or len(data) > MAX_MODEL_BYTES or type(epoch) is not int or not 0 <= epoch <= 1_000_000:
        raise ValueError('invalid model snapshot')
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError('model snapshot checksum mismatch')
    temporary = root / '.model-download.tmp'
    try:
        with temporary.open('wb') as output:
            output.write(data)
        os.replace(temporary, root / MODEL_FILE)
        meta = root / '.model-meta.tmp'
        try:
            meta.write_text(json.dumps({'sha256': digest, 'size_bytes': len(data), 'epoch': epoch}))
            os.replace(meta, root / MODEL_META)
        finally:
            meta.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def read_model(root: Path):
    # A download is a stable in-memory snapshot. An atomic replacement while a
    # client is downloading cannot truncate it or mix two different checkpoints.
    for _ in range(3):
        path, meta = root / MODEL_FILE, root / MODEL_META
        if path.is_symlink() or meta.is_symlink() or not path.is_file() or not meta.is_file():
            return None
        try:
            info = json.loads(meta.read_text())
            with path.open('rb') as source:
                data = source.read(MAX_MODEL_BYTES + 1)
            if len(data) <= MAX_MODEL_BYTES and hashlib.sha256(data).hexdigest() == info['sha256']:
                return info, data
        except (OSError, ValueError, KeyError):
            continue
    return None
