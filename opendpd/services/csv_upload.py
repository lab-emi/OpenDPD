"""Quarantine browser CSV uploads before exposing any sample or import path.

Only bounded UTF-8 CSV containing paired numeric samples is admitted. No file is
executed, unpickled, or handed to a spreadsheet engine. This is data validation,
not a promise that a filename or an antivirus signature proves a file safe.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import secrets
from pathlib import Path

from opendpd.services.csv_import import _is_header, _number
from opendpd.services.workspace import write_json_atomic

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SAMPLES = 1_000_000
MAX_LINE_BYTES = 4096
UPLOAD_PATH = re.compile(r"uploads/([a-f0-9]{32})\.csv\Z")


class CsvUploadRejected(ValueError):
    pass


def check_filename(filename: str):
    if (not filename or len(filename) > 200 or '/' in filename or '\\' in filename
            or any(ord(c) < 32 for c in filename) or Path(filename).suffix.lower() != '.csv'):
        raise CsvUploadRejected('Only .csv files are accepted. The file was not imported.')


def quarantine_path(ws):
    folder = ws.imports_dir / 'quarantine'
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    return folder / (secrets.token_hex(16) + '.part')


def validate_quarantine(path: Path):
    """Scan the entire file; reports never contain rejected file contents."""
    digest = hashlib.sha256()
    total = 0
    with path.open('rb') as source:
        while line := source.readline(MAX_LINE_BYTES + 1):
            total += len(line)
            if total > MAX_UPLOAD_BYTES or len(line) > MAX_LINE_BYTES:
                raise CsvUploadRejected('CSV size or line length exceeds the upload limit. Upload deleted.')
            if any(value < 32 and value not in (9, 10, 13) for value in line) or b'\x7f' in line:
                raise CsvUploadRejected('Binary or control characters are not allowed in CSV. Upload deleted.')
            digest.update(line)
    count = 0
    try:
        with path.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.reader(source, strict=True)
            first = next(reader, None)
            if first is None or len(first) not in (2, 4):
                raise CsvUploadRejected('Use two complex columns or four real I/Q columns. Upload deleted.')
            width = len(first)
            header = _is_header(first)
            if header and (len(set(first)) != width or any(len(c) > 64 or re.match(r'(?i)\s*(import|from|exec|eval)\b', c) for c in first)):
                raise CsvUploadRejected('CSV column names are invalid. Upload deleted.')

            def sample(row):
                if len(row) != width or any(len(cell) > 256 for cell in row):
                    raise CsvUploadRejected('CSV rows have missing, extra or oversized fields. Upload deleted.')
                for cell in row:
                    _number(cell, width == 2)

            if not header:
                sample(first)
                count = 1
            for row in reader:
                sample(row)
                count += 1
                if count > MAX_SAMPLES:
                    raise CsvUploadRejected('CSV contains too many samples. Upload deleted.')
            if not count:
                raise CsvUploadRejected('CSV has no sample rows. Upload deleted.')
    except (UnicodeError, csv.Error, ArithmeticError, ValueError) as exc:
        if isinstance(exc, CsvUploadRejected):
            raise
        raise CsvUploadRejected('CSV validation failed: only finite numeric samples are allowed. Upload deleted.') from None
    return {'status': 'passed', 'sha256': digest.hexdigest(), 'n_samples': count, 'columns': width}


def admit_upload(ws, path: Path):
    """Publish an unguessable source only after validation; delete on any failure."""
    destination = proof = None
    try:
        validation = validate_quarantine(path)
        identifier = secrets.token_hex(16)
        folder = ws.imports_dir / 'uploads'
        folder.mkdir(mode=0o700, exist_ok=True)
        destination = folder / (identifier + '.csv')
        proof = folder / (identifier + '.json')
        os.chmod(path, 0o600)
        os.replace(path, destination)
        write_json_atomic(proof, validation)
        return {'root_id': 'imports', 'path': f'uploads/{identifier}.csv',
                'size_bytes': destination.stat().st_size, 'validation': validation}
    except BaseException:
        for candidate in (path, destination, proof):
            if candidate is not None:
                candidate.unlink(missing_ok=True)
        raise


def validated_source(ws, source):
    """Public imports cannot select arbitrary paths, even in their own workspace."""
    reference = source.get('path') if isinstance(source, dict) else None
    match = UPLOAD_PATH.fullmatch(reference) if isinstance(reference, str) else None
    if not match or source.get('root_id') != 'imports':
        raise CsvUploadRejected('Select a CSV that passed upload validation in this workspace.')
    path = ws.imports_dir / source['path']
    proof = path.with_suffix('.json')
    if path.is_symlink() or proof.is_symlink() or not path.is_file() or not proof.is_file():
        raise CsvUploadRejected('The validated CSV is unavailable. Upload it again.')
    validation = json.loads(proof.read_text())
    if path.stat().st_size > MAX_UPLOAD_BYTES or hashlib.sha256(path.read_bytes()).hexdigest() != validation['sha256']:
        path.unlink(missing_ok=True)
        proof.unlink(missing_ok=True)
        raise CsvUploadRejected('CSV changed after validation. Upload deleted.')
    return path, validation
