"""Whole-file validation of paired PA captures, shared by Studio, CLI and Python.

The parser never evaluates expressions, drops bad rows, fills missing samples
or shuffles time order. Split coordinates come from the existing core protocol.
"""

import csv
import math
import re
from pathlib import Path

import numpy as np

from opendpd.core.splits import contiguous_boundaries
from opendpd.schemas.importing import CsvInspection, CsvIssue, CsvOptions, DatasetImportDefaults
from opendpd.services.workspace import sha256_file

ROLES = {"complex_pair": ("input", "output"), "iq_columns": ("I_in", "Q_in", "I_out", "Q_out")}
MAX_ISSUES = 20
FLOAT32_MAX = float(np.finfo(np.float32).max)
FORMAT_HELP = "Save a UTF-8, comma-separated CSV with input,output complex columns (e.g. 0.1+0.2j,0.3-0.4i), or I_in,Q_in,I_out,Q_out real columns. Each row must contain one paired sample."


def _number(value: str, complex_value: bool):
    value = "".join(value.split()).lower() if complex_value else value.strip()
    if complex_value:
        value = value.replace("i)", "j)")
        if value.endswith("i"):
            value = value[:-1] + "j"
    if not value or "_" in value:
        raise ValueError("empty or invalid numeric value")
    result = complex(value) if complex_value else float(value)
    parts = (result.real, result.imag) if complex_value else (result,)
    if not all(math.isfinite(part) for part in parts):
        raise ArithmeticError("NaN and infinity are not finite samples")
    if max(abs(part) for part in parts) > FLOAT32_MAX:
        raise OverflowError("value exceeds the float32 sample range")
    return result


def _is_header(row):
    # Do not hide a malformed first sample merely because one cell is text.
    # Reserved non-finite numeric spellings must be diagnosed as sample errors.
    return bool(row) and all(
        re.fullmatch(r"[^\W\d][\w .()/\-]*", cell.strip(), flags=re.UNICODE)
        and cell.strip().lower() not in {"nan", "inf", "infinity", "nanj", "infj", "j", "i"}
        for cell in row
    )


def inspect_csv(path: Path, options: CsvOptions | None = None, split: DatasetImportDefaults | None = None,
                *, collect: bool = False):
    """Return a bounded report and, only for materialisation, validated float32 arrays."""
    options, split = options or CsvOptions(), split or DatasetImportDefaults()
    report = CsvInspection(options=options, split=split)
    blocks, block = [], []

    def issue(code, message, fix, line=None, column=None):
        report.issue_count += 1
        if len(report.issues) < MAX_ISSUES:
            report.issues.append(CsvIssue(code=code, message=message, fix=fix, line=line, column=column))

    path = Path(path)
    if path.suffix.lower() != ".csv" or not path.is_file():
        issue("file", "The selected source is not a readable CSV file.", FORMAT_HELP)
        return report, None, None
    report.sha256 = sha256_file(path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            first = next(reader, None)
            if first is None:
                issue("empty", "The CSV is empty.", FORMAT_HELP)
                return report, None, None
            width = len(first)
            fmt = options.format if options.format != "auto" else {2: "complex_pair", 4: "iq_columns"}.get(width)
            if fmt is None or width != len(ROLES[fmt]):
                issue("columns", f"Found {width} columns; expected two complex or four real columns.", FORMAT_HELP, 1)
                return report, None, None
            has_header = options.header == "present" or (options.header == "auto" and _is_header(first))
            columns = [cell.strip() for cell in first] if has_header else [f"column_{i + 1}" for i in range(width)]
            report.columns = columns
            if any(not c for c in columns) or len(set(columns)) != width:
                issue("header", "Column names are empty or duplicated.", "Give each column a unique, non-empty header, or select 'No header' if the first row is data.", 1)
            roles = ROLES[fmt]
            if options.mapping:
                mapping = dict(options.mapping)
            elif fmt == "iq_columns" and has_header:
                from opendpd.services.datasets import suggest_mapping
                guessed = suggest_mapping(columns)
                mapping = {key: columns.index(guessed[key]) for key in roles} if set(guessed) == set(roles) else dict(zip(roles, range(width)))
            elif fmt == "complex_pair" and has_header:
                aliases = {"input": {"input", "x", "tx", "pa_input"}, "output": {"output", "y", "rx", "pa_output"}}
                guessed = {key: i for key, names in aliases.items() for i, c in enumerate(columns) if c.lower() in names}
                mapping = guessed if set(guessed) == set(roles) else dict(zip(roles, range(width)))
            else:
                mapping = dict(zip(roles, range(width)))
            mapping_ok = set(mapping) == set(roles) and set(mapping.values()) == set(range(width))
            if not mapping_ok:
                issue("mapping", "Every input/output role must use a different source column.", "Map input and output once each; for real columns also map both I and Q. Do not reuse a column.")
            report.options = CsvOptions(format=fmt, header="present" if has_header else "absent", mapping=mapping)

            def consume(row, line):
                report.n_samples += 1
                if len(report.preview) < 5:
                    report.preview.append([value[:120] for value in row])
                if len(row) != width:
                    issue("row_width", f"Expected {width} fields, found {len(row)}.", "Restore the missing fields or remove extra separators. Remove empty rows; never omit only one signal's sample.", line)
                    return
                values = []
                row_valid = True
                for i, cell in enumerate(row):
                    try:
                        values.append(_number(cell, fmt == "complex_pair"))
                    except (ValueError, ArithmeticError) as exc:
                        code = "non_finite" if type(exc) is ArithmeticError else "numeric"
                        issue(code, f"Invalid sample {cell[:60]!r}: {exc}.", "Use finite decimal/scientific numbers; complex cells accept a+bj or a+bi. Correct the source sample, including its paired input/output, instead of leaving a blank or NaN.", line, columns[i])
                        row_valid = False
                if collect and mapping_ok and row_valid:
                    ordered = [values[mapping[key]] for key in roles]
                    block.append([ordered[0].real, ordered[0].imag, ordered[1].real, ordered[1].imag] if fmt == "complex_pair" else ordered)
                    if len(block) >= 100_000:
                        blocks.append(np.asarray(block, dtype=np.float32))
                        block.clear()

            if not has_header:
                consume(first, 1)
            for row in reader:
                consume(row, reader.line_num)
    except UnicodeDecodeError:
        issue("encoding", "The file is not valid UTF-8 text.", "Export as CSV UTF-8 (comma delimited); an Excel workbook renamed .csv is not CSV.")
    except csv.Error as exc:
        issue("csv_syntax", f"Malformed CSV: {exc}.", "Check quotes and separators, then export as a comma-delimited CSV again.", reader.line_num)
    except OSError as exc:
        issue("read", f"The file could not be read: {exc}.", "Check the file is available and upload it again.")
    if report.n_samples == 0:
        issue("empty", "The CSV has no sample rows.", "Add paired input/output data below the header.")
    report.data_valid = report.issue_count == 0
    try:
        if any(not math.isfinite(r) or r <= 0 for r in split.ratios.values()):
            raise ValueError("Each train/validation/test ratio must be finite and greater than zero")
        report.boundaries = contiguous_boundaries(report.n_samples, split.ratios, split.guard_samples)
        report.split_counts = {key: end - start for key, (start, end) in report.boundaries.items()}
    except (ValueError, OverflowError) as exc:
        issue("split", str(exc), "Use three positive ratios summing to 100%. Provide enough samples for all three splits plus both boundary guards; alternatively reduce the guard only when appropriate for your frame length.")
    report.valid = report.issue_count == 0
    if not collect or not report.valid:
        return report, None, None
    if block:
        blocks.append(np.asarray(block, dtype=np.float32))
    samples = np.concatenate(blocks)
    return report, samples[:, :2], samples[:, 2:]
