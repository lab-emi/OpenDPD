"""MkDocs hook: publish repository assets that the single-source pages refer to.

The site pages include sections of ``README.md`` and ``benchmark/benchmark_report.md`` verbatim, and those files
reference images and evidence files that live outside ``docs/``. Adding them here keeps one copy in the repository:
no duplicates under ``docs/`` and no symlinks (which Windows checkouts would break).
"""

from __future__ import annotations

from pathlib import Path

from mkdocs.structure.files import File, Files

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIRS = ["pics"]
ASSET_FILES = [
    "benchmark/benchmark_results.png",
    "benchmark/benchmark_delta_dpd_results.png",
    "benchmark/reproduce_benchmark_report.sh",
    "benchmark/results/benchmark_report_results.json",
]


def on_files(files: Files, config) -> Files:
    paths = [p for d in ASSET_DIRS for p in sorted((ROOT / d).iterdir()) if p.is_file()]
    paths += [ROOT / f for f in ASSET_FILES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"docs asset {path.relative_to(ROOT)} is missing")
        files.append(File(str(path.relative_to(ROOT)), str(ROOT), config["site_dir"], config["use_directory_urls"]))
    return files
