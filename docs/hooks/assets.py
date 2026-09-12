"""Publish shared repository assets and resolve their links from standalone guides.

README snippets use root-relative ``pics/`` paths. Guides under ``docs/`` use
repository-relative paths so their Markdown also works directly on GitHub.
"""

from __future__ import annotations

import re
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


def on_page_markdown(markdown, page, **kwargs):
    """Drop only the extra hop out of docs/ in Markdown links to root pics/."""
    depth = page.file.src_uri.count("/")
    repo_prefix = "../" * (depth + 1) + "pics/"
    site_prefix = "../" * depth + "pics/"
    return re.sub(r"(?<=\]\()" + re.escape(repo_prefix), site_prefix, markdown)


def on_files(files: Files, config) -> Files:
    paths = [p for d in ASSET_DIRS for p in sorted((ROOT / d).iterdir()) if p.is_file()]
    paths += [ROOT / f for f in ASSET_FILES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"docs asset {path.relative_to(ROOT)} is missing")
        files.append(File(str(path.relative_to(ROOT)), str(ROOT), config["site_dir"], config["use_directory_urls"]))
    return files
