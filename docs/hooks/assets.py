"""Publish shared repository assets and resolve their links from standalone guides.

README snippets use root-relative ``pics/`` paths. Guides under ``docs/`` use
repository-relative paths so their Markdown also works directly on GitHub.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

from mkdocs.structure.files import File, Files

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIRS = ["pics"]
ASSET_FILES = [
    "frontend/src/assets/opendpd-studio-logo.svg",
    "frontend/src/assets/opendpd-studio-logo-inverse.svg",
    "frontend/src/assets/opendpd-studio-mark.svg",
    "frontend/public/favicon.svg",
    "frontend/src/assets/emi-logo.svg",
    "frontend/src/assets/emi-logo-inverse.svg",
    "benchmark/benchmark_results.png",
    "benchmark/benchmark_delta_dpd_results.png",
    "benchmark/reproduce_benchmark_report.sh",
    "benchmark/results/benchmark_report_results.json",
]


def _theme_logo(match):
    """Map the shared GitHub picture markup to Material's palette-aware images."""
    picture = ElementTree.fromstring(match.group(0))
    light = picture.find("img")
    source = picture.find("source")
    dark = deepcopy(light)
    dark.set("src", source.attrib["srcset"] + "#only-dark")
    light.set("src", light.attrib["src"] + "#only-light")
    return ElementTree.tostring(light, encoding="unicode") + ElementTree.tostring(dark, encoding="unicode")


def on_page_markdown(markdown, page, **kwargs):
    """Drop only the extra hop out of docs/ in Markdown links to root pics/."""
    depth = page.file.src_uri.count("/")
    repo_prefix = "../" * (depth + 1) + "pics/"
    site_prefix = "../" * depth + "pics/"
    return re.sub(r"(?<=\]\()" + re.escape(repo_prefix), site_prefix, markdown)


def on_page_content(html, **kwargs):
    """Apply site-specific theming after README snippets have been expanded."""
    return re.sub(r'<picture class="brand-logo">.*?</picture>', _theme_logo, html, flags=re.DOTALL)


def on_files(files: Files, config) -> Files:
    paths = [p for d in ASSET_DIRS for p in sorted((ROOT / d).iterdir()) if p.is_file()]
    paths += [ROOT / f for f in ASSET_FILES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"docs asset {path.relative_to(ROOT)} is missing")
        files.append(File(str(path.relative_to(ROOT)), str(ROOT), config["site_dir"], config["use_directory_urls"]))
    return files
