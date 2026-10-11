"""Build the bundled offline guide from its Markdown sources (pip install Markdown).

Run from any folder. Use --check in CI to detect stale packaged documentation.
Both modes validate bundled local links and HTML anchors.
"""
from __future__ import annotations

import argparse
import html
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1] / "Matlab" / "toolbox"
PAGES = {"index": "Get started", "gui": "MATLINK walkthrough", "workflow": "Script workflow",
         "reference": "Function reference", "architecture": "Integration design",
         "troubleshooting": "Troubleshooting"}


class PageLinks(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        for attribute in ("href", "src"):
            if values.get(attribute):
                self.links.append(values[attribute])


def validate_links(pages):
    parsed = {path: PageLinks(source) for path, source in pages.items()}
    failures = []
    for source_path, source in parsed.items():
        for link in source.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (source_path.parent / unquote(url.path)).resolve() if url.path else source_path
            if target not in parsed and not target.is_file():
                failures.append(f"{source_path.name}: missing {link}")
            elif url.fragment and target.suffix == ".html":
                if target not in parsed:
                    parsed_target = PageLinks(target.read_text(encoding="utf-8"))
                else:
                    parsed_target = parsed[target]
                if unquote(url.fragment) not in parsed_target.ids:
                    failures.append(f"{source_path.name}: missing anchor {link}")
    if failures:
        raise SystemExit("Broken MATLAB documentation links:\n" + "\n".join(failures))


def toolbox_version():
    """The one version the toolbox carries (Contents.m, checked against the build file and README by a test)."""
    return re.search(r"^% Version (\d+\.\d+\.\d+)$", (ROOT / "Contents.m").read_text(encoding="utf-8"), re.M).group(1)


def build(check=False):
    stale = []
    pages = {}
    version = toolbox_version()
    for slug, title in PAGES.items():
        source = (ROOT / "docs" / f"{slug}.md").read_text(encoding="utf-8")
        body = markdown.markdown(source, extensions=["tables", "fenced_code", "toc"])
        nav = "\n".join(f'<a href="{name}.html"' + (' aria-current="page"' if name == slug else '')
                        + f'>{html.escape(label)}</a>' for name, label in PAGES.items())
        page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · OpenDPD for MATLAB</title>
<link rel="stylesheet" href="guide.css"></head><body>
<a class="skip" href="#content">Skip to content</a>
<header><a class="brand" href="index.html">OpenDPD <span>FOR MATLAB</span></a>
<span class="version">{version}</span></header>
<nav aria-label="Guide">{nav}</nav>
<main id="content">{body}</main>
<footer>OpenDPD Toolbox for MATLAB {version} · Apache-2.0</footer>
</body></html>
'''
        target = ROOT / "resources" / "docs" / f"{slug}.html"
        pages[target] = page

    validate_links(pages)
    for target, page in pages.items():
        if check:
            if not target.is_file() or target.read_text(encoding="utf-8") != page:
                stale.append(str(target.relative_to(ROOT)))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(page, encoding="utf-8")
    if stale:
        raise SystemExit("Rebuild MATLAB documentation: " + ", ".join(stale))
    print(f"MATLAB documentation: {len(PAGES)} pages {'checked' if check else 'built'}; local links checked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(parser.parse_args().check)
