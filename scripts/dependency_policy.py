"""Data-only review of routine Dependabot updates; never import PR code."""
from __future__ import annotations

import copy
import json
import re
from urllib.parse import urlsplit

MANIFEST = "frontend/package.json"
LOCK = "frontend/package-lock.json"
DOCS = "docs/requirements.txt"
ACTIONS = {
    "actions/checkout", "actions/setup-python", "actions/setup-node",
    "actions/upload-artifact", "actions/download-artifact",
    "actions/upload-pages-artifact", "actions/deploy-pages", "astral-sh/setup-uv",
}
VERSION = re.compile(r"([~^]?)(\d+)\.(\d+)(?:\.(\d+))?")
USE = re.compile(r"(\s*(?:-\s*)?uses:\s+)([\w.-]+/[\w./-]+)@([a-f0-9]{40})\s+# v(\d+(?:\.\d+){0,2})\s*")


def routine_version(old, new):
    """Preserve the constraint kind; no downgrades, prereleases or breaking 0.x bumps."""
    if old == new:
        return isinstance(old, str) and bool(old)
    a, b = VERSION.fullmatch(old), VERSION.fullmatch(new)
    if not a or not b or a[1] != b[1]:
        return False
    av, bv = tuple(int(n or 0) for n in a.groups()[1:]), tuple(int(n or 0) for n in b.groups()[1:])
    return av <= bv and av[0] == bv[0] and (av[0] != 0 or av[1] == bv[1])


def dependency_paths(files):
    return bool(files) and all(
        f.get("status") == "modified" and not f.get("previous_filename")
        and (f["filename"] in {MANIFEST, LOCK, DOCS}
             or re.fullmatch(r"\.github/workflows/[\w-]+\.ya?ml", f["filename"]))
        for f in files
    )


def dependabot_pr(pr):
    return (pr.get("user", {}).get("login") == "dependabot[bot]"
            and pr.get("user", {}).get("type") == "Bot"
            and pr.get("base", {}).get("ref") == "main"
            and pr.get("head", {}).get("repo", {}).get("full_name")
            == pr.get("base", {}).get("repo", {}).get("full_name")
            and bool(pr.get("base", {}).get("repo", {}).get("full_name"))
            and pr.get("head", {}).get("ref", "").startswith("dependabot/")
            and not pr.get("draft"))


def manifest_update(old, new):
    a, b = json.loads(old), json.loads(new)
    for section in ("dependencies", "devDependencies"):
        before, after = a.get(section, {}), b.get(section, {})
        if before.keys() != after.keys() or not all(routine_version(v, after[k]) for k, v in before.items()):
            return False
        a[section] = after
    return a == b


def lock_update(old, new, manifest, *, direct_changed=False):
    a, b, package = json.loads(old), json.loads(new), json.loads(manifest)
    ap, bp = a.pop("packages", {}), b.pop("packages", {})
    if a != b or b.get("lockfileVersion") != 3 or not bp or len(bp) > 5000:
        return False
    expected = copy.deepcopy(ap.get("", {}))
    for section in ("dependencies", "devDependencies"):
        expected[section] = package.get(section, {})
    if bp.get("") != expected:
        return False
    for path, value in bp.items():
        if path == "":
            continue
        if not path.startswith("node_modules/") or ".." in path.split("/") or "\\" in path:
            return False
        if value.get("link") or "scripts" in value or "workspaces" in value:
            return False
        url = urlsplit(value.get("resolved", ""))
        if (url.scheme != "https" or url.netloc != "registry.npmjs.org" or url.query or url.fragment
                or not url.path.endswith(".tgz")
                or not re.fullmatch(r"sha(?:256|384|512)-[A-Za-z0-9+/]+=*", value.get("integrity", ""))):
            return False
        if path in ap and not direct_changed and not routine_version(ap[path].get("version", ""), value.get("version", "")):
            return False
        if path in ap and ap[path].get("version") == value.get("version"):
            if any(ap[path].get(key) != value.get(key) for key in ("resolved", "integrity")):
                return False
    return True


def docs_update(old, new):
    before, after = old.splitlines(), new.splitlines()
    if len(before) != len(after):
        return False
    for a, b in zip(before, after):
        if a == b:
            continue
        match = re.fullmatch(r"(mkdocs-material|mkdocstrings\[python\])>=(\d+(?:\.\d+){1,2})(,<\d+)", a)
        changed = re.fullmatch(r"(mkdocs-material|mkdocstrings\[python\])>=(\d+(?:\.\d+){1,2})(,<\d+)", b)
        if not match or not changed or match[1] != changed[1] or match[3] != changed[3] or not routine_version(match[2], changed[2]):
            return False
    return True


def action_updates(old, new):
    before, after = old.splitlines(), new.splitlines()
    if len(before) != len(after):
        raise ValueError("workflow structure changed")
    pins = []
    for a, b in zip(before, after):
        if a == b:
            continue
        x, y = USE.fullmatch(a), USE.fullmatch(b)
        if (not x or not y or x[1] != y[1] or x[2] != y[2] or y[2] not in ACTIONS
                or not routine_version(x[4], y[4]) or (x[4] == y[4] and x[3] != y[3])):
            raise ValueError("workflow changes require human review")
        pins.append((y[2], "v" + y[4], y[3]))
    return pins


def review(pr, files, before, after, verify_pin):
    """Readers are bound to immutable base/head SHAs; labels and PR prose are irrelevant."""
    if not dependabot_pr(pr) or not dependency_paths(files):
        return False
    try:
        names = {f["filename"] for f in files}
        if names & {MANIFEST, LOCK}:
            if MANIFEST in names and LOCK not in names:
                return False
            if not manifest_update(before(MANIFEST), after(MANIFEST)):
                return False
            old_manifest, new_manifest = json.loads(before(MANIFEST)), json.loads(after(MANIFEST))
            direct_changed = any(old_manifest.get(key, {}) != new_manifest.get(key, {})
                                 for key in ("dependencies", "devDependencies"))
            # A reviewed nonbreaking direct update may legitimately change its
            # internal dependency tree (including 0.x build tooling).
            if not lock_update(before(LOCK), after(LOCK), after(MANIFEST), direct_changed=direct_changed):
                return False
        for name in names - {MANIFEST, LOCK}:
            if name == DOCS:
                if not docs_update(before(name), after(name)):
                    return False
            else:
                for repo, tag, sha in action_updates(before(name), after(name)):
                    if not verify_pin(repo, tag, sha):
                        return False
        return True
    except (KeyError, TypeError, ValueError, UnicodeError):
        return False
