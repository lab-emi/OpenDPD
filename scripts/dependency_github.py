"""Bounded GitHub API access for trusted-base dependency maintenance."""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from dependency_policy import dependabot_pr, dependency_paths, review


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class GitHub:
    def __init__(self, repository=None, token=None):
        self.repository = repository or os.environ["GITHUB_REPOSITORY"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repository):
            raise ValueError("invalid repository")
        self.token = token or os.environ["GH_TOKEN"]
        self.http = urllib.request.build_opener(NoRedirect())
        self.trees = {}
        self.blobs = {}

    def request(self, path, payload=None, method=None, *, repository=None):
        repo = repository or self.repository
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("invalid API path")
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/{path}",
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28"},
            data=json.dumps(payload).encode() if payload is not None else None, method=method)
        with self.http.open(request, timeout=30) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("GitHub response exceeds limit")
        return json.loads(data) if data else None

    def pages(self, path):
        separator = "&" if "?" in path else "?"
        values = []
        for page in range(1, 21):
            chunk = self.request(f"{path}{separator}per_page=100&page={page}")
            if not isinstance(chunk, list):
                raise ValueError("invalid paginated response")
            values.extend(chunk)
            if len(chunk) < 100:
                return values
        raise ValueError("pagination limit exceeded")

    def pulls(self):
        return self.pages("pulls?state=open&base=main")

    def files(self, pr):
        files = self.pages(f"pulls/{pr['number']}/files")
        if len(files) != pr["changed_files"]:
            raise ValueError("incomplete changed-file list")
        return files

    def source(self, ref, path):
        if not re.fullmatch(r"[a-f0-9]{40}", ref):
            raise ValueError("source must be bound to a commit")
        sha = ref
        parts = path.split("/")
        for index, part in enumerate(parts):
            if sha not in self.trees:
                data = self.request(f"git/trees/{sha}")
                if data.get("truncated"):
                    raise ValueError("incomplete tree")
                self.trees[sha] = {entry["path"]: entry for entry in data["tree"]}
            entry = self.trees[sha][part]
            last = index == len(parts) - 1
            if entry["mode"] != ("100644" if last else "040000"):
                raise ValueError("symlinks and executable manifests are refused")
            sha = entry["sha"]
        if sha not in self.blobs:
            blob = self.request(f"git/blobs/{sha}")
            if blob["encoding"] != "base64" or blob["size"] > 4 * 1024 * 1024:
                raise ValueError("oversized or invalid manifest")
            data = base64.b64decode(blob["content"].replace("\n", ""), validate=True)
            if len(data) != blob["size"]:
                raise ValueError("blob size mismatch")
            self.blobs[sha] = data.decode("utf-8")
        return self.blobs[sha]

    def verify_pin(self, repository, tag, sha):
        ref = self.request("git/ref/tags/" + urllib.parse.quote(tag, safe=""), repository=repository)["object"]
        for _ in range(3):
            if ref["type"] == "commit":
                return ref["sha"] == sha
            if ref["type"] != "tag":
                return False
            ref = self.request(f"git/tags/{ref['sha']}", repository=repository)["object"]
        return False

    def comparison(self, pr):
        # PR.base.sha can remain the old base even after main advances.
        main = self.request("git/ref/heads/main")["object"]["sha"]
        return self.request(f"compare/{main}...{pr['head']['sha']}?per_page=1")

    def routine(self, pr, files=None):
        if not dependabot_pr(pr):
            return False
        files = self.files(pr) if files is None else files
        if not dependency_paths(files):
            return False
        base = self.comparison(pr)["merge_base_commit"]["sha"]
        return review(pr, files, lambda path: self.source(base, path),
                      lambda path: self.source(pr["head"]["sha"], path), self.verify_pin)


def automatic_dependency_review(pr):
    """The approval exception is intentionally narrower than the file allowlist."""
    if not dependabot_pr(pr):
        return False
    return GitHub().routine(pr)
