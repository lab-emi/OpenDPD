"""Trusted-base, data-only scan and independent current-head human review gate.

Never check out or execute PR code. This script and the scanner must be loaded
from the target branch. The resulting commit status is bound to the PR head,
including when invoked by pull_request_target (whose job check is on the base).
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependency_github import automatic_dependency_review
from opendpd.core.backbone_template import MAX_SOURCE_BYTES, scan_source, source_sha256

CONTEXT = "user-backbone-review"
CATALOG = "backbones/user_uploaded/"
PROTECTED = (CATALOG, ".github/workflows/", ".github/CODEOWNERS",
             "scripts/check_user_backbone_review.py", "opendpd/core/backbone_template.py",
             "opendpd/core/template_network.py", "opendpd/services/user_backbones.py",
             "opendpd/server/backbone_routes.py", "opendpd/core/backbone_builders.py",
             "opendpd/core/registry.py", "opendpd/services/dataset_publication.py",
             "opendpd/services/legacy_adapter.py", "models.py", "opendpd/web/")


def approved_by_human(pr, reviews, permission):
    """No bot, author, comment, dismissed approval or approval of an old head."""
    latest = {}
    for review in sorted(reviews, key=lambda r: r.get("id", 0)):
        user = review.get("user") or {}
        if review.get("state") in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest[user.get("login")] = review
    approved = False
    for login, review in latest.items():
        user = review["user"]
        if (not login or user.get("type") != "User" or login == pr["user"]["login"]
                or login.endswith("[bot]") or permission(login) not in {"admin", "maintain"}):
            continue
        if review["state"] == "CHANGES_REQUESTED":
            return False
        if review["state"] == "APPROVED" and review.get("commit_id") == pr["head"]["sha"]:
            approved = True
    return approved


def validate_changed_templates(files, get_source):
    written = {f['filename'] for f in files if f['status'] != 'removed'}
    departed = {f['filename'] for f in files if f['status']=='removed'} | {f['previous_filename'] for f in files if f.get('previous_filename')}
    paths = {path for path in written | departed if path.startswith(CATALOG)}
    folders = set()
    for path in paths:
        if path == CATALOG + "README.md":
            continue
        match = re.fullmatch(re.escape(CATALOG) + r"([a-z][a-z0-9_]{0,39}_[a-f0-9]{16})/(backbone\.py|manifest\.json)", path)
        if not match:
            if path in departed and path not in written:
                continue
            raise ValueError("Only backbone.py and manifest.json in a contribution directory are accepted.")
        folders.add(match[1])
    for folder in folders:
        pair = {f'{CATALOG}{folder}/{name}' for name in ('backbone.py','manifest.json')}
        if pair <= departed and not pair & written:
            continue
        source = get_source(f"{CATALOG}{folder}/backbone.py")
        definition = scan_source(source)
        if not definition["author"].strip():
            raise ValueError("Public author attribution is required.")
        metadata = json.loads(get_source(f"{CATALOG}{folder}/manifest.json"))
        digest = source_sha256(source)
        if metadata != {"format": "opendpd-backbone-template-v1", "source_sha256": digest} or not folder.endswith("_" + digest[:16]):
            raise ValueError("Backbone manifest or directory hash does not match the uploaded bytes.")


def evaluate(pr, files, get, get_source):
    protected = any(f["filename"].startswith(PROTECTED) or f.get("previous_filename", "").startswith(PROTECTED) for f in files)
    if not protected:
        return True, "No backbone security or catalog changes."
    if automatic_dependency_review(pr):
        return True, "Routine Dependabot version pins verified; full CI remains required."
    validate_changed_templates(files, get_source)
    if not reviewed(pr, get):
        return False, "Independent human maintainer approval of this exact head is required."
    return True, "Template scan and independent current-head human approval passed."


def reviewed(pr, get):
    reviews = []
    for page in range(1, 21):
        batch = get(f'pulls/{pr["number"]}/reviews?per_page=100&page={page}')
        reviews.extend(batch)
        if len(batch) < 100:
            break
    else:
        raise ValueError("Review history exceeds the verification limit.")
    return approved_by_human(pr, reviews, lambda login: get(f"collaborators/{login}/permission")["permission"])


def pr_files(pr, get):
    files = []
    for page in range(1, 31):
        batch = get(f'pulls/{pr["number"]}/files?per_page=100&page={page}')
        files.extend(batch)
        if len(batch) < 100:
            break
    if len(files) != pr['changed_files']:
        raise ValueError("Cannot verify the complete PR file list.")
    return files


def evaluate_merge_group(group, get, get_source):
    """Every protected queued blob must match an independently approved PR head.

    We don't transfer an approval to a synthetic commit by name or trust an old
    green status. Re-read the live reviews and compare the actual file identities.
    Conflicting combined edits fail closed; queue those protected PRs separately.
    """
    diff = get(f'compare/{group["base_sha"]}...{group["head_sha"]}?per_page=100')
    files = diff.get('files', [])
    if len(files) >= 300 or diff.get('total_commits', 0) > 100:
        raise ValueError('Merge group exceeds the complete-diff verification limit.')
    protected = [f for f in files if f['filename'].startswith(PROTECTED) or f.get('previous_filename','').startswith(PROTECTED)]
    if not protected:
        return True, 'No backbone security or catalog changes in the merge group.'
    validate_changed_templates(files, get_source)
    candidates = get('pulls?state=open&base=main&per_page=100')
    if len(candidates) >= 100:
        raise ValueError('Too many open PRs for bounded merge-group verification.')
    def identity(f):
        return (f['filename'], f.get('previous_filename'), f['status'], None if f['status']=='removed' else f['sha'])
    remaining = {identity(f) for f in protected}
    for candidate in candidates:
        pr = get(f'pulls/{candidate["number"]}')
        if pr['base']['ref'] != 'main' or pr['state'] != 'open':
            continue
        matching = remaining & {identity(f) for f in pr_files(pr, get)}
        if not matching or not reviewed(pr, get):
            continue
        if get(f'pulls/{pr["number"]}')['head']['sha'] != pr['head']['sha']:
            raise ValueError('A contributing PR changed during queue verification.')
        remaining -= matching
        if not remaining:
            return True, 'Queued backbone bytes match independently approved current PR heads.'
    return False, 'Each protected queued file must match a current human-approved PR head.'


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    repository = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise SystemExit("Invalid repository.")
    number = (event.get("pull_request") or event.get("issue") or {}).get("number") or event.get("inputs", {}).get("pr_number")
    group = event.get('merge_group')
    if not group and (not str(number).isdigit() or not 0 < int(number) < 10**9):
        raise SystemExit("Invalid PR number.")
    token = os.environ["GH_TOKEN"]

    def api(path, payload=None, repo=None):
        request = urllib.request.Request(f"https://api.github.com/repos/{repo or repository}/{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "Content-Type": "application/json"},
            data=json.dumps(payload).encode() if payload is not None else None)
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("GitHub response exceeds the verification limit.")
        return json.loads(data)

    pr = None if group else api(f"pulls/{number}")
    head = group['head_sha'] if group else pr["head"]["sha"]
    valid_base = (group.get('base_ref') == 'refs/heads/main' and re.fullmatch(r'[a-f0-9]{40}',group.get('base_sha',''))) if group else pr['base']['ref']=='main'
    if not re.fullmatch(r"[a-f0-9]{40}", head) or not valid_base:
        raise SystemExit("Only PRs into main are eligible.")
    def status(state, description):
        api(f"statuses/{head}", {"state": state, "context": CONTEXT, "description": description[:140]})
    status("pending", "Checking template format and independent human approval.")
    try:
        head_repo = repository if group else pr["head"]["repo"]["full_name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", head_repo):
            raise ValueError("Invalid source repository.")
        trees = {}
        def tree(sha):
            if sha not in trees:
                value = api(f"git/trees/{sha}", repo=head_repo)
                if value.get("truncated"):
                    raise ValueError("Incomplete source tree.")
                trees[sha] = {entry["path"]:entry for entry in value["tree"]}
            return trees[sha]
        def source(path):
            # Read blobs at the immutable PR head; no downloads, symlinks,
            # submodules, code execution, shell interpolation or PR checkout.
            sha = head
            parts = path.split('/')
            for index, part in enumerate(parts):
                entry = tree(sha)[part]
                last = index == len(parts)-1
                if (entry.get("type") != ("blob" if last else "tree")
                        or entry.get("mode") != ("100644" if last else "040000")):
                    raise ValueError("Symlinks, submodules and executable files are refused.")
                sha = entry["sha"]
                if not re.fullmatch(r"[a-f0-9]{40}",sha):
                    raise ValueError("Invalid blob or tree identity.")
            item = api(f"git/blobs/{sha}", repo=head_repo)
            if item.get("sha") != sha or item.get("encoding") != "base64" or not 0 < item.get("size", 0) <= MAX_SOURCE_BYTES:
                raise ValueError("Invalid or oversized template file.")
            if len(item.get("content", "")) > 48000:
                raise ValueError("Oversized source response.")
            value = base64.b64decode(item["content"].replace("\n", ""), validate=True)
            if len(value) != item["size"]:
                raise ValueError("Source size mismatch.")
            return value
        ok, description = evaluate_merge_group(group, api, source) if group else evaluate(pr, pr_files(pr,api), api, source)
        if not group and api(f"pulls/{number}")["head"]["sha"] != head:
            raise ValueError("PR head changed during review verification.")
        status("success" if ok else "failure", description)
        print(description)
        return 0 if ok else 1
    except Exception:
        status("failure", "Backbone verification failed. Review the template and repository review policy.")
        # Do not print API/credential diagnostics or attacker-controlled source.
        print("Backbone verification failed; no source was executed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
