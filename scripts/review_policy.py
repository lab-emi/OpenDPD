"""Independent human/agent review, read as data from the GitHub API.

GitHub authenticates accounts, not the agent using them. Shared-account agent
reviews are explicit maintainer attestations, never inferred from prose/labels.
"""
from __future__ import annotations

import json
import re
import urllib.error


def pages(get, path):
    values = []
    for page in range(1, 21):
        batch = get(f"{path}?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise ValueError("Invalid review history.")
        values.extend(batch)
        if len(batch) < 100:
            return values
    raise ValueError("Review history exceeds the verification limit.")


def block(body, kind):
    matches = re.findall(r"^```" + kind + r"[^\S\n]*\n(.*?)\n```[^\S\n]*$", body or "", re.M | re.S)
    if not matches:
        return None
    if len(matches) != 1 or len(matches[0]) > 10000:
        raise ValueError("Expected one bounded review metadata block.")
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise ValueError("Review metadata must be an object.")
    return value


def line(value):
    return isinstance(value, str) and 0 < len(value.strip()) <= 500 and not any(ord(c) < 32 or ord(c) == 127 for c in value)


def contributors(pr):
    data = block(pr.get("body"), "opendpd-contributors")
    if data is None:
        return []
    entries = data.get("contributors")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 20:
        raise ValueError("Declare the contributors and their development sessions.")
    if any(not isinstance(e, dict) or not line(e.get("name")) or not line(e.get("session")) for e in entries):
        raise ValueError("Invalid contributor identity or session.")
    return entries


def native_approvals(pr, reviews, permission):
    latest = {}
    for review in sorted(reviews, key=lambda r: r.get("id", 0)):
        if review.get("state") in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest[(review.get("user") or {}).get("login")] = review
    accepted = []
    for login, review in latest.items():
        if not login or login == pr["user"]["login"] or permission(login) not in {"admin", "maintain"}:
            continue
        if review["state"] == "CHANGES_REQUESTED":
            return [], True
        if review["state"] == "APPROVED" and review.get("commit_id") == pr["head"]["sha"]:
            accepted.append({"reviewer": login, "url": review.get("html_url", ""), "session": f"github-review:{review['id']}"})
    return accepted, False


def independent_approvals(pr, reviews, comments, permission):
    """Return approvals; outstanding changes requests always veto a merge."""
    accepted, blocked = native_approvals(pr, reviews, permission)
    if blocked:
        return []
    authors = contributors(pr)
    latest = {}
    for comment in sorted(comments, key=lambda c: (c.get("updated_at", ""), c.get("id", 0))):
        if not has_review_record(comment.get("body")):
            continue
        login = (comment.get("user") or {}).get("login")
        if not login or permission(login) not in {"admin", "maintain"}:
            continue
        data = block(comment.get("body"), "opendpd-review")
        if data is None:
            raise ValueError("Malformed review attestation.")
        if (not re.fullmatch(r"[a-f0-9]{40}", str(data.get("head", "")))
                or data.get("decision") not in {"approve", "request_changes", "dismiss"}
                or not all(line(data.get(k)) for k in ("reviewer", "session", "summary"))
                or data.get("independent") is not True):
            raise ValueError("Invalid independent review attestation.")
        if not authors or data["session"] in {a["session"] for a in authors}:
            raise ValueError("A review needs declared contributors and a separate review session.")
        if data["reviewer"] in {a["name"] for a in authors} and not line(data.get("independence_reason")):
            raise ValueError("Same-agent-family review needs an independence explanation.")
        latest[(login, data["reviewer"], data["session"])] = {**data, "url": comment.get("html_url", "")}
    for data in latest.values():
        if data["decision"] == "request_changes":
            return []
        if data["decision"] == "approve" and data["head"] == pr["head"]["sha"]:
            accepted.append(data)
    return accepted


def permission_reader(get):
    permissions = {}
    def permission(login):
        if not re.fullmatch(r"[A-Za-z0-9_\[\]-]+", login):
            return None
        if login not in permissions:
            try:
                details = get(f"collaborators/{login}/permission")
                # GitHub maps maintain to legacy permission=write; role_name
                # preserves the actual role. Plain write must not gain approval.
                permissions[login] = ("admin" if details.get("permission") == "admin"
                                      else details.get("role_name") or details.get("permission"))
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                # A GitHub App need not be a repository collaborator. Its
                # review cannot authorize a merge or invalidate another approval.
                permissions[login] = None
        return permissions[login]
    return permission


def has_review_record(body):
    return bool(re.search(r"^```opendpd-review[^\S\n]*(?:\n|$)", body or "", re.M))


def authorized_attestations(comments, permission):
    return [comment for comment in comments if has_review_record(comment.get("body"))
            and permission((comment.get("user") or {}).get("login", "")) in {"admin", "maintain"}]


def approvals(pr, get):
    return independent_approvals(
        pr, pages(get, f"pulls/{pr['number']}/reviews"),
        pages(get, f"issues/{pr['number']}/comments"), permission_reader(get))


def merge_message(pr, accepted):
    if not accepted:
        raise ValueError("Independent approval of the current head is required.")
    names = [entry["name"] for entry in contributors(pr)] or [pr["user"]["login"]]
    lines = [f"Merge independently reviewed PR #{pr['number']}.", ""]
    lines.extend(f"Contributed-By: {name}" for name in dict.fromkeys(names))
    lines.extend(f"Reviewed-By: {r['reviewer']}" for r in accepted)
    lines.append(f"Reviewed-Commit: {pr['head']['sha']}")
    lines.extend(f"Review-URL: {r['url']}" for r in accepted if r.get("url"))
    lines.extend(f"Review-Session: {r['session']}" for r in accepted)
    return "\n".join(lines)
