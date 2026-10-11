"""Publish an independent-review status on the immutable PR head.

Execute only trusted target-branch code. Never check out or execute PR code.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependency_github import GitHub
from review_policy import approvals, native_approvals, pages

CONTEXT = "independent-review"


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    number = (event.get("pull_request") or event.get("issue") or {}).get("number") or event.get("inputs", {}).get("pr_number")
    if not str(number).isdigit() or not 0 < int(number) < 10**9:
        raise SystemExit("Invalid PR number.")
    api = GitHub()
    pr = api.request(f"pulls/{number}")
    if pr["state"] != "open":
        return 0
    head = pr["head"]["sha"]
    def status(state, description):
        api.request(f"statuses/{head}", {"state": state, "context": CONTEXT, "description": description[:140]})
    status("pending", "Checking independent human or agent review of this commit.")
    try:
        ok = bool(approvals(pr, api.request))
        if not ok:
            # An objection overrides even the narrow dependency exception.
            _, blocked = native_approvals(pr, pages(api.request, f"pulls/{number}/reviews"),
                lambda login: api.request(f"collaborators/{login}/permission")["permission"])
            comments = pages(api.request, f"issues/{number}/comments")
            has_attestation = any("```opendpd-review" in (c.get("body") or "") for c in comments)
            ok = not blocked and not has_attestation and api.routine(pr)
        current = api.request(f"pulls/{number}")
        if current["head"]["sha"] != head or current.get("body") != pr.get("body"):
            raise ValueError("PR changed during review verification.")
        description = "Independent review passed for this commit." if ok else "Independent human or agent approval of this exact head is required."
        status("success" if ok else "failure", description)
        print(description)
        return 0 if ok else 1
    except Exception:
        status("failure", "Review verification failed; inspect the review metadata and rerun.")
        print("Review verification failed; no PR code was executed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
