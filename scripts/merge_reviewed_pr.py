"""Reviewer-operated merge with fresh approval, full CI and attribution trailers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependency_github import GitHub
from check_user_backbone_review import validate_changed_templates
from maintain_dependencies import require_strict_checks, validation
from review_policy import approvals, merge_message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int)
    parser.add_argument("--repo", default="lab-emi/OpenDPD")
    parser.add_argument("--head", required=True, help="Reviewed full commit SHA; refuse a newer head")
    parser.add_argument("--merge", action="store_true", help="Perform the merge; otherwise print the validated commit message")
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    api = GitHub(args.repo, token)
    pr = api.request(f"pulls/{args.number}")
    if pr["state"] != "open" or pr.get("draft") or pr["base"]["ref"] != "main" or pr["head"]["sha"] != args.head:
        raise SystemExit("Expected a ready, open PR into main at the reviewed head.")
    require_strict_checks(api)
    state, detail = validation(api, args.head, head_repository=pr["head"]["repo"]["full_name"])
    if state != "passed":
        raise SystemExit(detail)
    if api.comparison(pr).get("behind_by") != 0:
        raise SystemExit("Update the PR against main, rerun CI and review the new head.")
    # Recheck template bytes directly even during policy bootstrap, before the
    # new required statuses exist. Read fork blobs as data; never execute them.
    source_api = GitHub(pr["head"]["repo"]["full_name"], token)
    validate_changed_templates(api.files(pr), lambda path: source_api.source(args.head, path).encode("utf-8"))
    body = merge_message(pr, approvals(pr, api.request))
    current = api.request(f"pulls/{args.number}")
    if current["head"]["sha"] != args.head or current.get("body") != pr.get("body"):
        raise SystemExit("PR changed during merge preparation.")
    print(body)
    if args.merge:
        result = api.request(f"pulls/{args.number}/merge", {
            "sha": args.head, "merge_method": "squash",
            "commit_title": f"{pr['title']} (#{args.number})", "commit_message": body,
        }, method="PUT")
        if not result.get("merged"):
            raise SystemExit("GitHub refused the merge.")
        print(f"Merged: {result['sha']}")


if __name__ == "__main__":
    main()
