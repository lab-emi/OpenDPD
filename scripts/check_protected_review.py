"""Require a maintainer review tied to the current PR head for protected changes."""
import json
import os
import urllib.request


def main():
    event = json.load(open(os.environ['GITHUB_EVENT_PATH']))
    pr = event['pull_request']
    repository = os.environ['GITHUB_REPOSITORY']
    token = os.environ['GH_TOKEN']

    def get(path):
        request = urllib.request.Request(f'https://api.github.com/repos/{repository}/{path}',
            headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    # Read fresh state: reviews and new commits may arrive after this job was queued.
    current = get(f'pulls/{pr["number"]}')
    head = current['head']['sha']
    if head != pr['head']['sha']:
        raise SystemExit('The PR head changed; wait for the check on its new commit.')
    latest = {}
    page = 1
    while True:
        reviews = get(f'pulls/{pr["number"]}/reviews?per_page=100&page={page}')
        for review in reviews:
            if review['commit_id'] == head:
                latest[review['user']['login']] = review
        if len(reviews) < 100:
            break
        page += 1
    for login, review in latest.items():
        approved = review['state'] == 'APPROVED'
        signed_comment = review['state'] == 'COMMENTED' and f'Reviewed head: {head}' in (review['body'] or '')
        if approved or signed_comment:
            permission = get(f'collaborators/{login}/permission')['permission']
            if permission in {'admin', 'maintain'}:
                print(f'Maintainer review recorded for {head}')
                return
    raise SystemExit('Protected changes require an admin/maintainer review of the current head; labels are not approvals.')


if __name__ == '__main__':
    main()
