"""Review security boundaries and final commit attribution, without GitHub writes."""
import json

import pytest

from scripts.review_policy import block, contributors, independent_approvals, merge_message, pages

HEAD = 'a' * 40
OTHER = 'b' * 40


def fenced(kind, value):
    return f'```{kind}\n{json.dumps(value)}\n```'


def pr():
    return {'number': 81, 'head': {'sha': HEAD}, 'user': {'login': 'owner'},
            'body': fenced('opendpd-contributors', {'contributors': [
                {'name': 'Codex <noreply@openai.com>', 'session': 'codex:development'}]})}


def attestation(**changes):
    data = {'head': HEAD, 'decision': 'approve', 'reviewer': 'Claude Code <noreply@anthropic.com>',
            'session': 'claude:review', 'independent': True, 'summary': 'Inspected diff and ran relevant tests.'}
    data.update(changes)
    return {'id': 1, 'user': {'login': 'owner', 'type': 'User'},
            'html_url': 'https://github.com/lab-emi/OpenDPD/pull/81#issuecomment-1',
            'body': fenced('opendpd-review', data)}


def decide(comments, reviews=(), permission=lambda _: 'admin', pull=None):
    return independent_approvals(pull or pr(), reviews, comments, permission)


def test_shared_account_cross_agent_review_and_attribution():
    accepted = decide([attestation()])
    assert len(accepted) == 1
    body = merge_message(pr(), accepted)
    assert 'Contributed-By: Codex <noreply@openai.com>' in body
    assert 'Reviewed-By: Claude Code <noreply@anthropic.com>' in body
    assert f'Reviewed-Commit: {HEAD}' in body
    assert 'Review-URL: https://github.com/' in body
    assert 'Review-Session: claude:review' in body


@pytest.mark.parametrize('changes', [
    {'session': 'codex:development'}, {'independent': False}, {'independent': 'true'},
    {'head': 'short'}, {'decision': 'APPROVED'}, {'summary': ''}, {'reviewer': 'name\nReviewed-By: fake'},
    {'reviewer': 'Codex <noreply@openai.com>'},
])
def test_invalid_or_self_review_fails_closed(changes):
    with pytest.raises(ValueError):
        decide([attestation(**changes)])


def test_same_family_separate_session_needs_reason():
    assert decide([attestation(reviewer='Codex <noreply@openai.com>', session='codex:independent',
                               independence_reason='Separate session reviewed diff without implementing it.')])


def test_stale_unprivileged_and_plain_comments_do_not_approve():
    assert not decide([attestation(head=OTHER)])
    assert not decide([attestation()], permission=lambda _: 'write')
    assert not decide([{**attestation(), 'body': f'Reviewed head: {HEAD} approved'}])
    with pytest.raises(ValueError):
        decide([attestation()], pull={**pr(), 'body': ''})


def test_change_request_on_old_head_vetoes_other_approval_until_withdrawn():
    request = attestation(head=OTHER, decision='request_changes', session='claude:other-review')
    assert not decide([request, {**attestation(), 'id': 2}])
    withdrawn = {**attestation(decision='dismiss', session='claude:other-review'), 'id': 3}
    assert decide([request, attestation(), withdrawn])


def test_edit_or_delete_approval_revokes_it():
    assert not decide([{**attestation(decision='dismiss'), 'updated_at': '2026-10-11T14:00:00Z'}])
    assert not decide([])
    old = {**attestation(), 'updated_at': '2026-10-11T12:00:00Z'}
    newer = {**attestation(decision='request_changes'), 'id': 2, 'updated_at': '2026-10-11T13:00:00Z'}
    assert not decide([newer, old])


def native(**changes):
    return {'id': 1, 'user': {'login': 'review-agent[bot]', 'type': 'Bot'},
            'state': 'APPROVED', 'commit_id': HEAD, **changes}


def test_native_agent_approval_and_latest_decision():
    assert decide([], [native()])
    assert not decide([], [native(user={'login': 'owner'})])
    assert not decide([], [native(commit_id=OTHER)])
    assert not decide([], [native(), native(id=2, state='DISMISSED')])
    assert decide([], [native(), native(id=2, state='COMMENTED')])
    assert not decide([attestation()], [native(state='CHANGES_REQUESTED', commit_id=OTHER)])


def test_metadata_rejects_duplicate_blocks_and_bad_contributors():
    with pytest.raises(ValueError):
        block(attestation()['body'] + '\n' + attestation()['body'], 'opendpd-review')
    with pytest.raises(ValueError):
        contributors({**pr(), 'body': fenced('opendpd-contributors', {'contributors': []})})
    with pytest.raises(ValueError):
        merge_message(pr(), [])


def test_bounded_pagination_includes_reviews_after_first_page():
    calls = []
    def get(path):
        calls.append(path)
        return [None] * 100 if path.endswith('page=1') else ['last']
    assert len(pages(get, 'pulls/81/reviews')) == 101
    assert len(calls) == 2
    with pytest.raises(ValueError):
        pages(lambda _: [None] * 100, 'pulls/81/reviews')


def test_gate_publishes_on_live_head_and_fails_when_head_changes(tmp_path, monkeypatch):
    from scripts import check_protected_review as gate
    event = tmp_path / 'event.json'
    event.write_text(json.dumps({'pull_request': {'number': 81, 'head': {'sha': OTHER}}}))
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(event))
    class API:
        def __init__(self):
            self.statuses = []
            self.changed = False
            self.reads = 0
        def request(self, path, payload=None):
            if path.startswith('statuses/'):
                self.statuses.append((path, payload)); return {}
            if path == 'pulls/81':
                self.reads += 1
                return {**pr(), 'state': 'open', 'head': {'sha': OTHER if self.changed and self.reads > 1 else HEAD}}
            raise AssertionError(path)
    api = API()
    monkeypatch.setattr(gate, 'GitHub', lambda: api)
    monkeypatch.setattr(gate, 'approvals', lambda *_: [attestation()])
    assert gate.main() == 0
    assert [s['state'] for _, s in api.statuses] == ['pending', 'success']
    assert all(path == f'statuses/{HEAD}' for path, _ in api.statuses)
    api.changed, api.reads = True, 0
    assert gate.main() == 1
    assert api.statuses[-1][1]['state'] == 'failure'
