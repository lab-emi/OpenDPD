# Governance of the OpenDPD benchmark and leaderboard (`leaderboard-v1`)

This document says how results get onto a board, who decides, and what
happens when a result is wrong. It binds maintainers and external submitters
alike. The machine-readable side is `opendpd/schemas/leaderboard.py`; the
tooling is `opendpd leaderboard …` (`docs/tutorials/leaderboard-submission.md`).

## 1. Roles

- **Submitter**: the person or group who answers for a submission. A
  maintainer who submits their own method is a submitter for that entry and
  does not review it.
- **Reviewer**: runs the checklist below and records a decision under their
  name. Maintainers review external submissions; the plan requires at least
  one **external researcher** (not a core maintainer) among the reviewers of
  the protocol, and this requirement is open until such a person is named
  (`docs/releases/studio-progress.md`, S20).
- **Maintainers**: keep the boards, publish versions, and decide protocol
  changes under the rules of `AGENTS.md` §3 (a protected-path change is a
  separate, science-reviewed change that never relaxes a threshold or edits an
  expected value).

## 2. What a board is

One board is one **track**: `pa_modeling`, `dpd_surrogate` and
`dpd_measured` are open; `standard_evaluation`, `robustness` and
`deployment` open only when the stage they depend on passes its human gate
(`TRACK_GATES` in the schema names the record to fill). Evidence types are
never mixed on one board: a DPD scored through a surrogate is not compared
with a measured DPD.

Inside a board, entries **rank only within one comparability group**: same
data (id and raw hash), operating point, metric profile and version, split
protocol, execution semantics and declared resource class. Other groups are
shown apart, never against each other. The ordering metric of a track is
stated on the board; the other metrics, the uncertainty (mean ± sample
standard deviation over the listed seeds, with `n`), the measured wall clock
on the stated device, the failure conditions and the evidence grade are
shown next to it and are not a tie-break. A single ACLR number never is the
board.

A board carries a **version** (`v2026.09`). A version is a file under
`docs/leaderboard/<version>/`; the protocol and the test data of that
version are fixed by the hashes inside it. A new version is a new directory
that names the board hash it supersedes; an old version is never overwritten
and never deleted.

## 3. Submissions

A submission is a card (`docs/community/submission-template.md`) plus one
**share package per seed** (`opendpd export … --kind share`: no user data, no
machine paths). `opendpd leaderboard prepare` writes both from finished runs;
the statements only a person can make are left `TODO` and block the check
until they are filled in:

- method card (name, description, reference, code, licence);
- data card: id, raw hash, availability (`public`, `on_request`, `private`)
  and how to obtain it. Data that is not public is shown as such and is
  **never presented as publicly reproducible**;
- licence check for code, weights and data, and whether redistribution is
  allowed. Without it the board links the method card and does not host the
  weights;
- conflict-of-interest statement; citation; where the packages were
  produced (**isolated validation**, §5);
- failure conditions: where the method fails or was not tried.

Every external publication of a dataset or a model by the project itself
needs its own approval (plan §10); a submission is the submitter's own
publication under the submitter's own licence.

## 4. Review checklist

`opendpd leaderboard check` runs the same list for every submitter; the
reviewer records it, with their decision, under their name:

| Item | Blocks? | What it checks |
|---|---|---|
| statements | yes | no `TODO` left in the method, data, licence, conflict, citation and isolation statements |
| packages | yes | every package exists, its hash matches the card, it is a **share** package, its run id, configuration hash, dataset and profile match the card's seed |
| duplicate | yes | the submission id and the package hashes are not on the board already |
| recomputation | yes, when run | every package imported into a **fresh workspace** and re-scored from its checkpoint under the card's profile; every metric within `1e-6` dB of the card |
| seeds | no | fewer than three seeds is shown as "uncertainty not established" |
| data | no | non-public data is marked; public data without a raw hash cannot be matched |
| licence | no | no redistribution means the board links, it does not host |
| isolation | no | recorded; an organisational rule the software cannot verify, the reviewer confirms it |

Decisions: `accepted`, `rejected`, `needs_changes`. An acceptance never
rests on a failed recomputation (the tool refuses it).

**Evidence grades** on the board: `self_reported` (the submitter's numbers,
also the maintainers' reference entries copied from a hash-bound
benchmark-v1 report), `reviewed` (checklist passed, decision recorded),
`independently_recomputed` (a reviewer who is not the submitter reproduced
every number from the packages).

## 5. Isolated validation

Packages of a submission are produced, and recomputed by reviewers, on a
machine that is **not** connected to laboratory data stores, credentials or
instrument networks, and never on the laboratory's self-hosted CI runners.
The share package is the only thing that moves. This restates the policy of
`docs/protocols/benchmark-protocol.md` ("Leaderboard submissions"); it is
recorded on every card and confirmed by the reviewer, not verified by
software.

## 6. Corrections and retractions

An accepted entry is corrected or retracted by its submitter or by a
maintainer with a written reason (`opendpd leaderboard amend`). The entry
stays on the board with its new status; the numbers that were shown before
are kept in the entry's history, on the board and in its Markdown rendering.
Nothing is erased. A retracted entry is not ranked and cannot be corrected
back; a new submission is the way forward.

## 7. Protocol changes

Metric definitions, splits, evaluation ranges, normalisation, checkpoint
selection and seed sets are protected (`AGENTS.md` §3, plan §10). A change
is a new profile or protocol version with a science review; it never
rewrites the numbers of an existing board version. Results under different
protocol versions are different comparability groups.

## 8. Conflicts of interest and appeals

A reviewer does not review their own group's submission or a direct
competitor's under a declared conflict; the conflict statement on the card
and the reviewer's name in the record make this checkable. A submitter who
disagrees with a decision writes to the maintainers with the entry id; a
second reviewer who was not involved records a decision within the same
rules, and both decisions stay in the history. Maintainers' own methods go
through the same checklist, the same grades and the same appeal path.

## 9. The label

Until at least **three** complete external submissions are accepted and at
least **two** of them are independently recomputed by someone who is not
their author, every board calls itself a **reference benchmark** and the
project does not call it a community standard. The tool computes the label
from the entries; it cannot be set by hand.
