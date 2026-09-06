# Leaderboard submission and review (`leaderboard-v1`)

This tutorial goes through both sides of the board: a submitter drafting and
checking a submission, and a maintainer seeding a board version, adding the
entry, reviewing it with a recomputation, and amending it. The rules are in
`docs/community/governance.md`; the boards are files under
`docs/leaderboard/`.

Every command below is executed by `tests/integration/test_docs_commands.py`.

## 1. Draft a submission (submitter)

Train the runs you want to submit, one per seed, of one method on one
dataset (the headless tutorial: `pa-gru-smoke-v1` on `dpa-200mhz`). Then:

```bash
opendpd leaderboard prepare <run-id-seed-0> <run-id-seed-1> <run-id-seed-2> --workspace ws --out submission --id mygroup-gru-2026-09 --submitter "My Group"
```

This writes `submission/submission.json` and one share package per run
(`opendpd export --kind share`: no user data, no machine paths). The card
already holds what the workspace knows (model, data hashes, every metric per
seed, the aggregate, wall clock); the statements only you can make are
`TODO`: method description and licence, data availability, licence check,
conflict of interest, citation, where the packages were produced.

## 2. Check it (submitter, then reviewer)

```bash
opendpd leaderboard check submission/submission.json
```

exits 1 while a `TODO` is left. Fill them in
(`docs/community/submission-template.md`), then run the full check with a
recomputation: every package is imported into a fresh, temporary workspace
and re-scored from its checkpoint under the card's profile; every metric
must reproduce within `1e-6` dB:

```bash
opendpd leaderboard check submission/submission.json --recompute --by "My Group" --json
```

`seeds` warns below three seeds (the entry is shown, its uncertainty is
marked as not established); `data` warns when the data is not public;
`licence` warns when redistribution is not allowed. Warnings are shown on
the board, failures block. Send the directory to the maintainers as the
leaderboard README says.

## 3. Seed a board version (maintainer)

A version starts from a hash-bound benchmark-v1 report; its entries are the
maintainers' reference results, marked `self_reported`:

```bash
opendpd leaderboard seed benchmark/regression/cpu-regression-dpa-200mhz/report.json --track pa_modeling --board-id opendpd-pa-modeling --version v2026.09 --out docs/leaderboard/v2026.09/pa_modeling.json
```

The JSON is sealed; the Markdown rendering is written next to it. A new
version is a new directory (`--supersedes` names the previous board hash);
an old one is never overwritten.

## 4. Add, review, amend (maintainer)

```bash
opendpd leaderboard add docs/leaderboard/v2026.09/pa_modeling.json submission/submission.json
```

runs the checklist against the board (duplicates are refused) and adds the
entry as `submitted`. A reviewer who is not the submitter records the
decision; with `--recompute` the packages (found next to the board, or under
`--packages`) are reproduced again and an accepted entry becomes
`independently_recomputed`:

```bash
opendpd leaderboard review docs/leaderboard/v2026.09/pa_modeling.json mygroup-gru-2026-09 --reviewer "A Reviewer" --kind external --decision accepted --notes "checklist passed; recomputed on my machine" --recompute --packages ../../../submission
```

An accepted entry is corrected (`--action correct --card corrected.json`)
or retracted with a reason; the numbers shown before stay in the history:

```bash
opendpd leaderboard amend docs/leaderboard/v2026.09/pa_modeling.json mygroup-gru-2026-09 --action retract --by "My Group" --reason "wrong dataset version"
```

The board's first line says whether it is a reference benchmark or a
community leaderboard; the tool computes that from the accepted external
entries and the independent recomputations, never from a flag.
