# Submission template (`leaderboard-v1`)

`opendpd leaderboard prepare <run>… --workspace WS --out DIR --id ID --submitter NAME`
writes `DIR/submission.json` and one share package per run. The fields a
person must fill in are `TODO` in the draft; `opendpd leaderboard check`
refuses a card with any `TODO` left. A filled-in example is
`docs/community/submission-example.json`.

| Field | Who fills it | Meaning |
|---|---|---|
| `submission_id` | you | a slug that is yours, e.g. `mygroup-gru-2026-09` |
| `track` | tool | from the runs' task: `pa_modeling`, `dpd_surrogate` or `dpd_measured` |
| `submitter` | you | `name`, `kind` (`external` or `maintainer`), `affiliation` |
| `method.name` / `description` / `reference` / `code_url` / `licence` | you | the method card; licence as SPDX id, `proprietary` or `not_provided` |
| `model` | tool | model key, parameters, parameter count, look-ahead, execution semantics, training path — from the stored results |
| `data.dataset_id` / `raw_sha256` | tool | from the workspace |
| `data.availability` / `statement` | you | `public`, `on_request` or `private`, and how to obtain it. Non-public data is shown as such |
| `data.operating_point` | you (measured tracks) | PA, drive, power, chain, rate as declared in the measurement |
| `result` | tool | profile and version, split version, evidence type, one `SeedScore` per run (run id, configuration hash, checkpoint hash, every metric, wall clock), the aggregate (mean, std, min, max, n) and the resource budget |
| `result.resources.budget_class` | you, optional | the declared class you compete in (default `unbounded`), e.g. `params<=1000` |
| `packages` | tool | path (next to the card), sha256 and run id of every share package |
| `failure_conditions` | you | where the method fails or was not tried; an empty list is shown as "none stated" |
| `licence` | you | licences of code, weights, data; `redistribution_allowed`; a statement |
| `conflict_of_interest` | you | e.g. "none", or the relation to the maintainers or to entries on the board |
| `citation` | you | how you want to be cited |
| `isolated_validation` | you | where the packages were produced (never a machine with laboratory access, `docs/community/governance.md` §5) |

What the tool checks, what blocks and what is only shown is in
`docs/community/governance.md` §4. Send the directory (card and packages)
to the maintainers as stated in the leaderboard README; the review and its
recomputation are recorded on the board under the reviewer's name.
