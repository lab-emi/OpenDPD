# OpenDPD leaderboard — reference benchmark

**Status: reference benchmark, not a community standard.** No external
submission has been accepted yet and nobody outside the maintainers has
recomputed an entry. The boards below hold the maintainers' own reference
entries, copied from the hash-bound weekly benchmark report
(`benchmark/regression/cpu-regression-dpa-200mhz/report.json`) and marked
`self_reported`. The label changes by itself once at least three external
submissions are accepted and two of them are independently recomputed
(`docs/community/governance.md` §9); it is never set by hand.

## Tracks and versions

| Track | Evidence | Board (current version) | Status |
|---|---|---|---|
| `pa_modeling` | PA behavioural model against the measured PA output | [v2026.09/pa_modeling.md](v2026.09/pa_modeling.md) | open; reference entries only |
| `dpd_surrogate` | DPD scored through the frozen PA surrogate of the same seed | [v2026.09/dpd_surrogate.md](v2026.09/dpd_surrogate.md) | open; reference entries only |
| `dpd_measured` | DPD scored on a capture of a physical PA (`docs/protocols/measured-dpd.md`) | [v2026.09/dpd_measured.md](v2026.09/dpd_measured.md) | open; **no entry**: no physical PA has been measured with the S16 path |
| `standard_evaluation` | data-aided EVM / ACLR on the reference waveform | — | closed until the S15 cross-validation record is filled (`docs/protocols/waveform-profiles.md` §7) |
| `robustness` | adaptation across conditions (`conditions-v1`) | — | closed until a measured condition set meets the S17 evidence bar (`docs/protocols/conditions-v1.md` §7) |
| `deployment` | fixed-point packages with verified targets | — | closed until the S19 fixed-point rules are approved (`docs/protocols/fixed-point-v1.md` §8) |

The three reference boards are `cpu_regression` tier: smoke budgets on the
built-in DPA 200 MHz data, regression references for that data and that
hardware, never research results. They exist so that the submission,
review, correction and versioning path is real before anybody uses it.

A version is a directory. Its boards are sealed by a hash that covers every
entry, review and history event; a new version is a new directory that
names the hash it supersedes. Nothing here is overwritten or deleted.

## How entries are shown

Entries rank only inside one comparability group (data id and raw hash,
operating point, metric profile and version, split protocol, execution
semantics, declared resource class). A board states its ordering metric;
every other metric, the mean ± standard deviation over the listed seeds,
the measured wall clock on the stated device, the submitter's failure
conditions, the evidence grade and the traceability (packages with hashes,
or the benchmark report and plan hashes, plus run and checkpoint ids) are
shown next to it and are not a tie-break. Non-public data is marked as such.

## How to submit

1. `opendpd leaderboard prepare <run>… --workspace WS --out DIR --id ID --submitter NAME`
2. fill in every `TODO` of `DIR/submission.json` (`docs/community/submission-template.md`)
3. `opendpd leaderboard check DIR/submission.json --recompute` until it passes
4. open an issue titled `leaderboard submission: <id>` on the repository and
   attach the directory (card and share packages; nothing else leaves your
   machine). A reviewer runs the same check, recomputes, and records the
   decision on the board under their name.

The tutorial `docs/tutorials/leaderboard-submission.md` walks through it; the
rules are `docs/community/governance.md`; how to cite a board version is
`docs/community/citation.md`.
