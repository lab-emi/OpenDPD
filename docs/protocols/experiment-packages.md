# Reproducible experiment packages

Status: implemented (S11). Package format `package_version: 1`
(`opendpd/schemas/package.py`); the exporter, verifier and importer live in
`opendpd/services/packages.py`, the reports in `opendpd/services/reports.py`.

## What a package is

A zip written by `opendpd export RUN --workspace WS [--kind full|share]`,
`POST /exports {run_id, kind}` or the **Export** buttons on a result page. It
carries one run and everything needed to state what its numbers mean:

| Member | Content |
|---|---|
| `package.json` | the manifest: kind, software provenance, run id, task, resolved configuration hash, seed, result id and metric profile, dataset identity (id, raw sha256, data version, split version, source kind), referenced runs with their checkpoint hashes, every member with its sha256 and size, named reproduction commands, what was left out (`redaction`), what is still needed (`missing`), the retraining note |
| `run/<id>/…` | the run directory: `config.user.json`, `config.resolved.json`, `provenance.json`, `run.json`, `artifacts.json`, `result.json`, `results/<profile>.json`, `plots/*.json`, `save/` (checkpoints), `dpd_out/` |
| `refs/<id>/…` | every referenced run (PA surrogate of a DPD run, DPD model of a `run_dpd` run): its configurations, provenance, results, metric logs and checkpoint; its `artifacts.json` lists exactly these (worker logs and plot data stay behind) |
| `dataset/manifest.json` | the dataset manifest (versions, split, hashes) |
| `dataset/raw`, `dataset/versions/<v>` | the raw copy and the used data version — **full packages of non-built-in datasets only** |
| `report.html`, `report.md` | the reports (below) |

Everything a package says about a result is copied from the run directory.
Nothing is recomputed on export.

## Full versus share

| | `full` (private) | `share` (redacted) |
|---|---|---|
| Purpose | move an experiment to another machine of the same owner | give a collaborator the evidence without the private parts |
| Run directory | complete | without `logs/` (worker logs) and `events.jsonl`; `artifacts.json` is rewritten without the entries whose files stay behind, so an imported run never lists a file it does not have |
| `provenance.json`, `run.json` | verbatim | machine paths replaced by `<workspace>` / `<home>` / `<host>`; the worker identity (host, pid) removed |
| Dataset manifest | verbatim | the original import path removed |
| User PA data | raw copy and the used version included | **never included** |
| Built-in dataset | not copied (it ships with every install); the importer registers it and checks the raw sha256 | same |

Every redaction is listed in `manifest.redaction`; every prerequisite the
package does not carry is listed in `manifest.missing` with how to obtain it
(the dataset id, its raw sha256, and the import command). Nothing is sent
anywhere: a package is a file in `<workspace>/exports/` until the user moves
it.

## Import

`opendpd import FILE --workspace WS [--inspect]`, `POST /imports`
(multipart, 2 GB cap, streamed into `<workspace>/imports/packages/`) or
**Import package…** on the Experiments page.

1. **Verify first, write nothing.** The manifest is parsed and its version
   checked; every archive member must be listed in the manifest with a
   matching sha256 and size, every listed member must exist, no member may
   escape the workspace. A package failing any check is refused with a
   specific code: `not_a_package`, `manifest_missing`, `manifest_invalid`,
   `unsupported_version`, `unlisted_file`, `missing_file`, `hash_mismatch`,
   `unsafe_path` (`test_damaged_packages_are_refused_with_a_specific_diagnostic`).
2. **Conflicts are refused, not merged.** A run id that already exists
   (`run_exists`), a referenced run that exists with a different checkpoint
   hash (`reference_conflict`), or a dataset that exists with a different raw
   sha256 (`dataset_conflict`) stops the import before anything is written.
3. **Dataset status** in the import report: `imported` (data travelled with
   the package), `existing` (same id and raw hash already in the workspace),
   `registered_builtin` (the built-in example was registered and its hash
   checked) or `missing` (a share package of user data: the run and its
   stored results are imported, the report lists what to ask the author for).
4. Referenced runs are imported under `runs/` first, then the run itself; the
   service indexes them on the next listing.

A checksum verifies that data someone provides later is the same data; it
cannot stand in for the data. An imported share package without its dataset
keeps the stored results, and `opendpd evaluate` says the data is missing.

## What "reproduced" means

| Claim | How it is checked | Tolerance |
|---|---|---|
| Re-evaluating the packaged checkpoint in a new workspace reproduces the stored metrics | `test_full_package_round_trips_into_a_new_workspace_and_re_evaluates`: import a full package, run `evaluate_run` under the stored profile, compare every metric with the packaged result | rel 1e-4 / abs 1e-5 (the frozen-checkpoint tolerance in `acceptance-thresholds.md`) |
| A share package re-evaluates once the author's data is imported under the same id with the same hash | `test_share_package_re_evaluates_once_the_data_is_imported_with_the_same_hash` | same |
| Re-training from `config.user.json` gives the same numbers | **not implied by any of the above.** `manifest.retraining_note` says so; retraining is a new experiment whose agreement depends on the reproducibility mode, device and software versions and is reported by the S12 regression protocol | n/a |

## Reports

`opendpd report RUN --workspace WS [--format html|md]`,
`GET /results/{id}/report?format=html|md`, or the **Report** links on a result
page. A report is bound to its sources: every number is copied from the stored
result (`result.json`), the metric columns of the no-DPD baselines come from
the same result, the spectrum image in the HTML report is drawn from the
stored `plots/spectrum.json` (plots-v1), and the resolved configuration,
provenance, lineage and artifact hashes are quoted verbatim. Nothing is
recomputed in a report, and the frontend never recomputes a metric either
(charts are drawn from separate, decimated plot data — see
`metric-profiles.md`). `test_reports_are_bound_to_the_stored_result` checks
the numbers in both formats against `result.json`.
