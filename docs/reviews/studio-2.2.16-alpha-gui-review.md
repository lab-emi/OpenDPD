# OpenDPD Studio 2.2.16-alpha GUI review

Reviewed locally on 18–19 September 2026. Changes stay on `2.2.16-alpha`; nothing was pushed or published.

## Method

Used the running Studio GUI with Chromium attached to the installed local browser service. Clicked controls, entered data, downloaded files, and inspected rendered screenshots against the real backend. The review used an isolated workspace, leaving the desktop user's saved datasets and runs intact. Screen sizes included 1440 × 1000, 1366 × 900, and 390 × 844, with light/dark appearance and English/Chinese checks.

Local evidence (screenshots, downloads, test logs, and review workspace) is under `/tmp/opendpd-ui-review-2216/`. These temporary files are not part of a release.

## Changes from observed problems

- Navigation returns to the top of a newly opened page. Previously, cached navigation could open a page with its heading above the viewport.
- Dataset pages give **Train PA & DPD Models** clear priority, wrap long names, and put secondary tools in their own row. Dataset provenance and sharing remain accessible in a details section. Dataset lists show the total samples and sample-rate range of a collection.
- PA Library and Analyzer use searchable, named dataset selectors. Returning to a multi-signal input collection restores all its members. Removing a collection is reversible and preserves shared signal files.
- Opening a dataset, saved run, or result restores the relevant workflow context. Re-running a DPD configuration keeps the correct DPD heading, dataset, and data version.
- Analyzer CSV uploads show the original filename. Result headings use the saved experiment name. Result task labels use the same readable names as the experiment list.
- Generator plots use the theme's light/dark colors, with larger legends. Analyzer headings and downloads wrap cleanly. On narrow screens, summary metrics use readable rows instead of cramped columns. Dataset time-domain, I/Q and AM/AM plots retain enough height and legend space on laptop screens. Duplicate arrow text was removed from the Virtual PA action.
- Result AM/AM and AM/PM legends distinguish synthetic output, measured output, and PA-model predictions. Utility pages omit the unrelated workflow strip. Broken Plotly cloud-sharing controls were removed; PNG download, zoom, pan, and enlargement remain available.
- Multi-result comparison now displays the input traces by default. Previously both input panels were empty. Legend space is shared only between charts on the same row.
- The current `opendpd-spectral-v2` profile now supplies display integration bands matching its actual carrier-width definition. This changes the shading, not the metric calculations.
- Short DPD captures are rejected before queueing when an evaluated split cannot fill one PSD segment. The real GUI previously accepted a dataset with 3,174 validation samples against a 4,096-sample segment, then failed inside training. Validation now explains the required sample count and corrective options.

## GUI coverage

| Area | Actions exercised | Outcome |
| --- | --- | --- |
| Home and navigation | Dataset guide, Generate/Preview navigation, cached-page scroll, workflow expansion, reset page and Reset Studio | Worked; saved datasets and results survive resets |
| Signal Generator | NR and Wi-Fi batch generation, dataset naming, signal selection, all preview tabs, enlarged plots, dataset ZIP, signal CSV and metadata downloads | Worked |
| Signal Analyzer | Dataset/member selection, CSV upload, physical rate/bandwidth entry, analysis, spectrum/spectrogram, time/frequency, I/Q/statistics, eye and definitions tabs, report controls | Worked; mobile layout refined |
| Virtual PA Library | All nine model panels, parameter editing, named collection selection, Rapp AM/AM + AM/PM simulation, paired output, remove and Undo | Worked; simulation itself used the Rapp model |
| Datasets | Built-in APA import, paired CSV creation, inspection plots, source/capture selection, metadata, Doctor and preprocessing preview/new version | Worked; missing metadata correctly blocks formal training |
| Training and testing | TRes-GRU PA and DPD setup, invalid-field handling, live charts, logs, artifacts, configuration, saved-run re-run, PA testing and DPD application | Completed real smoke runs; short-capture failure fixed at preflight |
| ILC | Trained PA selection, two-iteration waveform/ILA fit with 8,192 fitting samples, resulting dashboard | Succeeded |
| Run control | Start, cancellation, terminal state, reset from a run, lineage inspection | Cancellation confirmed and no review jobs left running |
| Results | Named result, metric/evidence display, same-condition comparison, saved view, publication preview/save, figure ZIP and HTML report download, local share-package export | Worked |
| Package import | Re-import an exported run already present in the same workspace | Correctly refused the duplicate without overwriting it |
| Sweep Board | Copy a real PA configuration, one seed/one epoch matrix, preview, register, start and completed seed summary | Succeeded |
| Other screens | Robustness empty state, hardware cost plot/cards and report dialog, measured-capture dialog, dataset sharing controls, bug-report dialog, Settings, Server load and About | Inspected; actions requiring real external evidence or public submission were not fabricated/submitted |

## Executed review runs

| Run | Task | Result |
| --- | --- | --- |
| `run-20260918-205737-982bf5` | Synthetic TRes-GRU PA, 5 epochs | Succeeded |
| `run-20260918-210205-57fc67` | Short synthetic DPD | Exposed the split-length failure; retained as evidence, then confirmed preflight blocks the configuration |
| `run-20260918-212439-f7f4d2` | Measured APA TRes-GRU PA, 3 epochs | Succeeded |
| `run-20260918-213335-11ff4a` | Measured APA TRes-GRU DPD, 3 epochs | Succeeded |
| `run-20260918-213643-897763` | Apply DPD to the APA test split | Succeeded |
| `run-20260918-215011-1c9c0d` | One-epoch PA sweep cell | Succeeded |
| `run-20260918-215325-2cbffe` | APA PA-model testing | Succeeded |
| `run-20260918-215503-a80997` | ILC smoke run | Succeeded |
| `run-20260918-215542-d7b83f` | Cancellation exercise | Cancelled |

These are GUI/workflow smoke runs. Their metrics do not establish converged DPD performance, a −55 dBc benchmark, physical-PA linearization, or robustness across measured operating conditions.

## Verification

- Full frontend suite: **276 tests passed**. After the final spacing/copy adjustments, **19 affected frontend tests** and **6 dataset-page tests** passed again.
- Targeted Python suites for dataset collections, archive/restore, short-capture preflight, Analyzer, web boundaries, review bands, and figure flow passed. Review/band tests include 20, 160, and 200 MHz cases and out-of-capture/missing-metadata handling.
- Frontend lint, TypeScript production build, generated API types, committed OpenAPI equality, and whitespace checks passed.
- Axe WCAG A/AA checks on the tested Generate, Preview, Dataset detail and mobile Analyzer states found no violations. Screenshots were also visually inspected; automated accessibility checks alone do not assess scientific clarity.
- Real physical measurement import, hardware cost recording, public dataset submission, and GitHub issue submission were inspected up to their input/confirmation controls. No hardware evidence or external publication was invented for this review.
