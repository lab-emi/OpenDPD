# Backlog after S13/S14 (prioritised)

Ordered by what blocks the next gate first. Items marked **human** need a
person or a machine this environment does not have; the rest are ordinary
agent tasks bound by `AGENTS.md`.

| # | Item | Why it is here | Gate |
|---|---|---|---|
| 1 | **human** External trial with three groups and the ≥ 5-user onboarding session (`docs/releases/external-trial-protocol.md`) | the only S14 evidence that is not the maintainers' own opinion | G2 |
| 2 | **human** Real desktop checks on macOS and Windows (`opendpd gui`, default browser launch, Ctrl+C cleanup, paths with spaces/Unicode) and a real Safari pass | support matrix rows are "unverified" | G2 |
| 3 | **human** Maintainer approval of the draft regression baseline (`benchmark/regression/cpu-regression-dpa-200mhz/baseline.json`) and a `gpu_full` run on a named machine | regression checks stay advisory until then | G2 |
| 4 | **human** Release approval, tag, PyPI publication and demo material from the same candidate | agents never publish | G2 |
| 5 | Streaming CSV import (chunked split writing straight from the reader) and chunked preprocessing versions for Stress-size captures | `.npy` streams, CSV and preprocessing keep the capture in RAM | later |
| 6 | Screen-reader pass and a manual keyboard review of the dialogs (import, preprocess, apply) | axe-core covers rules, not the experience | later |
| 7 | Windows CI runner for the core library and the packaged service | no evidence at all on Windows today | later |
| 8 | Reserved-port and instance-reuse behaviour documented for SSH tunnels (`--no-browser`, port forwarding) | remote use is an SSH-tunnel story only | later |
| 8a | Expose `execution.num_threads` in the New experiment form (advanced section) | the field is applied on every path since S14 but only a downloaded configuration or the CLI can set it today | later |
| 9 | **human** S15 cross-validation of `ofdm-lte20-evm-v1` against MATLAB LTE Toolbox on the same signals, standard versions fixed, error budget approved (`docs/protocols/waveform-profiles.md` §5–7) | the waveform, profile, packages and tests exist; the profile stays hidden in the GUI until this record is filled | G3 |
| 10 | S16–S19 tracks (hardware capture import, instrument adapter, quantisation/deployment exports, community models) as scoped in the plan | after G2 | G3 |
| 11 | S20 versioned leaderboard policy once external submissions exist | needs a community first (risk R11) | G4 |
