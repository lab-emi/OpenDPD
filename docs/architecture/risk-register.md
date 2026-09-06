# Risk register

Living document; each row links to the mitigation the plan prescribes and the
current status. Update when a stage closes.

| # | Risk | Early signal | Mitigation / gate | Status (S00) |
|---|---|---|---|---|
| R1 | GUI and backend diverge | frontend computes its own metrics or defaults | all formal computation through shared services; contract tests | open — schemas start in S01 |
| R2 | Feature pile-up by agents | PR count grows, user loop unfinished | ≤ 2 pending agent PRs; acceptance by user journey | policy in AGENTS.md |
| R3 | Web server blocked by training | clicks freeze during training | separate worker process, event throttling, resource budget, stress tests | design fixed (S03) |
| R4 | Pretty UI, untrustworthy results | hidden profile, mixed measured/surrogate | evidence type + profile mandatory in result schema | open — S01/S08 |
| R5 | Packaging fails | only works from source with Node.js | wheel/sdist tests from G0; no-Node install test | baseline: wheel builds, no frontend yet |
| R6 | Big data freezes browser | full I/Q or logs sent to frontend | backend aggregation, viewport data, virtual lists | open — S05/S13 |
| R7 | Reproducibility over-promised | expected values edited per device | deterministic vs statistical regressions, recorded tolerances | protocol drafted |
| R8 | Local service insecure | any web page can write, any path readable | session, Origin/Host, CSRF, path sandbox, negative tests | open — S04 |
| R9 | Hardware scope creep | several instrument drivers in parallel | manual capture import first, then one adapter | out of first version |
| R10 | Standards work delays release | waiting for "full 3GPP" | S15 delivers one profile, does not block G2 | deferred |
| R11 | Leaderboard without community | only maintainer models | external trial first; otherwise "reference benchmark" | deferred |
| R12 | Legacy `sys.argv` API misuse | concurrent API calls corrupt each other | explicit config path (S02); legacy API kept single-threaded and documented | known, baseline |
| R13 | Legacy paths depend on CWD | artifacts land wherever the process runs | worker runs inside its own run directory; new services use explicit workspace paths | known, baseline |
