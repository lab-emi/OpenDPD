# opendpd-pa-modeling v2026.09 — reference benchmark

Track `pa_modeling` (pa_modeling evidence), protocol `leaderboard-v1`.

This is a **reference benchmark**, not a community standard: 0 of the 3 required external submissions are accepted and 0 of the 2 required independent recomputations exist. Entries by the maintainers are self-reported unless their evidence grade says otherwise.

Entries rank only inside one comparability group (data, operating point, metric profile version, split, execution semantics, resource class). Every number is mean ± sample standard deviation over the listed seeds; wall clock is a measurement on the stated device, never power. Failure conditions are the submitter's own statement of where the method fails or was not tried. Evidence grades: self_reported, reviewed, independently_recomputed.

## Group: dpa-200mhz (raw 9464a16d9842), as captured, legacy-opendpd-v1 v1, split contiguous-v1, offline_segmented, budget unbounded

Ordered by NMSE (lower is better); the other columns are not a tie-break.

| # | Method | Model (params) | NMSE | ACLR_AVG | ACLR_L | ACLR_R | EVM | Wall clock (s) | Grade | Status | Submitter | Failure conditions | Traceability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MP (least squares) | mp_ls (200) | -34.91 ± 0.00 (n=3) | -31.16 ± 0.00 (n=3) | -31.88 ± 0.00 (n=3) | -30.45 ± 0.00 (n=3) | -39.36 ± 0.00 (n=3) | 0.30 on cpu | self_reported | accepted | OpenDPD maintainers (maintainer) | not evaluated on any other dataset or operating point; cpu_regression tier: smoke budgets, regression references for this data and hardware, never research results | benchmark report `be10351c3fea`, plan `d79883def16e`; runs run-20260906-114018-790002 (8a7975cad510), run-20260906-114021-88b110 (b80aced066fd), run-20260906-114023-6a3b3b (6fe2ac3b7138) |
| 2 | GRU | gru (1911) | -21.60 ± 0.89 (n=3) | -32.36 ± 0.66 (n=3) | -33.51 ± 0.73 (n=3) | -31.22 ± 0.95 (n=3) | -22.71 ± 0.97 (n=3) | 1.23 on cpu | self_reported | accepted | OpenDPD maintainers (maintainer) | not evaluated on any other dataset or operating point; cpu_regression tier: smoke budgets, regression references for this data and hardware, never research results | benchmark report `be10351c3fea`, plan `d79883def16e`; runs run-20260906-114016-39ad61 (a67791136bc8), run-20260906-114020-8c19f1 (c94f5d5a59e2), run-20260906-114023-d5594d (d1a701282406) |

## History

- 2026-09-06T16:47:20.264551+00:00 created by opendpd leaderboard seed (report be10351c3fea): version v2026.09 seeded with 2 reference entries
- 2026-09-06T16:47:20.264450+00:00 entry pa-gru created by opendpd leaderboard seed (report be10351c3fea): copied from the hash-bound benchmark-v1 report; no independent recomputation
- 2026-09-06T16:47:20.264539+00:00 entry pa-mp-ls created by opendpd leaderboard seed (report be10351c3fea): copied from the hash-bound benchmark-v1 report; no independent recomputation

Board hash: `1d7e571bf3146455927d6424fa555699c00b63e49149857dba5e865df42e9442`
