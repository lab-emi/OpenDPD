# opendpd-dpd-surrogate v2026.09 — reference benchmark

Track `dpd_surrogate` (dpd_surrogate evidence), protocol `leaderboard-v1`.

This is a **reference benchmark**, not a community standard: 0 of the 3 required external submissions are accepted and 0 of the 2 required independent recomputations exist. Entries by the maintainers are self-reported unless their evidence grade says otherwise.

Entries rank only inside one comparability group (data, operating point, metric profile version, split, execution semantics, resource class). Every number is mean ± sample standard deviation over the listed seeds; wall clock is a measurement on the stated device, never power. Failure conditions are the submitter's own statement of where the method fails or was not tried. Evidence grades: self_reported, reviewed, independently_recomputed.

## Group: dpa-200mhz (raw 9464a16d9842), as captured, legacy-opendpd-v1 v1, split contiguous-v1, offline_segmented, budget unbounded

Ordered by ACLR_AVG (lower is better); the other columns are not a tie-break.

| # | Method | Model (params) | ACLR_AVG | ACLR_L | ACLR_R | EVM | NMSE | Wall clock (s) | Grade | Status | Submitter | Failure conditions | Traceability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MP (least squares) | mp_ls (200) | -34.70 ± 1.50 (n=3) | -36.17 ± 1.40 (n=3) | -33.23 ± 1.60 (n=3) | -22.87 ± 0.96 (n=3) | -21.87 ± 0.96 (n=3) | 0.55 on cpu | self_reported | accepted | OpenDPD maintainers (maintainer) | not evaluated on any other dataset or operating point; the surrogate is a model fitted to measured data; agreement between a DPD's simulated and measured outcome is not established by this benchmark (cross-validation of the surrogate is not physical validation); cpu_regression tier: smoke budgets, regression references for this data and hardware, never research results | benchmark report `be10351c3fea`, plan `d79883def16e`; runs run-20260906-114019-94885b (57e9cb582e53), run-20260906-114022-c66858 (b05cda8b2f96), run-20260906-114025-3876a4 (b55615effb53) |
| 2 | GRU | gru (887) | -30.39 ± 1.27 (n=3) | -30.15 ± 2.28 (n=3) | -30.63 ± 1.13 (n=3) | -21.16 ± 0.86 (n=3) | -19.97 ± 0.93 (n=3) | 1.17 on cpu | self_reported | accepted | OpenDPD maintainers (maintainer) | not evaluated on any other dataset or operating point; the surrogate is a model fitted to measured data; agreement between a DPD's simulated and measured outcome is not established by this benchmark (cross-validation of the surrogate is not physical validation); cpu_regression tier: smoke budgets, regression references for this data and hardware, never research results | benchmark report `be10351c3fea`, plan `d79883def16e`; runs run-20260906-114018-6448fc (1ce8b7f349d2), run-20260906-114021-f34df4 (0ae6a4f643c2), run-20260906-114024-1a01e2 (50c75d3af220) |

## History

- 2026-09-06T16:38:51.770986+00:00 created by opendpd leaderboard seed (report be10351c3fea): version v2026.09 seeded with 2 reference entries
- 2026-09-06T16:38:51.770924+00:00 entry dpd-gru created by opendpd leaderboard seed (report be10351c3fea): copied from the hash-bound benchmark-v1 report; no independent recomputation
- 2026-09-06T16:38:51.770980+00:00 entry dpd-mp-ila created by opendpd leaderboard seed (report be10351c3fea): copied from the hash-bound benchmark-v1 report; no independent recomputation

Board hash: `d0f41156cafe4a5af4fab911eb0dc7742e286c322f7200d502a9dc267c529040`
