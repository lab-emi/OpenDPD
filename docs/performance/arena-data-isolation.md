# Arena training data and test isolation

Arena v6 excludes ILC and ILC→MP from its model catalogue, submissions,
rankings and Pareto fronts. MP and GMP remain ordinary DPD competitors.

## What the polynomial models learn from

`x_train` below is the original measured training input. `PA` is the dataset's
frozen TRes-GRU, and `G` is the gain fixed from the training partition.

| Model | Regression or optimization input | Target | ILC used? |
|---|---|---|---|
| MP least squares (`mp_ls`) | `PA(x_train)/G` | `x_train` | No |
| GMP least squares (`gmp_ls`) | `PA(x_train)/G` | `x_train` | No |
| Gradient-trained GMP (`gmp`, `gmp_stream`) | Frames from `x_train`, passed through DPD→PA | `G*x_train` | No |
| Neural DPD | Frames from `x_train`, passed through DPD→PA | `G*x_train` | No |
| Historical ILC→MP, excluded | ILC operates only on `x_train[:16384]`; MP then fits `PA(u_ILC)/G → u_ILC` | Learned training-waveform input | Training prefix only |

The least-squares MP/GMP models use indirect learning (ILA). Their regression
features are PA-model feedback derived from the measured training waveform;
they are not an ILC-generated dataset. Polynomial orders and SVD cutoffs are
fixed before evaluation. Neither validation nor test samples fit coefficients.
The original measured `y_train` identifies the PA; it is not the target of
these inverse-model fits.

The old ILC entry performed offline waveform learning followed by MP fitting;
its online predictor was that fixed MP. The ILC routine itself received only
the training prefix and a frozen-PA callable, with no validation/test data.
It is nevertheless excluded under the requested Arena scope. This change
does not remove the separate ILC workflow elsewhere in Studio.

## Separation of model selection and final scoring

1. PA fitting uses measured training input/output. Validation NMSE selects PA
   checkpoints and seeds; test NMSE is reported only after selection. APA PAs
   were retrained after repartitioning, so old weights that saw the new test
   segment are excluded.
2. Arena fitting workers load only `x_train`, `y_train`, `x_val`, `y_val`.
   Neural updates use all training windows for 240 epochs; equal-weight
   validation in-band error and worse-side ACLR select a checkpoint, with
   feasible validation output power first. Classical MP/GMP fit only training data.
3. Every configuration and seed in a submission finishes fitting before the
   test context is constructed. Selected models are put in evaluation mode,
   gradients are disabled, and checkpoint hashes are checked before testing.
   A fitting failure does not trigger any test evaluation.
4. Only held-out `x_test` is used for final DPD→PA evaluation and the common
   baseline. EVM, output ACLR, the final power gate and quality improvement all
   use test observations. FoM adds the frozen model's parameter/operation costs.
   Validation diagnostics do not contribute to FoM or ranks.

The v4 fitting functions already avoided test samples in optimization and
checkpoint selection. Two orchestration weaknesses were tightened in v5:
the old shared context eagerly loaded all splits, and a standalone submission
could interleave fitting one seed with testing it before fitting later seeds.
No test score fed back into fitting, but the new loader and two-stage execution
make that boundary explicit and enforceable.

## Verification

`tests/unit/test_arena_data_isolation.py` checks the actual split loader,
blocks held-out-array access during polynomial fitting, fails if MP/GMP calls
ILC, and checks the historical ILC training-prefix boundary. It also verifies
that validation diagnostics cannot change FoM and that train/validation
observations are rejected by the scorer. Pipeline tests assert that every fit
precedes the first test load, including interrupted submissions.

The APA_200MHz_b release contains 218 independent v6 fits with its frozen PA and training settings. All are projected from the completed matrix with identical weights and test observations; no older DPD checkpoints enter these rankings.
The distributed coordinator verifies every checkpoint before freezing a
global manifest. Official test workers cannot train a missing checkpoint.
The 23 model keys on APA_200MHz_b produce 23 results and 233 test cases,
including stateful reevaluation of the relevant base weights.

<!-- arena-v6-isolation:start -->
All **218 refitted checkpoints** were frozen before the official test phase. The full audit verifies **233 test cases**, the operation ledgers and **15 streaming replays**. [Numerical isolation evidence](arena-data-isolation.json).
<!-- arena-v6-isolation:end -->

[Current results](arena-reference-results.md) ·
[Complete audit](arena-reference-audit.json) ·
[Protocol](../protocols/dpd-arena-v6.md)
