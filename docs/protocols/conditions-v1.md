# Multi-condition adaptation (`conditions-v1`): protocol, S17

Status: protocol implemented and tested on synthetic three-condition cards
and on the built-in two-batch APA card; **no condition set that meets the
evidence bar (§6) exists yet** and no external recomputation has been made
(§8). Every report below the bar says so in its limitations and calls itself
a rehearsal.

## 1. What the protocol measures

A DPD or PA model tuned at one operating point is engineering evidence only if
it is known what happens when the condition changes. The protocol separates
three questions and never averages them into one score:

| Task | The weights come from | New samples of the target condition |
|---|---|---|
| `zero_update` | the source-condition model, unchanged | 0 |
| `few_shot` | the source-condition model, then trained on the first *budget* samples of the target's train split | the budget |
| `full_retrain` | training from scratch on the target's train split | the whole train split |

Every task is scored on the target condition's **test split** under the plan's
metric profile, by the same executor as every other run (`evaluate_pa`,
`train_pa` with `initialization` + `training.train_samples`, `train_pa`;
`run_dpd` with `dpd_reference.transfer`, `train_dpd` with `initialization` +
`training.train_samples`, `train_dpd`). A DPD cell of a condition always goes
through the PA model **trained on that condition** with the same seed
(surrogate evidence, `dpd_surrogate`); the surrogate's own quality is a cell of
the report, not an assumption.

## 2. The condition set (data card)

`ConditionSet` (`opendpd/schemas/conditions.py`) is one device, one varied
dimension and at least two conditions:

| Field | Rule |
|---|---|
| `device` | the one PA every condition comes from; the report's first limitation always says nothing generalises beyond it |
| `dimension` | what varies: `capture_batch`, `output_power_dbm`, `temperature_c`, `waveform`, … |
| `conditions[].dataset_id` | a registered workspace dataset, one per condition; the audit (`opendpd adaptation card … --workspace`) refuses two conditions whose raw captures have the same sha256: one capture split two ways is not two conditions |
| `conditions[].capture_batch` | the independent acquisition the dataset comes from; the evidence bar needs one batch per condition |
| `conditions[].role` | exactly one `source`; the rest are `target`. Roles are declared in the card **before any run** and sealed by `card_sha256` |
| `held_out_policy` | fixed text: targets are never used to select hyper-parameters; every number is read from the target's test split after the run finished |

The card is sealed by `card_sha256` over everything but its creation time; an
edited card is refused. A built-in card, `apa-200mhz-batches-v1`, pairs the
packaged `APA_200MHz` and `APA_200MHz_b` captures (the same set-up, two
acquisitions): a real `capture_batch` dimension with **two** conditions, below
the bar.

## 3. The plan (pre-registration)

`AdaptationPlan` fixes, before anything runs: the sealed card, the entries
(a PA recipe and optionally a DPD recipe that names its PA entry), the tasks,
the few-shot budgets, the training seeds, the metric profile, the device and an
optional target rule (`metric`, `threshold`, `better`). The plan is sealed by
`plan_sha256`; an edited plan is refused. Every cell's run carries the
idempotency key

    adapt-<plan_sha256[:12]>-<entry>-<task>-<condition>-b<budget>-s<seed>

so re-running a plan reuses finished runs and fills gaps, never duplicates. A
cell whose configuration the executor refuses (a budget below the frame
length, a missing dependency) is recorded next to the plan with the reason and
appears in the report as `missing`; a run that failed appears as `failed` with
its error. Hyper-parameters are those of the named recipes: there is no search
inside a plan, so nothing can be tuned on a target.

## 4. The report

`AdaptationReport` reads stored results only (nothing is recomputed):

- `cells`: one per (entry, task, condition, budget, seed) with status,
  run id, metrics under the plan's profile, **new samples**, wall clock,
  device, configuration and checkpoint hashes, and whether the target rule was
  reached;
- `aggregates`: mean / std / min / max over seeds per cell group with the
  number of failed seeds;
- `repeats`: seeds and capture batches counted separately; measurement
  repeats are conditions of the card and are never counted as seeds;
- `evidence_bar` (§6) and `limitations`;
- `report_sha256` over the whole content; the Markdown rendering carries the
  report, plan and card hashes.

The Studio's Robustness page shows the same matrix with filters by entry,
metric and task, the failed cells with their reasons, the cost columns and the
limitations; it reads reports, it never runs plans.

## 5. Budgets and what "new samples" means

`training.train_samples = N` keeps the first `N` samples of the target's
**train** split for fitting; the validation and test splits and the reference
gain stay those of the full dataset (`docs/protocols/simulation-chain.md`
for the gain). `new_samples` in a cell is therefore `0`, the budget, or the
train split size; the time column is the run's wall clock on the stated
device. Both are costs, and both are reported next to every score.

## 6. Evidence bar

A card is evidence about adaptation on its device when all of the following
hold, and the report's `evidence_bar.met` is true only then:

1. at least `MIN_CONDITIONS_FOR_EVIDENCE = 3` conditions along one real
   dimension;
2. every condition comes from its own capture batch;
3. every condition is a `measured` dataset (`DatasetManifest.origin`).

Below the bar the report is a rehearsal of the protocol and says so. Fewer
than three seeds add the limitation that the seed spread is not established.
The bar is a constant of the schema; agents do not lower it (AGENTS.md §3).

## 7. Record of condition sets that meet the bar

| Card | Device | Dimension | Conditions | Batches | Origin | Bar |
|---|---|---|---|---|---|---|
| `apa-200mhz-batches-v1` (built in) | APA | `capture_batch` | 2 | 2 | measured | **not met** (2 < 3) |
| synthetic cards in `tests/integration/test_adaptation.py` and the tutorial | synthetic PA | `drive` | 3 | 3 | synthetic | not met (origin) |

**Pending human**: a measured condition set with three or more conditions
along one dimension (power, temperature, waveform or batch), each from its own
acquisition. No such data exists in this repository.

## 8. External recomputation

**Pending human**: nobody outside the original implementation has recomputed
a report. Recomputation means: the same card and plan (hashes), `opendpd
adaptation run` and `report` on another machine, and the reported metrics
within the profile's stated tolerance; record the plan hash, the machine and
the software block of both reports here.
