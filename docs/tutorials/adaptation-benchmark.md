# Adaptation across conditions: a `conditions-v1` report

One score at one operating point says nothing about what happens when the
drive, the temperature or the capture batch changes. This tutorial builds a
condition card, pre-registers an adaptation plan, runs every cell and reads
the report that keeps failures and costs next to the numbers. The card here is
synthetic, so the report says it is a rehearsal; the flow is the one a
measured card takes.

Every command below is executed by `tests/integration/test_docs_commands.py`.

## 1. Look at the built-in card

```bash
opendpd adaptation card apa-200mhz-batches-v1
```

The built-in card pairs the packaged `APA_200MHz` and `APA_200MHz_b` captures:
the same set-up acquired twice, the real `capture_batch` dimension with two
conditions. Two is below the evidence bar (three conditions, each from its own
batch, all measured), so a report over it is a rehearsal too. Import both
datasets (`opendpd datasets import-builtin`) and add `--workspace ws` to audit
the card against the workspace.

## 2. Write a card for your own conditions

One dataset per condition, each from its own acquisition. Roles are fixed here,
before any run: one `source`, the rest `target`.

```json
{
  "set_id": "my-pa-drive-v1",
  "device": "GaN Doherty, unit 2",
  "dimension": "output_power_dbm",
  "conditions": [
    {"condition_id": "p30", "dataset_id": "unit2-p30", "role": "source", "capture_batch": "2026-09-01-a", "values": {"output_power_dbm": 30}},
    {"condition_id": "p32", "dataset_id": "unit2-p32", "role": "target", "capture_batch": "2026-09-01-b", "values": {"output_power_dbm": 32}},
    {"condition_id": "p34", "dataset_id": "unit2-p34", "role": "target", "capture_batch": "2026-09-02-a", "values": {"output_power_dbm": 34}}
  ]
}
```

The audit refuses a card whose conditions share a raw capture (same sha256):
one capture split two ways is not two conditions.

```bash
opendpd adaptation card card.json --workspace ws --out card.sealed.json
```

## 3. Pre-register the plan

```bash
opendpd adaptation plan card.sealed.json --workspace ws --pa-recipe pa-gru-smoke-v1 --dpd-recipe dpd-gru-smoke-v1 --budgets 2000 --seeds 0 --target-metric NMSE --target-threshold -25 --out plan.json
```

The plan fixes the entries, the tasks (`zero_update`, `few_shot` per budget,
`full_retrain`), the seeds, the profile and the target rule, and is sealed by
its hash. Every cell's run is keyed by that hash. Use three or more seeds for a
report whose seed spread means something; the smoke recipes here are for the
flow, not for numbers.

## 4. Run every cell

```bash
opendpd adaptation run plan.json --workspace ws
```

PA models are trained on every condition first (each DPD cell of a condition
goes through the PA of that condition). Then, per target: the source model
scored unchanged (`zero_update`), warm-started and trained on the first 2000
samples of the target's train split (`few_shot`), and trained from scratch
(`full_retrain`). A refused cell (a budget below the recipe's frame length,
a failed dependency) is recorded with its reason and the command exits 1;
running the plan again reuses finished runs and only fills gaps.

## 5. Read the report

```bash
opendpd adaptation report plan.json --workspace ws --markdown adaptation-report.md
```

The report is stored under `ws/adaptation/` and served by the Studio's
**Robustness** page, where the matrix can be filtered by entry, metric and
task and every failed cell shows its reason. Read, in this order:

1. the **evidence bar** line: met or not, and why;
2. the **repeats** line: training seeds and capture batches are counted
   separately;
3. the matrix: `mean ± std (n)` per cell, `FAILED k/n` with the reason where a
   seed produced no number, and whether the target rule was reached;
4. the **cost** table: new samples of the condition and mean wall clock per
   cell, with the device;
5. the limitations, starting with "single device".

The protocol, the definition of each task and the records that are still
empty (a measured card that meets the bar, an external recomputation) are in
`docs/protocols/conditions-v1.md`.
