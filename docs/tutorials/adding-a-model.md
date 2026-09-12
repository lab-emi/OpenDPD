# Adding a model to OpenDPD Studio

One registry entry makes a model usable from the CLI, the Python API and the
GUI at once; nothing is hand-written twice. This page walks through the two
supported kinds — a gradient-trained backbone and a least-squares baseline —
and the evidence the registry demands before a model is shown as *supported*.

## 1. Gradient-trained backbones

1. **Implement the backbone** under `backbones/` and register it in
   `models.CoreModel` exactly as the existing ones (`gru`, `tcn`, …). The
   trainer, quantiser and evaluation stages call it through `CoreModel`.
2. **Describe it** in `opendpd/core/registry.py` with a `ModelDescriptor`:

   ```python
   ModelDescriptor(
       key="my_rnn", display_name="My RNN", family="recurrent", legacy_backbone="my_rnn",
       training_method="gradient", roles=("pa", "dpd"),
       params=(_hidden(16), _layers()),          # ParamSpec: name, type, default, bounds, legacy flag
       status="experimental",                    # "supported" only with device evidence
       devices_tested=("cpu",), lookahead_samples=0, lookahead_note=CAUSAL,
       reference="paper or report", evidence="which test or report produced the numbers",
   )
   ```

   - `params` become the CLI flags, the API schema and the **Advanced
     settings** form of the GUI; each `ParamSpec` carries type, default,
     bounds and the legacy argparse name per role. Validation errors name
     the field (`model.parameters.<name>`).
   - `lookahead_samples` is the number of *future* samples the model reads.
     State it, or set it to `None` with a `lookahead_note` containing
     "not characterised" — the registry test refuses silence on this point.
   - `devices_tested` must be non-empty and `evidence` must say where the
     evidence comes from (a test name or a report). "Supported" means a
     device with real evidence, not a device the code should work on.
3. **Run the registry tests:** `pytest tests/unit/test_registry.py`. They
   check that every registered model is constructible, that every working
   legacy backbone is registered, that keys are unique and that look-ahead is
   declared.
4. **Give it a smoke run** on the built-in data through both entry points:

   ```bash
   opendpd run --config my-rnn.json --workspace WS --json    # {"model": {"key": "my_rnn", ...}}
   ```

   and once through the GUI (New experiment → model → Start run). The result
   page shows the evidence type and metric profile like for any other model.

## 2. Least-squares baselines

Polynomial baselines fitted in closed form (MP, GMP) do not go through the
legacy trainer. They are registry models with `training_method="least_squares"`
whose basis functions live in `opendpd/core/polynomial.py` and whose fitting
and application live in `opendpd/services/polynomial.py`:

- add a basis to `opendpd/core/polynomial.py` (segmented, causal: the memory
  is reset at every `nperseg` segment start) and a coefficient count;
- add the descriptor with `_poly(...)` parameter specs and an `rcond` cutoff;
- the PA role is a direct least-squares fit on the train split; the DPD role
  is indirect learning on the measured train split, scored through a
  gradient-trained PA surrogate (never usable as a surrogate itself);
- rank, condition number and cutoff are recorded in `fit.json` and shown as
  fit diagnostics; results state the training path
  (`least_squares` / `ila_least_squares`).

Tests: `tests/unit/test_polynomial.py` (basis shapes, coefficient counts, fit
stability) and `tests/integration/test_baselines.py` (both roles as ordinary
runs, refusals as surrogate and under quantisation).

## 3. What you must not do in the same change

Metric definitions, data splits, golden references, acceptance thresholds and
the benchmark protocol are protected paths (`AGENTS.md` §3): a new model
never edits them. If a model needs a new evaluation, add a metric *profile*
in a separate, science-reviewed change (`docs/protocols/metric-profiles.md`).

## 4. Making it a benchmark entry

Benchmark plans (`benchmark-v1`) list model entries per tier. A new entry
needs a pre-registered plan (`opendpd benchmark plan`), at least three seeds,
and a report that states the training path and the per-seed spread; see
`docs/protocols/benchmark-protocol.md`. A model that only exists in a
notebook is not a benchmark entry.
