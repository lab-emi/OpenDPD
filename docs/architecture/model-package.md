# The `opendpd-model-v1` package

A trained PA or DPD leaves OpenDPD as one zip file of **data**. It holds the weights, a statement of what the model
computes and how it was scored, and a golden test vector that lets whoever receives it check that their implementation
reproduces OpenDPD's outputs. Nothing in it is code, and a reader never executes anything from it.

This page is the format specification. The writer is `opendpd/services/model_export.py` (`opendpd export-model RUN_ID
--workspace WS --out FILE`, `Job.export` in the Python SDK, `opendpd.export` in the MATLAB toolbox); the reference reader
is the MATLAB toolbox (`opendpd.load`, `opendpd.verify`, `opendpd.apply(model, x)`), which runs the model in plain MATLAB
with no Python.

## What a package is for, and what it is not

* **For** moving a model out of the Python environment it was trained in: into MATLAB or Simulink code, a lab machine
  without Python, a colleague's laptop. It carries enough to reproduce `opendpd.apply` on a waveform.
* **Not** a statement that the model is good for any amplifier. The golden vector comes from the same file as the weights,
  so a pass shows that an implementation computes this model correctly. It does not show that the file came from where it
  says. Compare the package's SHA-256 with the value published by whoever gave it to you.
* **Not** a quantised or fixed-point model. `fixed-point-v1` (the C99 export) is a separate format.
* **Not** normalised: samples are used as supplied, in the units and at the sample rate of the training dataset. A DPD's
  output is the predistorted PA input.

## Container

A zip file with exactly these entries, no others, none twice:

| Entry | Content |
|---|---|
| `manifest.json` | UTF-8 JSON, keys sorted, two-space indent, trailing newline, no `NaN`/`Infinity`. Described below. |
| `weights.npz` | The model's arrays, one `.npy` per array (`numpy.savez_compressed` layout). |
| `weights.mat` | The same arrays and names as a MAT v5 file of numeric arrays, for the reader's own code. |
| `golden/golden.npz` | The test vector: `input` and one `output_*` array per execution semantics. |
| `golden/golden.mat` | The same as a MAT v5 file. |
| `README.md` | A short human-readable description. |

The writer is deterministic: the same run gives the same bytes. Entry timestamps are 1980-01-01, entries are sorted by
name, the `.npz` members are sorted, and the creation time that SciPy puts into a MAT header is replaced by a fixed text
(`MATLAB 5.0 MAT-file, written by OpenDPD (opendpd-model-v1); numeric arrays only`).

`manifest.files` lists the SHA-256 of every other entry. A reader refuses a package in which an entry is missing from that
list or does not match it.

## `manifest.json`

| Key | Content |
|---|---|
| `format` | `"opendpd-model-v1"`. A reader refuses anything else. |
| `created_by` | `opendpd` (the version that wrote it) and `exporter`. |
| `run` | `run_id`, `role` (`"pa"` or `"dpd"`), `task`, `checkpoint_sha256`, `dataset_id`: where the weights came from. |
| `model` | `key` (the registry key), `parameters` (the run's model parameters), `architecture` (what the arrays mean, below) and `weights`: for every array its `name`, `shape`, `dtype` and `source` (the PyTorch state-dict name). |
| `signal` | `sample_rate_hz`, `bandwidth_hz`, `nperseg`, `amplitude_units`: the training dataset's. `nperseg` is also the segment length of the offline semantics. |
| `scaling` | `reference_gain`, `train_input` (`rms`, `peak`, `n_samples` of the training input: inputs far outside are extrapolation) and a note. |
| `execution` | The two execution semantics, below. |
| `evidence` | `type` (`simulation`, `measurement` or `unknown`, from the dataset's origin), `dataset_origin`, a note. It states that this is model inference, not a hardware measurement or a new metric evaluation. |
| `golden` | `input`, `samples`, `tolerance_abs`, `streaming_chunk_samples`, `input_description`, `outputs` (the names of the `output_*` arrays). |
| `files` | The SHA-256 of every entry except `manifest.json`. |

Readers should ignore keys they do not know. A change that a v1 reader would misread gets a new format name, not a
changed meaning of an existing key.

## Arrays

All arrays are little-endian and unnormalised. Shapes follow PyTorch.

| Model key | Arrays |
|---|---|
| `mp_ls` | `coefficients`: complex128, flat, `w[k*Q+q]` multiplies `x(n-q)·|x(n-q)|^k`, lag fastest (`k` is the power index from 0, `Q` the memory depth). |
| `gmp_ls` | `coefficients`: complex128, flat, in the order of `opendpd.core.polynomial.gmp_basis`: aligned terms (`k*La+l`), then lagging terms `(k,l,m)`, then leading terms `(k,l,m)`. `architecture.parameters` gives the ranges and `architecture.coefficient_order` states the order. |
| `gmp` | `gmp_weight`: float32, flat, `memory_length·(1+(degree-1)·memory_length)` real weights (`architecture.terms`), one per term of the `gmp` backbone: the `memory_length` linear taps first, then the cross terms by power and lag pair, in the order of `backbones/gmp.py`. The complex output is the real-weighted sum of the terms. |
| `gru` | `rnn_weight_ih_l{n}`, `rnn_weight_hh_l{n}` and, if present, `rnn_bias_ih_l{n}`, `rnn_bias_hh_l{n}` for every layer; `fc_weight` and, if present, `fc_bias`. float32. PyTorch GRU, gate order r, z, n; the new gate is `n = tanh(W_in x + b_in + r·(W_hn h + b_hn))`; the head is a linear map from the hidden size to two outputs. Input features are `I`, `Q`. |
| `tres_gru` | The `gru` arrays, plus `tcn_conv1_weight` (3×2×3) and `tcn_conv2_weight` (2×3×1). Input features are `I`, `Q`, `|x|`, `|x|^3`, `I(n+1)`, `Q(n+1)`; the TCN branch is a dilated (16) convolution, hardswish, a 1×1 convolution and hardswish, added to the head's output. |

`model.architecture` repeats these facts in words and gives the sizes (`hidden_size`, `num_layers`, whether the RNN and the
head have biases), so that a reader can check an array's shape before it uses the array.

## Execution semantics

A model's output depends on how a waveform is cut up, so a package states both ways it can be run, and a result must say
which one produced it.

* **`offline_segmented`**: how the run was scored. The waveform is cut into segments of `signal.nperseg` samples, the state
  is zero at the start of every segment (and of every call), the last segment is zero padded and the padding is trimmed
  from the output. `lookahead_samples` says how far the model reads ahead inside a segment (16 for `tres_gru`, whose
  recurrent features also read the next sample, wrapping within the segment).
* **`streaming_stateful`**: one state across the whole waveform, for the models with a registered streaming variant
  (`gru`: the recurrent state; `gmp`: a window of `history_samples` past inputs, which is carried instead of zero filled).
  `available` is `false`, with a reason, for the others.

The two give different waveforms for the same model. Which one to use for a waveform that goes to hardware is a decision
for the user; neither is a measurement.

## Golden test

`golden/golden.npz` holds `input` (float32, `N×2`, seeded noise with the training input's rms and peak: never a slice of
the user's data, so a package can be shared without sharing a measurement) and the outputs OpenDPD's `apply` produced for
it, one per available semantics (`output_offline_segmented`, `output_streaming_stateful`). The streaming output was made
with chunks of `streaming_chunk_samples` samples, an awkward size on purpose: chunk boundaries must not matter.

A reader runs its implementation on `input` and compares with the outputs: the maximum absolute difference must be at
most `tolerance_abs`, which is `1e-5` for float32 outputs. A package may ask for a stricter bound and never a looser one
(the MATLAB reader uses the smaller of the package's value and `1e-5`), and a non-finite output never passes.

## Writing a reader safely

The MATLAB reader is the reference. It was written so that a package from an unknown source cannot make it run anything:

* Accept only the six entry names above and refuse duplicates, whatever the archive says. Use no path from the archive.
* Copy entries out with a hard byte limit and refuse an archive that inflates beyond the size it declares (the defaults are
  256 MB per entry and 512 MB in total).
* Check every entry against `manifest.files` before using it.
* Parse the `.npy` members strictly: little-endian float32, float64, complex64 or complex128 only, the exact header NumPy
  writes, a data length that matches the shape. Never unpickle, never evaluate a header.
* **Do not open the MAT files of a package you did not make.** In MATLAB, `load`, `whos -file` and `matfile` all call
  `loadobj` for a class that is on the path, so inspecting an untrusted MAT file can itself run code (shown with a trap
  class in R2026a; only `who -file` is inert). The `.npz` and `.mat` files carry the same arrays, so a reader loses
  nothing by using the `.npz`. The `.mat` files are hash-checked and left for the user's own code.

## Which models a package can hold

`mp_ls`, `gmp_ls`, `gmp`, `gru` and `tres_gru`, unquantised. A model joins the list only together with a test that compares
the reader's output with the Python evaluator (`opendpd.services.inference.APPLY_MODELS` and `EXPORT_MODELS` are the same
set), and with a runtime in each reader that implements it. The registry does not yet carry an `export_formats` field for
this; adding one touches the model registry, which needs maintainer review.

## Reference implementations and tests

* Writer: `opendpd/services/model_export.py`; drift test `tests/unit/test_model_export.py` (rebuilds each committed fixture
  from its own weights with today's PyTorch code and compares with the golden outputs).
* MATLAB reader and runtime: `Matlab/toolbox/+opendpd/` (`load`, `verify`, `Model`, `+internal/readPackage.m`, `+runtime/`);
  tests `TestModel`, `TestPackageSecurity`; fixtures `Matlab/toolbox/tests/data/*.opendpd.zip` (six small packages).
* Evidence recorded with the first implementation: the pure-MATLAB kernels reproduce the evaluator's outputs for the six
  fixtures to at most 1.2e-7 (the polynomial models exactly), on MATLAB R2026a, Linux.
