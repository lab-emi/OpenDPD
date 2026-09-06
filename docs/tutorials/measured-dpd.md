# Measured DPD evidence: from an exported signal to a `dpd_measured` result

Simulated DPD results (`dpd_surrogate`) say how a DPD behaves through a learned
PA model. This tutorial closes the loop with a physical PA: play the exported
pre-distorted signal, capture the output, import the capture, and read a result
that is labelled **user-provided, not independently verified** with every
condition you declared next to it. No instrument connection is needed; the
mock adapter runs the same procedure so you can rehearse it.

Every command below is executed by `tests/integration/test_docs_commands.py`.

## 1. Export the signal to play

Train a PA surrogate and a DPD as in the headless tutorial, then apply the DPD
to the test split:

```bash
opendpd apply <dpd-run-id> --workspace ws
```

The apply run's `dpd-output` artifact holds `x` (columns `I`, `Q`) and
`u = DPD(x)` (`I_dpd`, `Q_dpd`) at the dataset's sample rate, with a sidecar
that names the dataset, the DPD weights and the scaling. `u` is a PA *input*:
its existence does not show that anything was linearised.

## 2. Rehearse with the mock adapter (no hardware, no RF)

```bash
opendpd instruments list
```

Only `mock` ships. RF output is off by default on every adapter: a session
must be armed by a named person, so this refuses and plays nothing:

```bash
opendpd instruments dry-run --apply-run <apply-run-id> --out captures --workspace ws
```

Armed, the dry run plays `u` and then `x` through a synthetic stand-in PA,
turns the output off again, and writes `with_dpd.npy`, `without_dpd.npy`, a
session record per capture (adapter, limits, operator, interlock log, hashes)
and a `conditions.json` template:

```bash
opendpd instruments dry-run --apply-run <apply-run-id> --out captures --workspace ws --arm "Your Name"
```

Import the pair. `--mock` marks the result as mock evidence:

```bash
opendpd measurements import --apply-run <apply-run-id> --with-dpd captures/with_dpd.npy --without-dpd captures/without_dpd.npy --conditions captures/conditions.json --mock --workspace ws
```

The output shows, per capture, the recovered delay, the correlation with the
played signal and the least-squares gain, then the metrics, the limitations
and the no-DPD baseline. A mock result carries `"is_mock": true` and the
attestation "mock instrument adapter: a synthetic stand-in for a PA, not a
measurement"; it is never evidence about a PA.

## 3. A real measurement, by hand

1. Play `u` from the apply run's `dpd-output` file through your PA (looped)
   and capture the output with your I/Q analyser; then play `x` from the same
   file under **identical** settings and capture again. Do not change the
   drive between the two captures.
2. Write the conditions you can vouch for:

```json
{
  "pa": "GaN Doherty, unit 2",
  "capture_chain": "SMW200A -> PA -> 30 dB pad -> FSW (I/Q analyser)",
  "sample_rate_hz": 800000000.0,
  "drive": "generator -12 dBm, PA input +8 dBm",
  "gain_db": 28.5,
  "calibration": "none",
  "measured_at": "2026-09-06T14:00:00Z",
  "temperature_c": 25.0,
  "operator": "Your Name"
}
```

3. Import, declaring the output power you read for each capture:

```bash
opendpd measurements import --apply-run <apply-run-id> --with-dpd pa-with-dpd.csv --without-dpd pa-without-dpd.csv --conditions conditions.json --power-with 30.0 --power-without 30.0 --workspace ws
```

Captures may be `.csv` (columns `I`,`Q` or `I_out`,`Q_out`; `--columns` names
others), `.npy` (`(n, 2)` or complex) or `.npz`. A capture at another rate is
converted when the ratio to the dataset rate is a small rational number.
A capture that does not correlate with the played signal, or is shorter than
one period, fails the run with `capture_rejected` and the reason; nothing is
scored.

In the Studio the same import is the run page's **Import measured captures…**
button on a succeeded `run_dpd` run.

## 4. Read the result

```bash
opendpd evaluate <measured-run-id> --workspace ws --profile general-spectral-v1
```

```bash
opendpd report <measured-run-id> --workspace ws --format md --out measured-report.md
```

Before any metric, read the **Measurement** block: the attestation, the
declared conditions, and for each capture its raw file hash, delay,
correlation, gain and rms in capture units. The **output level with DPD
relative to without** is reported in capture units and, beyond 0.5 dB, becomes
a limitation: the difference between the captures is then not attributable to
the DPD alone. Lowering the drive for the with-DPD capture shows up there and
in the declared powers; it never shows up as an improvement.

Measured and simulated results are shown side by side and never ranked; two
measured results rank only when their declared operating point (PA, drive,
output power, capture chain, rate) agrees. The protocol, the alignment
definition and the hardware trial record (pending) are in
`docs/protocols/measured-dpd.md`; the adapter contract and the interlock in
`docs/architecture/instruments.md`.
