# Dataset constellation correction

Date: 2026-09-12 (Europe/Amsterdam). This follows the
[Studio workbench update](studio-workbench-2026-09-11.md).

## Cause and correction

The first workbench drew raw complex waveform samples in its I/Q panel. It
never called the dataset demodulators. The user correctly identified that this
cloud was not the expected 64QAM symbol constellation.

The shared inspection service now calls the existing
`datasets/<dataset>/demod.py` through `Demodulator.from_dataset`, matching
`datasets/plot_utils.py`: demodulate the input, then demodulate the PA output
with the input as synchronization reference and `equalize=True`. The new
`opendpd/services/dataset_constellation.py` adapts identity, sample coordinates,
statuses and bounded plot payloads; it does not implement another receiver.
GUI, HTTP and CLI still use `analyze_dataset`.

The default view for a supported capture is now **64QAM · Constellation**, with
**Symbols / Raw I/Q** controls. The output legend explicitly says **equalized**.
Unsupported imported data stays a raw I/Q plot with a reason; naming a capture
`DPA_200MHz` or setting its modulation to `64QAM` does not bind a receiver.

## Receiver selection and actual data

| Packaged dataset | Existing receiver | Parameters used |
|---|---|---|
| DPA_200MHz | `IFFTFrameDemodulator` | 2,560 samples/frame, 10 carriers, 64 extracted bins/carrier; no CP removal or resampling. |
| DPA_160MHz | `IFFTFrameDemodulator` | 16,384 samples/frame, 4 carriers, 1,024 extracted bins/carrier. |
| APA_200MHz / APA_200MHz_b | `OFDMCPDemodulator` | FFT 32,768, CP synchronization, 5 carriers, 600 extracted bins/carrier; the existing receiver selects the first recovered symbol per carrier. |

These values come from the packaged specs and classes, not frontend defaults.
The DPA_200MHz spec's descriptive text still mentions LTE resampling, which
conflicts with its explicit IFFT-frame receiver and the dataset README. That
prose is not used to select processing. The actual 38,400-sample capture yields
15 complete frames and a visible 8 × 8 constellation with the existing receiver.
This is evidence for that dataset's visualization, not for standards-compliant
LTE demodulation.

DPA_200MHz produces 9,600 input symbols and 8,700 equalized output symbols. The
existing equalizer masks weak reference subcarriers, so the two counts differ.
The browser receives 3,200 and 2,900 points respectively after display-only
subsampling. The integration test checks the returned I and Q coordinates
exactly against the existing receiver, and the live browser check verifies
that both displayed traces match the HTTP response.

The separate raw I/Q view contains 3,840 displayed input/output waveform points.
No hard decisions, snapping to ideal constellation coordinates, or new symbol
EVM calculation is used to make the plot look like QAM.

## Timing, equalization and limits

- The receiver sees contiguous samples before any chart subsampling. IFFT
  windows are rounded inward to complete frames in original capture coordinates,
  within the existing 131,072-sample analysis budget. Their own selected-version
  and raw-coordinate ranges are returned in the API and described in the plot
  tooltip.
- Input head crops are recovered from recorded `preprocess-v1` version ancestry.
  Tests cover positive/negative integer and fractional delay corrections.
  Unknown ancestry or changed receiver metadata yields an unavailable status.
- The CP receiver's mixer has no absolute-sample-offset argument. A version
  whose input head was cropped cannot preserve the same carrier phase with
  that receiver, so it falls back to raw I/Q with a reason. Both unmodified APA
  captures pass real computation checks. Their existing CP search takes about
  14–16 seconds for input and output on this CPU; it was not rewritten here.
- The output equalizer is fitted to the displayed capture. It can remove linear
  response and also affect apparent error. Noise, residual linear effects and
  nonlinear distortion are not separated by this plot. Input normalization is
  the existing per-carrier/per-symbol RMS convention, which can slightly spread
  the apparent QAM radii when frames/carriers are overlaid.
- A descriptive constellation does not define a standard EVM protocol or bind
  the separate `ofdm-lte20-evm-v1` reference waveform. Its numerical EVM status
  remains unchanged. No metric definition, protected path, seed, tolerance,
  expected scientific value or checkpoint-selection rule was changed.

## Verification

Environment: macOS 26.6.2 arm64, Python 3.13.12, Node 26.8.1, CPU. Full versions
and checkout identity are in [environment.json](studio-constellation-2026-09-12/environment.json).

| Command / check | Result |
|---|---|
| `.venv/bin/python -m pytest tests/integration/test_dataset_analysis_api.py -q` | 14 passed in 33.88 s; includes real HTTP DPA symbols, existing-receiver parity, version/frame coordinates, all four packaged receivers, metadata mismatch and invalid samples. |
| `npm --prefix frontend test -- --maxWorkers=2` | 88 passed. Includes default-symbol view, equalized legend, raw-I/Q switching and unavailable-receiver behavior. |
| `npm --prefix frontend run build` | Passed, including strict TypeScript; packaged frontend rebuilt. |
| `npm --prefix frontend run lint` and `npm --prefix frontend run types:check` | Passed. |
| In `frontend/`: `npx playwright test e2e/a11y.spec.ts --project=chromium-1366 --project=chromium-1920 --workers=2 --ignore-snapshots` | 8 passed in 23.6 s. Mock journeys cover accessibility, keyboard navigation and loopback-only operation; pixel baselines were not asserted. |
| Real server at `127.0.0.1:8876`, real DPA_200MHz capture, Chromium | Symbol/raw toggle, Chinese translation, enlargement and exact agreement between Plotly traces and HTTP data verified. |

Initial TypeScript checks caught an optional generated `traces` field and an
unsupported Testing Library query option; both were corrected without changing
assertions. The live accessibility probe initially hit the server's inline-script
CSP. It then ran the installed axe bundle through browser automation evaluation;
the application CSP was not changed.

The Chinese 1366 × 768 page has document size 1366 × 768 and workbench bottom
721.39 px, with zero axe WCAG 2.1 AA violations. The horizontal legend remains
above the chart. At 1920 × 1080 the document is also 1920 × 1080, its
workbench ends at 925.39 px, and axe reports no violations. Browser checks and
screenshots are linked below.

- [Chinese overview](studio-constellation-2026-09-12/constellation-zh-1366.png)
- [Chinese 1920 × 1080 overview](studio-constellation-2026-09-12/constellation-zh-1920.png)
- [Actual CLI analysis summary](studio-constellation-2026-09-12/cli-analysis-summary.json)
- [Enlarged 64QAM constellation](studio-constellation-2026-09-12/constellation-zh-enlarged.png)
- [English overview](studio-constellation-2026-09-12/constellation-en-1366.png)
- [1366 layout, accessibility and API/plot parity](studio-constellation-2026-09-12/ui-zh-1366.json)
- [1920 layout, accessibility and API/plot parity](studio-constellation-2026-09-12/ui-zh-1920.json)
- [Backend checks](studio-constellation-2026-09-12/backend-tests.txt)
- [Frontend checks](studio-constellation-2026-09-12/frontend-full-tests.txt)
- [Browser checks](studio-constellation-2026-09-12/browser-tests.txt)

Implementation and receiver/spec hashes are recorded in
[source-sha256.json](studio-constellation-2026-09-12/source-sha256.json).

No new dependency or Sionna adapter was needed. The previous broader UI stage's
tests and CPU training evidence remain in its own report. Windows, GPU, RF
hardware and standards-conformance validation were not performed in this
follow-up. Temporary probes, the owned browser/server and the preview workspace were
removed after archiving the evidence.
