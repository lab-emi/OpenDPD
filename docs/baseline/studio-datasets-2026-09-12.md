# Studio dataset catalog and guided CSV creation

Verified on 2026-09-12, macOS arm64, CPU. This extends the
[dataset constellation correction](studio-constellation-2026-09-12.md).
The implementation is an uncommitted working tree; exact versions and source
hashes are recorded in [environment.json](studio-datasets-2026-09-12/environment.json)
and [source-sha256.json](studio-datasets-2026-09-12/source-sha256.json).

## Result

Studio now discovers every packaged `datasets/*/spec.json` and offers it through
**Built-in datasets**. The catalog contains all five datasets in this installation.
The shared `load_version_arrays` delegates to the existing training loader, so
both six-file split CSVs and single-file CSVs are read in capture order. No
frontend list of dataset names or separate split calculation was introduced.

| Dataset | Paired samples read | Modulation | Recovered input / equalized output symbols |
|---|---:|---|---:|
| APA_200MHz | 98,304 | 256QAM | 3,000 / 2,985 |
| APA_200MHz_b | 98,304 | 256QAM | 3,000 / 2,985 |
| DPA_160MHz | 491,520 | 1024QAM | 32,768 / 31,424 |
| DPA_200MHz | 38,400 | 64QAM | 9,600 / 8,700 |
| MyCustomPA | 102,400 | 64QAM, synthetic tutorial | 2,560 / 2,560 |

Each was selected through the real browser catalog. Frequency, time, I/Q,
AM/AM and demodulated constellation payloads came from the real local API.
Both rendered constellation traces matched their HTTP I/Q arrays exactly;
all five workbenches fit 1366×768. The existing bounded analysis window and
presentation subsampling remain in use. The differing input/output symbol
counts reflect the existing equalizer's weak-reference mask.

See the [input constellation comparison](studio-datasets-2026-09-12/builtin-input-constellations.png),
[catalog screenshot](studio-datasets-2026-09-12/builtin-catalog-1366.png), and
[browser checks with trace hashes](studio-datasets-2026-09-12/all-builtin-browser.json).
Individual screenshots are saved alongside them as `<dataset>-1366.png`.
These are recovered symbols without ideal-point snapping. The existing receiver's
per-frame/carrier normalization and fitted output equalization remain descriptive
visualization conventions; this work does not establish standard EVM compliance.

## MyCustomPA tutorial

The old single-CSV example had inconsistent capture metadata and receiver
selection. Following the user's explicit instruction, it was replaced with an
independently generated fake PA capture, labeled **dummy dataset for tutorial
purpose** and `origin: synthetic`. All measured APA and DPA dataset files remain
unchanged; `git diff` for those four directories is empty.

The [tutorial README](../../datasets/MyCustomPA/README.md) and
[generator](../../datasets/MyCustomPA/generate.py) explain the CSV and signal:
40 IFFT frames, 2,560 samples/frame, 64 active bins, no CP, 80 MHz sample rate,
2 MHz channel, followed by illustrative memoryless compression and phase.
Each frame uses all 64 QAM points once, shuffled with the recorded generator seed.
An independent grid check verifies the recovered tutorial input symbols.

The catalog and manifest expose the synthetic/tutorial label. If a workspace
already contains an older packaged copy, the new content hash receives a separate
stable dataset ID; old raw data and manifests are retained. See the
[Chinese tutorial workbench](studio-datasets-2026-09-12/tutorial-zh-1920.png).

## Create Your Own Dataset

The new button opens three steps: **CSV & columns → Split & metadata → Review**.
Accept either two complex columns (`input,output`, with `i` or `j`) or four real
columns (`I_in,Q_in,I_out,Q_out`). UTF-8 BOM, scientific notation, optional headers
and explicit column mapping are supported. Each row must be a paired sample.

Validation scans the entire file, not only the five preview rows. Missing fields,
ragged rows, invalid numbers, NaN/Inf, float32 overflow, duplicated column roles,
bad encoding and malformed CSV are rejected with line/column context and repair
instructions. The displayed error list is bounded to 20 entries. Fix instructions
are localized; the server's exact technical cause remains visible.
Changing column roles or split settings invalidates the corresponding green check.
Creation validates again against a copied source and the reviewed SHA-256, then
uses the existing materialization service. Original bytes and hashes are retained.

The default train/val/test ratio is **60/20/20**, obtained from the shared core.
Users can change it. The existing default guard is 256 samples at each boundary;
percentages apply to the usable samples after those two guards. The review shows
actual counts and half-open ranges. No protected split definition was changed.

Two complete real-browser checks were performed:

- Four real columns: 102,400 rows, 70/20/10, counts 71,321 / 20,377 / 10,190.
  The created dataset then completed the real three-epoch CPU GRU smoke recipe
  in about 5.3 seconds. [Run record](studio-datasets-2026-09-12/csv-smoke-run.json),
  [worker log](studio-datasets-2026-09-12/csv-smoke-worker.txt),
  [created manifest](studio-datasets-2026-09-12/csv-created-manifest.json).
- Two complex columns: a NaN at line 901 blocked progression and showed
  [repair guidance in Chinese](studio-datasets-2026-09-12/create-invalid-zh-1920.png).
  Replacing the file allowed creation of all 8,192 rows with the default ratio:
  4,608 / 1,536 / 1,536. Imported arrays matched the shared parser exactly.
  [Review screenshot](studio-datasets-2026-09-12/create-review-zh-1366.png),
  [array/split verification](studio-datasets-2026-09-12/complex-csv-live.json).

Unknown imported waveforms show raw I/Q with an explanation. A filename or
modulation label alone does not bind a demodulator. Optional physical metadata
can be added during creation or later; missing sample rate means normalized
frequency, and experiment readiness continues to use the shared validation.

The same CSV service is available from Python and CLI, for example:

```sh
opendpd datasets import capture.csv --workspace ./workspace --inspect-csv
opendpd datasets import capture.csv --workspace ./workspace \
  --csv-format auto --ratios 0.7 0.2 0.1 --origin measured
```

Use `--csv-header present|absent|auto` and `--csv-map input=0 --csv-map output=1`
when the complex-column roles need explicit assignment. Real-column roles use
`I_in`, `Q_in`, `I_out`, and `Q_out`, with zero-based indices.

## Verification and limits

| Check | Command / evidence | Result |
|---|---|---|
| Real backend regressions | `.venv/bin/python -m pytest tests/integration/test_datasets_api.py tests/integration/test_dataset_analysis_api.py tests/integration/test_dataset_csv_creation.py -q` | [33 passed](studio-datasets-2026-09-12/backend-tests.txt), 104 s |
| Frontend suite | `npm --prefix frontend test` | [92 passed](studio-datasets-2026-09-12/frontend-tests.txt) |
| Browser regression gate | `npm --prefix frontend run e2e -- --project=chromium-1366 --project=chromium-1920 --ignore-snapshots` | [22 passed](studio-datasets-2026-09-12/browser-tests.txt), six opt-in live/performance tests skipped; real API evidence is separately recorded above |
| Build / lint / contracts | `npm --prefix frontend run build`, `run lint`, `run types:check` | Passed; logs alongside this report |
| Live accessibility | axe WCAG 2.0/2.1 A/AA tags, English review 1366, Chinese review 1366, Chinese tutorial 1920 | Zero violations; `create-review-axe.json`, `create-zh-axe.json`, `tutorial-zh-axe.json` |
| Scientific scope | `git diff --name-only` on protected paths and measured datasets | Empty |

Initial implementation checks caught MUI prop/type mismatches and a NumPy warning
in the invalid-input range check; these were corrected without changing any
scientific expected value or tolerance. Existing FastAPI/TestClient deprecation
warnings remain in the backend log. Pixel-baseline assertions, Windows, GPU and
standard-conformance EVM are not claimed as verified.

No new dependency, protected metric/split change, release, external upload or RF
operation was introduced. Unused diagnostic screenshots and temporary test files
were removed after retaining the acceptance evidence. Existing Studio processes
need a restart to load the updated backend and built frontend.
