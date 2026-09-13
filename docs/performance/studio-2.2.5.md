# Studio 2.2.5 validation

Validation date: 2026-09-14. Synthetic runs establish software behavior, not physical RF performance.

## Exercised workflows

- Actual Chromium UI at 1366×768 and 1920×1080: explicit generation, custom independent OFDMA channels, input CSV/JSON, Virtual PA simulation, exact paired exports, dataset creation, PA/DPD training and testing. Eight CPU jobs succeeded; each dataset contains 32,768 pairs and 6,452 test samples.
- All nine Virtual PA catalogs render without LaTeX errors. Selecting parameters highlights matching coefficients. PA Input Dataset removal and Undo pass through the actual API.
- Default ILC/ILA settings execute through the GUI on CPU and CUDA, with 19,353 available training samples and 6,452 separate test samples. Both stop when improvement falls below tolerance. On this one-epoch smoke surrogate, pooled Ideal tracking NMSE improves from approximately −3.52 dB to −19.85 dB; it does **not** reach the −45 dB target. This is a workflow check, not a model-quality benchmark.
- ILC PSDs contain one DPD Input trace, two PA Input traces and five PA Output traces. The Ideal waveform is explicitly labelled. Ordinary PSD review at 1366×768, 1920×1080 and 390×844 retains exact saved bins and independent zoom/legend controls.
- An isolated Python 3.12 installation resolves all default dependencies and runs CPU tensor arithmetic with PyTorch 2.14.0+cpu, pywebview 6.2.1 and Qt/WebEngine 6.11.0. Separate `uv --torch-backend=auto` resolution detects this host's CUDA driver and selects the CUDA 13.2 wheel. The installed GUI serves from outside the source repository.
- Chromium mock journeys and accessibility checks: 24 passed, six optional live cases skipped; the actual-worker workflows above exercise the service separately. Packaging, golden metrics and ILC integration: 10 passed before the final full regression.
- LaTeX adversarial tests reject URLs, resource loads, injected markup and arbitrary HTML classes/styles. The remaining frontend stays under the HTML-sink scanner.

Records: [local workflow](studio-2.2.5/local-workflow.json), [ILC and formula checks](studio-2.2.5/ilc.json), [spectral checks](studio-2.2.5/signal-chain.json), [screenshots](studio-2.2.5/screenshots.json).

## Release verification

The PR checks the final Python/frontend regression, distribution metadata, API contract, documentation build and default installs on Linux/macOS/Windows. Release/deployment evidence is attached to the GitHub release after merged source, public API, GPU image, website and PyPI package are checked together.

Native windows on all three platforms and MPS hardware are not claimed by a Linux smoke check. See the [support matrix](../releases/support-matrix.md).
