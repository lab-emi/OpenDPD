# What's new

## 2.2.7: more workspaces and an automatic waiting room

Hosted capacity grows from 16 to 256 workspaces with up to 1,024 waiting tickets. A full service shows a position and admits visitors automatically. One shared scheduler replaces per-visitor background loops, while GPU jobs keep their own bounded FIFO queue. Workspaces now expire at 11:55 and 23:55 UTC; Server load includes waiting visitors and an explicit End workspace action.

[Release notes](releases/release-notes-2.2.7.md) · [Waiting room and cleanup](guides/server-load.md) · [Capacity validation](performance/studio-2.2.7.md)

## 2.2.6: server load and stronger isolation

- Live aggregate CPU, RAM, GPU and queue status, with an active-session estimate and clear stale-data handling.
- One persistent UTC cleanup timestamp, including a compact mobile layout.
- Stricter filesystem identifiers and request boundaries, serialized workspace writes, bounded shared analysis and reserved temporary storage.
- Updated runtime dependencies and GPU image, with dependency audits in CI.

[Release notes](releases/release-notes-2.2.6.md) · [Server load](guides/server-load.md) · [Security review](releases/security-review-2.2.6.md)

## 2.2.5: ILC, interactive LaTeX and simpler installation

- ILC + ILA DPD with a separate waveform-specific Ideal test reference, configurable bounds and convergence evidence.
- LaTeX Virtual PA equations with linked controls, and LaTeX metric definitions.
- Shared/per-channel OFDMA controls, explicit-only input generation and reversible input removal.
- Next-step controls above settings; clearer metric calculation labels; About without GitHub activity.
- Default PyTorch/Studio/pywebview dependencies, uv environment instructions, MPS selection and clearer localhost/SSH diagnostics.

[Release notes](releases/release-notes-2.2.5.md) · [Installation](install.md) · [ILC guide](guides/ilc-dpd.md)

## 2.2.4: Virtual PA Library and signal-chain PSD views

Signal Generator exports input-only I/Q. PA Library provides nine mathematical Virtual PAs, parameter-linked formulas, explicit output simulation and paired dataset creation. The expandable workflow diagram separates dataset making from PA/DPD model training and recognizes existing paired datasets.

PSD views now separate DPD input, DPD output / PA input, and PA output throughout inspection, live previews, results, comparisons and exported figures/reports. Compact legends retain synthetic/measured provenance, output baselines remain together, and saved views preserve per-position zoom.

[Release notes](releases/release-notes-2.2.4.md) · [PA Library](guides/virtual-pa-library.md) · [Signal-chain spectra](guides/signal-chain-spectra.md)


## 2.2.3: Signal Generator and research workflows

Start with Signal Generator, an existing dataset or your own CSV. PA Model and DPD Model each bring training and testing together, with exact test-set I/Q counts. The generator combines twenty presets, advanced waveform controls, signal plots and metrics, exact I/Q export and synthetic PA datasets. The standards-based presets are uncoded engineering stimuli; Wi-Fi 8 remains experimental.

Local Studio adds research review, saved publication figures, full reproduction bundles, measurement sessions, condition sweeps, qualified hardware costs and optional dataset PRs that await human review. See the [2.2.3 release notes](releases/release-notes-2.2.3.md), [generator guide](guides/signal-generator.md) and [research review guide](guides/research-review.md).

## 2.2: OpenDPD Studio

Studio provides a packaged desktop/browser workbench with guided PA and DPD experiments, live plots and terminals, reusable configurations, result comparisons and export packages. CSV upload, shared hosted CUDA compute and checkpoint downloads are available. Version 2.2.1 improved training speed; 2.2.2 added bug reporting from Studio.

[Install Studio](install.md) or start with the [walkthrough](tutorials/gui-quickstart.md). Verification evidence is tracked in the [support matrix](releases/support-matrix.md).

## 2.1: visualization

The original CLI and Python API gained per-epoch figures, animated GIFs, a training dashboard, training curves and without/with-DPD comparison plots. The [visualization guide](visualization.md) is the maintained reference for commands and options.

## Research generations: OpenDPDv1 and OpenDPDv2

The research names below describe algorithms and their paper experiments; they are separate from the Python package's release number. The performance values are historical paper results under their respective settings, not a ranking of every model available in Studio. See the [benchmark](benchmark/index.md) for comparable evidence and the [reproduction guide](reproducing.md) for scripts.

**OpenDPDv2** embeds a new temporal residual (TRes)-DeltaGRU NN DPD algorithm and a new TM3.1a 5-channel x 40 MHz (200 MHz) test signal dataset, measured from a 3.5 GHz Ampleon GaN PA at 41.5 dBm average output power, named APA_200MHz, enabling fast prototyping of accurate and power-efficient NN-based DPD algorithms by streamlining learning and optimization through DPD quantization and temporal sparsity exploitation.

| Version    | Related Papers|Dataset                                      |    Supported Backbones      | Performance <br> on APA_200MHz
|------------|----|-------------------|----------------------|---------|
| OpenDPDv1 | Algorithms <br>  [![paper](https://img.shields.io/badge/OpenDPD-ISCAS2024-orange)](https://ieeexplore.ieee.org/abstract/document/10558162) <br> [![paper](https://img.shields.io/badge/MP--DPD-MWTL2024-orange)](https://ieeexplore.ieee.org/document/10502240) <br> [![paper](https://img.shields.io/badge/DeltaDPD-MWTL2025-orange)](https://ieeexplore.ieee.org/abstract/document/11006082) <br> [![paper](https://img.shields.io/badge/TCN--DPD-IMS2025-orange)](https://www.arxiv.org/abs/2506.12165) <br> Hardware [![paper](https://img.shields.io/badge/DPD--NeuralEngine-ISCAS2025-blue)](https://ieeexplore.ieee.org/document/11043563) <br> [![paper](https://img.shields.io/badge/SparseDPD-FPL2025-blue)](https://arxiv.org/abs/2506.16591)|Collected from a 40nm CMOS DTX @ 2.4 GHz<br>DPA_100MHz,<br>DPA_160MHz,<br>DPA_200MHz | GRU,<br>LSTM,<br>GMP,<br>RVTDCNN, <br>VDLSTM,<br> DGRU,<br> TCN        | DGRU with 1041 params: <br> ACPR of -58.4 dBc, <br> EVM of -39.1 dB|
| OpenDPDv2  |   [![paper](https://img.shields.io/badge/OpenDPDv2-arXiv-red)](https://arxiv.org/abs/2507.06849)|Collected from a GaN Doherty @ 3.5 GHz<br>APA_200MHz,<br>APA_200MHz_b |  PGJANET,<br>DVRJANET,<br>TRes-DeltaGRU | TRes-DeltaGRU with 996 params: <br> ACPR of -59.4 dBc, <br> EVM of -42.1 dB|
|Experiment code|[![paper](https://img.shields.io/badge/OpenDPDv2-arXiv-red)](https://arxiv.org/abs/2507.06849)[![paper](https://img.shields.io/badge/QianWu-UCD-red)](https://ieeexplore.ieee.org/author/37088931208)| Controlling, <br>I/Q data upload/download MATLAB code | Rohde & Schwarz SMW200A, <br> Keysight N9042B,| Matlab/.m
