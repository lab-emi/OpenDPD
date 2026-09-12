# What's new

## 2.2 development preview: OpenDPD Studio

Studio adds a local desktop/browser workbench with guided PA and DPD experiments, live plots and step-specific terminals, reusable configurations, result comparisons and export packages. The interface supports nine languages and system light/dark themes. Custom dataset upload in the UI is marked **Coming soon**.

Install this preview [from source](install.md); the published PyPI 2.1.0 package does not include Studio. Start with the [walkthrough](tutorials/gui-quickstart.md). Advanced evaluation and export workflows are indexed in [Advanced guides](advanced.md), and verification evidence is tracked in the [support matrix](releases/support-matrix.md).

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
