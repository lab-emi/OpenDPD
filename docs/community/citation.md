# Citing OpenDPD, a board version and an entry

**The software.** `CITATION.cff` at the repository root is what GitHub's
"Cite this repository" reads; it names the software (version, licence,
repository) and the preferred paper:

> Y. Wu, G. D. Singh, M. Beikmirza, L. C. N. de Vreede, M. Alavi and C. Gao,
> "OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for
> Wideband Power Amplifier Modeling and Digital Pre-Distortion," *2024 IEEE
> International Symposium on Circuits and Systems (ISCAS)*, 2024, pp. 1–5,
> doi:10.1109/ISCAS58744.2024.10558162.

BibTeX for the framework, MP-DPD and DeltaDPD is collected below. Related
work includes [OpenDPDv2](https://arxiv.org/abs/2507.06849) and
[TCN-DPD](https://arxiv.org/abs/2506.12165); use the publication metadata
for the version you cite.

**A board version.** A board is a file whose hash covers every entry; cite
the board id, the version, the hash and the date you read it, so a reader
finds exactly what you saw even after later versions exist:

> OpenDPD leaderboard `opendpd-pa-modeling`, version v2026.09, board hash
> `<board_sha256>`, `docs/leaderboard/v2026.09/pa_modeling.md`, accessed
> 2026-09-06.

The hash is the last line of the Markdown rendering and the `board_sha256`
field of the JSON. Say the label the board carried (reference benchmark or
community leaderboard): the label is computed from the entries and can
change between versions.

**An entry.** Cite the submitter's own `citation` statement (it is on the
card and in the board JSON), plus the entry id and the board version. A
retracted or corrected entry keeps its history on the board; cite the
version you used.

**Data.** The built-in datasets are cited through the papers above; a
submission's own data is cited as its data card says.

## Paper BibTeX

If you find this repository helpful, please cite our work:

- \[ISCAS 2024\] [OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion](https://ieeexplore.ieee.org/abstract/document/10558162)
```
@INPROCEEDINGS{Wu2024ISCAS,
  author={Wu, Yizhuo and Singh, Gagan Deep and Beikmirza, Mohammadreza and de Vreede, Leo C. N. and Alavi, Morteza and Gao, Chang},
  booktitle={2024 IEEE International Symposium on Circuits and Systems (ISCAS)},
  title={OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion},
  year={2024},
  volume={},
  number={},
  pages={1-5},
  keywords={Codes;Transmitters;OFDM;Power amplifiers;Artificial neural networks;Documentation;Benchmark testing;digital pre-distortion;behavioral modeling;deep neural network;power amplifier;digital transmitter},
  doi={10.1109/ISCAS58744.2024.10558162}}
```

- \[IMS/MWTL 2024\] [MP-DPD: Low-Complexity Mixed-Precision Neural Networks for Energy-Efficient Digital Pre-distortion of Wideband Power Amplifiers](https://ieeexplore.ieee.org/document/10502240)
```
@ARTICLE{Wu2024IMS,
  author={Wu, Yizhuo and Li, Ang and Beikmirza, Mohammadreza and Singh, Gagan Deep and Chen, Qinyu and de Vreede, Leo C. N. and Alavi, Morteza and Gao, Chang},
  journal={IEEE Microwave and Wireless Technology Letters},
  title={MP-DPD: Low-Complexity Mixed-Precision Neural Networks for Energy-Efficient Digital Predistortion of Wideband Power Amplifiers},
  year={2024},
  volume={},
  number={},
  pages={1-4},
  keywords={Deep neural network (DNN);digital predistortion (DPD);digital transmitter (DTX);power amplifier (PA);quantization},
  doi={10.1109/LMWT.2024.3386330}}
```

- \[IMS/MWTL 2025\] [DeltaDPD: Exploiting Dynamic Temporal Sparsity in Recurrent Neural Networks for Energy-Efficient Wideband Digital Predistortion](https://ieeexplore.ieee.org/abstract/document/11006082/)
```
@article{Wu2025MWTL,
   title={DeltaDPD: Exploiting Dynamic Temporal Sparsity in Recurrent Neural Networks for Energy-Efficient Wideband Digital Predistortion},
   ISSN={2771-957X},
   url={http://dx.doi.org/10.1109/LMWT.2025.3565004},
   DOI={10.1109/lmwt.2025.3565004},
   journal={IEEE Microwave and Wireless Technology Letters},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Wu, Yizhuo and Zhu, Yi and Qian, Kun and Chen, Qinyu and Zhu, Anding and Gajadharsing, John and de Vreede, Leo C. N. and Gao, Chang},
   year={2025},
   pages={1–4} }
```
