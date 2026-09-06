# Citing OpenDPD, a board version and an entry

**The software.** `CITATION.cff` at the repository root is what GitHub's
"Cite this repository" reads; it names the software (version, licence,
repository) and the preferred paper:

> Y. Wu, G. D. Singh, M. Beikmirza, L. C. N. de Vreede, M. Alavi and C. Gao,
> "OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for
> Wideband Power Amplifier Modeling and Digital Pre-Distortion," *2024 IEEE
> International Symposium on Circuits and Systems (ISCAS)*, 2024, pp. 1–5,
> doi:10.1109/ISCAS58744.2024.10558162.

The BibTeX entries of this and the follow-up papers (OpenDPDv2, MP-DPD,
DeltaDPD, TCN-DPD) are in the README.

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
