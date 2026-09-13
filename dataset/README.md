# Shared dataset catalog

This directory holds datasets in one validated, data-only format. Existing
`datasets/` benchmark layouts remain readable for compatibility.

- `synthetic/studio-research-v1/`: generated examples with three normalized drive
  settings and two synthetic realizations per setting. No physical measurements.
- `community/<origin>/<dataset-id>/<package-sha256>/`: user-selected public data
  contributions. A human reviews each PR before deciding whether to merge it.

Each entry contains `data.csv` (I_in,Q_in,I_out,Q_out), `dataset.json` (signal,
split, license, attribution, provenance and full hashes), and `README.md`.
Private workspace paths, executables, credentials and private notes do not
belong in this catalog. Published PRs are public before they are merged.

The Studio upload wizard defaults to private. Selecting public review opens an
exact package preview and license/rights confirmation. Studio creates an
isolated branch, pushes it using the operator's GitHub CLI account (forking when
needed), and opens a PR. It never automatically merges. Failed submissions can
resume the same branch and recover an existing PR.

After merge, an updated Studio checkout discovers the entry in **Add built-in
dataset**. Existing installations receive new entries when they update. Origin
is the contributor's declaration, not independent measurement certification.

Questions about datasets or review: [emi.lab@outlook.com](mailto:emi.lab@outlook.com).

Regenerate the synthetic fixtures from the repository root:

```bash
python scripts/studio_synthetic_datasets.py --catalog dataset/synthetic/studio-research-v1
```
