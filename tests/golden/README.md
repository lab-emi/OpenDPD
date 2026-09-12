# Golden references (protected path)

Everything in this directory freezes scientific behaviour. Edits need a
separate PR with the `science-review-approved` label (see `AGENTS.md` §3).

| File | What it freezes | Regenerate |
|---|---|---|
| `legacy_metrics_v1.json` | `utils.metrics` NMSE / EVM / ACLR and `set_target_gain` on a deterministic multi-tone stimulus for the DPA_200MHz and APA_200MHz spec configurations. Metric profile id: `legacy-opendpd-v1`. | `python tests/golden/generate_legacy_metric_goldens.py` (only after approval) |
| `legacy_checkpoints/PA_S_0_M_GRU_H_23_F_50_P_1911.pt` | A real checkpoint written by the legacy pipeline (`state_dict` of `models.CoreModel`, GRU, hidden 23). Guarantees new code keeps loading historical checkpoints with `weights_only=True`. | see provenance below |
| `legacy_checkpoints/PA_S_0_M_GRU_H_23_F_50_P_1911.csv` | The `log/.../best` row written next to that checkpoint. | same run |

## Checkpoint provenance

- Command (run from an empty working directory):
  `python main.py --step train_pa --n_epochs 2 --dataset_name DPA_200MHz --accelerator cpu --frame_length 50 --frame_stride 16 --batch_size 64 --batch_size_eval 256`
- Commit: `7426bbf8a47624b59bd7f045a86641b403023f3c` (S00 baseline)
- Environment: Linux x86-64, Python 3.13.14, torch 2.13.0+cu132 (CPU execution), numpy 2.4.4
- SHA-256: `f4e2cac55f1e479070b4b839f084d3892e4b22b4c4f1c15c972e01c3b1540672`
- Purpose: compatibility fixture only (2-epoch smoke); it is **not** a benchmark model.
