#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ -n "${OPENDPD_CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${OPENDPD_CONDA_ENV}"
  else
    echo "[WARN] OPENDPD_CONDA_ENV is set but 'conda' was not found on PATH. Skipping activation." >&2
  fi
elif [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[INFO] Activate your Python environment before running this script (set OPENDPD_CONDA_ENV to auto-activate)." >&2
fi

PYTHON_BIN=${PYTHON:-python}

cd "${REPO_ROOT}"

dataset_name=${DATASET_NAME:-DPA_200MHz}
accelerator=${ACCELERATOR:-cpu}
devices=${DEVICES:-0}

seed_values=(0 1 2 3 4)
if [[ -n "${SEEDS:-}" ]]; then
  IFS=',' read -ra seed_values <<< "${SEEDS}"
fi
PA_backbone=(gru vdlstm rvtdcnn gmp)
PA_hidden_size=(11 8 12 8)
PA_num_layers=(1 1 1 1)

frame_length=${FRAME_LENGTH:-50}
frame_stride=${FRAME_STRIDE:-1}
loss_type=${LOSS_TYPE:-l2}
opt_type=${OPT_TYPE:-adamw}
batch_size=${BATCH_SIZE:-64}
batch_size_eval=${BATCH_SIZE_EVAL:-256}
n_epochs=${N_EPOCHS:-100}
lr_schedule=${LR_SCHEDULE:-1}
lr=${LR:-1e-3}
lr_end=${LR_END:-1e-6}
decay_factor=${DECAY_FACTOR:-0.5}
patience=${PATIENCE:-10}

for i_seed in "${seed_values[@]}"; do
  for ((i=0; i<${#PA_backbone[@]}; i++)); do
    step=train_pa
    "${PYTHON_BIN}" main.py \
      --dataset_name "${dataset_name}" \
      --seed "${i_seed}" \
      --step "${step}" \
      --accelerator "${accelerator}" \
      --devices "${devices}" \
      --PA_backbone "${PA_backbone[$i]}" \
      --PA_hidden_size "${PA_hidden_size[$i]}" \
      --PA_num_layers "${PA_num_layers[$i]}" \
      --frame_length "${frame_length}" \
      --frame_stride "${frame_stride}" \
      --loss_type "${loss_type}" \
      --opt_type "${opt_type}" \
      --batch_size "${batch_size}" \
      --batch_size_eval "${batch_size_eval}" \
      --n_epochs "${n_epochs}" \
      --lr_schedule "${lr_schedule}" \
      --lr "${lr}" \
      --lr_end "${lr_end}" \
      --decay_factor "${decay_factor}" \
      --patience "${patience}" || exit 1
  done

done
