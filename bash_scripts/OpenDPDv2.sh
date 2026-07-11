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

dataset_name=${DATASET_NAME:-APA_200MHz}
if [[ ! -d "datasets/${dataset_name}" ]]; then
  echo "[ERROR] Dataset directory 'datasets/${dataset_name}' not found. Set DATASET_NAME to one of:" >&2
  ( cd datasets && ls -1d */ 2>/dev/null | sed 's#/$##' | grep -v -e '^__pycache__$' -e '^MATLAB$' || true ) >&2
  exit 1
fi

accelerator=${ACCELERATOR:-cpu}
devices=${DEVICES:-0}

frame_length=${FRAME_LENGTH:-200}
frame_stride=${FRAME_STRIDE:-1}
loss_type=${LOSS_TYPE:-l2}
opt_type=${OPT_TYPE:-adamw}
batch_size=${BATCH_SIZE:-64}
batch_size_eval=${BATCH_SIZE_EVAL:-256}
n_epochs=${N_EPOCHS:-240}
lr_schedule=${LR_SCHEDULE:-1}
lr=${LR:-5e-3}
lr_end=${LR_END:-1e-4}
decay_factor=${DECAY_FACTOR:-0.5}
patience=${PATIENCE:-10}
cuda_graph_training=${CUDA_GRAPH_TRAINING:-1}
cuda_graph_opts=()
if [[ "${cuda_graph_training}" == "1" ]]; then
  cuda_graph_opts+=(--cuda_graph_training)
fi

seed_values=(0)
PA_backbone=dgru
PA_hidden_size=23
PA_num_layers=1
DPD_backbone=(tres_deltagru)
DPD_hidden_size=(15)
DPD_num_layers=(1)
thx=${THX:-0.01}
thh=${THH:-0.05}
quant_n_bits_w=${QUANT_BITS_W:-16}
quant_n_bits_a=${QUANT_BITS_A:-16}
quant_opts=(--quant --n_bits_w "${quant_n_bits_w}" --n_bits_a "${quant_n_bits_a}")

for i_seed in "${seed_values[@]}"; do
  # Train the PA model per seed with the full architecture and training config;
  # the DPD stage looks up the PA checkpoint by seed.
  printf '\033[32m==== Train PA model (seed=%s, dataset=%s, backbone=%s) ====\033[0m\n' "${i_seed}" "${dataset_name}" "${PA_backbone}"
  "${PYTHON_BIN}" main.py \
    --dataset_name "${dataset_name}" \
    --seed "${i_seed}" \
    --step train_pa \
    --accelerator "${accelerator}" \
    --devices "${devices}" \
    --PA_backbone "${PA_backbone}" \
    --PA_hidden_size "${PA_hidden_size}" \
    --PA_num_layers "${PA_num_layers}" \
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
    --patience "${patience}"

  for ((i=0; i<${#DPD_backbone[@]}; i++)); do
    printf '\033[32m==== Pre-training DPD (seed=%s, backbone=%s) ====\033[0m\n' "${i_seed}" "${DPD_backbone[$i]}"
    "${PYTHON_BIN}" main.py \
      --dataset_name "${dataset_name}" \
      --seed "${i_seed}" \
      --step train_dpd \
      --accelerator "${accelerator}" \
      --devices "${devices}" \
      --PA_backbone "${PA_backbone}" \
      --PA_hidden_size "${PA_hidden_size}" \
      --PA_num_layers "${PA_num_layers}" \
      --DPD_backbone "${DPD_backbone[$i]}" \
      --DPD_hidden_size "${DPD_hidden_size[$i]}" \
      --DPD_num_layers "${DPD_num_layers[$i]}" \
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
      --patience "${patience}" \
      --thx "${thx}" \
      --thh "${thh}" \
      "${cuda_graph_opts[@]}"

    quant_dir_label="w${quant_n_bits_w}a${quant_n_bits_a}"

    # The dense DPD checkpoint is saved under the PA-model subfolder. Resolve it
    # and require it to exist so QAT fine-tunes it instead of training anew.
    pa_subdir="PA_S_${i_seed}_M_${PA_backbone^^}_H_${PA_hidden_size}_F_${frame_length}"
    pretrained_pattern="./save/${dataset_name}/train_dpd/${pa_subdir}/DPD_S_${i_seed}_M_${DPD_backbone[$i]^^}_H_${DPD_hidden_size[$i]}_F_${frame_length}"
    pretrained_model=$(ls -1t ${pretrained_pattern}*.pt 2>/dev/null | head -n1 || true)
    if [[ -z "${pretrained_model}" || ! -f "${pretrained_model}" ]]; then
      echo "[ERROR] Pretrained dense DPD model not found for pattern ${pretrained_pattern}*.pt" >&2
      exit 1
    fi
    printf '[INFO] QAT will fine-tune from: %s\n' "${pretrained_model}"

    printf '\033[32m==== Quantized aware training (label %s) ====\033[0m\n' "${quant_dir_label}"
    "${PYTHON_BIN}" main.py \
      --dataset_name "${dataset_name}" \
      --seed "${i_seed}" \
      --step train_dpd \
      --accelerator "${accelerator}" \
      --devices "${devices}" \
      --PA_backbone "${PA_backbone}" \
      --PA_hidden_size "${PA_hidden_size}" \
      --PA_num_layers "${PA_num_layers}" \
      --DPD_backbone "${DPD_backbone[$i]}" \
      --DPD_hidden_size "${DPD_hidden_size[$i]}" \
      --DPD_num_layers "${DPD_num_layers[$i]}" \
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
      --patience "${patience}" \
      --quant_dir_label "${quant_dir_label}" \
      --pretrained_model "${pretrained_model}" \
      --thx "${thx}" \
      --thh "${thh}" \
      "${quant_opts[@]}"

    printf '\033[32m==== Run DPD (label %s) ====\033[0m\n' "${quant_dir_label}"
    "${PYTHON_BIN}" main.py \
      --dataset_name "${dataset_name}" \
      --seed "${i_seed}" \
      --step run_dpd \
      --accelerator "${accelerator}" \
      --devices "${devices}" \
      --PA_backbone "${PA_backbone}" \
      --PA_hidden_size "${PA_hidden_size}" \
      --PA_num_layers "${PA_num_layers}" \
      --DPD_backbone "${DPD_backbone[$i]}" \
      --DPD_hidden_size "${DPD_hidden_size[$i]}" \
      --DPD_num_layers "${DPD_num_layers[$i]}" \
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
      --patience "${patience}" \
      --quant_dir_label "${quant_dir_label}" \
      --pretrained_model "${pretrained_model}" \
      --thx "${thx}" \
      --thh "${thh}" \
      "${quant_opts[@]}"
  done

done
