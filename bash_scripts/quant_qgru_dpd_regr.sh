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

resolve_pretrained_model() {
  local pattern=$1
  local dir="$(dirname "${pattern}")"
  local base="$(basename "${pattern}")"

  if [[ -f "${pattern}" ]]; then
    echo "${pattern}"
    return 0
  fi

  if [[ -d "${pattern}" ]]; then
    local candidate
    candidate=$(ls -1t "${pattern}"/*.pt 2>/dev/null | head -n1 || true)
    [[ -n "${candidate}" ]] && { echo "${candidate}"; return 0; }
  fi

  if [[ -d "${dir}" ]]; then
    local candidate
    candidate=$(ls -1t "${dir}"/"${base}"*.pt 2>/dev/null | head -n1 || true)
    [[ -n "${candidate}" ]] && { echo "${candidate}"; return 0; }
  fi

  return 1
}

log_info()  { printf '[INFO] %s\n'  "$*"; }
log_warn()  { printf '[WARN] %s\n'  "$*" >&2; }
log_error() { printf '[ERROR] %s\n' "$*" >&2; }

# Global Settings
dataset_name=${DATASET_NAME:-DPA_160MHz}
accelerator=${ACCELERATOR:-cpu}
devices=${DEVICES:-0}

# Hyperparameters
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

seed_values=(0)
PA_backbone=dgru
PA_hidden_size=8
PA_num_layers=1
DPD_backbone=(qgru qgru qgru qgru qgru)
DPD_hidden_size=(6 9 13 20 30)
DPD_num_layers=(1 1 1 1 1)

quant_n_bits_w=${QUANT_BITS_W:-16}
quant_n_bits_a=${QUANT_BITS_A:-16}
quant_opts=(--quant --n_bits_w "${quant_n_bits_w}" --n_bits_a "${quant_n_bits_a}")

# Optional: force creation of float models before QAT
q_pretrain=${Q_PRETRAIN:-True}

run_python() {
  "${PYTHON_BIN}" main.py "$@" || exit 1
}

for i_seed in "${seed_values[@]}"; do
  for ((i=0; i<${#DPD_backbone[@]}; i++)); do
    if [[ "${q_pretrain}" == "True" ]]; then
      log_info "Pre-training (float) DPD backbone ${DPD_backbone[$i]} hidden=${DPD_hidden_size[$i]}"
      run_python \
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
        --patience "${patience}"
    fi

    quant_dir_label="amp_qgru_h${DPD_hidden_size[$i]}_w${quant_n_bits_w}a${quant_n_bits_a}"
    float_model_path="./save/${dataset_name}/train_dpd"
    float_model_pattern="${float_model_path}/DPD_S_${i_seed}_M_${DPD_backbone[$i]^^}_H_${DPD_hidden_size[$i]}_F_${frame_length}"
    pretrained_model=""
    if pretrained_model=$(resolve_pretrained_model "${float_model_pattern}"); then
      log_info "Using pretrained model: ${pretrained_model}"
    else
      log_warn "No pretrained model found matching ${float_model_pattern}*.pt"
      pretrained_model=""
    fi

    log_info "Quantized aware training (${quant_dir_label})"
    run_python \
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
      "${quant_opts[@]}"

    log_info "Running DPD inference (${quant_dir_label})"
    run_python \
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
      "${quant_opts[@]}"
  done

done
