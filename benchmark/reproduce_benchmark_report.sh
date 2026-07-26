#!/usr/bin/env bash
#
# Reproduce benchmark/benchmark_report.md from the two bundled RF datasets.
#
# The benchmark matrix is fixed before execution:
#   PA (~2,700 real parameters): MP, GMP, GRU, TRes-GRU,
#                                TRes-DeltaGRU (THX=THH=0)
#   DPD (~1,000 real parameters): MP/ILA, GMP/ILA, GRU, TRes-GRU
#   PA-surrogate sensitivity: TRes-DeltaGRU DPD (THX=THH=0) trained
#                             independently through TRes-GRU and
#                             TRes-DeltaGRU PA models
#
# All neural jobs use the same 300-epoch recipe.  The primary DPD comparison
# uses the dataset's validation-selected TRes-GRU PA; the sensitivity runs use
# each of the two named PA surrogates.
#
# Default invocation:
#
#   bash benchmark/reproduce_benchmark_report.sh
#
# Evidence is written to an immutable timestamped directory under
# benchmark/results/reproduced/. Architecture-addressed save/log artifacts are
# never overwritten unless --overwrite is supplied; in that case, the exact
# colliding files are archived into the run directory first.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

dry_run=0
overwrite=0
device=${DEVICE:-0}
output_dir=""

usage() {
  echo "Usage: $0 [--dry-run] [--overwrite] [--device INDEX] [--output-dir PATH]"
  echo
  echo "  --dry-run          Record and print every job without executing it."
  echo "  --overwrite        Archive, then replace exact colliding save/log files."
  echo "  --device INDEX     CUDA device index (default: ${device})."
  echo "  --output-dir PATH  Run-specific evidence directory."
}

while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    --device)
      if (($# < 2)); then
        echo "[ERROR] --device requires an index." >&2
        exit 2
      fi
      device=$2
      shift 2
      ;;
    --output-dir)
      if (($# < 2)); then
        echo "[ERROR] --output-dir requires a path." >&2
        exit 2
      fi
      output_dir=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${OPENDPD_PYTHON:-}" ]]; then
  python_bin=${OPENDPD_PYTHON}
elif [[ -n "${PYTHON:-}" ]]; then
  python_bin=${PYTHON}
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  python_bin=${REPO_ROOT}/.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  python_bin=$(command -v python3)
else
  echo "[ERROR] Python was not found. Set OPENDPD_PYTHON." >&2
  exit 1
fi

if [[ "${python_bin}" == */* ]]; then
  if [[ "${python_bin}" != /* ]]; then
    python_dir=$(cd "$(dirname "${python_bin}")" && pwd)
    python_bin="${python_dir}/$(basename "${python_bin}")"
  fi
  if [[ ! -x "${python_bin}" ]]; then
    echo "[ERROR] Python interpreter is not executable: ${python_bin}" >&2
    exit 1
  fi
else
  resolved_python=$(type -P -- "${python_bin}" || true)
  if [[ -z "${resolved_python}" ]]; then
    echo "[ERROR] Python interpreter was not found on PATH: ${python_bin}" >&2
    exit 1
  fi
  python_bin=${resolved_python}
fi

if [[ ! "${device}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --device must be a non-negative integer: ${device}" >&2
  exit 2
fi

if [[ -z "${output_dir}" ]]; then
  run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
  output_dir="${REPO_ROOT}/benchmark/results/reproduced/${run_id}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${REPO_ROOT}/${output_dir}"
fi

benchmark_lock_fd=""
if ((dry_run == 0)); then
  if ! command -v flock >/dev/null 2>&1; then
    echo "[ERROR] flock is required to serialize benchmark reproductions." >&2
    exit 1
  fi
  if ! benchmark_lock_path=$(
    git -C "${REPO_ROOT}" rev-parse --git-path opendpd-benchmark-report.lock
  ); then
    echo "[ERROR] Could not resolve the repository benchmark lock path." >&2
    exit 1
  fi
  if [[ "${benchmark_lock_path}" != /* ]]; then
    benchmark_lock_path="${REPO_ROOT}/${benchmark_lock_path}"
  fi
  benchmark_lock_dir=$(cd "$(dirname "${benchmark_lock_path}")" && pwd)
  benchmark_lock_path="${benchmark_lock_dir}/$(basename "${benchmark_lock_path}")"
  exec {benchmark_lock_fd}> "${benchmark_lock_path}"
  if ! flock -n "${benchmark_lock_fd}"; then
    echo "[ERROR] Another benchmark reproduction is already running for this repository." >&2
    echo "Lock: ${benchmark_lock_path}" >&2
    exit 1
  fi
fi

if [[ -e "${output_dir}" ]]; then
  echo "[ERROR] Output path already exists: ${output_dir}" >&2
  echo "Choose a new --output-dir or RUN_ID; run directories are immutable." >&2
  exit 1
fi

for dataset in APA_200MHz DPA_160MHz; do
  if [[ ! -f "${REPO_ROOT}/datasets/${dataset}/spec.json" ]]; then
    echo "[ERROR] Required dataset is missing: datasets/${dataset}" >&2
    exit 1
  fi
done

git_commit_before=$(git -C "${REPO_ROOT}" rev-parse HEAD)
git_branch_before=$(git -C "${REPO_ROOT}" branch --show-current)
git_status_before=$(git -C "${REPO_ROOT}" status --short --untracked-files=all)

mkdir -p "$(dirname "${output_dir}")"
mkdir "${output_dir}"
mkdir "${output_dir}/logs"
commands_log="${output_dir}/commands.log"
jobs_log="${output_dir}/jobs.tsv"
pa_bindings_log="${output_dir}/pa_checkpoint_bindings.tsv"
: > "${commands_log}"
printf 'label\tstarted_at_utc\tfinished_at_utc\tduration_seconds\texit_status\n' > "${jobs_log}"
printf 'dataset\tdpd_model_key\tpa_model_key\tcheckpoint\tsha256\n' \
  > "${pa_bindings_log}"
printf '%s\n' "${git_commit_before}" > "${output_dir}/git_commit_before.txt"
printf '%s\n' "${git_branch_before}" > "${output_dir}/git_branch_before.txt"
printf '%s\n' "${git_status_before}" > "${output_dir}/git_status_before.txt"

cd "${REPO_ROOT}"

"${python_bin}" - "${output_dir}/recipe.json" <<'PY'
import json
from pathlib import Path
import sys

recipe = {
    "schema_version": 6,
    "datasets": ["APA_200MHz", "DPA_160MHz"],
    "parameter_count_convention": (
        "Neural parameters are real trainable scalars; each complex polynomial "
        "coefficient counts as two real degrees of freedom."
    ),
    "neural": {
        "optimizer": "AdamW",
        "optimizer_weight_decay": 0.01,
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "loss": "MSE",
        "epochs": 300,
        "batch_size": 64,
        "evaluation_batch_size": 64,
        "initial_learning_rate": 5e-3,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "scheduler_threshold": 1e-4,
        "scheduler_threshold_mode": "rel",
        "scheduler_cooldown": 0,
        "scheduler_eps": 1e-8,
        "minimum_learning_rate": 5e-5,
        "frame_length": 200,
        "frame_stride": 1,
        "gradient_clip": 200,
        "seed": 0,
        "reproducibility": "soft",
        "pa_scheduler_and_selection_metric": "validation NMSE",
        "dpd_scheduler_and_selection_metric": "validation ACLR_AVG",
    },
    "pa_models": {
        "mp": {"K": 9, "Q": 150, "real_parameters": 2700},
        "gmp": {
            "Ka": 5, "La": 30,
            "Kb": 4, "Lb": 30, "Mb": 5,
            "Kc": 4, "Lc": 30, "Mc": 5,
            "real_parameters": 2700,
        },
        "gru": {"hidden_size": 28, "real_parameters": 2746},
        "tres_gru": {"hidden_size": 27, "real_parameters": 2751},
        "tres_deltagru": {
            "hidden_size": 27,
            "real_parameters": 2751,
            "delta_thresholds": {"input": 0.0, "hidden": 0.0},
        },
    },
    "dpd_models": {
        "mp": {"K": 5, "Q": 100, "real_parameters": 1000, "method": "ILA"},
        "gmp": {
            "Ka": 5, "La": 20,
            "Kb": 4, "Lb": 20, "Mb": 3,
            "Kc": 4, "Lc": 20, "Mc": 2,
            "real_parameters": 1000,
            "method": "ILA",
        },
        "gru": {"hidden_size": 16, "real_parameters": 994},
        "tres_gru": {"hidden_size": 15, "real_parameters": 999},
    },
    "dpd_tres_deltagru_pa_surrogate_comparison": {
        "model": {
            "hidden_size": 15,
            "real_parameters": 999,
            "delta_thresholds": {"input": 0.0, "hidden": 0.0},
        },
        "pa_surrogates": {
            "tres_gru": {
                "hidden_size": 27,
                "real_parameters": 2751,
                "delta_thresholds": None,
            },
            "tres_deltagru": {
                "hidden_size": 27,
                "real_parameters": 2751,
                "delta_thresholds": {"input": 0.0, "hidden": 0.0},
            },
        },
    },
    "dpd_pa_surrogate": {
        "backbone": "tres_gru",
        "hidden_size": 27,
        "real_parameters": 2751,
    },
    "polynomial_solvers": {
        "default": {
            "implementation": "torch.linalg.lstsq",
            "mode": "gels",
            "driver": "gels",
            "dtype": "torch.complex64",
            "column_scaling": "l2",
            "regularization": None,
        },
        "gmp_pa": {
            "implementation": "torch.linalg.svd",
            "mode": "truncated_svd",
            "driver": "gesvdj",
            "dtype": "torch.complex64",
            "column_scaling": "l2",
            "regularization": {
                "type": "truncated_svd",
                "relative_cutoff": 1e-4,
            },
        },
        "segment_boundary_policy": "zero delay state at every nperseg boundary",
    },
}

path = Path(sys.argv[1])
path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
PY

record_command() {
  local label=$1
  shift
  printf '# %s\n' "${label}" >> "${commands_log}"
  printf '%q ' "$@" >> "${commands_log}"
  printf '\n\n' >> "${commands_log}"
}

print_command() {
  local label=$1
  shift
  printf '\n[%s]\n' "${label}"
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

run_step() {
  local label=$1
  shift
  record_command "${label}" "$@"
  print_command "${label}" "$@"
  if ((dry_run)); then
    return
  fi

  local started_at finished_at started_epoch finished_epoch duration status
  started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  started_epoch=$(date +%s)
  set +e
  "$@" 2>&1 | tee "${output_dir}/logs/${label}.log"
  status=${PIPESTATUS[0]}
  set -e
  finished_epoch=$(date +%s)
  finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  duration=$((finished_epoch - started_epoch))
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${started_at}" "${finished_at}" "${duration}" "${status}" \
    >> "${jobs_log}"
  if ((status != 0)); then
    echo "[ERROR] Job failed (${status}): ${label}" >&2
    exit "${status}"
  fi
}

copy_with_parents() {
  local destination=$1
  shift
  local path
  for path in "$@"; do
    [[ -f "${path}" ]] || continue
    mkdir -p "${destination}/$(dirname "${path}")"
    cp -p "${path}" "${destination}/${path}"
  done
}

sha256_path() {
  "${python_bin}" - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[ERROR] Required checkpoint is missing: {path}")
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

record_pa_binding() {
  local dataset=$1
  local dpd_model_key=$2
  local pa_model_key=$3
  local checkpoint=$4
  if ((dry_run)); then
    return
  fi
  local digest
  digest=$(sha256_path "${checkpoint}")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${dataset}" "${dpd_model_key}" "${pa_model_key}" \
    "${checkpoint}" "${digest}" \
    >> "${pa_bindings_log}"
}

# Only these exact architecture-addressed files can collide with this recipe.
target_artifacts=()
add_training_artifacts() {
  local dataset=$1
  local step=$2
  local parent=$3
  local model_id=$4
  local save_base="save/${dataset}/${step}"
  local log_base="log/${dataset}/${step}"
  if [[ -n "${parent}" ]]; then
    save_base="${save_base}/${parent}"
    log_base="${log_base}/${parent}"
  fi
  target_artifacts+=(
    "${save_base}/${model_id}.pt"
    "${log_base}/best/${model_id}.csv"
    "${log_base}/history/${model_id}.csv"
  )
}

for dataset in APA_200MHz DPA_160MHz; do
  add_training_artifacts "${dataset}" train_pa "" \
    PA_S_0_M_GRU_H_28_F_200_P_2746
  add_training_artifacts "${dataset}" train_pa "" \
    PA_S_0_M_TRES_GRU_H_27_F_200_P_2751
  add_training_artifacts "${dataset}" train_pa "" \
    PA_S_0_M_TRES_DELTAGRU_H_27_F_200_P_2751

  dpd_parent=PA_S_0_M_TRES_GRU_H_27_F_200
  add_training_artifacts "${dataset}" train_dpd "${dpd_parent}" \
    DPD_S_0_M_GRU_H_16_F_200_P_994
  add_training_artifacts "${dataset}" train_dpd "${dpd_parent}" \
    DPD_S_0_M_TRES_GRU_H_15_F_200_P_999
  add_training_artifacts "${dataset}" train_dpd "${dpd_parent}" \
    DPD_S_0_M_TRES_DELTAGRU_H_15_F_200_P_999_THX_0.000_THH_0.000

  delta_dpd_parent=PA_S_0_M_TRES_DELTAGRU_H_27_F_200
  add_training_artifacts "${dataset}" train_dpd "${delta_dpd_parent}" \
    DPD_S_0_M_TRES_DELTAGRU_H_15_F_200_P_999_THX_0.000_THH_0.000
done

existing_artifacts=()
for path in "${target_artifacts[@]}"; do
  if [[ -L "${path}" ]]; then
    echo "[ERROR] Refusing benchmark artifact symlink: ${path}" >&2
    exit 1
  elif [[ -e "${path}" && ! -f "${path}" ]]; then
    echo "[ERROR] Refusing non-regular benchmark artifact: ${path}" >&2
    exit 1
  elif [[ -f "${path}" ]]; then
    existing_artifacts+=("${path}")
  fi
done

if ((dry_run == 0 && ${#existing_artifacts[@]} > 0 && overwrite == 0)); then
  echo "[ERROR] Matching benchmark artifacts already exist and would be overwritten:" >&2
  printf '  %s\n' "${existing_artifacts[@]}" >&2
  echo "Re-run with --overwrite to archive these exact files before training." >&2
  exit 1
fi

if ((dry_run == 0 && ${#existing_artifacts[@]} > 0)); then
  copy_with_parents "${output_dir}/preexisting" "${existing_artifacts[@]}"
  echo "[INFO] Archived ${#existing_artifacts[@]} pre-existing artifact(s)."
fi

if ((dry_run == 0)); then
  "${python_bin}" - "${device}" <<'PY'
import sys

try:
    import numpy
    import pandas
    import scipy
    import torch
except ImportError as exc:
    raise SystemExit(f"[ERROR] Missing benchmark dependency: {exc}") from exc

from models import CoreModel
from utils.util import count_net_params

device_index = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA is required for the reference benchmark.")
if device_index >= torch.cuda.device_count():
    raise SystemExit(
        f"[ERROR] CUDA device {device_index} is unavailable; "
        f"detected {torch.cuda.device_count()} device(s)."
    )
device = torch.device(f"cuda:{device_index}")
print(f"[INFO] Python {sys.version.split()[0]}")
print(f"[INFO] PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
print(f"[INFO] CUDA device {device_index}: {torch.cuda.get_device_name(device)}")

expected = {
    ("gru", 28): 2746,
    ("tres_gru", 27): 2751,
    ("tres_deltagru", 27): 2751,
    ("gru", 16): 994,
    ("tres_gru", 15): 999,
    ("tres_deltagru", 15): 999,
}
for (backbone, hidden_size), expected_parameters in expected.items():
    network = CoreModel(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=1,
        backbone_type=backbone,
        thx=0.0,
        thh=0.0,
    ).to(device)
    actual_parameters = count_net_params(network)
    if actual_parameters != expected_parameters:
        raise SystemExit(
            f"[ERROR] {backbone}-H{hidden_size} has {actual_parameters} "
            f"parameters, expected {expected_parameters}."
        )
    sample = torch.randn(2, 200, 2, device=device, requires_grad=True)
    output = network(sample)
    output.square().mean().backward()
    if not torch.isfinite(output).all() or sample.grad is None:
        raise SystemExit(f"[ERROR] Non-finite smoke test for {backbone}-H{hidden_size}.")
    del network, sample, output

matrix = torch.randn(512, 16, dtype=torch.complex64, device=device)
target = torch.randn(512, 1, dtype=torch.complex64, device=device)
solution = torch.linalg.lstsq(matrix, target, driver="gels").solution
if not torch.isfinite(solution).all():
    raise SystemExit("[ERROR] CUDA complex least-squares smoke test failed.")
torch.cuda.synchronize(device)
print("[INFO] Neural and complex least-squares CUDA smoke tests passed.")
PY
fi

neural_common=(
  --PA_num_layers 1
  --DPD_num_layers 1
  --frame_length 200
  --frame_stride 1
  --n_epochs 300
  --opt_type adamw
  --lr 5e-3
  --lr_schedule 1
  --lr_end 5e-5
  --decay_factor 0.5
  --patience 5
  --batch_size 64
  --batch_size_eval 64
  --loss_type l2
  --grad_clip_val 200
  --thx 0
  --thh 0
  --log_precision 8
  --seed 0
  --re_level soft
  --eval_val 1
  --eval_test 1
  --accelerator cuda
  --devices "${device}"
)

run_step snapshot_context \
  "${python_bin}" -m benchmark.collect_benchmark_report \
  --output-dir "${output_dir}" --device "${device}" --snapshot-context

run_neural_pa() {
  local dataset=$1
  local slug=$2
  local backbone=$3
  local hidden_size=$4
  run_step "train_pa_${slug}_${backbone}" \
    "${python_bin}" main.py \
    --dataset_name "${dataset}" \
    --step train_pa \
    --PA_backbone "${backbone}" \
    --PA_hidden_size "${hidden_size}" \
    "${neural_common[@]}"
}

run_neural_dpd() {
  local dataset=$1
  local slug=$2
  local backbone=$3
  local hidden_size=$4
  local pa_model_key=$5
  local pa_backbone=$6
  local pa_hidden_size=$7
  local pa_model_id=$8
  local pa_checkpoint="save/${dataset}/train_pa/${pa_model_id}.pt"
  record_pa_binding \
    "${dataset}" "${backbone}" "${pa_model_key}" "${pa_checkpoint}"
  run_step "train_dpd_${slug}_${backbone}_via_${pa_model_key}_pa" \
    "${python_bin}" main.py \
    --dataset_name "${dataset}" \
    --step train_dpd \
    --PA_backbone "${pa_backbone}" \
    --PA_hidden_size "${pa_hidden_size}" \
    --DPD_backbone "${backbone}" \
    --DPD_hidden_size "${hidden_size}" \
    "${neural_common[@]}"
}

run_polynomial_pa() {
  local dataset=$1
  local slug=$2
  local polynomial=$3
  shift 3
  local solver_arguments=(--solver-mode gels)
  if [[ "${polynomial}" == "gmp" ]]; then
    solver_arguments=(--solver-mode truncated_svd --svd-rcond 1e-4)
  fi
  run_step "pa_model_${slug}_${polynomial}" \
    "${python_bin}" -m benchmark.benchmark_volterra \
    --task pa_modeling \
    --dataset-name "${dataset}" \
    --model "${polynomial}" \
    --solver-device "cuda:${device}" \
    --solver-dtype complex64 \
    "${solver_arguments[@]}" \
    --device "${device}" \
    --json-out "${output_dir}/benchmark_report_${slug}_pa_${polynomial}.json" \
    "$@"
}

run_polynomial_dpd() {
  local dataset=$1
  local slug=$2
  local polynomial=$3
  shift 3
  run_step "dpd_ila_${slug}_${polynomial}" \
    "${python_bin}" -m benchmark.benchmark_volterra \
    --task dpd_ila \
    --dataset-name "${dataset}" \
    --model "${polynomial}" \
    --solver-device "cuda:${device}" \
    --solver-dtype complex64 \
    --solver-mode gels \
    --device "${device}" \
    --pa-backbone tres_gru \
    --pa-hidden-size 27 \
    --pa-num-layers 1 \
    --pa-checkpoint \
      "save/${dataset}/train_pa/PA_S_0_M_TRES_GRU_H_27_F_200_P_2751.pt" \
    --json-out "${output_dir}/benchmark_report_${slug}_dpd_${polynomial}.json" \
    "$@"
}

for dataset_spec in "APA_200MHz apa" "DPA_160MHz dpa"; do
  read -r dataset slug <<< "${dataset_spec}"
  run_neural_pa "${dataset}" "${slug}" gru 28
  run_neural_pa "${dataset}" "${slug}" tres_gru 27
  run_neural_pa "${dataset}" "${slug}" tres_deltagru 27
done

for dataset_spec in "APA_200MHz apa" "DPA_160MHz dpa"; do
  read -r dataset slug <<< "${dataset_spec}"
  run_polynomial_pa "${dataset}" "${slug}" mp \
    --K 9 --Q 150
  run_polynomial_pa "${dataset}" "${slug}" gmp \
    --Ka 5 --La 30 --Kb 4 --Lb 30 --Mb 5 \
    --Kc 4 --Lc 30 --Mc 5
done

for dataset_spec in "APA_200MHz apa" "DPA_160MHz dpa"; do
  read -r dataset slug <<< "${dataset_spec}"
  run_neural_dpd \
    "${dataset}" "${slug}" gru 16 tres_gru tres_gru 27 \
    PA_S_0_M_TRES_GRU_H_27_F_200_P_2751
  run_neural_dpd \
    "${dataset}" "${slug}" tres_gru 15 tres_gru tres_gru 27 \
    PA_S_0_M_TRES_GRU_H_27_F_200_P_2751
  run_neural_dpd \
    "${dataset}" "${slug}" tres_deltagru 15 tres_gru tres_gru 27 \
    PA_S_0_M_TRES_GRU_H_27_F_200_P_2751
  run_neural_dpd \
    "${dataset}" "${slug}" tres_deltagru 15 \
    tres_deltagru tres_deltagru 27 \
    PA_S_0_M_TRES_DELTAGRU_H_27_F_200_P_2751
done

for dataset_spec in "APA_200MHz apa" "DPA_160MHz dpa"; do
  read -r dataset slug <<< "${dataset_spec}"
  run_polynomial_dpd "${dataset}" "${slug}" mp \
    --K 5 --Q 100
  run_polynomial_dpd "${dataset}" "${slug}" gmp \
    --Ka 5 --La 20 --Kb 4 --Lb 20 --Mb 3 \
    --Kc 4 --Lc 20 --Mc 2
done

if ((dry_run)); then
  record_command collect_report \
    "${python_bin}" -m benchmark.collect_benchmark_report \
    --output-dir "${output_dir}" --device "${device}"
  print_command collect_report \
    "${python_bin}" -m benchmark.collect_benchmark_report \
    --output-dir "${output_dir}" --device "${device}"
  echo
  echo "[DRY RUN] Commands were written to ${commands_log}"
  exit 0
fi

copy_with_parents "${output_dir}/artifacts" "${target_artifacts[@]}"

run_step collect_report \
  "${python_bin}" -m benchmark.collect_benchmark_report \
  --output-dir "${output_dir}" --device "${device}"

echo
echo "[DONE] Reproduced benchmark evidence: ${output_dir}"
echo "[DONE] Markdown report: ${output_dir}/benchmark_report.md"
echo "[DONE] Machine-readable results: ${output_dir}/benchmark_report_results.json"
echo "[DONE] Results visualization: ${output_dir}/benchmark_results.png"
echo "[DONE] Delta-DPD sensitivity visualization: ${output_dir}/benchmark_delta_dpd_results.png"
