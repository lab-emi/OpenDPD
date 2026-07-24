#!/usr/bin/env bash
# Train every viable PA backbone at ~2700 params with the OpenDPDv2 recipe.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR=${LOGDIR:-/tmp/opendpd_pa2700_logs}
mkdir -p "$LOGDIR"
DATASET=${DATASET:-APA_200MHz}
PYTHON=${PYTHON:-python}
export LOGDIR DATASET PYTHON

# backbone:hidden_size  (fast/cheap first so results land early)
JOBS=(
  tcn:93
  rvtdcnn:68
  mcldnn:11
  gru:28
  lstm:24
  tres_gru:27
  dgru:23
  vdlstm:22
  qgru:27
  pgjanet:19
  dvrjanet:19
  deltajanet:32
  apnrru:34
  deltagru:26
)

run_one() {
  local spec="$1"
  local bb="${spec%%:*}"
  local h="${spec##*:}"
  local log="$LOGDIR/${DATASET}_${bb}_h${h}.log"
  echo "[START] $bb h=$h"
  "${PYTHON:-python}" main.py --step train_pa \
    --dataset_name "$DATASET" --PA_backbone "$bb" --PA_hidden_size "$h" \
    --n_epochs 240 --lr 5e-3 --lr_schedule 1 --lr_end 1e-4 \
    --decay_factor 0.5 --patience 10 --batch_size 64 > "$log" 2>&1
  echo "[DONE ] $bb h=$h exit=$?"
}
export -f run_one

printf '%s\n' "${JOBS[@]}" | xargs -P "${CONC:-5}" -I{} bash -c 'run_one "$@"' _ {}
echo "ALL TRAINING COMPLETE"
