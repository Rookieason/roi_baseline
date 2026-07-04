#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-.}"
shift || true

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-experiments/roi_heatmap_ablation/configs/default.json}"
LOG_DIR="${LOG_DIR:-outputs/roi_heatmap_ablation/logs}"

mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TS}.log"
PID_FILE="$LOG_DIR/run_${TS}.pid"

nohup "$PYTHON_BIN" -m experiments.roi_heatmap_ablation.run_experiment \
  --dataset-root "$DATASET_ROOT" \
  --config "$CONFIG" \
  "$@" \
  > "$LOG_FILE" 2>&1 &

PID="$!"
echo "$PID" > "$PID_FILE"

echo "started pid=$PID"
echo "log=$LOG_FILE"
echo "pid_file=$PID_FILE"
echo "tail -f $LOG_FILE"
