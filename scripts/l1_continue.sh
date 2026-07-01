#!/usr/bin/env bash
# Resume L1 Main step RL from iter_0010; stop after 3 iters without expert_match gain.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

GPU="${CUDA_VISIBLE_DEVICES:-5}"
CONFIG="${CONFIG:-l1/config/continue.yaml}"
START_ITER="${START_ITER:-11}"
MAIN_ADAPTER="${MAIN_ADAPTER:-artifacts/checkpoints/l1_main_step_rl_smoke/iter_0010/main}"
LOG="${LOG:-artifacts/l1/l1_continue_train.log}"

if [[ ! -f "${MAIN_ADAPTER}/adapter_model.safetensors" ]]; then
  echo "[l1-continue] missing main adapter: $MAIN_ADAPTER" >&2
  exit 1
fi

echo "[l1-continue] GPU=$GPU start_iter=$START_ITER main=$MAIN_ADAPTER"
echo "===== L1 continue $(date -Iseconds) =====" >> "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" nohup python3 -u -m l1.trainer \
  --config "$CONFIG" \
  --main-adapter "$MAIN_ADAPTER" \
  --start-iteration "$START_ITER" \
  "$@" >> "$LOG" 2>&1 &

echo "  pid=$! log=$LOG"
