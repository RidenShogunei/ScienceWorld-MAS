#!/usr/bin/env bash
# L1 Main-only step RL on stratified k2 decision states (30 tasks × 2 variations).
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${CUDA_VISIBLE_DEVICES:-4}"
CONFIG="${CONFIG:-l1/config/main_stratified_k2.yaml}"
STATES="${STATES:-artifacts/l1/decision_states_stratified_k2.json}"
START_ITER="${START_ITER:-1}"
MAIN_ADAPTER="${MAIN_ADAPTER:-}"
SUB_ADAPTER="${SUB_ADAPTER:-}"
LOG="${LOG:-artifacts/l1/l1_main_stratified_k2_train.log}"

for path in \
  "$STATES" \
  "artifacts/checkpoints/action_id_sft_smoke/gold_contract/sub_agent/best/adapter_model.safetensors" \
  "artifacts/checkpoints/l1_main_step_rl_smoke/iter_0011/main/adapter_model.safetensors"; do
  [[ -f "$path" ]] || { echo "[l1-main-k2] missing: $path" >&2; exit 1; }
done

ARGS=(--config "$CONFIG" --start-iteration "$START_ITER")
if [[ -n "$MAIN_ADAPTER" ]]; then
  ARGS+=(--main-adapter "$MAIN_ADAPTER")
fi
if [[ -n "$SUB_ADAPTER" ]]; then
  ARGS+=(--sub-adapter "$SUB_ADAPTER")
fi

echo "[l1-main-k2] GPU=$GPU config=$CONFIG states=$STATES start_iter=$START_ITER agents=main"
echo "===== L1 Main-only stratified k2 $(date -Iseconds) =====" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" nohup python3 -u -m l1.trainer \
  "${ARGS[@]}" \
  "$@" >> "$LOG" 2>&1 &

echo "  pid=$! log=$LOG"
