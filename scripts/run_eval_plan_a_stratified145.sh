#!/usr/bin/env bash
# Stratified-145 episodic eval: Plan A SFT (Main [plan] + Sub action-id).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-4}"
CKPT="${CKPT:-artifacts/checkpoints/plan_a_sft_smoke}"
MAIN="${MAIN:-${CKPT}/main_agent/best}"
SUB="${SUB:-${CKPT}/sub_agent/best}"
EPISODE_LIST="${EPISODE_LIST:-artifacts/eval/dev_stratified_k5_seed123.json}"
OUTPUT="${OUTPUT:-artifacts/eval/plan_a_sft_smoke_stratified145.json}"
LOG="${LOG:-artifacts/eval/plan_a_sft_smoke_stratified145.log}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

for path in "$MAIN" "$SUB"; do
  if [[ ! -f "${path}/adapter_model.safetensors" ]]; then
    echo "[eval-plan-a] missing checkpoint: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
echo "[eval-plan-a] GPU=$GPU main=$MAIN sub=$SUB"
echo "[eval-plan-a] interface=plan-a -> $OUTPUT"
echo "===== Plan A stratified-145 $(date -Iseconds) =====" | tee -a "$LOG"

python3 -u evaluate_environment.py \
  --base-model Qwen/Qwen3.5-9B \
  --main-adapter "$MAIN" \
  --sub-adapter "$SUB" \
  --episode-list "$EPISODE_LIST" \
  --agent-interface plan-a \
  --no-use-4bit \
  --max-input-length 6656 \
  --main-max-new-tokens 350 \
  --sub-max-new-tokens 32 \
  --max-candidate-actions 32 \
  --history-limit 4 \
  --output "$OUTPUT" \
  "$@" 2>&1 | tee -a "$LOG"

echo "[eval-plan-a] done -> $OUTPUT"
