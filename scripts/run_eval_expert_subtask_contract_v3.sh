#!/usr/bin/env bash
# Stratified-145 environment eval for V7 expert-subtask contract SFT.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-6}"
CKPT="${CKPT:-artifacts/checkpoints/sft_expert_subtask_contract_v3}"
TAG="${TAG:-sft_expert_subtask_contract_v3}"
EPISODE_LIST="${EPISODE_LIST:-artifacts/eval/dev_stratified_k5_seed123.json}"
OUTPUT="${OUTPUT:-artifacts/eval/${TAG}_stratified145.json}"
LOG="${LOG:-artifacts/eval/${TAG}_stratified145.log}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"

MAIN="${CKPT}/main_agent/best"
SUB="${CKPT}/sub_agent/best"
for path in "$MAIN" "$SUB"; do
  if [[ ! -f "${path}/adapter_model.safetensors" ]]; then
    echo "[eval-v7] missing checkpoint: $path" >&2
    exit 1
  fi
done

echo "[eval-v7] GPU=$GPU ckpt=$CKPT -> $OUTPUT"
echo "===== V7 stratified-145 $(date -Iseconds) =====" | tee -a "$LOG"

python3 -u evaluate_environment.py \
  --base-model Qwen/Qwen3.5-9B \
  --main-adapter "$MAIN" \
  --sub-adapter "$SUB" \
  --episode-list "$EPISODE_LIST" \
  --agent-interface contract-simple \
  --no-use-4bit \
  --max-input-length 6656 \
  --main-max-new-tokens 350 \
  --sub-max-new-tokens 96 \
  --output "$OUTPUT" \
  "$@" 2>&1 | tee -a "$LOG"

echo "[eval-v7] done -> $OUTPUT"
