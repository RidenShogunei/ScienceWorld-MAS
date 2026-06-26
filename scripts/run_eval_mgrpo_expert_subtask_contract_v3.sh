#!/usr/bin/env bash
# Stratified-145 eval: V7 SFT Main + MGRPO Sub checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAIN_ADAPTER="${MAIN_ADAPTER:-artifacts/checkpoints/sft_expert_subtask_contract_v3/main_agent/best}"
SUB_ADAPTER="${SUB_ADAPTER:?Set SUB_ADAPTER to mgrpo iter sub path}"
ITER="${ITER:-unknown}"
OUTPUT="${OUTPUT:-artifacts/eval/mgrpo_expert_subtask_contract_v3_${ITER}_stratified145.json}"
LOG="${LOG:-artifacts/eval/mgrpo_expert_subtask_contract_v3_${ITER}_stratified145.log}"
EPISODE_LIST="${EPISODE_LIST:-artifacts/eval/dev_stratified_k5_seed123.json}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"

for path in "$MAIN_ADAPTER" "$SUB_ADAPTER"; do
  if [[ ! -f "${path}/adapter_model.safetensors" ]]; then
    echo "[eval-rl] missing: $path" >&2
    exit 1
  fi
done

echo "[eval-rl] GPU=$GPU iter=$ITER sub=$SUB_ADAPTER -> $OUTPUT"
echo "===== MGRPO eval iter=$ITER $(date -Iseconds) =====" | tee -a "$LOG"

python3 -u evaluate_environment.py \
  --base-model Qwen/Qwen3.5-9B \
  --main-adapter "$MAIN_ADAPTER" \
  --sub-adapter "$SUB_ADAPTER" \
  --episode-list "$EPISODE_LIST" \
  --agent-interface contract-simple \
  --no-use-4bit \
  --max-input-length 6656 \
  --main-max-new-tokens 350 \
  --sub-max-new-tokens 96 \
  --output "$OUTPUT" \
  "$@" 2>&1 | tee -a "$LOG"

echo "[eval-rl] done -> $OUTPUT"
