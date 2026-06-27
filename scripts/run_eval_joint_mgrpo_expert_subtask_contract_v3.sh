#!/usr/bin/env bash
# Stratified-145 eval: joint MGRPO iter (Main + Sub from same checkpoint).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
ITER_CKPT="${ITER_CKPT:?Set ITER_CKPT to mgrpo iter dir, e.g. artifacts/checkpoints/mgrpo_expert_subtask_contract_v3_joint_v1/iter_0005}"
ITER="${ITER:-$(basename "$ITER_CKPT")}"
MAIN_ADAPTER="${MAIN_ADAPTER:-$ITER_CKPT/main}"
SUB_ADAPTER="${SUB_ADAPTER:-$ITER_CKPT/sub}"
OUTPUT="${OUTPUT:-artifacts/eval/mgrpo_expert_subtask_contract_v3_joint_${ITER}_stratified145.json}"
LOG="${LOG:-artifacts/eval/mgrpo_expert_subtask_contract_v3_joint_${ITER}_stratified145.log}"
EPISODE_LIST="${EPISODE_LIST:-artifacts/eval/dev_stratified_k5_seed123.json}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"

for path in "$MAIN_ADAPTER" "$SUB_ADAPTER"; do
  if [[ ! -f "${path}/adapter_model.safetensors" ]]; then
    echo "[eval-joint] missing: $path" >&2
    exit 1
  fi
done

echo "[eval-joint] GPU=$GPU iter=$ITER main=$MAIN_ADAPTER sub=$SUB_ADAPTER"
echo "===== Joint MGRPO eval $ITER $(date -Iseconds) =====" | tee -a "$LOG"

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

echo "[eval-joint] done -> $OUTPUT"
