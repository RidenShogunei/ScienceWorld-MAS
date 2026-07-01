#!/usr/bin/env bash
# Stratified-145 episodic eval: L1 RL Main (iter_0010) + V7 Sub (same pipeline as baseline).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-6}"
ITER="${ITER:-0010}"
MAIN="${MAIN:-artifacts/checkpoints/l1_main_step_rl_smoke/iter_${ITER}/main}"
SUB="${SUB:-artifacts/checkpoints/sft_expert_subtask_contract_v3/sub_agent/best}"
EPISODE_LIST="${EPISODE_LIST:-artifacts/eval/dev_stratified_k5_seed123.json}"
OUTPUT="${OUTPUT:-artifacts/eval/l1_main_step_rl_smoke_iter${ITER}_stratified145.json}"
LOG="${LOG:-artifacts/eval/l1_main_step_rl_smoke_iter${ITER}_stratified145.log}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

for path in "$MAIN" "$SUB"; do
  if [[ ! -f "${path}/adapter_model.safetensors" ]]; then
    echo "[eval-l1] missing checkpoint: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
echo "[eval-l1] GPU=$GPU main=$MAIN sub=$SUB -> $OUTPUT"
echo "===== L1 iter${ITER} stratified-145 $(date -Iseconds) =====" | tee -a "$LOG"

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

echo "[eval-l1] done -> $OUTPUT"
