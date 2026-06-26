#!/usr/bin/env bash
# Sub-only MGRPO on V7 expert-subtask contract SFT.
# v3: rollout fp16 (Main JSON quality) + train Sub 4bit; Main greedy, Sub sampled.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
CKPT="${CKPT:-artifacts/checkpoints/sft_expert_subtask_contract_v3}"
SAVE_DIR="${SAVE_DIR:-artifacts/checkpoints/mgrpo_expert_subtask_contract_v3_sub_only_v3}"
LOG="${LOG:-artifacts/mgrpo_expert_subtask_contract_v3_sub_only_v3.log}"
ITERATIONS="${ITERATIONS:-20}"
RESUME="${RESUME:-}"
EXTRA_ARGS=("$@")

ARGS=(
  --base-model Qwen/Qwen3.5-9B
  --main-adapter "${CKPT}/main_agent/best"
  --sub-adapter "${CKPT}/sub_agent/best"
  --protocol minimal
  --split dev
  --groups 8
  --group-size 4
  --iterations "$ITERATIONS"
  --agents sub
  --use-4bit
  --no-rollout-use-4bit
  --no-rollout-main-do-sample
  --rollout-do-sample
  --max-input-length 6656
  --rollout-temperature 0.7
  --rollout-main-repetition-penalty 1.0
  --sub-lr 1e-5
  --beta 0.05
  --reward-action-validity 0.3
  --main-max-new-tokens 350
  --sub-max-new-tokens 96
  --max-completion-tokens 96
  --save-dir "$SAVE_DIR"
)

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo "[mgrpo-v3] Sub-only | GPU=$GPU ckpt=$CKPT -> $SAVE_DIR (rollout fp16, train 4bit)"
echo "===== Sub-only MGRPO v3 $(date -Iseconds) =====" >> "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup python3 -u mgrpo_trainer.py \
  "${ARGS[@]}" "${EXTRA_ARGS[@]}" >> "$LOG" 2>&1 &
echo "  pid=$! log=$LOG"
