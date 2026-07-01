#!/usr/bin/env bash
# Plan A SFT: Main [plan]{subgoal, focus_objects} + Sub action-id.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="${DATA_DIR:-data/plan_a_sft_smoke}"
CKPT="${CKPT:-artifacts/checkpoints/plan_a_sft_smoke}"
LOG="${LOG:-artifacts/plan_a_sft_smoke.log}"
INIT_SUB="${INIT_SUB:-artifacts/checkpoints/action_id_sft_smoke/gold_contract/sub_agent/best}"

mkdir -p "$(dirname "$LOG")" "$CKPT"

echo "[plan-a-sft] GPU=$GPU data=$DATA_DIR -> $CKPT"
echo "===== Plan A SFT smoke $(date -Iseconds) =====" >> "$LOG"

nohup python3 -u sft_trainer.py \
  --base-model Qwen/Qwen3.5-9B \
  --train-data "${DATA_DIR}/train.jsonl" \
  --val-data "${DATA_DIR}/val.jsonl" \
  --save-dir "$CKPT" \
  --agents both \
  --init-sub-adapter "$INIT_SUB" \
  --epochs "${EPOCHS:-10}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-3}" \
  --early-stop-overfit-patience 2 \
  --early-stop-min-delta 1e-4 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr "${LR:-2e-5}" \
  --max-length "${MAX_LENGTH:-4096}" \
  --no-use-4bit \
  --gradient-checkpointing \
  --log-every-updates 10 \
  "$@" >> "$LOG" 2>&1 &

echo "  pid=$! log=$LOG"
