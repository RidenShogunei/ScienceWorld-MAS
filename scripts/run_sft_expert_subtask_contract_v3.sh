#!/usr/bin/env bash
# SFT on V7 causal expert-subtask contract data (minimal protocol, 6656 context).
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-5}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="${DATA_DIR:-data/expert_subtask_contract_sft_v3_simple_minimax_sample1000}"
CKPT="${CKPT:-artifacts/checkpoints/sft_expert_subtask_contract_v3}"
LOG="${LOG:-artifacts/sft_expert_subtask_contract_v3.log}"
INIT_BASE="${INIT_BASE:-artifacts/checkpoints/sft_minimax_native_train50}"

echo "[sft-v7] GPU=$GPU data=$DATA_DIR -> $CKPT"
echo "===== V7 expert_subtask_contract SFT $(date -Iseconds) =====" >> "$LOG"

nohup python3 -u sft_trainer.py \
  --base-model Qwen/Qwen3.5-9B \
  --train-data "${DATA_DIR}/train.jsonl" \
  --val-data "${DATA_DIR}/val.jsonl" \
  --save-dir "$CKPT" \
  --agents both \
  --init-main-adapter "${INIT_BASE}/main_agent/best" \
  --init-sub-adapter "${INIT_BASE}/sub_agent/best" \
  --epochs "${EPOCHS:-10}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-3}" \
  --early-stop-overfit-patience 2 \
  --early-stop-min-delta 1e-4 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr "${LR:-2e-5}" \
  --max-length "${MAX_LENGTH:-6656}" \
  --no-use-4bit \
  --gradient-checkpointing \
  --log-every-updates 10 \
  "$@" >> "$LOG" 2>&1 &

echo "  pid=$! log=$LOG"
