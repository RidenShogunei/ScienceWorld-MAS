#!/usr/bin/env bash
# Wait for V7 SFT to finish, then run Stratified-145 eval on a free GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_PID="${TRAIN_PID:-}"
EVAL_GPU="${EVAL_GPU:-6}"
CKPT="${CKPT:-artifacts/checkpoints/sft_expert_subtask_contract_v3}"
LOG="${LOG:-artifacts/eval/sft_expert_subtask_contract_v3_wait_eval.log}"

if [[ -z "$TRAIN_PID" ]]; then
  TRAIN_PID="$(pgrep -f 'sft_trainer.py.*sft_expert_subtask_contract_v3' | head -1 || true)"
fi

{
  echo "===== wait-for-v7-sft $(date -Iseconds) ====="
  echo "train_pid=${TRAIN_PID:-none} eval_gpu=$EVAL_GPU ckpt=$CKPT"
  if [[ -n "$TRAIN_PID" ]]; then
    echo "[wait] polling pid $TRAIN_PID ..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      sleep 120
      tail -1 artifacts/sft_expert_subtask_contract_v3.log 2>/dev/null | strings | tail -1 || true
    done
    echo "[wait] training pid $TRAIN_PID exited at $(date -Iseconds)"
  else
    echo "[wait] no train pid found; sleeping 60s then checking checkpoints"
    sleep 60
  fi

  for i in $(seq 1 30); do
    if [[ -f "${CKPT}/main_agent/best/adapter_model.safetensors" && -f "${CKPT}/sub_agent/best/adapter_model.safetensors" ]]; then
      break
    fi
    echo "[wait] checkpoints not ready ($i/30), sleep 30s"
    sleep 30
  done

  CUDA_VISIBLE_DEVICES="$EVAL_GPU" bash scripts/run_eval_expert_subtask_contract_v3.sh
  echo "[wait] eval finished at $(date -Iseconds)"
} >> "$LOG" 2>&1 &

echo "wait_eval_pid=$! log=$LOG train_pid=${TRAIN_PID:-auto}"
