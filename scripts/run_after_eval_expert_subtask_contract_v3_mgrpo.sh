#!/usr/bin/env bash
# Wait for V7 Stratified-145 eval to finish, then launch Sub-only MGRPO.
set -euo pipefail
cd "$(dirname "$0")/.."

EVAL_OUTPUT="${EVAL_OUTPUT:-artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json}"
EVAL_LOG="${EVAL_LOG:-artifacts/eval/sft_expert_subtask_contract_v3_stratified145.log}"
RL_GPU="${RL_GPU:-0}"
LOG="${LOG:-artifacts/mgrpo_expert_subtask_contract_v3_after_eval.log}"

{
  echo "===== wait-for-v7-eval $(date -Iseconds) ====="
  echo "eval_output=$EVAL_OUTPUT rl_gpu=$RL_GPU"

  while true; do
    if [[ -f "$EVAL_OUTPUT" ]]; then
      echo "[wait] eval json ready at $(date -Iseconds)"
      break
    fi
    if pgrep -f 'evaluate_environment.py.*sft_expert_subtask_contract_v3' >/dev/null 2>&1; then
      n=$(grep -c '  score=' "$EVAL_LOG" 2>/dev/null || echo 0)
      echo "[wait] eval running: ${n}/145"
    else
      n=$(grep -c '  score=' "$EVAL_LOG" 2>/dev/null || echo 0)
      if [[ "$n" -ge 145 ]]; then
        echo "[wait] eval log complete (${n}/145) but json missing; continuing anyway"
        break
      fi
      echo "[wait] no eval process; log=${n}/145 — sleep 60s"
    fi
    sleep 120
  done

  for path in \
    artifacts/checkpoints/sft_expert_subtask_contract_v3/main_agent/best/adapter_model.safetensors \
    artifacts/checkpoints/sft_expert_subtask_contract_v3/sub_agent/best/adapter_model.safetensors; do
    [[ -f "$path" ]] || { echo "[wait] missing $path"; exit 1; }
  done

  CUDA_VISIBLE_DEVICES="$RL_GPU" bash scripts/run_sub_only_mgrpo_expert_subtask_contract_v3.sh
  echo "[wait] MGRPO launched at $(date -Iseconds)"
} >> "$LOG" 2>&1 &

echo "wait_mgrpo_pid=$! log=$LOG rl_gpu=$RL_GPU"
