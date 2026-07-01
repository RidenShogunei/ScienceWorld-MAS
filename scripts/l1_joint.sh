#!/usr/bin/env bash
# L1 single-step joint GRPO: Main + Sub both update (action-id Sub, fixed states).
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${CUDA_VISIBLE_DEVICES:-5}"
CONFIG="${CONFIG:-l1/config/joint.yaml}"
START_ITER="${START_ITER:-1}"
MAIN_ADAPTER="${MAIN_ADAPTER:-}"
SUB_ADAPTER="${SUB_ADAPTER:-}"
LOG="${LOG:-artifacts/l1/l1_joint_train.log}"

for path in \
  "artifacts/l1/decision_states_smoke.json" \
  "artifacts/checkpoints/action_id_sft_smoke/gold_contract/sub_agent/best/adapter_model.safetensors"; do
  [[ -f "$path" ]] || { echo "[l1-joint] missing: $path" >&2; exit 1; }
done

ARGS=(--config "$CONFIG" --start-iteration "$START_ITER")
if [[ -n "$MAIN_ADAPTER" ]]; then
  ARGS+=(--main-adapter "$MAIN_ADAPTER")
fi
if [[ -n "$SUB_ADAPTER" ]]; then
  ARGS+=(--sub-adapter "$SUB_ADAPTER")
fi

echo "[l1-joint] GPU=$GPU config=$CONFIG start_iter=$START_ITER"
echo "===== L1 joint Main+Sub $(date -Iseconds) =====" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" nohup python3 -u -m l1.trainer \
  "${ARGS[@]}" \
  "$@" >> "$LOG" 2>&1 &

echo "  pid=$! log=$LOG"
