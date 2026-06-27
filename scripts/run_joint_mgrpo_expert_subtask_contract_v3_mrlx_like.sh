#!/usr/bin/env bash
# MrlX-like joint Main+Sub MGRPO ablation on V7 expert-subtask contract SFT.
#
# Differences from the conservative joint_v1 script:
# - Main and Sub are both sampled during rollout.
# - GRPO group size defaults to 8 samples per prompt.
# - Sampling temperature defaults to 1.0.
# - PPO clip matches the wider MrlX-style high side: [0.8, 1.28].
# - KL coefficient is set to 0.0; this local trainer currently uses old-policy
#   clipping but does not apply a separate reference KL loss.
# - Sub reward inherits the rollout-level Main reward, with a Sub format gate.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
CKPT="${CKPT:-artifacts/checkpoints/sft_expert_subtask_contract_v3}"
SAVE_DIR="${SAVE_DIR:-artifacts/checkpoints/mgrpo_expert_subtask_contract_v3_mrlx_like_v1}"
LOG="${LOG:-artifacts/mgrpo_expert_subtask_contract_v3_mrlx_like_v1.log}"
EPISODE_LIST="${EPISODE_LIST:-artifacts/eval/dev_stratified_k5_seed123.json}"
ITERATIONS="${ITERATIONS:-20}"
MGRPO_GROUPS="${MGRPO_GROUPS:-8}"
MGRPO_GROUP_SIZE="${MGRPO_GROUP_SIZE:-8}"
RESUME="${RESUME:-}"
EXTRA_ARGS=("$@")

for path in \
  "${CKPT}/main_agent/best/adapter_model.safetensors" \
  "${CKPT}/sub_agent/best/adapter_model.safetensors" \
  "$EPISODE_LIST"; do
  [[ -f "$path" ]] || { echo "[mrlx-like-mgrpo] missing: $path" >&2; exit 1; }
done

ARGS=(
  --base-model Qwen/Qwen3.5-9B
  --main-adapter "${CKPT}/main_agent/best"
  --sub-adapter "${CKPT}/sub_agent/best"
  --protocol minimal
  --episode-list "$EPISODE_LIST"
  --agents both
  --groups "$MGRPO_GROUPS"
  --group-size "$MGRPO_GROUP_SIZE"
  --iterations "$ITERATIONS"
  --use-4bit
  --no-rollout-use-4bit
  --rollout-do-sample
  --rollout-main-do-sample
  --rollout-sub-do-sample
  --rollout-temperature 1.0
  --rollout-top-p 0.95
  --rollout-main-repetition-penalty 1.0
  --max-input-length 6656
  --main-lr 2e-6
  --sub-lr 1e-5
  --beta 0.0
  --clip-low 0.2
  --clip-high 0.28
  --main-invalid-format-advantage -1.0
  --sub-reward-mode rollout_format
  --reward-global-score 1.0
  --reward-progress 0.0
  --format-validity 0.05
  --reward-action-validity 0.0
  --reward-no-progress-penalty 0.0
  --reward-repetition-penalty 0.0
  --reward-premature-done-penalty 0.0
  --main-max-new-tokens 350
  --sub-max-new-tokens 96
  --max-completion-tokens 96
  --save-dir "$SAVE_DIR"
)

if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo "[mrlx-like-mgrpo] GPU=$GPU ckpt=$CKPT"
echo "  pool=$EPISODE_LIST groups=${MGRPO_GROUPS}x${MGRPO_GROUP_SIZE} -> $SAVE_DIR"
echo "===== MrlX-like Joint MGRPO v1 $(date -Iseconds) =====" >> "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup python3 -u mgrpo_trainer.py \
  "${ARGS[@]}" "${EXTRA_ARGS[@]}" >> "$LOG" 2>&1 &
echo "  pid=$! log=$LOG"
