#!/usr/bin/env bash
# Sub-only M-GRPO: freeze SFT Main, RL-update Sub from sft_sub_v2.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"

python3 -u mgrpo_trainer.py \
  --base-model Qwen/Qwen3.5-9B \
  --main-adapter artifacts/checkpoints/sft/main_agent/best \
  --sub-adapter artifacts/checkpoints/sft_sub_v2/sub_agent/best \
  --agents sub \
  --split dev \
  --groups 8 \
  --group-size 4 \
  --iterations 20 \
  --rollout-temperature 0.9 \
  --reward-action-validity 0.3 \
  --save-dir artifacts/checkpoints/mgrpo_sub_only \
  "$@"
