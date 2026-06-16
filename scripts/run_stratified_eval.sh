#!/usr/bin/env bash
# Run hierarchical eval on the fixed stratified episode list.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAIN_ADAPTER="${MAIN_ADAPTER:-artifacts/checkpoints/sft/main_agent/best}"
SUB_ADAPTER="${SUB_ADAPTER:?Set SUB_ADAPTER to the sub checkpoint path}"
EPISODE_LIST="${EPISODE_LIST:-artifacts/eval/dev_stratified_k5_seed123.json}"
OUTPUT="${OUTPUT:-artifacts/eval/stratified_dev150.json}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"

python3 -u evaluate_environment.py \
  --base-model Qwen/Qwen3.5-9B \
  --main-adapter "$MAIN_ADAPTER" \
  --sub-adapter "$SUB_ADAPTER" \
  --episode-list "$EPISODE_LIST" \
  --no-use-4bit \
  --output "$OUTPUT" \
  "$@"
