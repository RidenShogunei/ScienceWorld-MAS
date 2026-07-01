#!/usr/bin/env bash
# L1 eval only (baseline or trained Main adapter).
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG="${CONFIG:-l1/config/smoke.yaml}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
MAIN_ADAPTER="${MAIN_ADAPTER:-}"

ARGS=(--config "$CONFIG")
if [[ -n "$MAIN_ADAPTER" ]]; then
  ARGS+=(--main-adapter "$MAIN_ADAPTER")
fi

CUDA_VISIBLE_DEVICES="$GPU" python3 -u -m l1.eval "${ARGS[@]}"
