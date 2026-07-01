#!/usr/bin/env bash
# L1 smoke: collect states (if needed) + Main step RL train.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG="${CONFIG:-l1/config/smoke.yaml}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"

mkdir -p artifacts/l1

STATES_OUT="$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['states']['output'])")"
if [[ ! -f "$STATES_OUT" && -f artifacts/diagnostics/decision_states_smoke.json ]]; then
  echo "[l1] reusing artifacts/diagnostics/decision_states_smoke.json"
  cp artifacts/diagnostics/decision_states_smoke.json "$STATES_OUT"
fi

if [[ ! -f "$STATES_OUT" ]]; then
  echo "[l1] collecting decision states..."
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u -m l1.states --config "$CONFIG"
fi

echo "[l1] training Main step RL..."
CUDA_VISIBLE_DEVICES="$GPU" python3 -u -m l1.trainer --config "$CONFIG"

echo "[l1] eval latest checkpoint..."
MAIN_ADAPTER="$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['train']['save_dir'] + '/iter_' + str(c['train']['iterations']).zfill(4))")"
CUDA_VISIBLE_DEVICES="$GPU" python3 -u -m l1.eval --config "$CONFIG" --main-adapter "$MAIN_ADAPTER"
