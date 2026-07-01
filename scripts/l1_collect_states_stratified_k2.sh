#!/usr/bin/env bash
# Build L1 decision states: all task types in stratified eval list, k=2 variations each.
set -euo pipefail
cd "$(dirname "$0")/.."

export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG="${CONFIG:-l1/config/joint_stratified_k2.yaml}"
LOG="${LOG:-artifacts/l1/collect_states_stratified_k2.log}"

mkdir -p "$(dirname "$LOG")"
echo "[l1.states] config=$CONFIG"
echo "===== collect stratified k2 states $(date -Iseconds) =====" | tee "$LOG"

python3 -u -m l1.states --config "$CONFIG" 2>&1 | tee -a "$LOG"

echo "[l1.states] done -> see config states.output"
