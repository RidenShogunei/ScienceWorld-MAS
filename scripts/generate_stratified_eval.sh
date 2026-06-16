#!/usr/bin/env bash
# Generate fixed stratified dev eval list: K variations per task type.
set -euo pipefail
cd "$(dirname "$0")/.."

K="${K_PER_TASK:-5}"
SEED="${SEED:-123}"
SPLIT="${SPLIT:-dev}"
OUT="${OUT:-artifacts/eval/${SPLIT}_stratified_k${K}_seed${SEED}.json}"

export JAVA_HOME="${JAVA_HOME:-/home/jinxu/jdk-21.0.11+10-jre}"
export PATH="$JAVA_HOME/bin:$PATH"

python3 -u evaluate_environment.py \
  --split "$SPLIT" \
  --k-per-task "$K" \
  --seed "$SEED" \
  --write-episode-list "$OUT"

echo "[generate_stratified_eval] wrote $OUT"
