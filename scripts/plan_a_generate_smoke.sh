#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

OUTPUT="${OUTPUT:-data/plan_a_sft_smoke}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"

python3 -u -m plan_a.generate_sft_data \
  --output-dir "$OUTPUT" \
  --sample-size "$SAMPLE_SIZE"
