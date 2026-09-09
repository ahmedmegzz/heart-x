#!/usr/bin/env bash
# End-to-end HEART-X MIT-BIH pipeline
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY="${ROOT}/.venv/bin/python"

MAX_BEATS="${MAX_BEATS:-800}"
EPOCHS="${EPOCHS:-10}"

echo "== Prepare MIT-BIH (max ${MAX_BEATS} beats/record) =="
"$PY" scripts/prepare_mitbih.py --max-beats-per-record "$MAX_BEATS" --method stft

echo "== Train dual-stream HEART-X =="
"$PY" scripts/train_heartx.py --model dual --epochs "$EPOCHS" --balance random

echo "== MM-GradCAM figures =="
"$PY" scripts/explain_mm_gradcam.py --num-samples 8

echo "Done. See outputs/"
