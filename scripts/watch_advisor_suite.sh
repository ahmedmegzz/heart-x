#!/usr/bin/env bash
# Keep run_advisor_suite.py alive across OOM / crashes (resume skips finished runs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/outputs/results/advisor_suite/suite.log"
mkdir -p "$(dirname "$LOG")"
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 PYTHONPATH=src

while true; do
  if [[ -f "$ROOT/outputs/results/advisor_suite/advisor_summary.txt" ]] \
     && [[ -f "$ROOT/outputs/results/advisor_suite/ablation_mean_std.csv" ]] \
     && [[ -f "$ROOT/outputs/results/advisor_suite/fusion_mean_std.csv" ]] \
     && [[ -f "$ROOT/outputs/results/advisor_suite/imbalance_mean_std.csv" ]]; then
    echo "$(date -Is) suite artifacts present — watcher exiting" | tee -a "$LOG"
    exit 0
  fi
  if pgrep -f 'scripts/run_advisor_suite.py' >/dev/null; then
    sleep 120
    continue
  fi
  echo "$(date -Is) restarting run_advisor_suite.py --epochs 8 --batch-size 32 --seeds 42 43 44" | tee -a "$LOG"
  .venv/bin/python scripts/run_advisor_suite.py --epochs 8 --batch-size 32 --seeds 42 43 44 >>"$LOG" 2>&1 || true
  sleep 15
done
