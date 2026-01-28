#!/usr/bin/env bash
set -euo pipefail

# Batch scoring launcher that always uses the project's .venv.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv Python at: $PY" >&2
  echo "Create it first:" >&2
  echo "  cd \"$ROOT_DIR\"" >&2
  echo "  python3.12 -m venv .venv" >&2
  echo "  $ROOT_DIR/.venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

# Defaults (override via env vars or args):
#   INVOICES_DIR, OUT_CSV, THRESHOLD
INVOICES_DIR="${INVOICES_DIR:-data/public_benchmark/_smoke_cord10/invoices_test}"
OUT_CSV="${OUT_CSV:-reports/scores.csv}"
THRESHOLD="${THRESHOLD:-0.5}"

# Usage examples:
#   ./run_batch.sh
#   INVOICES_DIR=/path/to/invoices OUT_CSV=reports/my_scores.csv ./run_batch.sh
#   DONUT_DISABLE=1 XGB_MODEL_PATH=models/xgb_fraud.json ./run_batch.sh

exec "$PY" "$ROOT_DIR/scripts/batch_score_invoices.py" \
  --invoices "$INVOICES_DIR" \
  --out "$OUT_CSV" \
  --threshold "$THRESHOLD"
