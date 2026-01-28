#!/usr/bin/env bash
set -euo pipefail

# Always run Streamlit using the project's .venv to avoid conda/system Python conflicts.
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

# Usage:
#   ./run_app.sh                # full mode (Donut enabled)
#   DONUT_DISABLE=1 ./run_app.sh # offline/fast mode
#   XGB_MODEL_PATH=models/xgb_fraud.json DONUT_DISABLE=1 ./run_app.sh

exec "$PY" -m streamlit run "$ROOT_DIR/app.py"
