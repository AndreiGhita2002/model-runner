#!/usr/bin/env bash
# run_project.sh
# Build editable, then run the app from the project root

set -euo pipefail
IFS=$'\n\t'

# --- check uv ---------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not installed – install it with: pip install uv" >&2
  exit 1
fi

# --- install package editable ------------------
echo "Installing project..."
uv pip install -e .

# --- run the application ------------------------
echo "Launching main.py ..."
uv run python -m src.main

echo "Done!"
