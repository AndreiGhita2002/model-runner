#!/usr/bin/env bash
# run_project.sh
# Build editable, then run the app from the project root

echo "Running uv sync ..."
uv sync

echo "Launching main.py ..."
uv run python -m src.main

echo "Done!"
