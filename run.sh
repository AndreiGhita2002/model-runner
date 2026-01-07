#!/bin/bash
# run_project.sh
# Build editable, then run the app from the project root

echo "Running setup script ..."
./setup_env.sh

echo "Launching main.py ..."
uv run python -m src.main

echo "Done!"
