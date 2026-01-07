#!/bin/bash
set -e

echo "Building CUDA extension..."

# Ensure we're in venv
source .venv/bin/activate

# Sync main dependencies first (gets torch)
uv sync

# Build CUDA extension without build isolation
cd src/gpu_timer
uv pip install -e . --no-build-isolation -v

cd ../..

echo "✓ CUDA extension built successfully"
echo "Test: python -c 'import gpu_timer_cpp; print(\"Success!\")'"
