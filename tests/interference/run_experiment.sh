#!/bin/bash
# Interference experiment orchestrator.
#
# Runs evaluation with the adaptive pipeline, optionally with interference
# (benchmark processes that stress CPU/memory) running alongside.
#
# Usage:
#   bash tests/interference/run_experiment.sh                     # with interference, 10 min
#   bash tests/interference/run_experiment.sh --no-interference   # without interference
#   bash tests/interference/run_experiment.sh --duration 300      # 5 minutes
#
# Safe to Ctrl+C — all child processes are cleaned up.

set -euo pipefail

# Defaults
DURATION=600
NPROC="${NPROC:-4}"
OMP_THREADS="${OMP_THREADS:-8}"
INTERFERENCE=true
INTERFERENCE_MODE="deterministic"
INTERFERENCE_INTERVAL=60
OUTPUT_DIR="./data/interference"
EVAL_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --no-interference) INTERFERENCE=false; shift ;;
        --mode) INTERFERENCE_MODE="$2"; shift 2 ;;
        --interval) INTERFERENCE_INTERVAL="$2"; shift 2 ;;
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        --) shift; EVAL_ARGS="$*"; break ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR="$OUTPUT_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "============================================"
echo "Interference Experiment"
echo "============================================"
echo "Duration:      ${DURATION}s"
echo "Interference:  $INTERFERENCE ($INTERFERENCE_MODE, interval=${INTERFERENCE_INTERVAL}s)"
echo "NPROC:         $NPROC"
echo "Output:        $RUN_DIR"
echo "============================================"
echo ""

# Track child PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Done."
}
trap cleanup EXIT INT TERM

# 1. Start interference (if enabled) — runs in background
if [ "$INTERFERENCE" = true ]; then
    echo "[1/2] Starting interference ($INTERFERENCE_MODE)..."
    uv run python -m tests.interference.interfere \
        --duration "$DURATION" \
        --mode "$INTERFERENCE_MODE" \
        --interval "$INTERFERENCE_INTERVAL" \
        -o "$RUN_DIR/interference.json" \
        2>&1 | tee "$RUN_DIR/interference.log" &
    INTERFERENCE_PID=$!
    PIDS+=($INTERFERENCE_PID)
else
    echo "[1/2] Skipping interference (--no-interference)"
fi

# 2. Run evaluation in continuous mode (foreground)
echo "[2/2] Starting evaluation (continuous, ${DURATION}s)..."
OMP_NUM_THREADS=$OMP_THREADS uv run --no-sync torchrun \
    --nproc_per_node="$NPROC" -m tests.evaluation \
    --duration "$DURATION" \
    -o "$RUN_DIR" \
    $EVAL_ARGS \
    2>&1 | tee "$RUN_DIR/eval.log"

echo ""
echo "Experiment complete. Results in $RUN_DIR/"
ls -la "$RUN_DIR/"
