#!/bin/bash
# Hyperparameter sweep for the Shisha optimizer.
# Runs evaluations with different parameter combinations and logs results.
#
# Usage:
#   bash tests/sweep.sh                  # run all combos
#   bash tests/sweep.sh 2000             # override REQUEST_NUM
#
# Output:
#   - Each eval run saves to data/runs/ (timestamped JSON)
#   - A sweep log is saved to data/sweeps/<timestamp>.txt
#

set -euo pipefail

REQUEST_NUM="${1:-1000}"
SWEEP_DIR="./data/sweeps"
RUNS_DIR="./data/runs"

mkdir -p "$SWEEP_DIR" "$RUNS_DIR"

SWEEP_LOG="$SWEEP_DIR/$(date +%Y-%m-%d_%H-%M-%S).txt"
echo "Sweep started at $(date)" > "$SWEEP_LOG"
echo "REQUEST_NUM=$REQUEST_NUM" >> "$SWEEP_LOG"
echo "========================================" >> "$SWEEP_LOG"

# Trap Ctrl+C for clean exit
trap 'echo; echo "Sweep interrupted at $(date)." | tee -a "$SWEEP_LOG"; exit 0' INT

# Parameter grid
TOLERANCES=(0.01 0.05 0.1 0.15 0.25 0.5 0.75)
ALPHAS=(3 5 7 10)
REBALANCE_INTERVALS=(2 3 4 5)

TOTAL=0
for _ in "${TOLERANCES[@]}"; do
    for _ in "${ALPHAS[@]}"; do
        for _ in "${REBALANCE_INTERVALS[@]}"; do
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo "Total combinations: $TOTAL"
echo ""

COUNT=0
for TOL in "${TOLERANCES[@]}"; do
    for A in "${ALPHAS[@]}"; do
        for RI in "${REBALANCE_INTERVALS[@]}"; do
            COUNT=$((COUNT + 1))
            LABEL="tol=$TOL a=$A ri=$RI"
            echo "[$COUNT/$TOTAL] Running: $LABEL"

            # Find the latest file before this run
            BEFORE=$(ls -t "$RUNS_DIR"/*.json 2>/dev/null | head -1)

            # Run eval
            if ! make eval \
                REQUEST_NUM="$REQUEST_NUM" \
                TOLERANCE="$TOL" \
                ALPHA="$A" \
                REBALANCE_INTERVAL="$RI" \
                2>&1 | tail -5; then
                echo "  FAILED" | tee -a "$SWEEP_LOG"
                echo "" >> "$SWEEP_LOG"
                continue
            fi

            # Find the new file
            AFTER=$(ls -t "$RUNS_DIR"/*.json 2>/dev/null | head -1)
            if [ "$BEFORE" = "$AFTER" ]; then
                echo "  No new output file found" | tee -a "$SWEEP_LOG"
                echo "" >> "$SWEEP_LOG"
                continue
            fi

            # Analyse the run
            SUMMARY=$(python3 -c "
import json, sys
with open('$AFTER') as f:
    data = json.load(f)
models = []
total_rps = 0
total_rebalances = 0
total_optimum = 0
for model, result in data['results'].items():
    batches = result.get('batches', [])
    rebalances = sum(1 for b in batches if b.get('rebalance', {}).get('did_rebalance', False))
    at_optimum = sum(1 for b in batches if b.get('rebalance', {}).get('at_optimum', False))
    rps = result.get('requests_per_second', 0)
    total_rps += rps
    total_rebalances += rebalances
    total_optimum += at_optimum
    models.append(f'  {model}: rps={rps:.2f} reb={rebalances} opt={at_optimum}')
avg_rps = total_rps / len(data['results']) if data['results'] else 0
print(f'avg_rps={avg_rps:.2f} total_reb={total_rebalances} total_opt={total_optimum}')
for m in models:
    print(m)
" 2>&1)

            # Log to sweep file
            {
                echo "$LABEL | $(echo "$SUMMARY" | head -1) | file=$(basename "$AFTER")"
                echo "$SUMMARY" | tail -n +2
                echo ""
            } >> "$SWEEP_LOG"

            # Print summary
            echo "  $(echo "$SUMMARY" | head -1)"
            echo ""
        done
    done
done

echo "Sweep completed at $(date)" | tee -a "$SWEEP_LOG"
echo "Results saved to $SWEEP_LOG"
