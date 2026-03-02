#!/usr/bin/env bash
# Run eval-gpipe across all NPROC/OMP_THREADS configurations.
# Each combo keeps total_threads = NPROC * OMP_THREADS = 32.
#
# Outputs: data/baselines/gpipe_<nproc>x<omp>.json

CONFIGS=(
    "1  32"
    "2  16"
    "4  8"
    "8  4"
    "16 2"
    "32 1"
)

failed=()

for cfg in "${CONFIGS[@]}"; do
    read -r nproc omp <<< "$cfg"
    output="gpipe_${nproc}x${omp}.json"
    echo "=== GPipe sweep: NPROC=$nproc OMP_THREADS=$omp → $output ==="
    if make eval-gpipe NPROC="$nproc" OMP_THREADS="$omp" GPIPE_OUTPUT="$output"; then
        echo "  Done."
    else
        echo "  FAILED (NPROC=$nproc OMP_THREADS=$omp), continuing..."
        failed+=("${nproc}x${omp}")
    fi
    echo ""
done

echo "GPipe sweep complete."
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed configurations: ${failed[*]}"
fi
