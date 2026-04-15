"""Hardware benchmark: auto-tune OMP_NUM_THREADS and NPROC.

Phase 1 — OMP_NUM_THREADS sweep (single process, ~40s):
    Tests powers-of-2 thread counts, scores by median * (1 + 0.5 * cv)
    to penalise noisy configs that cause bad rebalance decisions.

Phase 2 — NPROC sweep (distributed via torchrun, ~2 min per candidate):
    Launches benchmark_worker.py for each NPROC candidate to measure
    actual pipeline throughput with the optimal OMP setting.

Usage:
    python -m tests.benchmark
    python -m tests.benchmark --phase1-only
    python -m tests.benchmark --phase2-only --omp-threads 2
    python -m tests.benchmark --output results.json
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn as nn

from tests.testing_models import DEFAULT_MODEL_SET as evaluation_models


# ── Phase 1: OMP_NUM_THREADS sweep ──────────────────────────────────────


def _generate_thread_candidates(cpu_count: int) -> list[int]:
    """Powers of 2 from 1 up to cpu_count (inclusive)."""
    candidates = []
    n = 1
    while n <= cpu_count:
        candidates.append(n)
        n *= 2
    # Always include the actual core count if it's not a power of 2
    if candidates[-1] != cpu_count:
        candidates.append(cpu_count)
    return candidates


def phase1_omp_sweep() -> dict:
    """Sweep OMP_NUM_THREADS and return results + best thread count.

    Returns:
        Dict with keys: "best_threads", "candidates", "per_model", "summary".
    """
    cpu_count = os.cpu_count() or 4
    candidates = _generate_thread_candidates(cpu_count)

    print(f"Phase 1: OMP_NUM_THREADS sweep (cores={cpu_count})")
    print(f"  Candidates: {candidates}")
    print()

    # per_model[model_name][threads] = {"median": ..., "cv": ..., "score": ...}
    per_model: dict[str, dict[int, dict]] = {}

    for model_name, load_model, rand_input_fn in evaluation_models:
        print(f"  [{model_name}] loading...")
        model = load_model()
        sample = rand_input_fn()
        per_model[model_name] = {}

        for threads in candidates:
            torch.set_num_threads(threads)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    model(sample)

            # Timed passes
            times = []
            with torch.no_grad():
                for _ in range(10):
                    t0 = time.perf_counter()
                    model(sample)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

            median = statistics.median(times)
            mean = statistics.mean(times)
            stdev = statistics.stdev(times) if len(times) > 1 else 0.0
            cv = stdev / mean if mean > 0 else 0.0
            score = median * (1 + 0.5 * cv)

            per_model[model_name][threads] = {
                "median": median,
                "cv": cv,
                "score": score,
            }
            print(f"    threads={threads:3d}  median={median:.4f}s  cv={cv:.3f}  score={score:.4f}")

        del model
        print()

    # Pick best: lowest average score across models
    avg_scores: dict[int, float] = {}
    for threads in candidates:
        scores = [per_model[m][threads]["score"] for m in per_model]
        avg_scores[threads] = statistics.mean(scores)

    best_threads = min(avg_scores, key=avg_scores.get)

    print(f"  Average scores: { {t: f'{s:.4f}' for t, s in sorted(avg_scores.items())} }")
    print(f"  Best OMP_NUM_THREADS: {best_threads}")
    print()

    return {
        "best_threads": best_threads,
        "candidates": candidates,
        "per_model": {
            m: {str(t): v for t, v in scores.items()}
            for m, scores in per_model.items()
        },
        "avg_scores": {str(t): s for t, s in avg_scores.items()},
    }


# ── Phase 2: NPROC sweep ────────────────────────────────────────────────


def count_max_stages(model: nn.Module, depth: int = 3) -> int:
    """Count leaf children at the given depth via DFS — mirrors PipelineOptimizer logic.

    This operates on the raw nn.Module (no TimedModule needed) to determine
    the maximum number of pipeline stages a model can support.
    """
    top_children = list(model.children())
    if not top_children:
        return 1

    leaves = []

    def dfs(module: nn.Module, current_depth: int):
        children = list(module.children())
        if not children or current_depth >= depth:
            leaves.append(module)
        else:
            for child in children:
                dfs(child, current_depth + 1)

    for child in top_children:
        dfs(child, 1)

    return len(leaves)


def phase2_nproc_sweep(omp_threads: int, num_requests: int = 40,
                       depth: int = 3) -> dict:
    """Sweep NPROC values and return results + best nproc.

    Returns:
        Dict with keys: "best_nproc", "candidates", "per_nproc".
    """
    cpu_count = os.cpu_count() or 4

    # Determine max stages each model supports
    max_stages_per_model = {}
    for model_name, load_model, _ in evaluation_models:
        model = load_model()
        max_stages_per_model[model_name] = count_max_stages(model, depth)
        del model

    min_max_stages = min(max_stages_per_model.values())
    max_nproc = min(cpu_count, min_max_stages)
    nproc_candidates = list(range(2, max_nproc + 1))

    print(f"Phase 2: NPROC sweep (omp_threads={omp_threads})")
    print(f"  Max stages per model: {max_stages_per_model}")
    print(f"  NPROC candidates: {nproc_candidates}")
    print()

    if not nproc_candidates:
        print("  No valid NPROC candidates (need at least 2 ranks).")
        return {"best_nproc": 2, "candidates": [], "per_nproc": {}}

    # For each nproc, only include models that support enough stages
    per_nproc: dict[int, dict] = {}

    for nproc in nproc_candidates:
        eligible = [m for m, s in max_stages_per_model.items() if s >= nproc]
        if not eligible:
            print(f"  nproc={nproc}: no eligible models, skipping")
            continue

        print(f"  nproc={nproc}: testing with {len(eligible)} model(s)...")

        config_data = {
            "nproc": nproc,
            "omp_threads": omp_threads,
            "num_requests": num_requests,
            "depth": depth,
            "eligible_models": eligible,
        }

        # Write config and results paths to temp files
        config_fd, config_path = tempfile.mkstemp(suffix=".json", prefix="bench_cfg_")
        results_fd, results_path = tempfile.mkstemp(suffix=".json", prefix="bench_res_")
        os.close(results_fd)

        try:
            with os.fdopen(config_fd, "w") as f:
                json.dump(config_data, f)

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(omp_threads)
            env["BENCHMARK_RESULTS_PATH"] = results_path

            result = subprocess.run(
                [
                    "uv", "run", "--no-sync",
                    "torchrun", "--nproc_per_node", str(nproc),
                    "-m", "tests.benchmark_worker",
                    "--config", config_path,
                ],
                env=env,
                timeout=300,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"    FAILED (exit code {result.returncode})")
                if result.stderr:
                    # Print last few lines of stderr for debugging
                    lines = result.stderr.strip().split("\n")
                    for line in lines[-5:]:
                        print(f"      {line}")
                continue

            # Read results
            with open(results_path, "r") as f:
                nproc_results = json.load(f)

            # Compute combined throughput
            total_throughput = 0.0
            for model_name, data in nproc_results.items():
                total_throughput += data.get("throughput", 0.0)
                wall = data.get("wall_time", 0)
                tp = data.get("throughput", 0)
                avg = data.get("final_config_avg", 0)
                print(f"    {model_name}: wall={wall:.2f}s  throughput={tp:.2f} req/s"
                      f"  final_batch_avg={avg:.4f}s")

            per_nproc[nproc] = {
                "results": nproc_results,
                "total_throughput": total_throughput,
            }
            print(f"    Combined throughput: {total_throughput:.2f} req/s")

        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT (>300s)")
        except Exception as e:
            print(f"    ERROR: {e}")
        finally:
            for path in (config_path, results_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

        print()

    if not per_nproc:
        print("  All NPROC candidates failed. Defaulting to 2.")
        return {"best_nproc": 2, "candidates": nproc_candidates, "per_nproc": {}}

    best_nproc = max(per_nproc, key=lambda n: per_nproc[n]["total_throughput"])

    print(f"  Throughput summary:")
    for n in sorted(per_nproc):
        tp = per_nproc[n]["total_throughput"]
        marker = " <-- best" if n == best_nproc else ""
        print(f"    nproc={n}: {tp:.2f} req/s{marker}")
    print(f"  Best NPROC: {best_nproc}")
    print()

    return {
        "best_nproc": best_nproc,
        "candidates": nproc_candidates,
        "per_nproc": {
            str(n): {"total_throughput": v["total_throughput"]}
            for n, v in per_nproc.items()
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Hardware benchmark: find optimal OMP_NUM_THREADS and NPROC"
    )
    parser.add_argument("--phase1-only", action="store_true",
                        help="Only run Phase 1 (OMP sweep)")
    parser.add_argument("--phase2-only", action="store_true",
                        help="Only run Phase 2 (NPROC sweep)")
    parser.add_argument("--omp-threads", type=int, default=None,
                        help="Skip Phase 1 and use this OMP_NUM_THREADS for Phase 2")
    parser.add_argument("--num-requests", type=int, default=40,
                        help="Requests per model in Phase 2 (default: 40)")
    parser.add_argument("--depth", type=int, default=3,
                        help="TimedModule depth for stage counting (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write full results to JSON file")
    args = parser.parse_args()

    if args.phase1_only and args.phase2_only:
        print("Cannot specify both --phase1-only and --phase2-only")
        sys.exit(1)

    print("=" * 60)
    print("Hardware Benchmark")
    print("=" * 60)
    print()

    full_results = {}
    best_threads = args.omp_threads
    best_nproc = None

    # Phase 1
    if not args.phase2_only:
        if args.omp_threads is not None:
            print(f"Skipping Phase 1: using --omp-threads={args.omp_threads}")
            best_threads = args.omp_threads
        else:
            phase1 = phase1_omp_sweep()
            best_threads = phase1["best_threads"]
            full_results["phase1"] = phase1

    # Phase 2
    if not args.phase1_only:
        if best_threads is None:
            print("Error: Phase 2 requires OMP_NUM_THREADS. Run Phase 1 or use --omp-threads.")
            sys.exit(1)
        phase2 = phase2_nproc_sweep(
            omp_threads=best_threads,
            num_requests=args.num_requests,
            depth=args.depth,
        )
        best_nproc = phase2["best_nproc"]
        full_results["phase2"] = phase2

    # Summary
    print("=" * 60)
    print("Recommendation")
    print("=" * 60)
    if best_threads is not None:
        print(f"  Recommended OMP_NUM_THREADS: {best_threads}")
    if best_nproc is not None:
        print(f"  Recommended NPROC: {best_nproc}")
    if best_threads is not None and best_nproc is not None:
        print(f"  Apply with: make eval OMP_THREADS={best_threads} NPROC={best_nproc}")
    print()

    # Write results JSON
    if args.output:
        full_results["recommendation"] = {
            "omp_threads": best_threads,
            "nproc": best_nproc,
        }
        with open(args.output, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()
