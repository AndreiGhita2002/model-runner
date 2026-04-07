"""Experiment orchestrator — runs all experiment runs (A through E).

A. No interference, no rebalancing (GPipe baseline)
B. No interference, with rebalancing (Shisha)
C. Interference, no rebalancing (GPipe under interference)
D. Interference, first optimum only (Shisha, stop at first optimum)
E. Interference, full rebalancing (Shisha)

Runs C, D, E use the same random interference seed for a fair comparison.

Usage:
    uv run python -m tests.experiment
    uv run python -m tests.experiment --repetitions 3
    uv run python -m tests.experiment --model-set reduced --nproc 4
"""

import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

from tests.interference.interfere import SCHEDULES
from tests.testing_models import MODEL_SETS


def run_cmd(cmd: list[str], env: dict | None = None, log_file: Path | None = None) -> int:
    """Run a command, log output to file. Returns exit code."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            proc.wait()
        return proc.returncode
    else:
        proc = subprocess.Popen(cmd, env=env)
        proc.wait()
        return proc.returncode


def torchrun_cmd(nproc: int, module: str, args: list[str], taskset_cores: str | None = None) -> list[str]:
    """Build a torchrun command."""
    cmd = []
    if taskset_cores:
        cmd.extend(["taskset", "-c", taskset_cores])
    cmd.extend([
        "uv", "run", "--no-sync", "torchrun",
        "--nproc_per_node", str(nproc),
        "-m", module,
    ])
    cmd.extend(args)
    return cmd


def run_experiment(run_dir: Path, label: str, cmd: list[str], env: dict) -> bool:
    """Run an experiment, saving output and handling errors. Returns True on success."""
    log_file = run_dir / f"{label}.log"
    print(f"  [{label}] Starting...")
    start = time.perf_counter()

    exit_code = run_cmd(cmd, env=env, log_file=log_file)

    elapsed = time.perf_counter() - start
    if exit_code != 0:
        print(f"  [{label}] FAILED (exit code {exit_code}, {elapsed:.0f}s)")
        # Save error info
        error_file = run_dir / "errors.txt"
        with open(error_file, "a") as f:
            f.write(f"=== {label} failed with exit code {exit_code} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Log: {log_file}\n\n")
        return False
    else:
        print(f"  [{label}] Done ({elapsed:.0f}s)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run full experiment suite")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Number of repetitions (default: 1)")
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", "4")),
                        help="Number of torchrun processes (default: 4)")
    parser.add_argument("--omp-threads", type=int, default=int(os.environ.get("OMP_THREADS", "8")),
                        help="OMP_NUM_THREADS (default: 8)")
    parser.add_argument("--model-set", choices=list(MODEL_SETS.keys()), default="reduced",
                        help="Model set to evaluate (default: reduced)")
    parser.add_argument("--num-requests", type=int, default=5000,
                        help="Requests for non-interference runs A, B (default: 5000)")
    parser.add_argument("--schedule", choices=list(SCHEDULES.keys()), default="gradient",
                        help="Interference schedule for C, D, E (default: gradient)")
    parser.add_argument("--interference-seed", type=int, default=None,
                        help="Seed for random interference schedule (default: random)")
    parser.add_argument("-o", "--output", type=str, default="./data/experiments",
                        help="Output directory (default: ./data/experiments)")
    parser.add_argument("--skip", nargs="*", choices=["A", "B", "C", "D", "E"], default=[],
                        help="Skip specific runs (e.g. --skip A C)")
    parser.add_argument("--only", nargs="*", choices=["A", "B", "C", "D", "E"], default=None,
                        help="Run only these (e.g. --only A B)")
    args = parser.parse_args()

    # --only overrides --skip
    if args.only is not None:
        args.skip = [r for r in ["A", "B", "C", "D", "E"] if r not in args.only]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.omp_threads)

    # Generate interference seed if not specified
    interf_seed = args.interference_seed if args.interference_seed is not None else random.randint(0, 2**31)

    print("=" * 60)
    print("Experiment Suite")
    print("=" * 60)
    print(f"Repetitions:   {args.repetitions}")
    print(f"Model set:     {args.model_set}")
    print(f"Requests:      {args.num_requests} (runs A, B)")
    print(f"Schedule:      {args.schedule} (runs C, D, E)")
    print(f"Interf. seed:  {interf_seed}")
    print(f"NPROC:         {args.nproc}")
    print(f"Skip:          {args.skip or 'none'}")
    print(f"Output:        {output_dir}")
    print("=" * 60)
    print()

    for rep in range(args.repetitions):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.repetitions > 1:
            print(f"\n{'#' * 60}")
            print(f"# Repetition {rep + 1}/{args.repetitions}")
            print(f"# Output: {run_dir}")
            print(f"{'#' * 60}")

        # Save experiment metadata
        meta = {
            "timestamp": timestamp,
            "repetition": rep + 1,
            "total_repetitions": args.repetitions,
            "model_set": args.model_set,
            "num_requests": args.num_requests,
            "schedule": args.schedule,
            "interference_seed": interf_seed,
            "nproc": args.nproc,
            "omp_threads": args.omp_threads,
            "skipped": args.skip,
        }
        with open(run_dir / "experiment_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # ── Run A: GPipe baseline (no interference, no rebalancing) ──
        if "A" not in args.skip:
            print(f"\n--- Run A: GPipe baseline (no interference) ---")
            cmd = torchrun_cmd(args.nproc, "tests.evaluation", [
                "-n", "100",
                "-b", "1", "-m", "32",
                "--optimizer", "gpipe",
                "--model-set", args.model_set,
                "-o", str(run_dir / "run_A.json"),
            ])
            run_experiment(run_dir, "run_A", cmd, env)

        # ── Run B: Shisha (no interference, with rebalancing) ──
        if "B" not in args.skip:
            print(f"\n--- Run B: Reactive Shisha (no interference) ---")
            cmd = torchrun_cmd(args.nproc, "tests.evaluation", [
                "-n", str(args.num_requests),
                "-b", "1", "-m", "32",
                "--optimizer", "reactive",
                "--model-set", args.model_set,
                "-o", str(run_dir / "run_B.json"),
            ])
            run_experiment(run_dir, "run_B", cmd, env)

        # ── Run C: GPipe under interference ──
        if "C" not in args.skip:
            print(f"\n--- Run C: GPipe under interference ---")
            cmd = [
                "uv", "run", "python", "-m", "tests.interference.interfere_eval",
                "--optimizer", "gpipe",
                "--mode", "deterministic",
                "--schedule", args.schedule,
                "--model-set", args.model_set,
                "--nproc", str(args.nproc),
                "--omp-threads", str(args.omp_threads),
                "-o", str(run_dir),
            ]
            # interfere_eval writes <timestamp>.json, so we rename after
            success = run_experiment(run_dir, "run_C", cmd, env)
            if success:
                _rename_latest_json(run_dir, "run_C.json")

        # ── Run D: Shisha, stop at first optimum, under interference ──
        if "D" not in args.skip:
            config_prefix = output_dir / "exhaustive_config"
            # Check if cached configs exist for all models
            models = [name for name, _, _ in MODEL_SETS[args.model_set]]
            cached_configs = all(
                Path(f"{config_prefix}_{m}.json").exists() for m in models
            )

            if cached_configs:
                print(f"\n--- Run D: Using cached exhaustive config ---")
                cmd = [
                    "uv", "run", "python", "-m", "tests.interference.interfere_eval",
                    "--load-config", str(config_prefix),
                    "--mode", "deterministic",
                    "--schedule", args.schedule,
                    "--model-set", args.model_set,
                    "--nproc", str(args.nproc),
                    "--omp-threads", str(args.omp_threads),
                    "-o", str(run_dir),
                ]
            else:
                print(f"\n--- Run D: Exhaustive Shisha (explore then freeze) under interference ---")
                cmd = [
                    "uv", "run", "python", "-m", "tests.interference.interfere_eval",
                    "--optimizer", "exhaustive",
                    "--wait-for-optimum",
                    "--save-config", str(config_prefix),
                    "--mode", "deterministic",
                    "--schedule", args.schedule,
                    "--model-set", args.model_set,
                    "--nproc", str(args.nproc),
                    "--omp-threads", str(args.omp_threads),
                    "-o", str(run_dir),
                ]
            success = run_experiment(run_dir, "run_D", cmd, env)
            if success:
                _rename_latest_json(run_dir, "run_D.json")

        # ── Run E: Shisha with full rebalancing under interference ──
        if "E" not in args.skip:
            print(f"\n--- Run E: Reactive Shisha (full rebalancing) under interference ---")
            cmd = [
                "uv", "run", "python", "-m", "tests.interference.interfere_eval",
                "--optimizer", "reactive",
                "--mode", "deterministic",
                "--schedule", args.schedule,
                "--model-set", args.model_set,
                "--nproc", str(args.nproc),
                "--omp-threads", str(args.omp_threads),
                "-o", str(run_dir),
            ]
            success = run_experiment(run_dir, "run_E", cmd, env)
            if success:
                _rename_latest_json(run_dir, "run_E.json")

        # Print summary
        print(f"\n--- Repetition {rep + 1} complete ---")
        print(f"Results in {run_dir}/")
        for f in sorted(run_dir.iterdir()):
            if f.suffix == ".json":
                print(f"  {f.name}")


def _rename_latest_json(run_dir: Path, target_name: str):
    """Rename the most recent timestamped JSON file to target_name.

    interfere_eval.py writes output as <timestamp>.json. This renames it to
    something like run_C.json for clarity.
    """
    json_files = sorted(run_dir.glob("20??-*.json"), key=lambda f: f.stat().st_mtime)
    if json_files:
        latest = json_files[-1]
        target = run_dir / target_name
        if not target.exists():
            latest.rename(target)


if __name__ == "__main__":
    main()
