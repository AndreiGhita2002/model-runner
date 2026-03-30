"""Interference evaluation runner.

Runs each model through its own full interference schedule. Uses
PipelineServer.run_continuous() via continuous_eval.py for maximum
throughput — the eval runs until SIGTERM, with interference switching
in the background.

Usage:
    uv run python -m tests.interference.interfere_eval
    uv run python -m tests.interference.interfere_eval --duration 120
    uv run python -m tests.interference.interfere_eval --no-interference
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from tests.testing_models import MODEL_SETS
from tests.interference.interfere import SCHEDULES, InterferenceManager, run_deterministic, step_label


def run_eval_background(cmd: list[str], env: dict | None = None,
                        log_file: Path | None = None) -> subprocess.Popen:
    """Start an evaluation subprocess in the background."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_file, "w")
    else:
        f = None

    return subprocess.Popen(
        cmd,
        env=env,
        stdout=f or sys.stdout,
        stderr=subprocess.STDOUT,
    )


def merge_results(run_dir: Path, output_file: Path, models: list[str], schedule, args):
    """Merge per-model eval JSONs and interference logs into one combined JSON."""
    combined = {
        "meta": {
            "experiment": "interference",
            "schedule": args.schedule,
            "schedule_steps": [list(specs) for specs in schedule],
            "step_duration": args.duration,
            "num_requests": "continuous",
            "model_set": args.model_set,
            "nproc": args.nproc,
            "omp_threads": args.omp_threads,
            "interference": not args.no_interference,
        },
        "results": {},
        "interference": {},
    }

    # Load per-model interference logs
    for model in models:
        interf_file = run_dir / f"interference_{model}.json"
        if interf_file.exists():
            with open(interf_file) as f:
                combined["interference"][model] = json.load(f)

    # Load per-model eval JSONs
    first_meta = None
    for model in models:
        eval_file = run_dir / f"eval_{model}.json"
        if not eval_file.exists():
            continue
        with open(eval_file) as f:
            data = json.load(f)

        if first_meta is None:
            first_meta = data.get("meta", {})
            for key in ["optimizer", "n_microbatches", "batch_size",
                        "world_size", "clock", "git_commit"]:
                if key in first_meta:
                    combined["meta"][key] = first_meta[key]

        for model_name, model_result in data.get("results", {}).items():
            combined["results"][model_name] = model_result

    # Write combined file
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2)

    # Clean up temp directory
    shutil.rmtree(run_dir)


def main():
    parser = argparse.ArgumentParser(description="Interference evaluation runner")
    parser.add_argument("--duration", type=int, default=120,
                        help="Seconds per schedule step (default: 120)")
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", "4")),
                        help="Number of torchrun processes (default: 4)")
    parser.add_argument("--omp-threads", type=int, default=int(os.environ.get("OMP_THREADS", "8")),
                        help="OMP_NUM_THREADS for evaluation (default: 8)")
    parser.add_argument("--no-interference", action="store_true",
                        help="Run evaluation without interference")
    parser.add_argument("--schedule", choices=list(SCHEDULES.keys()), default="gradient",
                        help="Interference schedule (default: gradient)")
    parser.add_argument("--model-set", choices=list(MODEL_SETS.keys()), default="small",
                        help="Which model set to evaluate (default: small)")
    parser.add_argument("-o", "--output", type=str, default="./data/interference",
                        help="Output directory (default: ./data/interference)")
    args = parser.parse_args()

    models = [name for name, _, _ in MODEL_SETS[args.model_set]]
    run_interference = not args.no_interference
    schedule = SCHEDULES[args.schedule]
    model_duration = args.duration * len(schedule)

    # Set up temp working directory and final output path
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"{timestamp}.json"
    run_dir = output_dir / f".tmp_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 44)
    print("Interference Experiment")
    print("=" * 44)
    print(f"Step duration: {args.duration}s")
    print(f"Schedule:      {args.schedule} ({len(schedule)} steps, {model_duration}s per model)")
    print(f"Models:        {len(models)} ({args.model_set} set): {', '.join(models)}")
    print(f"Interference:  {run_interference}")
    print(f"NPROC:         {args.nproc}")
    print(f"Output:        {output_file}")
    print("=" * 44)
    print()

    eval_env = os.environ.copy()
    eval_env["OMP_NUM_THREADS"] = str(args.omp_threads)

    managers: list[InterferenceManager] = []

    def cleanup(*_):
        print("\nCleaning up...")
        for m in managers:
            m.stop_all()
        print("Done.")

    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(1)))

    for i, model in enumerate(models):
        print()
        print("=" * 44)
        print(f"[{i + 1}/{len(models)}] {model} ({len(schedule)} steps × {args.duration}s = {model_duration}s)")
        print("=" * 44)

        manager = InterferenceManager(log_file=run_dir / f"interference_{model}.json")
        managers.append(manager)

        # 1. Start eval in background (runs until SIGTERM)
        print(f"  Starting evaluation...")
        eval_cmd = [
            "taskset", "-c", "0-31",
            "uv", "run", "--no-sync", "torchrun",
            "--nproc_per_node", str(args.nproc),
            "-m", "tests.interference.continuous_eval",
            "--model-set", args.model_set,
            "--model", model,
            "-o", str(run_dir / f"eval_{model}.json"),
        ]
        eval_proc = run_eval_background(
            eval_cmd, env=eval_env,
            log_file=run_dir / f"eval_{model}.log",
        )

        # 2. Run interference schedule (blocking, takes model_duration seconds)
        if run_interference:
            run_deterministic(manager, args.duration, schedule=schedule)
        else:
            # No interference — just wait for the same duration
            print(f"  No interference — waiting {model_duration}s...")
            time.sleep(model_duration)

        # 3. Stop eval via SIGTERM
        manager.stop_all()
        manager.save_log()

        print(f"  Stopping evaluation...")
        eval_proc.terminate()
        try:
            eval_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            eval_proc.kill()
            eval_proc.wait()

        print(f"  {model} complete.")

    # Merge all results into one JSON
    print()
    print("Merging results...")
    merge_results(run_dir, output_file, models, schedule, args)

    print()
    print("=" * 44)
    print(f"Experiment complete: {output_file}")
    print("=" * 44)


if __name__ == "__main__":
    main()
