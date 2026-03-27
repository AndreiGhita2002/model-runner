"""Interference evaluation runner.

Runs each model through its own full interference schedule. Each schedule
step runs for a fixed number of batches, so every model gets the same
amount of work under each interference condition.

Usage:
    uv run python -m tests.interference.interfere_eval
    uv run python -m tests.interference.interfere_eval --batches 500
    uv run python -m tests.interference.interfere_eval --no-interference
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tests.testing_models import MODEL_SETS
from tests.interference.interfere import SCHEDULES, InterferenceManager


def run_eval(cmd: list[str], env: dict | None = None,
             log_file: Path | None = None) -> int:
    """Run an evaluation command, returning the exit code."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_file, "w")
    else:
        f = None

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=f or sys.stdout,
        stderr=subprocess.STDOUT,
    )
    proc.wait()
    if f:
        f.close()
    return proc.returncode


def merge_results(run_dir: Path, output_file: Path, models: list[str], schedule, args) -> Path:
    """Merge all per-step eval JSONs into one combined JSON file and clean up temp dir."""
    combined = {
        "meta": {
            "experiment": "interference",
            "schedule": args.schedule,
            "schedule_steps": [(name, threads) for name, threads in schedule],
            "batches_per_step": args.batches,
            "model_set": args.model_set,
            "nproc": args.nproc,
            "omp_threads": args.omp_threads,
            "interference": not args.no_interference,
        },
        "results": {},
        "interference": None,
    }

    # Load interference log
    interf_file = run_dir / "interference.json"
    if interf_file.exists():
        with open(interf_file) as f:
            combined["interference"] = json.load(f)

    # Merge per-step eval JSONs
    step_files = sorted(run_dir.glob("step_*.json"))
    first_meta = None

    for step_file in step_files:
        with open(step_file) as f:
            data = json.load(f)

        # Grab shared meta from the first file
        if first_meta is None:
            first_meta = data.get("meta", {})
            for key in ["optimizer", "optimizer_kwargs", "n_microbatches",
                        "batch_size", "seed", "world_size", "clock", "git_commit"]:
                if key in first_meta:
                    combined["meta"][key] = first_meta[key]

        # Extract step info from filename: step_<model>_<step_i>_<label>.json
        stem = step_file.stem  # e.g. "step_conv_next_0_idle"

        for model_name, model_result in data.get("results", {}).items():
            if model_name not in combined["results"]:
                combined["results"][model_name] = {"batches": []}

            # Tag each batch with the interference step
            step_tag = stem.split(f"{model_name}_", 1)[-1] if model_name in stem else stem
            for batch in model_result.get("batches", []):
                batch["interference_step"] = step_tag
                combined["results"][model_name]["batches"].append(batch)

    # Compute overall RPS per model
    for model_name, result in combined["results"].items():
        batches = result["batches"]
        timed = [b for b in batches if "timing" in b]
        if timed:
            total_time = timed[-1]["timing"]["end"] - timed[0]["timing"]["start"]
            total_requests = len(timed)
            result["requests_per_second"] = total_requests / total_time if total_time > 0 else 0
        else:
            result["requests_per_second"] = 0

    # Write combined file
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2)

    # Clean up temp directory
    shutil.rmtree(run_dir)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Interference evaluation runner")
    parser.add_argument("--batches", type=int, default=2000,
                        help="Number of batches per schedule step (default: 2000)")
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", "4")),
                        help="Number of torchrun processes (default: 4)")
    parser.add_argument("--omp-threads", type=int, default=int(os.environ.get("OMP_THREADS", "8")),
                        help="OMP_NUM_THREADS for evaluation (default: 8)")
    parser.add_argument("--no-interference", action="store_true",
                        help="Run evaluation without interference")
    parser.add_argument("--schedule", choices=list(SCHEDULES.keys()), default="small",
                        help="Deterministic schedule name (default: small)")
    parser.add_argument("--model-set", choices=list(MODEL_SETS.keys()), default="small",
                        help="Which model set to evaluate (default: small)")
    parser.add_argument("-o", "--output", type=str, default="./data/interference",
                        help="Output directory (default: ./data/interference)")
    parser.add_argument("eval_args", nargs="*",
                        help="Additional arguments passed to evaluation.py")
    args = parser.parse_args()

    models = [name for name, _, _ in MODEL_SETS[args.model_set]]
    run_interference = not args.no_interference
    schedule = SCHEDULES[args.schedule]
    total_batches_per_model = args.batches * len(schedule)

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
    print(f"Batches/step:  {args.batches}")
    print(f"Schedule:      {args.schedule} ({len(schedule)} steps, {total_batches_per_model} batches per model)")
    print(f"Models:        {len(models)} ({args.model_set} set): {', '.join(models)}")
    print(f"Interference:  {run_interference}")
    print(f"NPROC:         {args.nproc}")
    print(f"Output:        {run_dir}")
    print("=" * 44)
    print()

    # Interference manager (manages benchmark processes directly)
    manager = InterferenceManager(log_file=run_dir / "interference.json")

    def cleanup(*_):
        print("\nCleaning up...")
        manager.stop_all()
        manager.save_log()
        print("Done.")

    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(1)))

    eval_env = os.environ.copy()
    eval_env["OMP_NUM_THREADS"] = str(args.omp_threads)

    for i, model in enumerate(models):
        print()
        print("=" * 44)
        print(f"[{i + 1}/{len(models)}] {model} ({len(schedule)} steps × {args.batches} batches)")
        print("=" * 44)

        for step_i, (bench_name, threads) in enumerate(schedule):
            # Start/stop interference for this step
            manager.stop_all()
            if run_interference and bench_name != "idle":
                manager.start_benchmark(bench_name, threads)
            else:
                label = bench_name if bench_name == "idle" else f"{bench_name} (no-interference mode)"
                print(f"  Step {step_i + 1}/{len(schedule)}: {label}")
                manager.log_event("start", bench_name, threads)

            # Run evaluation for this step
            step_label = f"{bench_name}_{threads}t" if bench_name != "idle" else "idle"
            step_output = run_dir / f"step_{model}_{step_i}_{step_label}.json"

            eval_cmd = [
                "uv", "run", "--no-sync", "torchrun",
                "--nproc_per_node", str(args.nproc),
                "-m", "tests.evaluation",
                "-n", str(args.batches),
                "--model-set", args.model_set,
                "--model", model,
                "-o", str(step_output),
                *args.eval_args,
            ]
            log_file = run_dir / f"eval_{model}_step{step_i}_{step_label}.log"

            exit_code = run_eval(eval_cmd, env=eval_env, log_file=log_file)

            if exit_code != 0:
                print(f"  Warning: {model} step {step_i + 1} exited with code {exit_code}")
            else:
                print(f"  Step {step_i + 1}/{len(schedule)} complete ({step_label})")

        manager.stop_all()
        print(f"  {model} complete.")

    manager.save_log()

    # Merge all per-step results into one JSON
    print()
    print("Merging results...")
    merge_results(run_dir, output_file, models, schedule, args)

    print()
    print("=" * 44)
    print(f"Experiment complete: {output_file}")
    print("=" * 44)


if __name__ == "__main__":
    main()
