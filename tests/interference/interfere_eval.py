"""Interference evaluation runner.

Runs each model through its own full interference schedule, so every model
experiences the same interference pattern independently.

Usage:
    uv run python -m tests.interference.interfere_eval
    uv run python -m tests.interference.interfere_eval --duration 300
    uv run python -m tests.interference.interfere_eval --no-interference
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from tests.testing_models import MODEL_SETS
from tests.interference.interfere import SCHEDULES


def run_cmd(cmd: list[str], env: dict | None = None, log_file: Path | None = None,
            background: bool = False) -> subprocess.Popen | int:
    """Run a command, optionally in the background. Returns Popen if background, else exit code."""
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

    if background:
        return proc

    proc.wait()
    if f:
        f.close()
    return proc.returncode


def stop_process(proc: subprocess.Popen):
    """Terminate a process, falling back to kill if needed."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Interference evaluation runner")
    parser.add_argument("--duration", type=int, default=60,
                        help="Seconds per schedule step (default: 60)")
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", "4")),
                        help="Number of torchrun processes (default: 4)")
    parser.add_argument("--omp-threads", type=int, default=int(os.environ.get("OMP_THREADS", "8")),
                        help="OMP_NUM_THREADS for evaluation (default: 8)")
    parser.add_argument("--no-interference", action="store_true",
                        help="Run evaluation without interference")
    parser.add_argument("--mode", choices=["deterministic", "random"], default="deterministic",
                        help="Interference mode (default: deterministic)")
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
    model_duration = args.duration * len(schedule)

    # Set up output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    total_time = model_duration * len(models)
    print("=" * 44)
    print("Interference Experiment")
    print("=" * 44)
    print(f"Step duration: {args.duration}s")
    print(f"Schedule:      {args.schedule} ({len(schedule)} steps, {model_duration}s per model)")
    print(f"Models:        {len(models)} ({args.model_set} set): {', '.join(models)}")
    print(f"Interference:  {run_interference} ({args.mode})")
    print(f"NPROC:         {args.nproc}")
    print(f"Output:        {run_dir}")
    print(f"Total time:    ~{total_time}s ({total_time // 60}m)")
    print("=" * 44)
    print()

    # Track background processes for cleanup
    bg_processes: list[subprocess.Popen] = []

    def cleanup(*_):
        print("\nCleaning up...")
        for proc in bg_processes:
            stop_process(proc)
        print("Done.")

    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(1)))

    eval_env = os.environ.copy()
    eval_env["OMP_NUM_THREADS"] = str(args.omp_threads)

    for i, model in enumerate(models):
        print()
        print("=" * 44)
        print(f"[{i + 1}/{len(models)}] Running: {model} ({model_duration}s)")
        print("=" * 44)

        interference_proc = None

        # 1. Start interference in background
        if run_interference:
            print(f"  Starting interference ({args.mode})...")
            interference_cmd = [
                sys.executable, "-m", "tests.interference.interfere",
                "--duration", str(args.duration),
                "--mode", args.mode,
                "--schedule", args.schedule,
                "-o", str(run_dir / f"interference_{model}.json"),
            ]
            interference_proc = run_cmd(
                interference_cmd,
                log_file=run_dir / f"interference_{model}.log",
                background=True,
            )
            bg_processes.append(interference_proc)

        # 2. Run evaluation for this model (foreground)
        print(f"  Starting evaluation...")
        eval_cmd = [
            "uv", "run", "--no-sync", "torchrun",
            "--nproc_per_node", str(args.nproc),
            "-m", "tests.evaluation",
            "--duration", str(model_duration),
            "--model-set", args.model_set,
            "--model", model,
            "-o", str(run_dir),
            *args.eval_args,
        ]
        exit_code = run_cmd(
            eval_cmd,
            env=eval_env,
            log_file=run_dir / f"eval_{model}.log",
        )

        # 3. Stop interference before next model
        if interference_proc:
            stop_process(interference_proc)
            bg_processes.remove(interference_proc)

        if exit_code != 0:
            print(f"  Warning: evaluation for {model} exited with code {exit_code}")

        print(f"  {model} complete.")

    print()
    print("=" * 44)
    print(f"Experiment complete. Results in {run_dir}/")
    print("=" * 44)
    for f in sorted(run_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
