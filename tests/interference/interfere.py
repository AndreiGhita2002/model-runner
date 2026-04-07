import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Benchmark commands — adjust paths as needed for your system
BENCHMARKS = {
    "cpu_stress": {
        "cmd": [os.environ.get("GEMM_PATH", "gemm")],
        "args_fn": lambda threads: ["9999", str(threads)],
        "description": "CPU-intensive workload (GEMM)",
    },
    "memory_bandwidth": {
        # STREAM with array size > L3 cache (80MB = 10M doubles)
        "cmd": [os.environ.get("STREAM_C_PATH", "stream_c")],
        "cwd": str(Path(os.environ.get("STREAM_C_PATH", "stream_c")).parent) or None,
        "env": {"STREAM_ARRAY_SIZE": "10000000"},
        "description": "Memory bandwidth benchmark (STREAM, 80MB array)",
    },
    "idle": {
        "cmd": None,  # no process — just a gap
        "description": "No interference",
    },
}

# Schedule format:
#   "all": default settings for all models
#   "<model_name>": per-model overrides (merged on top of "all")
# Each entry can have:
#   "step_duration": seconds per step (overrides CLI --duration)
#   "steps": list of steps, each a list of BenchSpec tuples
#
# On fisherman: cores 0-31 are real, 32-63 are hyper threads.
# Adaptive pipeline runs on 0-31, benchmarks on 32-63.
SCHEDULES = {
    "experiment": {
        "all": {
            "step_duration": 300,
            "steps": [
                # baseline — no interference
                [],
                # step 1 - just CPU
                [("cpu_stress", 8, "32-39")],
                # step 2 - just memory
                [("memory_bandwidth", 8, "40-48")],
                # step 3 - both
                [("cpu_stress", 4, "32-36"),
                 ("memory_bandwidth", 4, "36-40")],
            ],
        },
    },
    "gradient": {
        "all": {
            "step_duration": 120,
            "steps": [
                # baseline — no interference
                [],
                # light
                [("cpu_stress", 4, "32-35")],
                # medium
                [("cpu_stress", 8, "32-39")],
                # medium with memory
                [("cpu_stress", 8, "32-39"),
                 ("memory_bandwidth", 1, "48")],
                # heavy with memory
                [("cpu_stress", 8, "32-39"),
                 ("cpu_stress", 8, "39-47"),
                 ("memory_bandwidth", 1, "48"),
                 ("memory_bandwidth", 1, "49")],
            ],
        },
        "efficientnet_b6": {
            "step_duration": 240,
        }
    },
    "small": {
        "all": {
            "step_duration": 120,
            "steps": [
                [],  # idle
                [("cpu_stress", 2, "32-33")],
                [("memory_bandwidth", 1, "32")],
            ],
        },
    },
    "full": {
        "all": {
            "step_duration": 120,
            "steps": [
                [],  # idle
                [("cpu_stress", 2, "32-33")],
                [("cpu_stress", 4, "32-35")],
                [("memory_bandwidth", 1, "32")],
                [("cpu_stress", 8, "32-39")],
                [("memory_bandwidth", 2, "32-33")],
                [],  # idle
                [("cpu_stress", 1, "32")],
                [("memory_bandwidth", 4, "32-35")],
            ],
        },
    },
}


# A benchmark instance: (name, threads, cores)
BenchSpec = tuple[str, int, str]


class InterferenceManager:
    def __init__(self, log_file: Path | None = None):
        # Map from BenchSpec -> Popen process
        self.active: dict[BenchSpec, subprocess.Popen] = {}
        self.log: list[dict] = []
        self.log_file = log_file

    def start_benchmark(self, name: str, num_threads: int = 1, cores: str = "") -> bool:
        """Start a benchmark process. Returns True if started successfully."""
        spec = (name, num_threads, cores)

        # Already running with same spec — skip
        if spec in self.active and self.active[spec].poll() is None:
            return True

        bench = BENCHMARKS.get(name)
        if bench is None:
            print(f"  Unknown benchmark: {name}", file=sys.stderr)
            return False

        if bench["cmd"] is None:
            self.log_event("start", name, num_threads, pid=None)
            return True

        # Set up the environment
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_threads)
        if "env" in bench:
            env.update(bench["env"])

        cmd = list(bench["cmd"])
        resolved = shutil.which(cmd[0])
        if resolved:
            cmd[0] = resolved
        if bench.get("args_fn"):
            cmd.extend(bench["args_fn"](num_threads))

        if cores:
            cmd = ["taskset", "-c", cores] + cmd

        cwd = bench.get("cwd")

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.active[spec] = proc
            self.log_event("start", name, num_threads, pid=proc.pid)
            cores_str = f", cores={cores}" if cores else ""
            print(f"  Started {name} (pid={proc.pid}, threads={num_threads}{cores_str})")
            return True
        except FileNotFoundError:
            print(f"  Benchmark not found: {bench['cmd'][0]}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"  Failed to start {name}: {e}", file=sys.stderr)
            return False

    def stop_benchmark(self, spec: BenchSpec):
        """Stop a specific benchmark by spec."""
        proc = self.active.pop(spec, None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        name, threads, cores = spec
        self.log_event("stop", name, threads, pid=proc.pid)
        cores_str = f", cores={cores}" if cores else ""
        print(f"  Stopped {name} (pid={proc.pid}, threads={threads}{cores_str})")

    def apply_step(self, step: list[BenchSpec]):
        """Transition to a new set of benchmarks.

        Stops all running benchmarks and starts the new ones fresh, ensuring
        each stage begins with clean processes (no long-running carry-over).
        """
        self.stop_all()

        for name, threads, cores in step:
            self.start_benchmark(name, threads, cores=cores)

    def stop_all(self):
        """Stop all running benchmark processes."""
        for spec in list(self.active.keys()):
            self.stop_benchmark(spec)

    def log_event(self, event_type: str, benchmark: str, threads: int = 0, pid: int = None):
        entry = {
            "time": time.perf_counter(),
            "event": event_type,
            "benchmark": benchmark,
            "threads": threads,
            "pid": pid,
        }
        self.log.append(entry)

    def save_log(self):
        if self.log_file and self.log:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                json.dump({"events": self.log}, f, indent=2)
            print(f"Interference log saved to {self.log_file}")


def resolve_model_schedule(schedule_name: str, model_name: str,
                           duration_override: int | None = None) -> dict:
    """Resolve schedule settings for a specific model.

    Merges "all" defaults with model-specific overrides.
    Returns dict with "step_duration" and "steps".
    """
    schedule = SCHEDULES[schedule_name]
    defaults = dict(schedule.get("all", {}))
    overrides = schedule.get(model_name, {})

    result = {**defaults, **overrides}

    # CLI --duration overrides schedule step_duration
    if duration_override is not None:
        result["step_duration"] = duration_override

    return result


def step_label(step: list[BenchSpec]) -> str:
    """Short readable label for a schedule step."""
    if not step:
        return "idle"
    parts = []
    for name, threads, cores in step:
        cores_str = f"_c{cores}" if cores else ""
        parts.append(f"{name}_{threads}t{cores_str}")
    return "+".join(parts)


def run_deterministic(manager: InterferenceManager, step_duration: int,
                      schedule: list[list[BenchSpec]] | None = None):
    """Run a deterministic interference schedule.

    Args:
        manager: InterferenceManager instance.
        step_duration: Seconds per schedule step.
        schedule: List of steps, where each step is a list of BenchSpec tuples.
    """
    if schedule is None:
        schedule = SCHEDULES["full"]

    total_duration = step_duration * len(schedule)
    start = time.perf_counter()

    print(f"Deterministic interference: {len(schedule)} steps × {step_duration}s = {total_duration}s")
    try:
        for step_i, step in enumerate(schedule):
            manager.apply_step(step)

            if not step:
                print(f"  Step {step_i + 1}/{len(schedule)}: idle ({step_duration}s)")
                manager.log_event("start", "idle")
            else:
                print(f"  Step {step_i + 1}/{len(schedule)}: {step_label(step)} ({step_duration}s)")

            wait_until = start + step_duration * (step_i + 1)
            while time.perf_counter() < wait_until:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterference interrupted.")
    finally:
        manager.stop_all()


def run_random(manager: InterferenceManager, step_duration: int,
               schedule: list[list[BenchSpec]] | None = None,
               seed: int | None = None, first_step: int | None = 0):
    """Run schedule steps in a random order.

    Each step runs exactly once for ``step_duration`` seconds, so total time
    is the same as deterministic mode. The order is shuffled based on ``seed``.

    Args:
        manager: InterferenceManager instance.
        step_duration: Seconds per schedule step.
        schedule: List of steps (same format as deterministic).
        seed: Random seed for reproducibility. None = random seed.
        first_step: Index of the step to run first (default 0, typically idle).
            None = fully random order (no pinned first step).
    """
    if schedule is None:
        schedule = SCHEDULES["full"]["all"]["steps"]

    if seed is None:
        seed = random.randint(0, 2**31)

    random.seed(seed)

    # Build shuffled order, optionally pinning the first step
    indices = list(range(len(schedule)))
    if first_step is not None and 0 <= first_step < len(indices):
        indices.remove(first_step)
        random.shuffle(indices)
        indices.insert(0, first_step)
    else:
        random.shuffle(indices)

    manager.log_event("config", "random", pid=None)
    manager.log.append({"seed": seed, "step_order": indices})

    total_duration = step_duration * len(schedule)
    start = time.perf_counter()

    order_str = " -> ".join(str(i) for i in indices)
    print(f"Random interference: {len(schedule)} steps × {step_duration}s = {total_duration}s, "
          f"seed={seed}, order=[{order_str}]")

    try:
        for run_i, step_i in enumerate(indices):
            step = schedule[step_i]
            manager.apply_step(step)

            if not step:
                print(f"  [{run_i + 1}/{len(schedule)}] Step {step_i}: idle ({step_duration}s)")
                manager.log_event("start", "idle")
            else:
                print(f"  [{run_i + 1}/{len(schedule)}] Step {step_i}: {step_label(step)} ({step_duration}s)")

            wait_until = start + step_duration * (run_i + 1)
            while time.perf_counter() < wait_until:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterference interrupted.")
    finally:
        manager.stop_all()


def main():
    parser = argparse.ArgumentParser(description="Interference generator")
    parser.add_argument("--duration", type=int, default=60,
                        help="Seconds per schedule step (default: 60)")
    parser.add_argument("--mode", choices=["deterministic", "random"], default="deterministic",
                        help="Interference mode (default: deterministic)")
    parser.add_argument("--schedule", choices=list(SCHEDULES.keys()), default="small",
                        help="Deterministic schedule to use (default: small)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output JSON file for interference log")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for random mode (default: random)")
    args = parser.parse_args()

    # Check that all benchmark binaries exist before starting
    missing = []
    for name, bench in BENCHMARKS.items():
        if bench["cmd"] is None:
            continue
        path = Path(bench["cmd"][0])
        if not path.exists() and not shutil.which(str(path)):
            missing.append((name, path))
    if len(missing) > 0:
        print("Error: missing benchmarks:", file=sys.stderr)
        for name, path in missing:
            env_var = name.upper().replace(" ", "_") + "_PATH"
            print(f"  '{name}' not found at '{path}' (set {env_var})", file=sys.stderr)
        sys.exit(1)

    manager = InterferenceManager(log_file=args.output)

    # Clean up on SIGTERM
    signal.signal(signal.SIGTERM, lambda *_: (manager.stop_all(), sys.exit(0)))

    schedule_steps = SCHEDULES[args.schedule]["all"]["steps"]

    if args.mode == "deterministic":
        run_deterministic(manager, args.duration, schedule=schedule_steps)
    else:
        run_random(manager, args.duration, schedule=schedule_steps, seed=args.seed)

    manager.save_log()


if __name__ == "__main__":
    main()
