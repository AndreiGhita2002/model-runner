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


class InterferenceManager:
    def __init__(self, log_file: Path | None = None):
        self.active_processes: list[subprocess.Popen] = []
        self.log: list[dict] = []
        self.log_file = log_file

    def start_benchmark(self, name: str, num_threads: int = 1) -> bool:
        """Start a benchmark process. Returns True if started successfully."""
        bench = BENCHMARKS.get(name)
        if bench is None:
            print(f"  Unknown benchmark: {name}", file=sys.stderr)
            return False

        if bench["cmd"] is None:
            # "idle" benchmark — just log it
            self.log_event("start", name, num_threads, pid=None)
            return True

        # Set up the environment
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_threads)
        if "env" in bench:
            env.update(bench["env"])

        cmd = list(bench["cmd"])
        # Resolve binary path — subprocess.Popen may not search PATH the same way a shell does
        resolved = shutil.which(cmd[0])
        if resolved:
            cmd[0] = resolved
        if bench.get("args_fn"):
            cmd.extend(bench["args_fn"](num_threads))

        cwd = bench.get("cwd")

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.active_processes.append(proc)
            self.log_event("start", name, num_threads, pid=proc.pid)
            print(f"  Started {name} (pid={proc.pid}, threads={num_threads})")
            return True
        except FileNotFoundError:
            print(f"  Benchmark not found: {bench['cmd'][0]}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"  Failed to start {name}: {e}", file=sys.stderr)
            return False

    def stop_all(self):
        """Stop all running benchmark processes."""
        for proc in self.active_processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                self.log_event("stop", "all", pid=proc.pid)
        self.active_processes = []

    def stop_one(self):
        """Stop the oldest running benchmark process."""
        alive = [p for p in self.active_processes if p.poll() is None]
        if alive:
            proc = alive[0]
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            self.log_event("stop", "oldest", pid=proc.pid)
            self.active_processes.remove(proc)

    def log_event(self, event_type: str, benchmark: str, threads: int = 0, pid: int = None):
        entry = {
            "time": time.time(),
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


SCHEDULES = {
    "small": [
        ("idle", 0),
        ("cpu_stress", 2),
        ("memory_bandwidth", 1),
    ],
    "full": [
        ("idle", 0),
        ("cpu_stress", 2),
        ("cpu_stress", 4),
        ("memory_bandwidth", 1),
        ("cpu_stress", 8),
        ("memory_bandwidth", 2),
        ("idle", 0),
        ("cpu_stress", 1),
        ("memory_bandwidth", 4),
    ],
}


def run_deterministic(manager: InterferenceManager, step_duration: int,
                      schedule: list[tuple[str, int]] | None = None):
    """Run a deterministic interference schedule.

    Args:
        manager: InterferenceManager instance.
        step_duration: Seconds per schedule step.
        schedule: List of (benchmark_name, num_threads) tuples.
    """
    if schedule is None:
        schedule = SCHEDULES["full"]

    total_duration = step_duration * len(schedule)
    start = time.time()

    print(f"Deterministic interference: {len(schedule)} steps × {step_duration}s = {total_duration}s")
    try:
        for step, (name, threads) in enumerate(schedule):
            manager.stop_all()

            if name != "idle":
                manager.start_benchmark(name, threads)
            else:
                print(f"  Idle period ({step_duration}s)")
                manager.log_event("start", "idle")

            # Wait for step_duration
            wait_until = start + step_duration * (step + 1)
            while time.time() < wait_until:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterference interrupted.")
    finally:
        manager.stop_all()


def run_random(manager: InterferenceManager, duration: int,
               min_interval: int = 30, max_interval: int = 120,
               seed: int = 42, idle_start: int = 60):
    """Run random interference."""
    random.seed(seed)
    manager.log_event("seed", str(seed))

    benchmarks = [k for k in BENCHMARKS if k != "idle"]
    thread_choices = [1, 2, 4, 8]

    start = time.time()
    print(f"Random interference: {duration}s, intervals {min_interval}-{max_interval}s, seed={seed}")

    # Initial idle period
    if idle_start > 0:
        print(f"  Initial idle period ({idle_start}s)")
        manager.log_event("start", "idle")
        wait_until = min(start + duration, start + idle_start)
        while time.time() < wait_until:
            time.sleep(1)

    try:
        while time.time() - start < duration:
            manager.stop_all()

            # 30% chance of idle
            if random.random() < 0.3:
                print(f"  Random idle period")
                manager.log_event("start", "idle")
            else:
                name = random.choice(benchmarks)
                threads = random.choice(thread_choices)
                manager.start_benchmark(name, threads)

            interval = random.randint(min_interval, max_interval)
            wait_until = min(start + duration, time.time() + interval)
            while time.time() < wait_until:
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for random mode (default: 42)")
    parser.add_argument("--idle-start", type=int, default=60,
                        help="Seconds of idle time at the start in random mode (default: 60)")
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

    schedule = SCHEDULES[args.schedule]
    total_duration = args.duration * len(schedule)

    if args.mode == "deterministic":
        run_deterministic(manager, args.duration, schedule=schedule)
    else:
        run_random(manager, total_duration, seed=args.seed, idle_start=args.idle_start)

    manager.save_log()


if __name__ == "__main__":
    main()
