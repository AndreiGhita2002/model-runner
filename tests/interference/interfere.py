"""Interference script for the interference experiment.

Starts and stops benchmark processes to simulate system interference
while the inference server is running.

Supports two modes:
- deterministic: follows a fixed schedule of interference patterns
- random: randomly starts/stops processes at random intervals

Benchmark processes:
- CPU stress: CPU_stress (github.com/nikela/CPU_stress)
- Memory bandwidth: MLC or STREAM (github.com/intel/memory-bandwidth-benchmarks)

Usage:
    python -m tests.interference.interfere [options]
    python -m tests.interference.interfere --duration 600 --mode deterministic
"""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Benchmark commands — adjust paths as needed for your system
BENCHMARKS = {
    "cpu_stress": {
        "cmd": ["CPU_stress"],
        "description": "CPU-intensive workload",
    },
    "memory_bandwidth": {
        # STREAM with array size > L3 cache (80MB = 10M doubles)
        "cmd": ["stream_c.exe"],
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

        try:
            proc = subprocess.Popen(
                bench["cmd"],
                env=env,
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


def run_deterministic(manager: InterferenceManager, duration: int, interval: int):
    """Run a deterministic interference schedule.

    Cycles through benchmarks, changing every `interval` seconds.
    Uses varying thread counts: 1, 2, 4, 8.
    """
    #TODO: this schedule is too 'random' for deterministic benchmark
    schedule = [
        ("idle", 0),
        ("cpu_stress", 2),
        ("cpu_stress", 4),
        ("memory_bandwidth", 1),
        ("cpu_stress", 8),
        ("memory_bandwidth", 2),
        ("idle", 0),
        ("cpu_stress", 1),
        ("memory_bandwidth", 4),
    ]

    start = time.time()
    step = 0

    print(f"Deterministic interference: {duration}s, change every {interval}s")
    try:
        while time.time() - start < duration:
            name, threads = schedule[step % len(schedule)]
            manager.stop_all()

            if name != "idle":
                manager.start_benchmark(name, threads)
            else:
                print(f"  Idle period ({interval}s)")
                manager.log_event("start", "idle")

            # Wait for the interval or until duration ends
            wait_until = min(start + duration, time.time() + interval)
            while time.time() < wait_until:
                time.sleep(1)

            step += 1
    except KeyboardInterrupt:
        print("\nInterference interrupted.")
    finally:
        manager.stop_all()


def run_random(manager: InterferenceManager, duration: int,
               min_interval: int = 30, max_interval: int = 120,
               seed: int = 42, idle_start: int = 60):
    """Run random interference.

    Randomly picks benchmarks and thread counts, with random durations.
    Starts with an idle period so the optimizer can settle before interference begins.

    Args:
        seed: Random seed for reproducibility. Saved in the interference log.
        idle_start: Seconds of idle time at the beginning before interference starts.
    """
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
    parser.add_argument("--duration", type=int, default=600,
                        help="Total duration in seconds (default: 600 = 10 min)")
    parser.add_argument("--mode", choices=["deterministic", "random"], default="deterministic",
                        help="Interference mode (default: deterministic)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between benchmark changes in deterministic mode (default: 60)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output JSON file for interference log")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for random mode (default: 42)")
    parser.add_argument("--idle-start", type=int, default=60,
                        help="Seconds of idle time at the start in random mode (default: 60)")
    args = parser.parse_args()

    manager = InterferenceManager(log_file=args.output)

    # Clean up on SIGTERM
    signal.signal(signal.SIGTERM, lambda *_: (manager.stop_all(), sys.exit(0)))

    if args.mode == "deterministic":
        run_deterministic(manager, args.duration, args.interval)
    else:
        run_random(manager, args.duration, seed=args.seed, idle_start=args.idle_start)

    manager.save_log()


if __name__ == "__main__":
    main()
