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
        # Needs unlimited stack for large static arrays compiled with icc
        "cmd": [os.environ.get("STREAM_C_PATH", "stream_c")],
        "cwd": str(Path(os.environ.get("STREAM_C_PATH", "stream_c")).parent) or None,
        "env": {"STREAM_ARRAY_SIZE": "10000000"},
        "shell_prefix": "ulimit -s unlimited && ",
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
            "step_duration": 600,
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


MAX_RESTARTS = 10
UTILIZATION_POLL_INTERVAL = 10  # seconds between CPU utilization checks
UTILIZATION_WARN_THRESHOLD = 10.0  # warn if CPU% below this


def _parse_core_range(cores: str) -> list[int]:
    """Parse a core range string like '32-39' or '32,34,36' into a list of core IDs."""
    result = []
    for part in cores.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def _read_per_cpu_ticks() -> dict[int, tuple[int, int]]:
    """Read (busy_ticks, total_ticks) per CPU from /proc/stat.

    Returns dict mapping cpu_id -> (busy, total).
    busy = user + nice + system + irq + softirq + steal
    total = busy + idle + iowait
    """
    result = {}
    try:
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu"):
                    continue
                parts = line.split()
                if parts[0] == "cpu":
                    continue  # skip aggregate line
                cpu_id = int(parts[0][3:])
                # user, nice, system, idle, iowait, irq, softirq, steal
                user, nice, system, idle, iowait, irq, softirq, steal = (
                    int(x) for x in parts[1:9]
                )
                busy = user + nice + system + irq + softirq + steal
                total = busy + idle + iowait
                result[cpu_id] = (busy, total)
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return result


class InterferenceManager:
    def __init__(self, log_file: Path | None = None, bench_log_dir: Path | None = None):
        # Map from BenchSpec -> Popen process
        self.active: dict[BenchSpec, subprocess.Popen] = {}
        # Map from BenchSpec -> open log file handle
        self._bench_log_files: dict[BenchSpec, object] = {}
        self.log: list[dict] = []
        self.log_file = log_file
        self.bench_log_dir = bench_log_dir
        self.restart_count = 0
        # CPU utilization tracking: per-core ticks from /proc/stat
        self._last_core_sample: dict[int, tuple[int, int]] = {}  # cpu_id -> (busy, total)
        self._last_utilization_check = 0.0
        self._start_time = time.monotonic()

    def _ts(self) -> str:
        """Timestamp prefix: [HH:MM:SS +Xs]"""
        elapsed = time.monotonic() - self._start_time
        clock = datetime.now().strftime("%H:%M:%S")
        return f"[{clock} +{elapsed:.0f}s]"

    def start_benchmark(self, name: str, num_threads: int = 1, cores: str = "") -> bool:
        """Start a benchmark process. Returns True if started successfully."""
        spec = (name, num_threads, cores)

        # Already running with same spec — skip
        if spec in self.active and self.active[spec].poll() is None:
            # Take an initial CPU sample so the first utilization check has a baseline
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

        # Wrap in shell if a prefix is needed (e.g. ulimit -s unlimited)
        shell_prefix = bench.get("shell_prefix")
        use_shell = shell_prefix is not None
        if use_shell:
            cmd = shell_prefix + " ".join(cmd)

        cwd = bench.get("cwd")

        try:
            # Log benchmark output to file if bench_log_dir is set
            if self.bench_log_dir:
                self.bench_log_dir.mkdir(parents=True, exist_ok=True)
                log_path = self.bench_log_dir / f"{name}_{num_threads}t_{cores}.log"
                bench_log = open(log_path, "a")
                self._bench_log_files[spec] = bench_log
                stdout_dest = bench_log
                stderr_dest = bench_log
            else:
                stdout_dest = subprocess.DEVNULL
                stderr_dest = subprocess.DEVNULL

            proc = subprocess.Popen(
                cmd,
                env=env,
                cwd=cwd,
                shell=use_shell,
                stdout=stdout_dest,
                stderr=stderr_dest,
                start_new_session=True,
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
        bench_log = self._bench_log_files.pop(spec, None)
        if bench_log is not None:
            bench_log.close()
        if proc is None:
            return
        if proc.poll() is None:
            # Kill the entire process group (needed for shell=True benchmarks
            # where terminate() would only kill the shell, not its children)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
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

        # Take a fresh /proc/stat baseline so the first utilization poll
        # doesn't compare against the previous (idle) step's samples.
        self._last_core_sample = _read_per_cpu_ticks()
        self._last_utilization_check = time.monotonic()

    def stop_all(self):
        """Stop all running benchmark processes."""
        for spec in list(self.active.keys()):
            self.stop_benchmark(spec)

    def check_and_restart(self):
        """Restart any benchmark processes that have exited."""
        for spec, proc in list(self.active.items()):
            if proc.poll() is not None:
                name, threads, cores = spec
                exit_code = proc.returncode
                self.restart_count += 1
                print(f"  WARNING: {name} (pid={proc.pid}, threads={threads}) exited "
                      f"with code {exit_code}, restarting ({self.restart_count}/{MAX_RESTARTS})",
                      file=sys.stderr)
                self.log_event("restart", name, threads, pid=proc.pid)
                del self.active[spec]
                if self.restart_count >= MAX_RESTARTS:
                    print(f"  ERROR: too many benchmark restarts ({MAX_RESTARTS}), aborting",
                          file=sys.stderr)
                    self.stop_all()
                    sys.exit(1)
                self.start_benchmark(name, threads, cores=cores)

    def check_utilization(self):
        """Sample CPU utilization of interference cores and log it.

        Reads /proc/stat to compute per-core CPU% since the last sample.
        Checks cores for each active benchmark and warns if usage is low.
        """
        now = time.monotonic()
        if now - self._last_utilization_check < UTILIZATION_POLL_INTERVAL:
            return
        self._last_utilization_check = now

        current = _read_per_cpu_ticks()
        if not current:
            return

        for spec, proc in list(self.active.items()):
            if proc.poll() is not None:
                continue
            name, threads, cores_str = spec
            if not cores_str:
                continue

            core_ids = _parse_core_range(cores_str)
            core_pcts = []

            for cid in core_ids:
                if cid not in current:
                    continue
                prev = self._last_core_sample.get(cid)
                cur = current[cid]
                if prev is None:
                    continue
                d_busy = cur[0] - prev[0]
                d_total = cur[1] - prev[1]
                if d_total > 0:
                    core_pcts.append(d_busy / d_total * 100.0)

            if not core_pcts:
                continue  # need two samples

            avg_pct = sum(core_pcts) / len(core_pcts)

            entry = {
                "time": time.perf_counter(),
                "event": "utilization",
                "benchmark": name,
                "threads": threads,
                "pid": proc.pid,
                "cores": cores_str,
                "avg_core_pct": round(avg_pct, 1),
                "per_core_pct": [round(p, 1) for p in core_pcts],
            }
            self.log.append(entry)

            if avg_pct < UTILIZATION_WARN_THRESHOLD:
                print(f"  {self._ts()} WARNING: {name} (pid={proc.pid}, threads={threads}, "
                      f"cores={cores_str}) avg core usage {avg_pct:.1f}% — expected ~100%",
                      file=sys.stderr)

        # Update stored samples for all cores
        self._last_core_sample = current

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
                manager.check_and_restart()
                manager.check_utilization()
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
                manager.check_and_restart()
                manager.check_utilization()
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

    bench_log_dir = args.output.parent / "bench_logs" if args.output else None
    manager = InterferenceManager(log_file=args.output, bench_log_dir=bench_log_dir)

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
