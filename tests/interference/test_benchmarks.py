"""Quick smoke test for interference benchmarks.

Runs each benchmark for a short duration and samples CPU utilization
at a high rate. Use this to verify benchmarks are actually working
on the current machine before running a full experiment.

Usage:
    uv run python -m tests.interference.test_benchmarks
    uv run python -m tests.interference.test_benchmarks --duration 20 --poll 0.5
    uv run python -m tests.interference.test_benchmarks --cores 32-39
"""

import argparse
import time
from pathlib import Path

from tests.interference.interfere import (
    BENCHMARKS,
    UTILIZATION_POLL_INTERVAL,
    InterferenceManager,
)


def main():
    parser = argparse.ArgumentParser(description="Smoke test interference benchmarks")
    parser.add_argument("--duration", type=int, default=10,
                        help="Seconds to run each benchmark (default: 10)")
    parser.add_argument("--poll", type=float, default=0.1,
                        help="Utilization poll interval in seconds (default: 0.1)")
    parser.add_argument("--cores", type=str, default="32-39",
                        help="CPU cores to pin benchmarks to (default: 32-39)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads per benchmark (default: 8)")
    args = parser.parse_args()

    # Temporarily override the poll interval
    import tests.interference.interfere as interfere_mod
    original_interval = interfere_mod.UTILIZATION_POLL_INTERVAL
    interfere_mod.UTILIZATION_POLL_INTERVAL = args.poll

    benchmarks = [name for name, b in BENCHMARKS.items() if b["cmd"] is not None]

    print(f"Testing {len(benchmarks)} benchmarks: {', '.join(benchmarks)}")
    print(f"Duration: {args.duration}s each, poll: {args.poll}s, "
          f"cores: {args.cores}, threads: {args.threads}")
    print()

    all_ok = True

    try:
        for bench_name in benchmarks:
            print(f"--- {bench_name} ---")
            manager = InterferenceManager(
                bench_log_dir=Path("/tmp/interference_test_logs"),
            )
            # Reset start time for clean elapsed display
            manager._start_time = time.monotonic()

            ok = manager.start_benchmark(bench_name, num_threads=args.threads, cores=args.cores)
            if not ok:
                print(f"  FAILED to start\n")
                all_ok = False
                continue

            # Collect utilization samples
            samples = []
            start = time.monotonic()
            while time.monotonic() - start < args.duration:
                manager.check_and_restart()
                manager.check_utilization()

                # Extract latest utilization sample if one was just added
                for entry in manager.log:
                    if entry["event"] == "utilization" and entry not in samples:
                        samples.append(entry)

                time.sleep(args.poll)

            manager.stop_all()

            # Report
            if not samples:
                print(f"  No utilization samples collected!")
                all_ok = False
            else:
                pcts = [s["avg_core_pct"] for s in samples]
                avg_cpu = sum(pcts) / len(pcts)
                min_cpu = min(pcts)
                max_cpu = max(pcts)
                status = "OK" if avg_cpu > 50.0 else "FAIL"
                if status == "FAIL":
                    all_ok = False

                print(f"  Samples: {len(pcts)}")
                print(f"  Avg core usage: avg={avg_cpu:.1f}%, min={min_cpu:.1f}%, "
                      f"max={max_cpu:.1f}% (expected ~100%)")
                print(f"  Status: {status}")

            # Show benchmark output
            log_path = Path(f"/tmp/interference_test_logs/{bench_name}_{args.threads}t_{args.cores}.log")
            if log_path.exists():
                content = log_path.read_text().strip()
                if content:
                    lines = content.splitlines()
                    preview = lines[:5]
                    print(f"  Output ({len(lines)} lines): {preview[0][:80]}")
                    if len(lines) > 1:
                        print(f"    ...{lines[-1][:80]}")
                else:
                    print(f"  Output: (empty)")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
        manager.stop_all()

    finally:
        interfere_mod.UTILIZATION_POLL_INTERVAL = original_interval

    if all_ok:
        print("All benchmarks OK.")
    else:
        print("Some benchmarks FAILED — check output above.")


if __name__ == "__main__":
    main()
