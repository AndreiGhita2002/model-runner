#!/usr/bin/env python3
"""Export interference run data to CSV. One row per model per run.

Usage:
    python data/interference_to_csv.py                                  # all runs, stdout
    python data/interference_to_csv.py -o data/interference_summary.csv # save to file
    python data/interference_to_csv.py -r data/interference/interf3.json
    python data/interference_to_csv.py --baseline data/runs/run41.json  # include baseline
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def get_interference_regions(interf_log: dict) -> list[tuple[float, float, str, int]]:
    """Extract (start, end, benchmark, threads) regions from an interference log."""
    events = interf_log.get("events", [])
    regions = []
    current_bench = None
    current_start = None
    current_threads = 0

    for e in events:
        if e["event"] == "start":
            if current_bench is not None:
                regions.append((current_start, e["time"], current_bench, current_threads))
            current_bench = e["benchmark"]
            current_start = e["time"]
            current_threads = e.get("threads", 0)
        elif e["event"] == "stop" and e.get("benchmark") == "all":
            if current_bench is not None:
                regions.append((current_start, e["time"], current_bench, current_threads))
                current_bench = None

    return regions


def compute_clock_offset(timed_batches: list[dict], regions: list[tuple]) -> float:
    """Compute offset between perf_counter (batches) and time.time (interference).

    Both start roughly at the same wall-clock moment, so we align the first
    batch timestamp with the first interference event.
    """
    if not timed_batches or not regions:
        return 0.0
    batch_start = timed_batches[0]["timing"]["start"]
    region_start = regions[0][0]
    return region_start - batch_start


def step_rps(timed_batches: list[dict], region: tuple[float, float, str, int],
             clock_offset: float = 0.0) -> float:
    """Compute RPS for batches within a time region, adjusting for clock offset."""
    r_start, r_end, _, _ = region
    # Convert region bounds to batch clock
    r_start_adj = r_start - clock_offset
    r_end_adj = r_end - clock_offset
    step_batches = [b for b in timed_batches if r_start_adj <= b["timing"]["start"] < r_end_adj]
    if len(step_batches) >= 2:
        wall = step_batches[-1]["timing"]["end"] - step_batches[0]["timing"]["start"]
        return len(step_batches) / wall if wall > 0 else 0
    return 0


def step_label(name: str, threads: int, cores: str = "") -> str:
    """Short label for a schedule step."""
    if name == "idle":
        return "idle"
    cores_str = f"_c{cores}" if cores else ""
    return f"{name}_{threads}t{cores_str}"


def schedule_summary(steps: list) -> str:
    """One-line readable schedule summary."""
    parts = []
    for s in steps:
        name, threads = s[0], s[1]
        cores = s[2] if len(s) > 2 else ""
        parts.append(step_label(name, threads, cores))
    return " -> ".join(parts)


def analyse_run(path: Path) -> list[dict]:
    """Analyse one interference JSON, returning one row per model."""
    with open(path) as f:
        data = json.load(f)

    meta = data.get("meta", {})
    schedule_steps = meta.get("schedule_steps", [])
    interference_logs = data.get("interference", {})
    results = data.get("results", {})
    run_name = path.stem

    rows = []
    for model_name, model_result in results.items():
        batches = model_result.get("batches", [])
        timed = [b for b in batches if "timing" in b]
        overall_rps = model_result.get("requests_per_second", 0)

        row = {
            "model/run": f"{model_name}/{run_name}",
            "overall_rps": f"{overall_rps:.2f}",
            "batches": len(timed),
        }

        # Per-step RPS
        interf_log = interference_logs.get(model_name, {})
        regions = get_interference_regions(interf_log)
        offset = compute_clock_offset(timed, regions)

        for i, step in enumerate(schedule_steps):
            name, threads = step[0], step[1]
            cores = step[2] if len(step) > 2 else ""
            label = step_label(name, threads, cores)

            if i < len(regions):
                rps = step_rps(timed, regions[i], clock_offset=offset)
                row[label] = f"{rps:.2f}" if rps > 0 else "N/A"
            else:
                row[label] = "N/A"

        row["schedule"] = schedule_summary(schedule_steps)
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Export interference runs to CSV")
    parser.add_argument("-d", "--dir", type=Path, default=Path("data/interference"),
                        help="Directory containing interference JSON files")
    parser.add_argument("-r", "--runs", nargs="+", type=Path, default=None,
                        help="Specific JSON files to process")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CSV path (default: stdout)")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Baseline run JSON (e.g. data/runs/run41.json)")
    args = parser.parse_args()

    if args.runs:
        files = args.runs
    else:
        files = sorted(args.dir.glob("*.json"))

    if not files:
        print("No interference runs found.", file=sys.stderr)
        sys.exit(1)

    # Load baseline if provided
    baseline_rps: dict[str, tuple[float, int]] = {}
    baseline_name = ""
    if args.baseline and args.baseline.exists():
        with open(args.baseline) as f:
            bl_data = json.load(f)
        baseline_name = args.baseline.stem
        for model_name, result in bl_data.get("results", {}).items():
            rps = result.get("requests_per_second", 0)
            n_batches = len([b for b in result.get("batches", []) if "timing" in b])
            baseline_rps[model_name] = (rps, n_batches)

    all_rows = []
    for f in files:
        try:
            all_rows.extend(analyse_run(f))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {f}: {e}", file=sys.stderr)

    # Add baseline rows (RPS in the idle column, N/A for interference steps)
    if baseline_rps and all_rows:
        # Use first run's step columns to build matching baseline rows
        sample_row = all_rows[0]
        step_keys = [k for k in sample_row if k not in ("model/run", "overall_rps", "batches", "schedule")]
        for model_name, (rps, n_batches) in baseline_rps.items():
            row = {
                "model/run": f"{model_name}/{baseline_name}",
                "overall_rps": f"{rps:.2f}",
                "batches": n_batches,
            }
            for i, key in enumerate(step_keys):
                row[key] = f"{rps:.2f}" if i == 0 else ""  # RPS in idle column only
            row["schedule"] = "no interference"
            all_rows.append(row)

    if not all_rows:
        print("No data to export.", file=sys.stderr)
        sys.exit(1)

    # Collect all column names (preserving order)
    columns = []
    for row in all_rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    for row in all_rows:
        writer.writerow(row)

    if args.output:
        out.close()
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
