#!/usr/bin/env python3
"""Export run data to a CSV for easy comparison. One row per run.

Usage:
    python data/runs_to_csv.py                          # all runs, print to stdout
    python data/runs_to_csv.py -o data/runs_summary.csv # save to file
    python data/runs_to_csv.py -r data/runs/run28.json data/runs/run29.json  # specific runs
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# The 4 models we always test
MODELS = ["conv_next", "conv_next_base", "efficientnet_b6", "regnet_x_16gf"]


def analyse_run(path: Path) -> dict | None:
    with open(path) as f:
        data = json.load(f)

    meta = data.get("meta", {})
    opt_kwargs = meta.get("optimizer_kwargs", {})

    row = {
        "run": path.stem,
        "commit": (meta.get("git_commit") or "?")[:8],
        "num_requests": meta.get("num_requests", ""),
        "optimizer": meta.get("optimizer", ""),
        "rebalance_interval": meta.get("rebalance_interval", ""),
        "deep_alpha": opt_kwargs.get("deep_alpha", ""),
        "sibling_alpha": opt_kwargs.get("sibling_alpha", ""),
        "tolerance": opt_kwargs.get("tolerance", ""),
        "optimum_escape": opt_kwargs.get("optimum_escape", ""),
    }

    for model in MODELS:
        if model not in data["results"]:
            row[f"{model}_rps"] = ""
            row[f"{model}_final_rps"] = ""
            row[f"{model}_rebalances"] = ""
            row[f"{model}_optimum"] = ""
            continue

        result = data["results"][model]
        batches = result.get("batches", [])
        timed = [b for b in batches if "timing" in b]
        rps = result.get("requests_per_second", 0)

        rebalances = sum(1 for b in batches if b.get("rebalance", {}).get("did_rebalance", False))
        at_optimum = sum(1 for b in batches if b.get("rebalance", {}).get("at_optimum", False))

        # Final RPS: last 10% of timed batches (wall-clock span, same method as rps)
        final_rps = 0.0
        if len(timed) >= 10:
            n_final = max(1, len(timed) // 10)
            final_batches = timed[-n_final:]
            wall_clock = final_batches[-1]["timing"]["end"] - final_batches[0]["timing"]["start"]
            final_rps = len(final_batches) / wall_clock if wall_clock > 0 else 0.0

        row[f"{model}_rps"] = round(rps, 2)
        row[f"{model}_final_rps"] = round(final_rps, 2)
        row[f"{model}_rebalances"] = rebalances
        row[f"{model}_optimum"] = at_optimum > 0

    # Average RPS and final RPS across models
    rps_vals = [row[f"{m}_rps"] for m in MODELS if isinstance(row.get(f"{m}_rps"), (int, float))]
    final_vals = [row[f"{m}_final_rps"] for m in MODELS if isinstance(row.get(f"{m}_final_rps"), (int, float))]
    row["avg_rps"] = round(sum(rps_vals) / len(rps_vals), 2) if rps_vals else ""
    row["avg_final_rps"] = round(sum(final_vals) / len(final_vals), 2) if final_vals else ""
    row["optimum_count"] = sum(1 for m in MODELS if row.get(f"{m}_optimum") is True)

    return row


def main():
    parser = argparse.ArgumentParser(description="Export run data to CSV (one row per run)")
    parser.add_argument("-r", "--runs", nargs="+", type=Path, default=None,
                        help="Specific run JSON files (default: all in data/runs/)")
    parser.add_argument("-d", "--runs-dir", type=Path, default=Path(__file__).parent / "runs",
                        help="Directory containing run JSON files")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    if args.runs:
        paths = args.runs
    else:
        paths = sorted(args.runs_dir.glob("*.json"))

    if not paths:
        print("No run files found", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in paths:
        try:
            row = analyse_run(path)
            if row:
                rows.append(row)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)

    if not rows:
        print("No data to export", file=sys.stderr)
        sys.exit(1)

    # Build columns: params first, then per-model columns
    param_cols = [
        "run", "commit", "num_requests", "optimizer",
        "rebalance_interval", "deep_alpha", "sibling_alpha", "tolerance", "optimum_escape",
    ]
    model_cols = []
    for model in MODELS:
        model_cols.extend([
            f"{model}_rps", f"{model}_final_rps",
            f"{model}_rebalances", f"{model}_optimum",
        ])

    summary_cols = ["avg_rps", "avg_final_rps", "optimum_count"]

    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=param_cols + summary_cols + model_cols)
    writer.writeheader()
    writer.writerows(rows)

    if args.output:
        out.close()
        print(f"Saved {len(rows)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
