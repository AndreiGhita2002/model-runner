import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Fixed baseline filenames and colors
BASELINE_FILES = {
    "sequential": "sequential.json",
    "tensor_parallel": "tensor_parallel.json",
    "gpipe": "gpipe.json",
}

BASELINE_COLORS = {
    "sequential": "red",
    "tensor_parallel": "blue",
    "gpipe": "green",
}

# Color palette for runs (first few are hand-picked for distinctness)
RUN_COLORS = ["purple", "orange", "brown", "teal", "magenta", "olive", "cyan"]


def _get_run_color(index: int) -> str:
    """Return a color for the given run index."""
    if index < len(RUN_COLORS):
        return RUN_COLORS[index]
    cmap = plt.cm.tab20
    return cmap((index - len(RUN_COLORS)) / 20)


def get_requests_per_second(dataset: dict) -> dict[str, float]:
    """Return per-model requests per second."""
    return {
        model_name: data["requests_per_second"]
        for model_name, data in dataset["results"].items()
    }


def _get_rebalance_events(batches: list[dict]) -> list[int]:
    """Find the first index of each contiguous did_rebalance=True block."""
    events = []
    prev = False
    for i, b in enumerate(batches):
        did = b.get("rebalance", {}).get("did_rebalance", False)
        if did and not prev:
            events.append(i)
        prev = did
    return events


def _collect_model_names(*dataset_dicts: dict[str, dict]) -> list[str]:
    """Collect unique model names across all dataset dicts, preserving order."""
    seen = set()
    names = []
    for datasets in dataset_dicts:
        for ds in datasets.values():
            for m in ds["results"]:
                if m not in seen:
                    seen.add(m)
                    names.append(m)
    return names


def _legend_kwargs(n_entries: int) -> dict:
    """Return legend kwargs, using multi-column layout for many entries."""
    if n_entries > 6:
        return {"fontsize": "x-small", "ncol": 2}
    return {"fontsize": "small"}


def plot_requests_per_second(
    baselines: dict[str, dict],
    runs: dict[str, dict],
    output_path: Path | None = None,
):
    # Ordered: baselines first (fixed order), then runs (alphabetical)
    all_datasets = list(baselines.items()) + list(runs.items())
    dataset_names = [name for name, _ in all_datasets]

    all_rps = {name: get_requests_per_second(ds) for name, ds in all_datasets}

    model_names = _collect_model_names(baselines, runs)
    x = np.arange(len(model_names))
    width = 0.8 / len(dataset_names)

    # Build color map
    colors = {}
    for name in baselines:
        colors[name] = BASELINE_COLORS.get(name, "gray")
    for i, name in enumerate(runs):
        colors[name] = _get_run_color(i)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ds_name in enumerate(dataset_names):
        values = [all_rps[ds_name].get(model, 0) for model in model_names]
        offset = (i - len(dataset_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=ds_name, color=colors[ds_name], capsize=3)

    ax.set_xlabel("Model")
    ax.set_ylabel("Requests per second")
    ax.set_title("Throughput per model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.legend(**_legend_kwargs(len(dataset_names)))
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_batch_times(
    baselines: dict[str, dict],
    runs: dict[str, dict],
    output_path: Path | None = None,
):
    """Plot per-batch elapsed time for each model, one subplot per model."""
    model_names = _collect_model_names(baselines, runs)

    if not model_names:
        print("No models found for batch time plot")
        return

    # Build color map
    colors = {}
    for name in baselines:
        colors[name] = BASELINE_COLORS.get(name, "gray")
    for i, name in enumerate(runs):
        colors[name] = _get_run_color(i)

    n_models = len(model_names)
    cols = min(n_models, 2)
    rows = math.ceil(n_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    # Find the longest batch series for x-axis range of horizontal lines
    max_batches = 0
    for ds in list(baselines.values()) + list(runs.values()):
        for model in model_names:
            if model in ds["results"]:
                n = len(ds["results"][model].get("batches", []))
                max_batches = max(max_batches, n)

    for idx, model in enumerate(model_names):
        ax = axes[idx // cols][idx % cols]

        # Draw baselines as horizontal dashed lines at mean elapsed time
        for ds_name, ds in baselines.items():
            if model not in ds["results"]:
                continue
            batches = ds["results"][model].get("batches", [])
            if not batches:
                continue
            elapsed = [b["timing"]["end"] - b["timing"]["start"] for b in batches if "timing" in b]
            if not elapsed:
                continue
            mean_elapsed = sum(elapsed) / len(elapsed)
            ax.axhline(mean_elapsed, color=colors[ds_name], linestyle="--", alpha=0.7, label=ds_name)

        # Draw runs as line plots with rebalance markers
        for ds_name, ds in runs.items():
            if model not in ds["results"]:
                continue
            batches = ds["results"][model].get("batches", [])
            if not batches:
                continue
            elapsed = [b["timing"]["end"] - b["timing"]["start"] for b in batches if "timing" in b]
            if not elapsed:
                continue
            color = colors[ds_name]
            ax.plot(range(len(elapsed)), elapsed, color=color, label=ds_name, marker=".", markersize=4)

            # Mark rebalance events
            rebalance_events = _get_rebalance_events(batches)
            for j, event_idx in enumerate(rebalance_events):
                ax.axvline(
                    event_idx, color=color, linestyle=":", alpha=0.3,
                    label=f"{ds_name} rebalance" if j == 0 else None,
                )

        ax.set_title(model)
        ax.set_xlabel("Batch index")
        ax.set_ylabel("Elapsed time (s)")
        ax.legend(**_legend_kwargs(len(baselines) + len(runs) * 2))

    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Per-batch elapsed time", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def load_baselines(baselines_dir: Path) -> dict[str, dict]:
    """Load baseline JSON files from the baselines directory."""
    baselines = {}
    for name, filename in BASELINE_FILES.items():
        path = baselines_dir / filename
        if not path.exists():
            print(f"Warning: {path} not found, skipping {name}")
            continue
        with open(path) as f:
            baselines[name] = json.load(f)
    return baselines


def load_runs(runs_dir: Path) -> dict[str, dict]:
    """Load all run JSON files from the runs directory, sorted alphabetically."""
    runs = {}
    if not runs_dir.is_dir():
        print(f"Warning: {runs_dir} is not a directory, no runs loaded")
        return runs
    for path in sorted(runs_dir.glob("*.json")):
        with open(path) as f:
            runs[path.stem] = json.load(f)
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate throughput graphs from baseline/evaluation data")
    parser.add_argument("-b", "--baselines-dir", type=Path, default=Path(__file__).parent / "baselines",
                        help="Directory containing baseline JSON files (default: data/baselines/)")
    parser.add_argument("-r", "--runs-dir", type=Path, default=Path(__file__).parent / "runs",
                        help="Directory containing run JSON files (default: data/runs/)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output image path (e.g. graphs.png). Displays interactively if not set.")
    args = parser.parse_args()

    baselines = load_baselines(args.baselines_dir)
    runs = load_runs(args.runs_dir)

    if not baselines and not runs:
        print(f"No data files found in {args.baselines_dir} or {args.runs_dir}")
        sys.exit(1)

    output = args.output
    if output:
        plot_requests_per_second(baselines, runs, output)
        batch_times_path = output.with_name(f"{output.stem}_batch_times{output.suffix}")
        plot_batch_times(baselines, runs, batch_times_path)
    else:
        plot_requests_per_second(baselines, runs)
        plot_batch_times(baselines, runs)
