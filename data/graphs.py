import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Expected filenames within the input directory
DATASET_FILES = {
    "sequential": "sequential_baseline.json",
    "tensor_parallel": "tensor_parallel_baseline.json",
    "gpipe": "gpipe_baseline.json",
    "adaptive": "evaluation_output.json",
}


def get_requests_per_second(baseline: dict) -> dict[str, float]:
    """Return per-model requests per second."""
    return {
        model_name: data["requests_per_second"]
        for model_name, data in baseline["results"].items()
    }


def plot_requests_per_second(datasets: dict[str, dict], output_path: Path | None = None):
    dataset_names = list(datasets.keys())
    all_rps = {name: get_requests_per_second(ds) for name, ds in datasets.items()}

    # Union of all model names (preserving order from first dataset that has each)
    seen = set()
    model_names = []
    for rps in all_rps.values():
        for m in rps:
            if m not in seen:
                seen.add(m)
                model_names.append(m)

    x = np.arange(len(model_names))
    width = 0.8 / len(dataset_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ds_name in enumerate(dataset_names):
        values = [all_rps[ds_name].get(model, 0) for model in model_names]
        offset = (i - len(dataset_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=ds_name, capsize=3)

    ax.set_xlabel("Model")
    ax.set_ylabel("Requests per second")
    ax.set_title("Throughput per model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_batch_times(datasets: dict[str, dict], output_path: Path | None = None):
    """Plot per-batch elapsed time for each model, one subplot per model."""
    # Collect all model names across datasets
    seen = set()
    model_names = []
    for ds in datasets.values():
        for m in ds["results"]:
            if m not in seen:
                seen.add(m)
                model_names.append(m)

    if not model_names:
        print("No models found for batch time plot")
        return

    n_models = len(model_names)
    cols = min(n_models, 2)
    rows = math.ceil(n_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    prev_did_rebalance = False

    DATASET_COLORS = {
        "sequential": "red",
        "tensor_parallel": "blue",
        "gpipe": "green",
        "adaptive": "purple",
    }

    # Find the longest batch series to set x-axis range for horizontal lines
    max_batches = 0
    for ds in datasets.values():
        for model in model_names:
            if model in ds["results"]:
                n = len(ds["results"][model].get("batches", []))
                max_batches = max(max_batches, n)

    for idx, model in enumerate(model_names):
        ax = axes[idx // cols][idx % cols]
        for ds_name, ds in datasets.items():
            if model not in ds["results"]:
                continue
            batches = ds["results"][model].get("batches", [])
            if not batches:
                continue
            elapsed = [b["timing"]["end"] - b["timing"]["start"] for b in batches if "timing" in b]
            if not elapsed:
                continue

            color = DATASET_COLORS.get(ds_name)

            # Baselines with few batches: draw as horizontal line at mean elapsed time
            if ds_name != "adaptive" and len(elapsed) < max_batches // 2:
                mean_elapsed = sum(elapsed) / len(elapsed)
                ax.axhline(mean_elapsed, color=color, linestyle="--", alpha=0.7, label=ds_name)
            else:
                ax.plot(range(len(elapsed)), elapsed, color=color, label=ds_name, marker=".", markersize=4)

            # Mark rebalance points for adaptive dataset
            if ds_name == "adaptive":
                for i, b in enumerate(batches):
                    rb = b.get("rebalance", {})

                    if rb.get("did_rebalance"):
                        if not prev_did_rebalance:
                            prev_did_rebalance = True
                            ax.axvline(i, color="red", linestyle="--", alpha=0.6,
                                       label="rebalance" if i == next(
                                           j for j, bb in enumerate(batches)
                                           if bb.get("rebalance", {}).get("did_rebalance")
                                       ) else None)
                    else:
                        prev_did_rebalance = False

        ax.set_title(model)
        ax.set_xlabel("Batch index")
        ax.set_ylabel("Elapsed time (s)")
        ax.legend(fontsize="small")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate throughput graphs from baseline/evaluation data")
    parser.add_argument("-i", "--input-dir", type=Path, default=Path(__file__).parent / "sample",
                        help="Directory containing JSON data files (default: data/sample/)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output image path (e.g. graphs.png). Displays interactively if not set.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    datasets = {}
    for name, filename in DATASET_FILES.items():
        path = input_dir / filename
        if not path.exists():
            print(f"Warning: {path} not found, skipping {name}")
            continue
        with open(path) as f:
            datasets[name] = json.load(f)

    if not datasets:
        print(f"No data files found in {input_dir}")
        sys.exit(1)

    output = args.output
    if output:
        plot_requests_per_second(datasets, output)
        batch_times_path = output.with_name(f"{output.stem}_batch_times{output.suffix}")
        plot_batch_times(datasets, batch_times_path)
    else:
        plot_requests_per_second(datasets)
        plot_batch_times(datasets)
