import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).parent / "sample"

DATASETS = {
    "sequential": DATA_DIR / "sequential_baseline.json",
    "tensor_parallel": DATA_DIR / "tensor_parallel_baseline.json",
    "gpipe": DATA_DIR / "gpipe_baseline.json",
    "adaptive": DATA_DIR / "evaluation_output.json",
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


if __name__ == "__main__":
    graph_output = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    datasets = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"Warning: {path} not found, skipping {name}")
            continue
        # loading the file
        with open(path) as f:
            datasets[name] = json.load(f)

    if not datasets:
        print("No data files found.")
        sys.exit(1)

    plot_requests_per_second(datasets, graph_output)
