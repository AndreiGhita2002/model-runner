import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).parent / "sample"

BASELINES = {
    "sequential": DATA_DIR / "sequential_baseline.json",
    "tensor_parallel": DATA_DIR / "tensor_parallel_baseline.json",
    "gpipe": DATA_DIR / "gpipe_baseline.json",
}


def load_baseline(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_requests_per_second(baseline: dict) -> dict[str, float]:
    """Return per-model requests per second."""
    return {
        model_name: data["requests_per_second"]
        for model_name, data in baseline["results"].items()
    }


def plot_requests_per_second(baselines: dict[str, dict], output_path: Path | None = None):
    baseline_names = list(baselines.keys())
    all_rps = {name: get_requests_per_second(bl) for name, bl in baselines.items()}

    model_names = list(next(iter(all_rps.values())).keys())

    x = np.arange(len(model_names))
    width = 0.8 / len(baseline_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, bl_name in enumerate(baseline_names):
        values = [all_rps[bl_name][model] for model in model_names]
        offset = (i - len(baseline_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=bl_name, capsize=3)

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
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    baselines = {}
    for name, path in BASELINES.items():
        if not path.exists():
            print(f"Warning: {path} not found, skipping {name}")
            continue
        baselines[name] = load_baseline(path)

    if not baselines:
        print("No baseline files found.")
        sys.exit(1)

    plot_requests_per_second(baselines, output)
