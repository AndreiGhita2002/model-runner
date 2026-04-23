"""Export paper-ready graphs as transparent PDFs with captions disabled.

Run from the repo root:

    uv run python paper_graphs.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import sys

from data.notebook_utils import (  # noqa: E402
    load_experiment, collect_model_names,
    plot_experiment_throughput_avg, plot_experiment_rps_per_stage_avg,
    plot_experiment_batch_times, plot_interf_optimizer_state,
)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "data"))

target_dir = "docs/paper/graphs"


## What graphs we want: (so far)
# - average RPS per benchmark per model graph
# - average RPS per benchmark per interference per model graph
#

# How the graphs should be:
# - all exported as pdf
# - no captions or other things that should be written in the latex paper
# - lightmode and transparent if possible
# - should fit in double column, it's better to export subgraphs as separate graphs,
#   and we will arrange them on the latex layer
# - keep these comments


EXPERIMENT_DIRS = [
    _HERE / "data/experiments/exp15-1",
    _HERE / "data/experiments/exp15-2",
    _HERE / "data/experiments/exp15-3",
    _HERE / "data/experiments/exp15-4",
    _HERE / "data/experiments/exp15-5",
]

RUN_INFO = {
    "A": {"label": "GPipe (no interf)",             "color": "#2196f3", "display": "A"},
    "B": {"label": "Shisha (no interf)",            "color": "#4caf50", "display": "B"},
    "C": {"label": "Benchmark A: GPipe",            "color": "#f44336", "display": "A"},
    "D": {"label": "Benchmark B: ExhaustiveShisha", "color": "#ff9800", "display": "B"},
    "E": {"label": "Benchmark C: ReactiveShisha",   "color": "#9c27b0", "display": "C"},
}

MODEL_ORDER = [
    "mobilenet_v3_large",
    "efficientnet_b6",
    "conv_next",
    "regnet_x_16gf",
]

MODEL_LABELS = {
    "mobilenet_v3_large": "MobileNet",
    "efficientnet_b6":    "EfficientNet",
    "conv_next":          "ConvNeXt-Small",
    "regnet_x_16gf":      "RegNet X 16GF",
}


def _load_runs() -> tuple[list, list[str]]:
    runs = []
    for d in EXPERIMENT_DIRS:
        if not d.exists():
            print(f"  skipping missing {d}")
            continue
        r, _ = load_experiment(d)
        runs.append(r)
    all_models = collect_model_names(runs[0])
    all_models = [m for m in MODEL_ORDER if m in all_models] + \
                 [m for m in all_models if m not in MODEL_ORDER]
    return runs, all_models


def _capture(plot_fn, *args, **kwargs):
    """Call a plot_* helper and return its figure instead of letting
    ``plt.show()`` drop it on the floor."""
    real_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        plot_fn(*args, **kwargs)
        return plt.gcf()
    finally:
        plt.show = real_show


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _export(path: Path, plot_fn, *args, **kwargs) -> None:
    """Skip if the PDF already exists; otherwise capture and save it."""
    if path.exists():
        print(f"  skipping existing {path.name}")
        return
    fig = _capture(plot_fn, *args, **kwargs)
    _save(fig, path)


# Which repetition / model to feature for the per-run detail figures.
FEATURED_REP_IDX = 2     # runs[2] → exp15-3
FEATURED_MODEL = "conv_next"


def main() -> None:
    out = _HERE / target_dir
    print(f"Exporting paper graphs to {out}/")
    runs, all_models = _load_runs()

    # Average throughput per benchmark per model — one figure covers all models.
    _export(out / "throughput_avg.pdf",
            plot_experiment_throughput_avg, runs, RUN_INFO, all_models,
            run_ids=["C", "D", "E"], show_caption=False)

    # Average RPS per interference stage — one PDF per model so the LaTeX
    # layout can arrange them independently in the double-column grid.
    for model in all_models:
        _export(out / f"rps_per_stage_{model}.pdf",
                plot_experiment_rps_per_stage_avg,
                runs, RUN_INFO, [model], show_caption=False)

    # Featured per-run detail figures: batch-time series under interference and
    # the matching optimiser state (from ReactiveShisha, run E) for the same
    # (rep, model) pairing shown in the notebook.
    featured_run = runs[FEATURED_REP_IDX]
    _export(out / f"batch_times_{FEATURED_MODEL}.pdf",
            plot_experiment_batch_times, featured_run, RUN_INFO,
            [FEATURED_MODEL], show_optimum=False, show_caption=False,
            shade_interference=False)

    e_data = featured_run["E"]
    e_filtered = {**e_data,
                  "results": {FEATURED_MODEL: e_data["results"][FEATURED_MODEL]}}
    _export(out / f"optimizer_state_{FEATURED_MODEL}.pdf",
            plot_interf_optimizer_state, e_filtered, show_caption=False)


if __name__ == "__main__":
    main()
