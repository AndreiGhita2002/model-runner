"""Shared plotting utilities for experiment notebooks.

All heavy plotting logic lives here. Notebooks should import functions
and call them with configuration options — keeping cells short and readable.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

STEP_PALETTE = ["#66bb6a", "#ffb74d", "#ef5350", "#ab47bc", "#42a5f5", "#8d6e63"]
STAGE_COLORS = ["#2196f3", "#4caf50", "#f44336", "#ff9800", "#9c27b0", "#00bcd4", "#795548", "#607d8b"]
_INTERF_CMAP = plt.cm.Grays


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_runs(directory: Path) -> dict:
    """Load all JSON run files from a directory, keyed by stem name."""
    runs = {}
    if not directory.exists():
        return runs
    for f in sorted(directory.glob("*.json")):
        with open(f) as fh:
            runs[f.stem] = json.load(fh)
    return runs


def load_experiment(experiment_dir: Path) -> tuple[dict, dict]:
    """Load all runs and metadata from an experiment directory.

    Returns (runs_dict, experiment_meta).
    """
    runs = {}
    for run_id in ["A", "B", "C", "D", "E"]:
        path = experiment_dir / f"run_{run_id}.json"
        if path.exists():
            with open(path) as f:
                runs[run_id] = json.load(f)
            print(f"Loaded run {run_id}: {path.name}")
        else:
            print(f"Missing run {run_id}: {path.name}")

    meta = {}
    meta_path = experiment_dir / "experiment_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\nExperiment: {meta}")

    return runs, meta


def collect_model_names(*datasets) -> list[str]:
    """Collect and sort all model names across multiple run dicts."""
    names = set()
    for ds in datasets:
        for data in ds.values():
            names.update(data.get("results", {}).keys())
    return sorted(names)


def build_color_map(*datasets) -> dict[str, str]:
    """Assign colors to run names from a colormap."""
    all_names = []
    for ds in datasets:
        all_names.extend(ds.keys())
    cmap = plt.cm.tab10
    return {name: cmap(i / max(len(all_names), 1)) for i, name in enumerate(all_names)}


# ──────────────────────────────────────────────
# Timing helpers
# ──────────────────────────────────────────────

def batch_elapsed(batch: dict, include_rebalance: bool = False) -> float:
    """Compute elapsed time for a single batch."""
    t = batch["timing"]
    base = t["end"] - t["start"]
    if include_rebalance:
        reb = batch.get("rebalance", {})
        base += reb.get("end", 0) - reb.get("start", 0)
    return base


def get_rebalance_events(batches: list[dict]) -> list[int]:
    """Return batch indices where rebalance occurred."""
    return [i for i, b in enumerate(batches) if b.get("rebalance", {}).get("did_rebalance", False)]


def get_optimum_transitions(batches: list[dict]) -> tuple[list[int], list[int]]:
    """Return (enter_indices, leave_indices) for optimum state transitions."""
    opt_flags = [b.get("rebalance", {}).get("at_optimum", False) for b in batches]
    enters, leaves = [], []
    for i in range(len(opt_flags)):
        prev = opt_flags[i - 1] if i > 0 else False
        if opt_flags[i] and not prev:
            enters.append(i)
        elif not opt_flags[i] and prev:
            leaves.append(i - 1)
    return enters, leaves


# ──────────────────────────────────────────────
# Interference schedule helpers
# ──────────────────────────────────────────────

def get_model_schedule(data: dict, model: str) -> tuple[list, int]:
    """Get (schedule_steps, step_duration) for a model from run data."""
    meta = data.get("meta", {})
    ms = meta.get("model_schedules", {})
    for key in (model, "all"):
        if key in ms:
            return ms[key].get("schedule_steps", []), ms[key].get("step_duration", 0)
    return meta.get("schedule_steps", []), meta.get("step_duration", 0)


def get_interference_regions(data: dict, model: str) -> list[tuple[float, float]]:
    """Get interference step boundaries as (start, end) wall-clock times."""
    interf_log = data.get("interference", {}).get(model, {})
    events = interf_log.get("events", [])
    if not events:
        return []
    schedule_steps, step_duration = get_model_schedule(data, model)
    first_time = events[0]["time"]
    return [(first_time + i * step_duration, first_time + (i + 1) * step_duration)
            for i in range(len(schedule_steps))]


def find_first_interference_time(data: dict, model: str) -> float | None:
    """Find the timestamp of the first non-idle interference event."""
    interf_log = data.get("interference", {}).get(model, {})
    for e in interf_log.get("events", []):
        if e.get("event") == "start" and e.get("benchmark") not in ("idle", "random", None):
            return e["time"]
    return None


def compute_clock_offset(timed_batches: list, regions: list) -> float:
    """Compute offset between batch clock and interference clock."""
    if not timed_batches or not regions:
        return 0.0
    diff = abs(regions[0][0] - timed_batches[0]["timing"]["start"])
    return regions[0][0] - timed_batches[0]["timing"]["start"] if diff > 1e9 else 0.0


def get_interference_periods(data: dict, model: str) -> list[tuple[float, float, str]]:
    """Build (t_start, t_end, label) list relative to first interference event."""
    first_interf = find_first_interference_time(data, model)
    if first_interf is None:
        return []
    ms = data.get("meta", {}).get("model_schedules", {})
    sched = ms.get(model, ms.get("all", {}))
    steps = sched.get("schedule_steps", [])
    step_dur = sched.get("step_duration", 0)

    interf_log = data.get("interference", {}).get(model, {})
    events = interf_log.get("events", [])
    step_order = None
    for e in events:
        if "step_order" in e:
            step_order = e["step_order"]
            break
    if step_order is None:
        step_order = list(range(len(steps)))

    if not steps or not step_dur:
        return []

    periods = []
    for i, step_idx in enumerate(step_order):
        if i == 0:
            continue  # skip idle
        t_start = (i - 1) * step_dur
        t_end = t_start + step_dur
        step_def = steps[step_idx] if step_idx < len(steps) else []
        label = stage_label(step_def)
        periods.append((t_start, t_end, label))
    return periods


def stage_label(step_def: list) -> str:
    """Human-readable short label for an interference step definition."""
    if not step_def:
        return "idle"
    parts = []
    for bench in step_def:
        name, threads = bench[0], bench[1]
        short = "CPU" if "cpu" in name else "MEM" if "memory" in name else name
        parts.append(f"{short}×{threads}")
    return " + ".join(parts)


def _count_threads(label: str) -> int:
    if label == "idle":
        return 0
    total = 0
    for part in label.split(" + "):
        if "×" in part:
            try:
                total += int(part.split("×")[1])
            except ValueError:
                total += 1
    return total


def interf_color(label: str, all_periods: list) -> str:
    """Map interference label to grayscale color based on relative intensity."""
    threads = _count_threads(label)
    if threads == 0:
        return "#ffffff"
    all_threads = [_count_threads(l) for _, _, l in all_periods if _count_threads(l) > 0]
    if not all_threads:
        return "#ffffff"
    lo, hi = min(all_threads), max(all_threads)
    norm = 0.5 if lo == hi else (threads - lo) / (hi - lo)
    return _INTERF_CMAP(0.15 + norm * 0.4)


def draw_interference_bg(ax, periods: list, alpha: float = 0.4):
    """Draw shaded interference regions and labels on an axis."""
    for t_start, t_end, label in periods:
        color = interf_color(label, periods)
        ax.axvspan(t_start, t_end, color=color, alpha=alpha, zorder=0)
        ax.text((t_start + t_end) / 2, 1.0, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5, color="gray")


def draw_interference_boundaries_by_index(ax, regions, timed_batches, clock_offset=0.0):
    """Draw vertical lines at interference step boundaries (batch index x-axis)."""
    for region_i, (start_t, _) in enumerate(regions):
        color = STEP_PALETTE[region_i % len(STEP_PALETTE)]
        start_adj = start_t - clock_offset
        x = 0
        for i, b in enumerate(timed_batches):
            if "timing" in b and b["timing"]["start"] >= start_adj:
                x = i
                break
        ax.axvline(x, color=color, linestyle="--", linewidth=1.5, alpha=0.8,
                   label=f"Step {region_i}")


# ──────────────────────────────────────────────
# Plotting: graphs.ipynb (non-interference runs)
# ──────────────────────────────────────────────

def plot_batch_times(baselines: dict, runs: dict,
                     include_rebalance: bool = False,
                     show_rebalance: bool = False,
                     show_optimum: bool = True):
    """Plot per-batch elapsed time for baselines and runs."""
    model_names = collect_model_names(baselines, runs)
    colors = build_color_map(baselines, runs)
    n_models = len(model_names)
    cols = min(n_models, 2)
    rows = math.ceil(n_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, model in enumerate(model_names):
        ax = axes[idx // cols][idx % cols]

        for ds_name, ds in baselines.items():
            if model not in ds.get("results", {}):
                continue
            batches = ds["results"][model].get("batches", [])
            elapsed = [batch_elapsed(b, include_rebalance) for b in batches if "timing" in b]
            if elapsed:
                ax.axhline(sum(elapsed) / len(elapsed), color=colors[ds_name],
                           linestyle="--", alpha=0.7, label=ds_name)

        for ds_name, ds in runs.items():
            if model not in ds.get("results", {}):
                continue
            batches = ds["results"][model].get("batches", [])
            elapsed = [batch_elapsed(b, include_rebalance) for b in batches if "timing" in b]
            if not elapsed:
                continue
            color = colors[ds_name]
            ax.plot(range(len(elapsed)), elapsed, color=color, label=ds_name,
                    marker=".", markersize=2)

            if show_rebalance and idx == n_models - 1:
                for j, ev in enumerate(get_rebalance_events(batches)):
                    ax.axvline(ev, color=color, linestyle=":", alpha=0.3,
                               label=f"{ds_name} rebalance" if j == 0 else None)

            if show_optimum:
                enters, leaves = get_optimum_transitions(batches)
                if enters:
                    ax.scatter(enters, [elapsed[i] for i in enters if i < len(elapsed)],
                               color=color, marker="^", s=100, alpha=0.5,
                               edgecolors="black", linewidths=0.8, zorder=5)
                if leaves:
                    ax.scatter(leaves, [elapsed[i] for i in leaves if i < len(elapsed)],
                               color=color, marker="v", s=100, alpha=0.5,
                               edgecolors="black", linewidths=0.8, zorder=5)

        ax.set_title(model)
        ax.set_xlabel("Batch index")
        ax.set_ylabel("Elapsed time (s)")
        ax.legend(fontsize="x-small")

    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Per-batch elapsed time", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_optimizer_state(runs: dict, baselines: dict = None,
                         show_combined_gamma: bool = True,
                         show_deep_gamma: bool = False,
                         show_sibling_gamma: bool = False,
                         show_optimum_escape: bool = True):
    """Plot optimizer gamma and escape state per model.

    Backward-compat: old logs store ``deep_gamma`` + ``sibling_gamma`` plus
    ``deep_alpha``; new logs store just ``gamma`` and ``alpha``. The
    "combined" / "sibling" plots only have data for old logs.
    """
    if baselines is None:
        baselines = {}
    model_names = collect_model_names(baselines, runs)
    colors = build_color_map(baselines, runs)

    gamma_plots = []
    if show_combined_gamma:
        gamma_plots.append(("combined gamma\n(deep + sibling * alpha)", "combined"))
    if show_deep_gamma:
        gamma_plots.append(("gamma", "gamma"))
    if show_sibling_gamma:
        gamma_plots.append(("sibling_gamma", "sibling"))
    if show_optimum_escape:
        gamma_plots.append(("optimum escape (s)", "escape_elapsed"))

    if not gamma_plots:
        print("No optimizer state graphs enabled")
        return

    for model in model_names:
        n_plots = len(gamma_plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots + 1),
                                 sharex=True, squeeze=False)
        fig.suptitle(f"{model} — Optimizer State", fontsize=14)

        for run_name, run_data in runs.items():
            if model not in run_data.get("results", {}):
                continue
            batches = run_data["results"][model].get("batches", [])
            # New logs write "gamma"; old logs wrote "deep_gamma".
            gamma = [b.get("rebalance", {}).get("gamma",
                      b.get("rebalance", {}).get("deep_gamma")) for b in batches]
            # Old-log-only; new logs will yield all-None.
            sibling_gamma = [b.get("rebalance", {}).get("sibling_gamma") for b in batches]

            escape_elapsed = []
            for b in batches:
                reb = b.get("rebalance", {})
                if "optimum_escape_elapsed" in reb:
                    escape_elapsed.append(reb["optimum_escape_elapsed"] or 0.0)
                elif "optimum_escape_i" in reb:
                    escape_elapsed.append(reb["optimum_escape_i"] or 0)
                else:
                    escape_elapsed.append(None)

            if not any(v is not None for v in gamma):
                continue

            opt_kwargs = run_data.get("meta", {}).get("optimizer_kwargs", {})
            # Backward-compat: old logs used deep_alpha.
            alpha = opt_kwargs.get("alpha", opt_kwargs.get("deep_alpha", 5))

            # "combined" only has meaning when sibling_gamma is present (old logs).
            if any(v is not None for v in sibling_gamma):
                combined = [
                    (d or 0) + (s or 0) * alpha
                    if d is not None and s is not None else None
                    for d, s in zip(gamma, sibling_gamma)
                ]
            else:
                combined = gamma

            series = {
                "combined": combined,
                "gamma": gamma,
                "sibling": sibling_gamma,
                "escape_elapsed": escape_elapsed,
            }

            enters, leaves = get_optimum_transitions(batches)
            xs = range(len(batches))
            color = colors.get(run_name, "gray")

            for plot_idx, (ylabel, key) in enumerate(gamma_plots):
                ax = axes[plot_idx][0]
                data = series[key]
                ax.plot(xs, data, color=color, label=run_name, alpha=0.8, markersize=2)

                is_first = plot_idx == 0
                if enters:
                    exs = [i for i in enters if i < len(data) and data[i] is not None]
                    ax.scatter(exs, [data[i] for i in exs], color=color, marker="^",
                               s=100, alpha=0.5, edgecolors="black", linewidths=0.8,
                               zorder=5, label=f"{run_name} enter opt" if is_first else None)
                if leaves:
                    lxs = [i for i in leaves if i < len(data) and data[i] is not None]
                    ax.scatter(lxs, [data[i] for i in lxs], color=color, marker="v",
                               s=100, alpha=0.5, edgecolors="black", linewidths=0.8,
                               zorder=5, label=f"{run_name} leave opt" if is_first else None)

        for plot_idx, (ylabel, _) in enumerate(gamma_plots):
            ax = axes[plot_idx][0]
            ax.set_ylabel(ylabel)
            if plot_idx == 0:
                ax.legend(fontsize="small")
        axes[-1][0].set_xlabel("Batch index")
        fig.tight_layout()
        plt.show()


def print_run_summary(runs: dict):
    """Print summary table for each run."""
    for run_name, run_data in sorted(runs.items()):
        meta = run_data.get("meta", {})
        commit = meta.get("git_commit", "?")[:8]
        n_requests = meta.get("num_requests", "?")
        optimizer = meta.get("optimizer", "?")
        opt_kwargs = meta.get("optimizer_kwargs", {})
        interval = opt_kwargs.get("rebalance_interval") or meta.get("rebalance_interval", "?")
        # Backward-compat: old runs used deep_alpha + sibling_alpha.
        alpha = opt_kwargs.get("alpha", opt_kwargs.get("deep_alpha", "?"))
        sibling_alpha = opt_kwargs.get("sibling_alpha")
        tolerance = opt_kwargs.get("tolerance", "?")
        optimum_tolerance = opt_kwargs.get("optimum_tolerance", "?")
        optimum_escape = opt_kwargs.get("optimum_escape_duration",
                                         opt_kwargs.get("optimum_escape", "-"))

        print(f"=== {run_name} === commit: {commit}")
        print(f"  optimizer: {optimizer}, interval: {interval}, requests: {n_requests}")
        sibling_part = f", sibling_alpha: {sibling_alpha}" if sibling_alpha is not None else ""
        print(f"  alpha: {alpha}{sibling_part}, "
              f"tolerance: {tolerance}, optimum_tolerance: {optimum_tolerance}, "
              f"optimum_escape: {optimum_escape}")

        for model, result in run_data["results"].items():
            batches = result.get("batches", [])
            rebalances = sum(1 for b in batches
                             if b.get("rebalance", {}).get("did_rebalance", False))
            at_optimum = sum(1 for b in batches
                             if b.get("rebalance", {}).get("at_optimum", False))
            rps = result.get("requests_per_second", 0)
            print(f"  {model}: rps={rps:.2f}, rebalances={rebalances}, at_optimum={at_optimum}")

            if at_optimum > 0:
                intervals = []
                start = None
                for i, b in enumerate(batches):
                    is_opt = b.get("rebalance", {}).get("at_optimum", False)
                    if is_opt and start is None:
                        start = i
                    elif not is_opt and start is not None:
                        intervals.append((start, i - 1))
                        start = None
                if start is not None:
                    intervals.append((start, len(batches) - 1))
                print(f"    optimum intervals ({len(intervals)}): {intervals}")
        print()


# ──────────────────────────────────────────────
# Plotting: experiment_graphs.ipynb
# ──────────────────────────────────────────────

def plot_experiment_throughput(runs: dict, run_info: dict, all_models: list):
    """Bar chart of overall throughput by run, with interference faded bars."""
    run_ids = sorted(runs.keys())
    n_runs = len(run_ids)
    n_models = len(all_models)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.5), 5))
    x = np.arange(n_models)
    bar_width = 0.8 / n_runs

    for i, run_id in enumerate(run_ids):
        info = run_info.get(run_id, {})
        color = info.get("color", "gray")
        offset = (i - n_runs / 2 + 0.5) * bar_width
        has_interference = run_id in ("C", "D", "E")

        for j, model in enumerate(all_models):
            if has_interference:
                idle_rps, interf_rps = _compute_step_rps(runs[run_id], model)
                ax.bar(x[j] + offset, interf_rps, bar_width, color=color, alpha=1.0,
                       label=f"{run_id}: {info.get('label', '?')}" if j == 0 else None)
                if idle_rps > interf_rps:
                    ax.bar(x[j] + offset, idle_rps - interf_rps, bar_width,
                           bottom=interf_rps, color=color, alpha=0.3)
            else:
                result = runs.get(run_id, {}).get("results", {}).get(model)
                rps = result["requests_per_second"] if result else 0
                ax.bar(x[j] + offset, rps, bar_width, color=color, alpha=1.0,
                       label=f"{run_id}: {info.get('label', '?')}" if j == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=15, ha="right")
    ax.set_ylabel("Requests per second")
    ax.set_title("Overall throughput by run (faded = pre-interference RPS)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.show()


def _compute_step_rps(data: dict, model: str) -> tuple[float, float]:
    """Compute (idle_rps, interference_rps) for a model in an interference run."""
    result = data.get("results", {}).get(model)
    if result is None:
        return 0, 0
    interf_log = data.get("interference", {}).get(model, {})
    events = interf_log.get("events", [])
    schedule_steps, step_duration = get_model_schedule(data, model)

    if not events or not step_duration or not schedule_steps:
        return result.get("requests_per_second", 0), 0

    first_time = None
    for e in events:
        if "time" in e and "event" in e:
            first_time = e["time"]
            break
    if first_time is None:
        return result.get("requests_per_second", 0), 0

    timed = [b for b in result.get("batches", []) if "timing" in b]
    if len(timed) < 2:
        return 0, 0

    def _rps_in_window(start, end):
        window = [b for b in timed if start <= b["timing"]["start"] < end]
        if len(window) >= 2:
            wall = window[-1]["timing"]["end"] - window[0]["timing"]["start"]
            return len(window) / wall if wall > 0 else 0
        return 0

    idle_rps = _rps_in_window(first_time, first_time + step_duration)
    interf_rps = _rps_in_window(first_time + step_duration,
                                 first_time + step_duration * len(schedule_steps))
    return idle_rps, interf_rps


def plot_experiment_rps_per_stage(runs: dict, run_info: dict, all_models: list):
    """Bar chart of RPS per interference stage for runs C/D/E."""
    interf_run_ids = [k for k in ("C", "D", "E") if k in runs]
    if not interf_run_ids:
        print("No interference runs (C/D/E) available")
        return

    n_models = len(all_models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)

    for idx, model in enumerate(all_models):
        ax = axes[0][idx]
        ref_data = runs[interf_run_ids[0]]
        ref_stages = _compute_rps_per_stage(ref_data, model)
        stage_labels = [s[0] for s in ref_stages]
        n_stages = len(stage_labels)

        if n_stages == 0:
            ax.set_title(f"{model}\n(no interference data)")
            continue

        x = np.arange(n_stages)
        n_runs_i = len(interf_run_ids)
        bar_width = 0.8 / n_runs_i

        for i, run_id in enumerate(interf_run_ids):
            info = run_info.get(run_id, {})
            color = info.get("color", "gray")
            offset = (i - n_runs_i / 2 + 0.5) * bar_width
            stage_rps = _compute_rps_per_stage(runs[run_id], model)
            rps_vals = [s[1] for s in stage_rps] if stage_rps else [0] * n_stages
            bars = ax.bar(x + offset, rps_vals, bar_width, color=color, alpha=0.85,
                          label=f"{run_id}: {info.get('label', '?')}")
            for bar, val in zip(bars, rps_vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.1f}", ha="center", va="bottom", fontsize=6)

        for bl_id, bl_label in [("A", "GPipe baseline"), ("B", "Shisha baseline")]:
            if bl_id in runs:
                bl_result = runs[bl_id].get("results", {}).get(model)
                if bl_result:
                    bl_rps = bl_result.get("requests_per_second", 0)
                    ax.axhline(bl_rps, color=run_info[bl_id]["color"], linestyle="--",
                               alpha=0.6, label=f"{bl_id}: {bl_label} ({bl_rps:.1f})")

        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Requests per second" if idx == 0 else "")
        ax.set_title(model)
        ax.legend(fontsize="x-small", loc="lower right")

    fig.suptitle("RPS per interference stage (C / D / E)", fontsize=14)
    fig.tight_layout()
    plt.show()


def _compute_rps_per_stage(data: dict, model: str) -> list[tuple[str, float]]:
    """Return [(stage_label, rps), ...] for each interference stage."""
    result = data.get("results", {}).get(model)
    if result is None:
        return []
    interf_log = data.get("interference", {}).get(model, {})
    events = interf_log.get("events", [])
    ms = data.get("meta", {}).get("model_schedules", {})
    sched = ms.get(model, ms.get("all", {}))
    steps = sched.get("schedule_steps", [])
    step_dur = sched.get("step_duration", 0)

    step_order = None
    first_time = None
    for e in events:
        if "step_order" in e:
            step_order = e["step_order"]
        if first_time is None and "time" in e and "event" in e:
            first_time = e["time"]
    if step_order is None:
        step_order = list(range(len(steps)))
    if not steps or not step_dur or first_time is None:
        return []

    timed = [b for b in result.get("batches", []) if "timing" in b]
    if len(timed) < 2:
        return []

    stage_rps = []
    for i, step_idx in enumerate(step_order):
        t_start = first_time + i * step_dur
        t_end = t_start + step_dur
        step_def = steps[step_idx] if step_idx < len(steps) else []
        label = stage_label(step_def)
        stage_batches = [b for b in timed if t_start <= b["timing"]["start"] < t_end]
        if len(stage_batches) >= 2:
            wall = stage_batches[-1]["timing"]["end"] - stage_batches[0]["timing"]["start"]
            rps = len(stage_batches) / wall if wall > 0 else 0
        elif len(stage_batches) == 1:
            dur = stage_batches[0]["timing"]["end"] - stage_batches[0]["timing"]["start"]
            rps = 1.0 / dur if dur > 0 else 0
        else:
            rps = 0
        stage_rps.append((label, rps))
    return stage_rps


def plot_experiment_batch_times(runs: dict, run_info: dict, all_models: list,
                                y_limits: dict = None,
                                show_optimum: bool = True):
    """Time series of batch times for interference runs C/D/E with shaded regions."""
    if y_limits is None:
        y_limits = {}
    interf_runs = {k: v for k, v in runs.items() if k in ("C", "D", "E")}
    if not interf_runs:
        print("Runs C, D, E not available")
        return

    fig, axes = plt.subplots(1, len(all_models), figsize=(7 * len(all_models), 4), squeeze=False)

    for idx, model in enumerate(all_models):
        ax = axes[0][idx]

        ref_data = next(iter(interf_runs.values()))
        periods = get_interference_periods(ref_data, model)
        draw_interference_bg(ax, periods, alpha=0.6)

        for run_id, data in sorted(interf_runs.items()):
            result = data.get("results", {}).get(model)
            if result is None:
                continue
            timed = [b for b in result.get("batches", []) if "timing" in b]
            if not timed:
                continue
            first_interf = find_first_interference_time(data, model)
            if first_interf is None:
                continue

            t_rel = [b["timing"]["start"] - first_interf for b in timed]
            elapsed = [b["timing"]["end"] - b["timing"]["start"] for b in timed]
            info = run_info.get(run_id, {})
            ax.plot(t_rel, elapsed, color=info.get("color", "gray"), alpha=0.8,
                    label=f"{run_id}: {info.get('label', '?')}",
                    marker=".", markersize=1, linewidth=0.8)

        if show_optimum:
            for run_id, data in sorted(interf_runs.items()):
                result = data.get("results", {}).get(model)
                if result is None:
                    continue
                timed = [b for b in result.get("batches", []) if "timing" in b]
                first_interf = find_first_interference_time(data, model)
                if not timed or first_interf is None:
                    continue
                info = run_info.get(run_id, {})
                color = info.get("color", "gray")
                enters, leaves = get_optimum_transitions(timed)
                if enters:
                    exs = [timed[i]["timing"]["start"] - first_interf for i in enters]
                    eys = [timed[i]["timing"]["end"] - timed[i]["timing"]["start"] for i in enters]
                    ax.scatter(exs, eys, marker="^", color=color, s=30, zorder=5,
                               edgecolors="black", linewidths=0.3)
                if leaves:
                    lxs = [timed[i]["timing"]["start"] - first_interf for i in leaves]
                    lys = [timed[i]["timing"]["end"] - timed[i]["timing"]["start"] for i in leaves]
                    ax.scatter(lxs, lys, marker="v", color=color, s=30, zorder=5,
                               edgecolors="black", linewidths=0.3)

        if "A" in runs:
            a_result = runs["A"].get("results", {}).get(model)
            if a_result:
                a_timed = [b for b in a_result.get("batches", []) if "timing" in b]
                a_times = [b["timing"]["end"] - b["timing"]["start"] for b in a_timed]
                if a_times:
                    ax.axhline(np.mean(a_times), color=run_info["A"]["color"],
                               linestyle="--", alpha=0.5, label="A: GPipe avg")

        ax.axvline(0, color="black", linestyle=":", alpha=0.4, label="Interference start")
        ax.set_title(model)
        ax.set_xlabel("Time since first interference (s)")
        ax.set_ylabel("Forward time (s)" if idx == 0 else "")
        if model in y_limits:
            ax.set_ylim(y_limits[model])
        ax.legend(fontsize="x-small")

    fig.suptitle("Under interference: GPipe vs Exhaustive vs Shisha", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_experiment_stage_times(runs: dict, run_info: dict, all_models: list):
    """Per-stage batch times over time for interference runs."""
    stage_runs = {k: v for k, v in runs.items() if k in ("C", "D", "E")}
    if not stage_runs:
        print("No interference runs (C/D/E) available")
        return

    run_ids_s = sorted(stage_runs.keys())
    n_rows = len(run_ids_s)
    n_cols = len(all_models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.5 * n_rows), squeeze=False)

    for row, run_id in enumerate(run_ids_s):
        data = stage_runs[run_id]
        info = run_info.get(run_id, {})

        for col, model in enumerate(all_models):
            ax = axes[row][col]
            result = data.get("results", {}).get(model)
            if result is None:
                ax.set_title(f"{model} — Run {run_id}: no data")
                continue

            timed = [b for b in result.get("batches", [])
                     if "timing" in b and "stage_times" in b]
            if not timed:
                ax.set_title(f"{model} — Run {run_id}: no stage_times")
                continue

            first_interf = find_first_interference_time(data, model)
            if first_interf is None:
                first_interf = timed[0]["timing"]["start"]

            t_rel = [b["timing"]["start"] - first_interf for b in timed]
            n_stages = len(timed[0]["stage_times"])

            for s in range(n_stages):
                vals = [b["stage_times"][s] / 1e9 for b in timed]
                color = STAGE_COLORS[s % len(STAGE_COLORS)]
                ax.plot(t_rel, vals, color=color, alpha=0.8, label=f"Stage {s}",
                        linewidth=0.8, marker=".", markersize=0.5)

            periods = get_interference_periods(data, model)
            draw_interference_bg(ax, periods)
            ax.axvline(0, color="black", linestyle=":", alpha=0.4)

            ax.set_title(f"{model} — {run_id}: {info.get('label', '?')}", fontsize=10)
            ax.set_xlabel("Time since first interference (s)" if row == n_rows - 1 else "")
            ax.set_ylabel("Stage time (s)" if col == 0 else "")
            ax.legend(fontsize="x-small", loc="upper right")

    fig.suptitle("Per-stage batch times under interference", fontsize=14)
    fig.tight_layout()
    plt.show()


def print_experiment_rebalance_activity(runs: dict, run_info: dict, all_models: list):
    """Print rebalance activity table."""
    run_ids = sorted(runs.keys())
    print(f"{'Model':<20} {'Run':<8} {'Batches':>10} {'Rebalances':>12} "
          f"{'At Optimum':>12} {'Rebal %':>10} {'Optimum %':>10}")
    print("-" * 82)
    for model in all_models:
        for run_id in run_ids:
            if run_id == "C":
                continue
            result = runs[run_id].get("results", {}).get(model)
            if result is None:
                continue
            batches = result.get("batches", [])
            timed = [b for b in batches if "timing" in b]
            n = len(timed)
            rebalances = sum(1 for b in batches
                             if b.get("rebalance", {}).get("did_rebalance", False))
            at_optimum = sum(1 for b in batches
                             if b.get("rebalance", {}).get("at_optimum", False))
            rebal_pct = (rebalances / n * 100) if n > 0 else 0
            opt_pct = (at_optimum / n * 100) if n > 0 else 0
            print(f"{model:<20} {run_id:<8} {n:>10} {rebalances:>12} "
                  f"{at_optimum:>12} {rebal_pct:>9.1f}% {opt_pct:>9.1f}%")


def print_experiment_throughput_table(runs: dict, run_info: dict, all_models: list,
                                      baseline_id: str = "A"):
    """Print throughput comparison table relative to a baseline run."""
    run_ids = sorted(runs.keys())
    header = f"{'Model':<20}"
    for run_id in run_ids:
        info = run_info.get(run_id, {})
        header += f"{run_id + ': ' + info.get('label', '?'):>25}"
    print(header)
    print("-" * len(header))

    for model in all_models:
        bl_result = runs.get(baseline_id, {}).get("results", {}).get(model)
        bl_rps = bl_result["requests_per_second"] if bl_result else 0

        row = f"{model:<20}"
        for run_id in run_ids:
            result = runs[run_id].get("results", {}).get(model)
            if result:
                rps = result["requests_per_second"]
                if bl_rps > 0:
                    pct = (rps / bl_rps) * 100
                    row += f"{rps:>12.2f} ({pct:>5.0f}%)"
                else:
                    row += f"{rps:>12.2f}       "
            else:
                row += f"{'N/A':>25}"
        print(row)


def print_experiment_schedule(runs: dict, all_models: list):
    """Print the interference schedule details."""
    ref_run = next((runs[k] for k in ("C", "D", "E") if k in runs), None)
    if ref_run is None:
        print("No interference runs available")
        return

    ref_model = all_models[0]
    interf_log = ref_run.get("interference", {}).get(ref_model, {})
    ms = ref_run.get("meta", {}).get("model_schedules", {})
    sched = ms.get(ref_model, ms.get("all", {}))
    steps = sched.get("schedule_steps", [])

    step_order = None
    for e in interf_log.get("events", []):
        if "step_order" in e:
            step_order = e["step_order"]
            break
    if step_order is None:
        step_order = list(range(len(steps)))

    if not steps:
        print("No schedule data")
        return

    dur_parts = []
    for model in all_models:
        ms_m = ms.get(model, ms.get("all", {}))
        dur_parts.append(f"{model}={ms_m.get('step_duration', '?')}s")
    print(f"Step durations: {', '.join(dur_parts)}")
    print(f"Step order: {step_order}")
    print()

    bench_types = set()
    for step_def in steps:
        for bench in step_def:
            short = "CPU" if "cpu" in bench[0] else "MEM" if "memory" in bench[0] else bench[0]
            bench_types.add(short)
    bench_types = sorted(bench_types)

    header = f"{'Order':<7} {'Step':<6}"
    for bt in bench_types:
        header += f"{bt:>6}"
    header += f"  {'Total':>6}"
    print(header)
    print("-" * len(header))

    for i, step_idx in enumerate(step_order):
        step_def = steps[step_idx] if step_idx < len(steps) else []
        type_threads = {bt: 0 for bt in bench_types}
        for bench in step_def:
            short = "CPU" if "cpu" in bench[0] else "MEM" if "memory" in bench[0] else bench[0]
            type_threads[short] += bench[1]
        total = sum(type_threads.values())
        row = f"{i:<7} {step_idx:<6}"
        for bt in bench_types:
            t = type_threads[bt]
            row += f"{t if t > 0 else '-':>6}"
        row += f"  {total if total > 0 else 'idle':>6}"
        print(row)


# ──────────────────────────────────────────────
# Plotting: interference_graphs.ipynb
# ──────────────────────────────────────────────

def plot_interf_batch_times(interf_data: dict, baseline_data: dict = None,
                            baseline_name: str = "baseline",
                            show_interference_regions: bool = True,
                            show_optimum: bool = True):
    """Plot per-batch forward times under interference."""
    models = list(interf_data["results"].keys())
    n_models = len(models)
    if n_models == 0:
        print("No model results to plot")
        return
    cols = min(n_models, 2)
    rows = math.ceil(n_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // cols][idx % cols]
        result = interf_data["results"][model]
        batches = result.get("batches", [])
        timed = [b for b in batches if "timing" in b]
        elapsed = [b["timing"]["end"] - b["timing"]["start"] for b in timed]

        if not elapsed:
            ax.set_title(f"{model} (no data)")
            continue

        ax.plot(range(len(elapsed)), elapsed, color="steelblue", marker=".", markersize=2)

        if baseline_data and model in baseline_data.get("results", {}):
            bl_batches = baseline_data["results"][model].get("batches", [])
            bl_times = [b["timing"]["end"] - b["timing"]["start"]
                        for b in bl_batches if "timing" in b]
            if bl_times:
                ax.axhline(np.mean(bl_times), color="green", linestyle="--",
                           alpha=0.7, label=f"{baseline_name} avg")

        if show_interference_regions:
            regions = get_interference_regions(interf_data, model)
            offset = compute_clock_offset(timed, regions)
            draw_interference_boundaries_by_index(ax, regions, timed, clock_offset=offset)

        if show_optimum:
            enters, leaves = get_optimum_transitions(timed)
            if enters:
                ax.scatter(enters, [elapsed[i] for i in enters], color="green", marker="^",
                           s=60, alpha=0.6, edgecolors="black", linewidths=0.5, zorder=5)
            if leaves:
                ax.scatter(leaves, [elapsed[i] for i in leaves], color="red", marker="v",
                           s=60, alpha=0.6, edgecolors="black", linewidths=0.5, zorder=5)

        ax.set_title(model)
        ax.set_xlabel("Batch index")
        ax.set_ylabel("Forward time (s)")
        ax.legend(fontsize="x-small", loc="upper left")

    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Per-batch forward time under interference", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_interf_throughput(interf_data: dict, baseline_data: dict = None,
                           baseline_name: str = "baseline"):
    """Bar chart of per-step throughput under interference."""
    models = list(interf_data["results"].keys())
    first_steps, _ = get_model_schedule(interf_data, models[0])
    n_steps = len(first_steps)

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 3), 5))
    x = np.arange(len(models))
    bar_width = 0.8 / (n_steps + (1 if baseline_data else 0))
    step_colors = ["#4caf50", "#ff9800", "#f44336", "#9c27b0", "#2196f3", "#795548"]

    for step_i in range(n_steps):
        rps_values = []
        for model in models:
            batches = interf_data["results"][model].get("batches", [])
            timed = [b for b in batches if "timing" in b]
            regions = get_interference_regions(interf_data, model)
            offset = compute_clock_offset(timed, regions)

            if step_i < len(regions):
                r_start, r_end = regions[step_i]
                step_batches = [b for b in timed
                                if r_start - offset <= b["timing"]["start"] < r_end - offset]
                if len(step_batches) >= 2:
                    wall = step_batches[-1]["timing"]["end"] - step_batches[0]["timing"]["start"]
                    rps_values.append(len(step_batches) / wall if wall > 0 else 0)
                else:
                    rps_values.append(0)
            else:
                rps_values.append(0)

        offset_bar = (step_i - n_steps / 2 + 0.5) * bar_width
        ax.bar(x + offset_bar, rps_values, bar_width, label=f"Step {step_i}",
               color=step_colors[step_i % len(step_colors)], alpha=0.8)

    if baseline_data:
        bl_rps = [baseline_data.get("results", {}).get(m, {}).get("requests_per_second", 0)
                  for m in models]
        offset_bar = (n_steps - n_steps / 2 + 0.5) * bar_width
        ax.bar(x + offset_bar, bl_rps, bar_width, label=f"{baseline_name} (no interf)",
               color="brown", alpha=0.5, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Requests per second")
    ax.set_title("Throughput by interference level")
    ax.legend(fontsize="small")
    fig.tight_layout()
    plt.show()


def print_interf_throughput_impact(interf_data: dict, baseline_data: dict = None,
                                    baseline_name: str = "baseline"):
    """Print per-step throughput as % of idle."""
    models = list(interf_data["results"].keys())
    for model in models:
        schedule_steps, _ = get_model_schedule(interf_data, model)
        batches = interf_data["results"][model].get("batches", [])
        timed = [b for b in batches if "timing" in b]
        regions = get_interference_regions(interf_data, model)
        offset = compute_clock_offset(timed, regions)

        step_rps = []
        for step_i in range(len(schedule_steps)):
            if step_i < len(regions):
                r_start, r_end = regions[step_i]
                step_batches = [b for b in timed
                                if r_start - offset <= b["timing"]["start"] < r_end - offset]
                if len(step_batches) >= 2:
                    wall = step_batches[-1]["timing"]["end"] - step_batches[0]["timing"]["start"]
                    step_rps.append(len(step_batches) / wall if wall > 0 else 0)
                else:
                    step_rps.append(0)
            else:
                step_rps.append(0)

        idle_rps = step_rps[0] if step_rps and step_rps[0] > 0 else 1
        print(f"--- {model} ---")
        for i, rps in enumerate(step_rps):
            if rps > 0:
                print(f"  Step {i:<5} {rps:>8.2f} rps ({rps / idle_rps * 100:>5.1f}%)")
            else:
                print(f"  Step {i:<5} {'N/A':>8}")
        if baseline_data and model in baseline_data.get("results", {}):
            bl_rps = baseline_data["results"][model].get("requests_per_second", 0)
            print(f"  {'baseline':<10} {bl_rps:>8.2f} rps ({bl_rps / idle_rps * 100:>5.1f}%)")
        print()


def plot_interf_optimizer_state(interf_data: dict,
                                show_interference_regions: bool = True,
                                show_optimum: bool = True):
    """Plot optimizer gamma and best throughput under interference."""
    models = list(interf_data["results"].keys())
    for model in models:
        result = interf_data["results"][model]
        batches = result.get("batches", [])
        # Backward-compat: old logs → deep_gamma + sibling_gamma; new → gamma only.
        gamma = [b.get("rebalance", {}).get("gamma",
                  b.get("rebalance", {}).get("deep_gamma")) for b in batches]
        sibling_gamma = [b.get("rebalance", {}).get("sibling_gamma") for b in batches]
        best_throughput = [b.get("rebalance", {}).get("best_throughput") for b in batches]

        if not any(v is not None for v in gamma):
            continue

        opt_kwargs = interf_data.get("meta", {}).get("optimizer_kwargs", {})
        alpha = opt_kwargs.get("alpha", opt_kwargs.get("deep_alpha", 5))

        if any(v is not None for v in sibling_gamma):
            combined = [
                (d or 0) + (s or 0) * alpha
                if d is not None and s is not None else None
                for d, s in zip(gamma, sibling_gamma)
            ]
        else:
            combined = gamma

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"{model} — Optimizer State", fontsize=14)
        xs = range(len(batches))

        axes[0].plot(xs, combined, color="steelblue", alpha=0.8)
        axes[0].set_ylabel("Combined gamma\n(deep + sibling * alpha)")
        axes[1].plot(xs, best_throughput, color="darkorange", alpha=0.8)
        axes[1].set_ylabel("Best throughput")
        axes[1].set_xlabel("Batch index")

        if show_interference_regions:
            regions = get_interference_regions(interf_data, model)
            timed = [b for b in batches if "timing" in b]
            offset = compute_clock_offset(timed, regions)
            for ax_i in axes:
                draw_interference_boundaries_by_index(ax_i, regions, timed, clock_offset=offset)

        if show_optimum:
            enters, leaves = get_optimum_transitions(batches)
            for ax_i in axes:
                for e in enters:
                    ax_i.axvline(e, color="green", linestyle="-", alpha=0.3, linewidth=0.8)
                for lv in leaves:
                    ax_i.axvline(lv, color="red", linestyle="-", alpha=0.3, linewidth=0.8)

        axes[0].legend(fontsize="small", loc="upper left")
        fig.tight_layout()
        plt.show()


def plot_interf_boxplot(interf_data: dict):
    """Boxplot of forward time distribution per interference step."""
    models = list(interf_data["results"].keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0][idx]
        schedule_steps, _ = get_model_schedule(interf_data, model)
        batches = interf_data["results"][model].get("batches", [])
        timed = [b for b in batches if "timing" in b]
        regions = get_interference_regions(interf_data, model)
        offset = compute_clock_offset(timed, regions)

        box_data, box_labels = [], []
        for step_i in range(len(schedule_steps)):
            if step_i < len(regions):
                r_start, r_end = regions[step_i]
                step_times = [b["timing"]["end"] - b["timing"]["start"]
                              for b in timed if r_start - offset <= b["timing"]["start"] < r_end - offset]
            else:
                step_times = []
            box_data.append(step_times if step_times else [0])
            box_labels.append(f"Step {step_i}")

        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        colors = ["#4caf50", "#ff9800", "#f44336", "#9c27b0", "#2196f3", "#795548"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title(model)
        ax.set_ylabel("Forward time (s)" if idx == 0 else "")
        ax.set_xlabel("Interference step")

    fig.suptitle("Forward time distribution by interference level", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_interf_stage_times(interf_data: dict,
                            show_interference_regions: bool = True,
                            x_axis: str = "time"):
    """Plot per-stage batch times over time for a single interference run.

    Args:
        x_axis: "time" for wall-clock seconds since start, "index" for batch index.
    """
    models = list(interf_data["results"].keys())
    n_models = len(models)
    if n_models == 0:
        print("No model results to plot")
        return

    cols = min(n_models, 2)
    rows = math.ceil(n_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // cols][idx % cols]
        result = interf_data["results"][model]
        batches = result.get("batches", [])
        timed = [b for b in batches if "timing" in b and "stage_times" in b]

        if not timed:
            ax.set_title(f"{model} (no stage_times)")
            continue

        if x_axis == "time":
            t0 = timed[0]["timing"]["start"]
            xs = [b["timing"]["start"] - t0 for b in timed]
            x_label = "Time (s)"
        else:
            xs = list(range(len(timed)))
            x_label = "Batch index"

        n_stages = len(timed[0]["stage_times"])

        for s in range(n_stages):
            vals = [b["stage_times"][s] / 1e9 for b in timed]
            color = STAGE_COLORS[s % len(STAGE_COLORS)]
            ax.plot(xs, vals, color=color, alpha=0.8, label=f"Stage {s}",
                    linewidth=0.8, marker=".", markersize=1)

        if show_interference_regions:
            regions = get_interference_regions(interf_data, model)
            offset = compute_clock_offset(timed, regions)
            if x_axis == "time":
                # Draw boundaries as vertical lines at wall-clock offsets
                for region_i, (start_t, _) in enumerate(regions):
                    x_pos = start_t - offset - timed[0]["timing"]["start"]
                    color = STEP_PALETTE[region_i % len(STEP_PALETTE)]
                    ax.axvline(x_pos, color=color, linestyle="--", linewidth=1.5,
                               alpha=0.8, label=f"Step {region_i}")
            else:
                draw_interference_boundaries_by_index(ax, regions, timed, clock_offset=offset)

        ax.set_title(model)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Stage time (s)" if idx % cols == 0 else "")
        ax.legend(fontsize="x-small", loc="upper right")

    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Per-stage batch times under interference", fontsize=14)
    fig.tight_layout()
    plt.show()
