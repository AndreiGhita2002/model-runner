"""Measure the overhead of wrapping a model in TimedModule.

For each model we keep both the raw and TimedModule-wrapped copies resident at
the same time and run them as back-to-back **pairs**: raw measurement
immediately followed by timed measurement. The pair order is fixed (always raw
then timed) so the cache-state cost of switching between the two model objects
is constant across pairs and cancels out in the overhead comparison.

Each pair runs NUM_BATCHES * REQUESTS_PER_BATCH forward calls at batch_size=32.
This mirrors evaluation's unit of work (one "batch" = 32 requests, each = one
forward of a batched input). Each model can be paired `--repetitions` times
(default 5) to estimate variance, all written to the same output JSON.
"""

import argparse
import gc
import json
import statistics
import time
from typing import Callable

import torch
import torch.nn as nn

from model_runner.timed_module import make_module_timed
from tests.testing_models import DEFAULT_MODEL_SET
from tests.util import generate_batch


NUM_BATCHES = 4
REQUESTS_PER_BATCH = 32
BATCH_SIZE = 32
SEED = 37
DEFAULT_DEPTH = 3  # matches PipelineRunner.default_timing_depth
DEFAULT_REPETITIONS = 5


def time_forwards(model: nn.Module, inputs: list[torch.Tensor]) -> float:
    """Run model on each input and return total elapsed seconds (perf_counter)."""
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for x in inputs:
            model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
    return end - start


def _summarise(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_model(model_name: str, load_model: Callable, rand_inputs: Callable,
              device: str, depth: int, repetitions: int) -> dict:
    total_requests = NUM_BATCHES * REQUESTS_PER_BATCH

    inputs = [
        generate_batch(rand_inputs, BATCH_SIZE, SEED + i).to(device)
        for i in range(total_requests)
    ]

    # Load both raw and wrapped copies. They are separate nn.Module instances
    # with their own weight buffers — switching between them costs cache, but
    # because we always run raw-then-timed in each pair, that cost is the same
    # in every pair and does not bias the comparison.
    raw_model = load_model(device=device)
    wrapped_model = make_module_timed(load_model(device=device), device=device, depth=depth)

    # One warmup forward per model to pull weights into cache before the first
    # measurement and trigger any lazy CUDA / torch.compile init.
    with torch.no_grad():
        raw_model(inputs[0])
        wrapped_model(inputs[0])

    pairs = []
    for rep in range(repetitions):
        raw_elapsed = time_forwards(raw_model, inputs)
        timed_elapsed = time_forwards(wrapped_model, inputs)

        overhead_abs = timed_elapsed - raw_elapsed
        overhead_pct = (overhead_abs / raw_elapsed) * 100 if raw_elapsed > 0 else float("nan")

        pairs.append({
            "repetition": rep,
            "raw_seconds": raw_elapsed,
            "timed_seconds": timed_elapsed,
            "overhead_seconds": overhead_abs,
            "overhead_percent": overhead_pct,
            "raw_per_forward_ms": (raw_elapsed / total_requests) * 1000,
            "timed_per_forward_ms": (timed_elapsed / total_requests) * 1000,
        })

    del raw_model, wrapped_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "raw_seconds": _summarise([p["raw_seconds"] for p in pairs]),
        "timed_seconds": _summarise([p["timed_seconds"] for p in pairs]),
        "overhead_seconds": _summarise([p["overhead_seconds"] for p in pairs]),
        "overhead_percent": _summarise([p["overhead_percent"] for p in pairs]),
    }

    return {
        "model": model_name,
        "num_forwards_per_rep": total_requests,
        "repetitions": repetitions,
        "pairs": pairs,
        "summary": summary,
    }


def print_table(results: list[dict]) -> None:
    header = (f"{'model':<22}{'raw mean (s)':>14}{'timed mean (s)':>16}"
              f"{'ovh mean %':>13}{'ovh stdev %':>14}")
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        s = r["summary"]
        print(f"{r['model']:<22}"
              f"{s['raw_seconds']['mean']:>14.4f}"
              f"{s['timed_seconds']['mean']:>16.4f}"
              f"{s['overhead_percent']['mean']:>12.2f}%"
              f"{s['overhead_percent']['stdev']:>13.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help=f"TimedModule depth (default: {DEFAULT_DEPTH})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: auto-detect cuda/cpu)")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS,
                        help=f"Number of paired raw/timed runs per model "
                             f"(default: {DEFAULT_REPETITIONS})")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Optional JSON output path (all repetitions go here)")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")

    print(f"Device: {args.device}")
    print(f"TimedModule depth: {args.depth}")
    print(f"Repetitions per model: {args.repetitions}")
    print(f"Forwards per condition per rep: {NUM_BATCHES} batches x "
          f"{REQUESTS_PER_BATCH} requests = {NUM_BATCHES * REQUESTS_PER_BATCH}")
    print(f"Batch size per forward: {BATCH_SIZE}")

    results = []
    for model_name, load_model, rand_inputs in DEFAULT_MODEL_SET:
        print(f"\n>>> {model_name}")
        result = run_model(model_name, load_model, rand_inputs,
                           args.device, args.depth, args.repetitions)
        for p in result["pairs"]:
            print(f"    rep {p['repetition']}: "
                  f"raw {p['raw_seconds']:.4f}s "
                  f"({p['raw_per_forward_ms']:.2f} ms/fwd)  "
                  f"timed {p['timed_seconds']:.4f}s "
                  f"({p['timed_per_forward_ms']:.2f} ms/fwd)  "
                  f"ovh {p['overhead_percent']:.2f}%")
        s = result["summary"]
        print(f"    summary: overhead {s['overhead_percent']['mean']:.2f}% "
              f"± {s['overhead_percent']['stdev']:.2f}% "
              f"(min {s['overhead_percent']['min']:.2f}%, "
              f"max {s['overhead_percent']['max']:.2f}%)")
        results.append(result)

    print_table(results)

    if args.output:
        payload = {
            "meta": {
                "device": args.device,
                "depth": args.depth,
                "num_batches": NUM_BATCHES,
                "requests_per_batch": REQUESTS_PER_BATCH,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "repetitions": args.repetitions,
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
