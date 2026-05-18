"""Measure the overhead of wrapping a model in TimedModule.

Runs each evaluation model twice: once raw, once wrapped in TimedModule at the
default profiling depth (3, matching PipelineRunner.default_timing_depth). Times
the whole forward-pass loop with time.perf_counter from outside the model.

Per model and condition, we run NUM_BATCHES * REQUESTS_PER_BATCH forward calls,
each with a batch_size=32 input. This mirrors the unit of work in evaluation
(one "batch" = 32 requests, each request = one forward pass of batched input).
"""

import argparse
import gc
import json
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


def run_model(model_name: str, load_model: Callable, rand_inputs: Callable,
              device: str, depth: int) -> dict:
    total_requests = NUM_BATCHES * REQUESTS_PER_BATCH

    inputs = [
        generate_batch(rand_inputs, BATCH_SIZE, SEED + i).to(device)
        for i in range(total_requests)
    ]

    # ---- Raw model ----
    raw_model = load_model(device=device)
    # Warmup
    with torch.no_grad():
        raw_model(inputs[0])
    raw_elapsed = time_forwards(raw_model, inputs)
    del raw_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- TimedModule-wrapped ----
    wrapped_model = make_module_timed(load_model(device=device), device=device, depth=depth)
    with torch.no_grad():
        wrapped_model(inputs[0])
    timed_elapsed = time_forwards(wrapped_model, inputs)
    del wrapped_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    overhead_abs = timed_elapsed - raw_elapsed
    overhead_pct = (overhead_abs / raw_elapsed) * 100 if raw_elapsed > 0 else float("nan")

    return {
        "model": model_name,
        "num_forwards": total_requests,
        "raw_seconds": raw_elapsed,
        "timed_seconds": timed_elapsed,
        "overhead_seconds": overhead_abs,
        "overhead_percent": overhead_pct,
        "raw_per_forward_ms": (raw_elapsed / total_requests) * 1000,
        "timed_per_forward_ms": (timed_elapsed / total_requests) * 1000,
    }


def print_table(results: list[dict]) -> None:
    header = f"{'model':<22}{'raw (s)':>12}{'timed (s)':>12}{'overhead (s)':>15}{'overhead %':>14}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<22}{r['raw_seconds']:>12.4f}{r['timed_seconds']:>12.4f}"
              f"{r['overhead_seconds']:>15.4f}{r['overhead_percent']:>13.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help=f"TimedModule depth (default: {DEFAULT_DEPTH})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: auto-detect cuda/cpu)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device}")
    print(f"TimedModule depth: {args.depth}")
    print(f"Forwards per condition: {NUM_BATCHES} batches x {REQUESTS_PER_BATCH} requests "
          f"= {NUM_BATCHES * REQUESTS_PER_BATCH}")
    print(f"Batch size per forward: {BATCH_SIZE}")

    results = []
    for model_name, load_model, rand_inputs in DEFAULT_MODEL_SET:
        print(f"\n>>> {model_name}")
        result = run_model(model_name, load_model, rand_inputs, args.device, args.depth)
        print(f"    raw:   {result['raw_seconds']:.4f}s "
              f"({result['raw_per_forward_ms']:.2f} ms/forward)")
        print(f"    timed: {result['timed_seconds']:.4f}s "
              f"({result['timed_per_forward_ms']:.2f} ms/forward)")
        print(f"    overhead: {result['overhead_seconds']:.4f}s "
              f"({result['overhead_percent']:.2f}%)")
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
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
