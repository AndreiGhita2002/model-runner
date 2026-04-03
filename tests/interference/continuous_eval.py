"""Continuous evaluation for interference experiments.

Uses PipelineServer.run_continuous() for maximum throughput — generates
inputs inline with no queue overhead. Runs until SIGTERM, then saves results.

Intended to be run via torchrun and controlled by interfere_eval.py.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from model_runner import PipelineServer
from model_runner.pipeline_optimizer import (
    GreedyPipelineOptimizer, StaticGPipeOptimizer,
    TimeBasedShishaPipelineOptimizer, ExhaustiveShishaOptimizer,
)
from tests.testing_models import MODEL_SETS
from tests.util import generate_batch

OPTIMIZER_CHOICES = {
    "shisha": TimeBasedShishaPipelineOptimizer,
    "exhaustive": ExhaustiveShishaOptimizer,
    "greedy": GreedyPipelineOptimizer,
    "gpipe": StaticGPipeOptimizer,
}


# Collect results on last rank
result_timings: list[dict | None] = []
result_req_ids: list[uuid.UUID] = []
_signal_file: str | None = None
_signal_written = False


def handle_output(request_id: uuid.UUID, model_name: str, output: Any, timing: dict | None):
    global _signal_written
    result_req_ids.append(request_id)
    result_timings.append(timing)

    # Write sentinel file when optimum is first reached
    if _signal_file and not _signal_written and timing is not None:
        reb = timing.get("rebalance", {})
        if reb.get("at_optimum", False):
            Path(_signal_file).touch()
            _signal_written = True


def main():
    parser = argparse.ArgumentParser(description="Continuous eval for interference experiments")
    parser.add_argument("--model-set", choices=list(MODEL_SETS.keys()), default="small")
    parser.add_argument("--model", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-m", "--n-microbatches", type=int, default=32)
    parser.add_argument("--optimizer", choices=list(OPTIMIZER_CHOICES.keys()), default="shisha")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Optimizer tolerance (default: from optimizer)")
    parser.add_argument("--signal-file", type=str, default=None,
                        help="Write this file when optimum is first reached (for external coordination)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose optimizer logging")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    # Find the model in the set
    selected_models = MODEL_SETS[args.model_set]
    model_entry = None
    for name, load_fn, rand_fn in selected_models:
        if name == args.model:
            model_entry = (name, load_fn, rand_fn)
            break
    if model_entry is None:
        available = [n for n, _, _ in selected_models]
        raise ValueError(f"Model '{args.model}' not in '{args.model_set}' set. Available: {available}")

    model_name, load_model, rand_inputs = model_entry

    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    last_rank = dist.get_world_size() - 1
    is_print_rank = rank == 0

    # Set up signal file for external coordination
    global _signal_file
    _signal_file = args.signal_file

    # Set up server with one model
    server = PipelineServer(handle_output, verbose=False)
    if is_print_rank:
        print(f"Loading {model_name}...")
    optimizer_kwargs = {}
    if args.tolerance is not None:
        optimizer_kwargs['tolerance'] = args.tolerance
    optimizer_class = OPTIMIZER_CHOICES[args.optimizer]
    server.add_model(model_name, load_model(), rand_inputs(),
                     optimizer_class=optimizer_class,
                     n_microbatches=args.n_microbatches,
                     async_optimization=False,
                     verbose=(args.verbose and is_print_rank),
                     **optimizer_kwargs)

    # Input generator — uses incrementing seed for determinism
    seed_counter = [0]

    def gen_input():
        s = seed_counter[0]
        seed_counter[0] += 1
        return generate_batch(rand_inputs, args.batch_size, s)

    # SIGTERM handler — signal run_continuous to stop
    def on_sigterm(*_):
        if is_print_rank:
            print("\nSIGTERM received, finishing current batch...")
        server.stop_continuous()

    signal.signal(signal.SIGTERM, on_sigterm)

    if is_print_rank:
        print(f"Running continuous evaluation for {model_name}...")

    server.run_continuous(model_name, gen_input, handle_output_fn=handle_output)

    # Save results (last rank has the timing data)
    if rank == last_rank:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash = None

        batches = []
        for i, timing in enumerate(result_timings):
            batch_entry = {"seed": i}
            if timing is not None:
                fwd = timing["forward"]
                reb = timing["rebalance"]
                batch_entry["timing"] = {"start": fwd["start"], "end": fwd["end"]}
                batch_entry["rebalance"] = {
                    "start": reb["start"],
                    "end": reb["end"],
                    "did_rebalance": reb["did_rebalance"],
                    "at_optimum": reb.get("at_optimum", False),
                    "deep_gamma": reb.get("deep_gamma"),
                    "sibling_gamma": reb.get("sibling_gamma"),
                    "best_throughput": reb.get("best_throughput"),
                    "optimum_escape_elapsed": reb.get("optimum_escape_elapsed"),
                }
            batches.append(batch_entry)

        timed = [b for b in batches if "timing" in b]
        if len(timed) >= 2:
            wall = timed[-1]["timing"]["end"] - timed[0]["timing"]["start"]
            rps = len(timed) * args.batch_size / wall if wall > 0 else 0
        else:
            rps = 0

        data = {
            "meta": {
                "mode": "continuous",
                "num_requests": len(batches),
                "batch_size": args.batch_size,
                "n_microbatches": args.n_microbatches,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "world_size": dist.get_world_size(),
                "clock": "time.perf_counter (cross-rank)",
                "git_commit": git_hash,
                "optimizer": "TimeBasedShishaPipelineOptimizer",
            },
            "results": {
                model_name: {
                    "batches": batches,
                    "requests_per_second": rps,
                }
            },
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        if is_print_rank:
            print(f"  {model_name}: {len(batches)} batches, {rps:.2f} rps")
            print(f"  Saved to {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
