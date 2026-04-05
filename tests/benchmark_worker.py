"""Distributed benchmark worker — launched by benchmark.py via torchrun.

Reads a config JSON, loads eligible models into the adaptive pipeline,
queues seeded random inputs, runs the pipeline, and writes timing results
to a JSON file for the orchestrator to read.
"""

import argparse
import json
import os
import time
import uuid
from typing import Any

import torch
import torch.distributed as dist

from model_runner import PipelineServer, uuids_to_tensor, tensor_to_uuids
from model_runner.pipeline_optimizer import ReactiveShishaOptimiser
from tests.testing_models import evaluation_models


# ── Output collection ────────────────────────────────────────────────────

_results: dict[uuid.UUID, Any] = {}
_timings: dict[uuid.UUID, dict | None] = {}


def handle_output(request_id: uuid.UUID, model_name: str, output: Any,
                  timing: dict | None):
    _results[request_id] = output
    _timings[request_id] = timing


# ── Main ─────────────────────────────────────────────────────────────────

def worker_main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    num_requests = config["num_requests"]
    depth = config.get("depth", 3)
    eligible_names = set(config["eligible_models"])

    # Filter evaluation_models to only eligible ones
    models = [(n, lf, rf) for n, lf, rf in evaluation_models if n in eligible_names]

    rank = dist.get_rank()
    last_rank = dist.get_world_size() - 1

    main = PipelineServer(handle_output, default_timing_depth=depth, verbose=False)

    # Register models (all ranks)
    for model_name, load_model, rand_input_fn in models:
        main.add_model(
            model_name, load_model(), rand_input_fn(),
            optimizer_class=ReactiveShishaOptimiser,
            rebalance_interval=4, n_microbatches=5, async_optimization=False,
        )

    # Queue work (rank 0 only) with seeded random inputs
    requests: dict[str, list[uuid.UUID]] = {}
    wall_start = time.perf_counter()

    if rank == 0:
        for model_name, _, rand_input_fn in models:
            requests[model_name] = []
            for i in range(num_requests):
                torch.manual_seed(42 + i)
                x = rand_input_fn()
                req_id = main.queue_work(model_name, x)
                requests[model_name].append(req_id)

        # Send request IDs to last rank (same pattern as evaluation.py)
        if last_rank != 0:
            for model_name, _, _ in models:
                uuids = requests[model_name]
                n = torch.tensor([len(uuids)], dtype=torch.int)
                dist.send(n, dst=last_rank)
                if len(uuids) > 0:
                    t = uuids_to_tensor(uuids, len(uuids))
                    dist.send(t, dst=last_rank)

    elif rank == last_rank:
        for model_name, _, _ in models:
            n = torch.zeros(1, dtype=torch.int)
            dist.recv(n, src=0)
            count = n.item()
            requests[model_name] = []
            if count > 0:
                t = torch.zeros(count * 4, dtype=torch.int)
                dist.recv(t, src=0)
                decoded = tensor_to_uuids(t)
                requests[model_name] = [u for u in decoded if u is not None]

    # Run pipeline (all ranks)
    main.run(exit_when_done=True)

    wall_end = time.perf_counter()

    # Collect timing on the last rank and write results
    if rank == last_rank:
        output = {}

        for model_name in requests:
            req_ids = requests[model_name]
            if not req_ids:
                continue

            wall_time = wall_end - wall_start
            throughput = len(req_ids) / wall_time if wall_time > 0 else 0.0

            # Group batches by shared timing object
            batches: list[list[uuid.UUID]] = []
            last_tid = None
            for req_id in req_ids:
                tid = id(_timings.get(req_id))
                if tid != last_tid:
                    batches.append([])
                    last_tid = tid
                batches[-1].append(req_id)

            batch_times = []
            for batch in batches:
                timing = _timings.get(batch[0])
                if timing is not None:
                    fwd = timing["forward"]
                    batch_times.append(fwd["end"] - fwd["start"])

            # Average of the last few batches as "final config" performance
            tail = batch_times[-3:] if len(batch_times) >= 3 else batch_times
            final_config_avg = sum(tail) / len(tail) if tail else 0.0

            output[model_name] = {
                "wall_time": wall_time,
                "throughput": throughput,
                "batch_times": batch_times,
                "final_config_avg": final_config_avg,
            }

        results_path = os.environ.get("BENCHMARK_RESULTS_PATH")
        if results_path:
            with open(results_path, "w") as f:
                json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    device = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(device)
    dist.init_process_group(backend=backend)

    worker_main(args.config)

    dist.destroy_process_group()
