import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from model_runner import PipelineServer, uuids_to_tensor, tensor_to_uuids
from model_runner.pipeline_optimizer import GreedyPipelineOptimizer, TimeBasedShishaPipelineOptimizer

from tests.testing_models import evaluation_models
from tests.util import generate_batch

# Module-level verbosity and progress state (set by evaluation_main)
_verbose = False
_total_requests = 0
_completed_requests = 0
_last_progress_pct = -1

optimizer_choices = {
    "shisha": TimeBasedShishaPipelineOptimizer,
    "greedy": GreedyPipelineOptimizer,
}

evaluation_results: dict[uuid.UUID, Any] = {}
"""Global dict storing pipeline outputs keyed by request UUID. Populated by ``handle_output``."""

evaluation_timings: dict[uuid.UUID, dict | None] = {}
"""Global dict storing pipeline wall-clock timings keyed by request UUID."""


def handle_output(request_id: uuid.UUID, model_name: str, output: Any, timing: dict | None):
    """Callback passed to ``PipelineServer`` to collect pipeline outputs.

    Stores the output and timing in the global dicts for later comparison.
    In verbose mode, prints each received output. In quiet mode, prints
    progress at 10% intervals.

    Args:
        request_id: UUID of the completed request.
        model_name: Name of the model that produced the output.
        output: The output tensor from the pipeline.
        timing: Wall-clock timing dict with ``"start"`` and ``"end"`` keys, or ``None``.
    """
    global _completed_requests, _last_progress_pct

    evaluation_results[request_id] = output
    evaluation_timings[request_id] = timing

    if _verbose:
        print(f"Received output from {model_name} from request: {request_id}. Output shape: {output.shape}")
    elif _total_requests > 0:
        _completed_requests += 1
        pct = (_completed_requests * 100) // _total_requests
        # Print at every 10% milestone
        milestone = (pct // 10) * 10
        if milestone > _last_progress_pct and milestone > 0:
            _last_progress_pct = milestone
            print(f"  Processing... {milestone}%")


def evaluation_main(
    num_requests=100, seed=37, batch_size=32,
    output_file="evaluation_output.json",
    baseline_file=None,
    verbose=False,
    store_hashes=False,
    n_microbatches=32,
    optimizer_class=TimeBasedShishaPipelineOptimizer,
    rebalance_interval=None,
    optimizer_kwargs=None,
):
    """Run evaluation: queue generated inputs through the adaptive pipeline.

    Generates inputs from seeds (deterministic), registers models, queues work
    (rank 0 only), runs the pipeline across all ranks, and saves results as JSON.
    Optionally compares output hashes against a baseline file.

    Args:
        num_requests: Total number of requests to run per model.
        seed: Base random seed for input generation.
        batch_size: Number of samples per batch.
        output_file: Path to write the output JSON file.
        baseline_file: Optional path to a baseline JSON file for hash comparison.
        verbose: If True, print detailed per-request output.
        store_hashes: If True, compute and store output hashes in the JSON.
        n_microbatches: How many requests to bundle into a forward pass.
        optimizer_class: What pipeline optimiser to use.
        optimizer_kwargs: Extra keyword arguments forwarded to the optimizer constructor.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if store_hashes:
        import hashlib
    global _verbose, _total_requests, _completed_requests, _last_progress_pct
    _verbose = verbose
    _total_requests = num_requests * len(evaluation_models)
    _completed_requests = 0
    _last_progress_pct = -1

    rank = dist.get_rank()
    last_rank = dist.get_world_size() - 1
    is_print_rank = rank == 0
    requests: dict[str, list[uuid.UUID]] = dict()

    # Init main — only enable verbose logging on rank 0 to avoid duplicate prints
    if is_print_rank:
        print(f"Evaluation: {len(evaluation_models)} model(s), "
              f"{num_requests} requests each, world_size={dist.get_world_size()}")
    main = PipelineServer(handle_output, verbose=(verbose and is_print_rank))

    # Adding models (all ranks)
    if not verbose and is_print_rank:
        print("Loading models...")
    for model_name, load_model, rand_input in evaluation_models:
        if verbose and is_print_rank:
            print(f"> Adding model {model_name} with load function {load_model.__name__}")
        main.add_model(model_name, load_model(), rand_input(),
                       optimizer_class=optimizer_class,
                       rebalance_interval=rebalance_interval, n_microbatches=n_microbatches,
                       async_optimization=False, **optimizer_kwargs)
    if not verbose and is_print_rank:
        print("Models loaded. Running pipeline...")

    # Queue work (rank 0 only) — generate inputs from seeds
    if rank == 0:
        for model_name, _, rand_inputs in evaluation_models:
            requests[model_name] = list()
            for i in range(num_requests):
                input_seed = seed + i
                x = generate_batch(rand_inputs, batch_size, input_seed)
                req_id = main.queue_work(model_name, x)
                requests[model_name].append(req_id)
                if verbose and is_print_rank:
                    print(f" > Work added with request id: {req_id}")

        # Send request IDs to last rank so it can match outputs
        if last_rank != 0:
            for model_name, _, _ in evaluation_models:
                uuids = requests[model_name]
                n = torch.tensor([len(uuids)], dtype=torch.int)
                dist.send(n, dst=last_rank)
                if len(uuids) > 0:
                    t = uuids_to_tensor(uuids, len(uuids))
                    dist.send(t, dst=last_rank)

    # Receive request IDs from rank 0
    elif rank == last_rank:  # Unless there is a single rank in the world
        for model_name, _, _ in evaluation_models:
            n = torch.zeros(1, dtype=torch.int)
            dist.recv(n, src=0)
            count = n.item()
            requests[model_name] = []
            if count > 0:
                t = torch.zeros(count * 4, dtype=torch.int)
                dist.recv(t, src=0)
                decoded = tensor_to_uuids(t)
                requests[model_name] = [u for u in decoded if u is not None]

    # Running the main service (all ranks):
    main.run(exit_when_done=True)

    # Build output JSON and optionally compare against baseline (last rank only)
    if rank == last_rank:
        meta = {
            "mode": "adaptive",
            "num_requests": num_requests,
            "seed": seed,
            "batch_size": batch_size,
            "store_hashes": store_hashes,
            "output_file": output_file,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "world_size": dist.get_world_size(),
            "clock": "time.perf_counter (cross-rank)",
            "argv": sys.argv,
        }
        results = {}

        for model_name in requests:
            batches = []
            for i, req_id in enumerate(requests[model_name]):
                input_seed = seed + i
                timing = evaluation_timings.get(req_id)

                batch_entry = {"seed": input_seed}

                if timing is not None:
                    fwd = timing["forward"]
                    reb = timing["rebalance"]
                    batch_entry["timing"] = {"start": fwd["start"], "end": fwd["end"]}
                    batch_entry["rebalance"] = {
                        "start": reb["start"],
                        "end": reb["end"],
                        "did_rebalance": reb["did_rebalance"],
                    }

                if store_hashes:
                    output = evaluation_results[req_id]
                    batch_entry["output_hashes"] = [
                        hashlib.sha256(output[j].numpy().tobytes()).hexdigest()
                        for j in range(output.shape[0])
                    ]

                batches.append(batch_entry)

            # Compute requests_per_second from wall clock of forward timings
            timed_batches = [b for b in batches if "timing" in b]
            if len(timed_batches) >= 2:
                wall_clock = timed_batches[-1]["timing"]["end"] - timed_batches[0]["timing"]["start"]
                total_samples = len(timed_batches) * batch_size
                rps = total_samples / wall_clock if wall_clock > 0 else 0.0
            elif len(timed_batches) == 1:
                t = timed_batches[0]["timing"]
                wall_clock = t["end"] - t["start"]
                rps = batch_size / wall_clock if wall_clock > 0 else 0.0
            else:
                rps = 0.0

            results[model_name] = {
                "batches": batches,
                "requests_per_second": rps,
            }

        # Save output JSON
        with open(output_file, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\nEvaluation results saved to {output_file}")

        # Print summary
        for model_name, model_results in results.items():
            n_batches = len(model_results["batches"])
            rps = model_results["requests_per_second"]
            print(f"  [{model_name}] {n_batches} requests, {rps:.2f} samples/sec")

            # Rebalance summary
            rebalance_count = sum(
                1 for b in model_results["batches"]
                if b.get("rebalance", {}).get("did_rebalance", False)
            )
            if rebalance_count > 0:
                print(f"    Rebalanced {rebalance_count} time(s)")

        # Optional baseline comparison (hash-based only)
        if baseline_file is not None and os.path.exists(baseline_file):
            print(f"\nComparing output hashes against baseline: {baseline_file}")
            with open(baseline_file, "r") as f:
                raw = json.load(f)
            baseline_results = raw.get("results", raw)

            pass_count = 0
            fail_count = 0
            skip_count = 0
            for model_name in requests:
                if model_name not in baseline_results:
                    print(f"  [{model_name}] not found in baseline, skipping")
                    skip_count += len(requests[model_name])
                    continue
                baseline_batches = baseline_results[model_name].get("batches", [])
                for i, req_id in enumerate(requests[model_name]):
                    if i >= len(baseline_batches):
                        skip_count += 1
                        continue
                    baseline_entry = baseline_batches[i]
                    if "output_hashes" not in baseline_entry:
                        skip_count += 1
                        continue
                    if not store_hashes:
                        skip_count += 1
                        continue
                    eval_entry = results[model_name]["batches"][i]
                    eval_hashes = eval_entry.get("output_hashes", [])
                    baseline_hashes = baseline_entry["output_hashes"]
                    if eval_hashes == baseline_hashes:
                        pass_count += 1
                        if verbose:
                            print(f"  [{model_name}] Request {i}: PASS")
                    else:
                        fail_count += 1
                        print(f"  [{model_name}] Request {i}: FAIL (hash mismatch)")

            total = pass_count + fail_count + skip_count
            print(f"  Hash comparison: {pass_count} pass, {fail_count} fail, {skip_count} skipped (total {total})")
        elif baseline_file is not None:
            print(f"\nBaseline file not found: {baseline_file} — skipping comparison")

    if verbose:
        print(f"rank:{dist.get_rank()} exiting!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline evaluation")
    parser.add_argument('-n', '--num-requests', type=int, default=100,
                        help='Total number of requests per model (default: 100)')
    parser.add_argument('-s', '--seed', type=int, default=37,
                        help='Base random seed (default: 37)')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('-o', '--output', default='evaluation_output.json',
                        help='Output path: a directory (auto-generates timestamped filename) '
                             'or a .json file path (used as-is)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output (print every request and batch detail)')
    parser.add_argument('--baseline-file', default=None,
                        help='Optional baseline JSON file for hash comparison')
    parser.add_argument('-m', '--n-microbatches', type=int, default=32,
                        help='Requests per forward pass (default: 32)')
    parser.add_argument('--store-hashes', action='store_true',
                        help='Store output hashes in JSON')
    parser.add_argument('--optimizer', choices=optimizer_choices.keys(), default='shisha',
                        help='Pipeline optimizer class (default: shisha)')
    parser.add_argument('--rebalance-interval', type=int, default=None,
                        help='Check rebalance every N batches (default: None, check every batch)')
    parser.add_argument('--assignment-choice', choices=['rank_w', 'rank_l'], default=None,
                        help='Shisha device assignment strategy (default: rank_w)')
    parser.add_argument('--balance-strategy', choices=['nearest_lightest_fep', 'nearest_fep'], default=None,
                        help='Shisha balance strategy (default: nearest_lightest_fep)')
    parser.add_argument('--alpha', type=int, default=None,
                        help='Shisha patience parameter (default: 10)')
    args = parser.parse_args()

    rebalance_interval = args.rebalance_interval
    if rebalance_interval is not None and rebalance_interval < 0:
        rebalance_interval = None

    # Resolve output path: directory → timestamped file, file → use as-is
    output_path = Path(args.output)
    if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
        # Treat as directory — generate timestamped filename
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = str(output_path / f"{timestamp}.json")
    else:
        output_file = str(output_path)

    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)

    # Build optimizer kwargs from CLI args (only include if explicitly set)
    optimizer_kwargs = {}
    if args.assignment_choice is not None:
        optimizer_kwargs['assignment_choice'] = args.assignment_choice
    if args.balance_strategy is not None:
        optimizer_kwargs['balance_strategy'] = args.balance_strategy
    if args.alpha is not None:
        optimizer_kwargs['alpha'] = args.alpha

    evaluation_main(
        num_requests=args.num_requests,
        seed=args.seed,
        batch_size=args.batch_size,
        output_file=output_file,
        baseline_file=args.baseline_file,
        verbose=args.verbose,
        store_hashes=args.store_hashes,
        n_microbatches=args.n_microbatches,
        optimizer_class=optimizer_choices[args.optimizer],
        rebalance_interval=rebalance_interval,
        optimizer_kwargs=optimizer_kwargs,
    )

    dist.destroy_process_group()
