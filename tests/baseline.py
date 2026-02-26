import hashlib
import json
import time

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe

from tests.testing_models import evaluation_models
from tests.util import generate_batch

DEFAULT_BASELINE_FILE = "baseline_outputs.json"


def baseline_pipeline(num_requests: int = 100, seed: int = 37,
                      output_file: str = DEFAULT_BASELINE_FILE, batch_size: int = 32):
    is_first_rank = dist.get_rank() == 0
    is_last_rank = dist.get_rank() == (dist.get_world_size() - 1)
    results = {}

    for model_name, load_model, rand_inputs in evaluation_models:
        print(f"Running baseline for model: {model_name}")
        model = load_model()
        results[model_name] = {"batches": []}
        # TODO: provide an explicit split_spec for a fair GPipe comparison.
        #  Currently relies on PyTorch's default heuristic. Consider even-split
        #  by layer/parameter count, or reuse GreedyPipelineOptimizer.initial_setup().
        pipe = pipeline(model, mb_args=(generate_batch(rand_inputs, batch_size, seed),))
        stage = pipe.get_stage_module(dist.get_rank())
        scheduler = ScheduleGPipe(stage, n_microbatches=batch_size)

        # Warmup pass (not recorded) to avoid lazy-init / CUDA kernel caching overhead
        with torch.no_grad():
            if is_first_rank:
                scheduler.step(generate_batch(rand_inputs, batch_size, seed))
            else:
                scheduler.step()

        for i in range(num_requests):
            input_seed = seed + i

            if is_first_rank:
                x = generate_batch(rand_inputs, batch_size, input_seed)

            # Run forward pass
            if is_first_rank:
                forward_start = time.perf_counter()

            with torch.no_grad():
                if is_first_rank:
                    output = scheduler.step(x)
                else:
                    output = scheduler.step()

            if is_last_rank:
                forward_end = time.perf_counter()

            # Send forward_start from rank 0 to last rank
            if dist.get_world_size() > 1:
                if is_first_rank:
                    t = torch.tensor([forward_start], dtype=torch.float64)
                    dist.send(t, dst=dist.get_world_size() - 1)
                elif is_last_rank:
                    t = torch.tensor([0.0], dtype=torch.float64)
                    dist.recv(t, src=0)
                    forward_start = t.item()

            # Last rank stores output hash + timing data (input reproducible from seed)
            if is_last_rank:
                # Hash each sample in the batch individually
                output_hashes = [
                    hashlib.sha256(output[0][j].numpy().tobytes()).hexdigest()
                    for j in range(output[0].shape[0])
                ]
                results[model_name]["batches"].append({
                    "seed": input_seed,
                    "output_hashes": output_hashes,
                    "timing": {"start": forward_start, "end": forward_end}
                })

        # Compute sustained throughput
        if is_last_rank:
            batches = results[model_name]["batches"]
            wall_clock = batches[-1]["timing"]["end"] - batches[0]["timing"]["start"]
            total_samples = num_requests * batch_size
            results[model_name]["requests_per_second"] = total_samples / wall_clock

    # Only the last rank has meaningful results
    if is_last_rank:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Baseline outputs saved to {output_file}")


def baseline_simple(num_requests: int = 100, seed: int = 37,
                    output_file: str = DEFAULT_BASELINE_FILE, batch_size: int = 32):
    results = {}

    for model_name, load_model, rand_inputs in evaluation_models:
        print(f"Running baseline for model: {model_name}")
        model = load_model()
        results[model_name] = {"batches": []}

        # Warmup pass (not recorded) to avoid PyTorch lazy-init overhead in timings
        with torch.no_grad():
            model(generate_batch(rand_inputs, batch_size, seed))

        for i in range(num_requests):
            input_seed = seed + i
            x = generate_batch(rand_inputs, batch_size, input_seed)

            # Run forward pass
            with torch.no_grad():
                start = time.perf_counter()
                output = model(x)
                end = time.perf_counter()

            # Hash each sample in the batch individually
            output_hashes = [
                hashlib.sha256(output[j].numpy().tobytes()).hexdigest()
                for j in range(output.shape[0])
            ]
            results[model_name]["batches"].append({
                "seed": input_seed,
                "output_hashes": output_hashes,
                "timing": {"start": start, "end": end}
            })

        # Compute sustained throughput
        batches = results[model_name]["batches"]
        wall_clock = batches[-1]["timing"]["end"] - batches[0]["timing"]["start"]
        total_samples = num_requests * batch_size
        results[model_name]["requests_per_second"] = total_samples / wall_clock

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline outputs saved to {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline benchmarks")
    parser.add_argument("mode", choices=["simple", "pipeline"], help="Which baseline to run")
    parser.add_argument("-o", "--output", default=DEFAULT_BASELINE_FILE, help="Output JSON file")
    parser.add_argument("-n", "--num-requests", type=int, default=100, help="Number of requests")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("-s", "--seed", type=int, default=37, help="Base random seed")
    args = parser.parse_args()

    if args.mode == "simple":
        baseline_simple(num_requests=args.num_requests, seed=args.seed,
                        output_file=args.output, batch_size=args.batch_size)
    elif args.mode == "pipeline":
        device = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(device)
        dist.init_process_group(backend=backend)
        baseline_pipeline(num_requests=args.num_requests, seed=args.seed,
                          output_file=args.output, batch_size=args.batch_size)
