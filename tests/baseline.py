import json
import time

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe

from tests.testing_models import evaluation_models
from tests.util import generate_batch, gpipe_split_spec

DEFAULT_BASELINE_FILE = "baseline_outputs.json"


class _ContiguousStageWrapper(torch.nn.Module):
    """Wraps a pipeline stage to make its output contiguous for P2P send."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return output.contiguous()
        return output


def gpipe_baseline(num_requests: int = 100, seed: int = 37,
                   output_file: str = DEFAULT_BASELINE_FILE, batch_size: int = 32,
                   store_hashes: bool = False):
    if store_hashes:
        import hashlib
    is_first_rank = dist.get_rank() == 0
    is_last_rank = dist.get_rank() == (dist.get_world_size() - 1)
    device = torch.accelerator.current_accelerator()
    results = {}

    for model_name, load_model, rand_inputs in evaluation_models:
        print(f"Running baseline for model: {model_name}")
        model = load_model()
        results[model_name] = {"batches": []}

        split_spec = gpipe_split_spec(model, dist.get_world_size())
        pipe = pipeline(model, mb_args=(generate_batch(rand_inputs, batch_size, seed),), split_spec=split_spec)

        stage_module = pipe.get_stage_module(dist.get_rank())
        stage = PipelineStage(
            _ContiguousStageWrapper(stage_module),
            stage_index=dist.get_rank(),
            num_stages=pipe.num_stages,
            device=device,
        )
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

            # Run forward pass — time on last rank only to avoid cross-process clock skew
            if is_last_rank:
                forward_start = time.perf_counter()

            with torch.no_grad():
                if is_first_rank:
                    output = scheduler.step(x)
                else:
                    output = scheduler.step()

            if is_last_rank:
                forward_end = time.perf_counter()
                batch_entry = {
                    "seed": input_seed,
                    "timing": {"start": forward_start, "end": forward_end},
                }
                if store_hashes:
                    batch_entry["output_hashes"] = [
                        hashlib.sha256(output[0][j].numpy().tobytes()).hexdigest()
                        for j in range(output[0].shape[0])
                    ]
                results[model_name]["batches"].append(batch_entry)

        # Compute sustained throughput
        if is_last_rank:
            batches = results[model_name]["batches"]
            wall_clock = batches[-1]["timing"]["end"] - batches[0]["timing"]["start"]
            total_samples = num_requests * batch_size
            results[model_name]["requests_per_second"] = total_samples / wall_clock
            results[model_name]["clock"] = "time.perf_counter (last rank)"

    # Only the last rank has meaningful results
    if is_last_rank:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Baseline outputs saved to {output_file}")


def simple_baseline(num_requests: int = 100, seed: int = 37,
                    output_file: str = DEFAULT_BASELINE_FILE, batch_size: int = 32,
                    store_hashes: bool = False):
    if store_hashes:
        import hashlib
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

            batch_entry = {
                "seed": input_seed,
                "timing": {"start": start, "end": end},
            }
            if store_hashes:
                batch_entry["output_hashes"] = [
                    hashlib.sha256(output[j].numpy().tobytes()).hexdigest()
                    for j in range(output.shape[0])
                ]
            results[model_name]["batches"].append(batch_entry)

        # Compute sustained throughput
        batches = results[model_name]["batches"]
        wall_clock = batches[-1]["timing"]["end"] - batches[0]["timing"]["start"]
        total_samples = num_requests * batch_size
        results[model_name]["requests_per_second"] = total_samples / wall_clock
        results[model_name]["clock"] = "time.perf_counter"

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline outputs saved to {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline benchmarks")
    parser.add_argument("mode", choices=["simple", "gpipe"], help="Which baseline to run")
    parser.add_argument("-o", "--output", default=DEFAULT_BASELINE_FILE, help="Output JSON file")
    parser.add_argument("-n", "--num-requests", type=int, default=100, help="Number of requests")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("-s", "--seed", type=int, default=37, help="Base random seed")
    parser.add_argument("--store-hashes", action="store_true", help="Store output hashes (disabled by default)")
    args = parser.parse_args()

    if args.mode == "simple":
        simple_baseline(num_requests=args.num_requests, seed=args.seed,
                        output_file=args.output, batch_size=args.batch_size,
                        store_hashes=args.store_hashes)
    elif args.mode == "gpipe":
        device = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(device)
        dist.init_process_group(backend=backend)
        gpipe_baseline(num_requests=args.num_requests, seed=args.seed,
                       output_file=args.output, batch_size=args.batch_size,
                       store_hashes=args.store_hashes)
