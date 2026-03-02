import json
import os
import sys
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
                   store_hashes: bool = False, n_microbatches: int = 32):
    if store_hashes:
        import hashlib
    is_first_rank = dist.get_rank() == 0
    is_last_rank = dist.get_rank() == (dist.get_world_size() - 1)
    # Use CUDA when available (nccl backend), otherwise CPU (gloo).
    # torch.accelerator.current_accelerator() can return MPS which doesn't
    # support distributed P2P, so we derive the device from the backend.
    backend = dist.get_backend()
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    meta = {
        "mode": "gpipe",
        "num_requests": num_requests,
        "seed": seed,
        "batch_size": batch_size,
        "n_microbatches": n_microbatches,
        "store_hashes": store_hashes,
        "output_file": output_file,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "world_size": dist.get_world_size(),
        "device": str(device),
        "clock": "time.perf_counter (last rank)",
        "argv": sys.argv,
    }
    results = {}

    for model_name, load_model, rand_inputs in evaluation_models:
        print(f"Running baseline for model: {model_name}")
        try:
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
            scheduler = ScheduleGPipe(stage, n_microbatches=n_microbatches)

            # Warmup pass (not recorded) to avoid lazy-init / CUDA kernel caching overhead
            warmup_inputs = [generate_batch(rand_inputs, batch_size, seed) for _ in range(n_microbatches)]
            with torch.no_grad():
                if is_first_rank:
                    scheduler.step(torch.cat(warmup_inputs, dim=0))
                else:
                    scheduler.step()

            for chunk_start in range(0, num_requests, n_microbatches):
                chunk_size = min(n_microbatches, num_requests - chunk_start)
                seeds = [seed + chunk_start + j for j in range(chunk_size)]

                if is_first_rank:
                    inputs = [generate_batch(rand_inputs, batch_size, s) for s in seeds]
                    # Pad to n_microbatches by replicating last input (matches pipeline_runner.py)
                    while len(inputs) < n_microbatches:
                        inputs.append(inputs[-1])
                    x = torch.cat(inputs, dim=0)

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
                        "seeds": seeds,
                        "timing": {"start": forward_start, "end": forward_end},
                    }
                    if store_hashes:
                        # Only hash the real outputs (not padded ones)
                        real_count = chunk_size * batch_size
                        batch_entry["output_hashes"] = [
                            hashlib.sha256(output[0][j].numpy().tobytes()).hexdigest()
                            for j in range(real_count)
                        ]
                    results[model_name]["batches"].append(batch_entry)

            # Compute sustained throughput
            if is_last_rank:
                batches = results[model_name]["batches"]
                wall_clock = batches[-1]["timing"]["end"] - batches[0]["timing"]["start"]
                total_samples = num_requests * batch_size
                results[model_name]["requests_per_second"] = total_samples / wall_clock

        except Exception as e:
            print(f"  Skipping {model_name}: {e}")
            results.pop(model_name, None)

    # Only the last rank has meaningful results
    if is_last_rank:
        with open(output_file, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"Baseline outputs saved to {output_file}")


def simple_baseline(num_requests: int = 100, seed: int = 37,
                    output_file: str = DEFAULT_BASELINE_FILE, batch_size: int = 32,
                    store_hashes: bool = False, n_microbatches: int = 32):
    if store_hashes:
        import hashlib
    meta = {
        "mode": "simple",
        "num_requests": num_requests,
        "seed": seed,
        "batch_size": batch_size,
        "n_microbatches": n_microbatches,
        "store_hashes": store_hashes,
        "output_file": output_file,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "clock": "time.perf_counter",
        "argv": sys.argv,
    }
    results = {}

    for model_name, load_model, rand_inputs in evaluation_models:
        print(f"Running baseline for model: {model_name}")
        model = load_model()
        results[model_name] = {"batches": []}

        # Warmup pass (not recorded) to avoid PyTorch lazy-init overhead in timings
        with torch.no_grad():
            model(generate_batch(rand_inputs, batch_size, seed))

        for chunk_start in range(0, num_requests, n_microbatches):
            chunk_size = min(n_microbatches, num_requests - chunk_start)
            seeds = [seed + chunk_start + j for j in range(chunk_size)]
            inputs = [generate_batch(rand_inputs, batch_size, s) for s in seeds]
            x = torch.cat(inputs, dim=0)

            # Run forward pass
            with torch.no_grad():
                start = time.perf_counter()
                output = model(x)
                end = time.perf_counter()

            batch_entry = {
                "seeds": seeds,
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

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"Baseline outputs saved to {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline benchmarks")
    parser.add_argument("mode", choices=["simple", "gpipe"], help="Which baseline to run")
    parser.add_argument("-o", "--output", default=DEFAULT_BASELINE_FILE, help="Output JSON file")
    parser.add_argument("-n", "--num-requests", type=int, default=100, help="Number of requests")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("-s", "--seed", type=int, default=37, help="Base random seed")
    parser.add_argument("-m", "--n-microbatches", type=int, default=32, help="Requests per forward pass (default: 32)")
    parser.add_argument("--store-hashes", action="store_true", help="Store output hashes (disabled by default)")
    args = parser.parse_args()

    if args.mode == "simple":
        simple_baseline(num_requests=args.num_requests, seed=args.seed,
                        output_file=args.output, batch_size=args.batch_size,
                        store_hashes=args.store_hashes, n_microbatches=args.n_microbatches)
    elif args.mode == "gpipe":
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
        gpipe_baseline(num_requests=args.num_requests, seed=args.seed,
                       output_file=args.output, batch_size=args.batch_size,
                       store_hashes=args.store_hashes, n_microbatches=args.n_microbatches)
