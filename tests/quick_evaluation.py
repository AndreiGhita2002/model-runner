import json
import os
import sys
import uuid
from typing import Any

import torch
import torch.distributed as dist

from model_runner import MainService, uuids_to_tensor, tensor_to_uuids
from model_runner.pipeline_optimizer import TimeBasedShishaPipelineOptimizer
from tests.baseline import baseline_main, DEFAULT_BASELINE_FILE
from tests.testing_models import load_conv_next, conv_next_rand_inputs


QUICK_MODEL_NAME = "conv_next"


quick_results: dict[uuid.UUID, Any] = {}
quick_timings: dict[uuid.UUID, dict | None] = {}


def handle_output(request_id: uuid.UUID, model_name: str, output: Any, timing: dict | None):
    quick_results[request_id] = output
    quick_timings[request_id] = timing
    print(f"Received output from {model_name} request: {request_id}. Shape: {output.shape}")


def quick_evaluation_main(baseline_file: str = DEFAULT_BASELINE_FILE):
    """Run a single model with a single batch against baseline to verify correctness."""

    # Load baseline (generate if missing)
    if not os.path.exists(baseline_file):
        print(f"Baseline file not found, generating {baseline_file}...")
        if dist.get_rank() == 0:
            baseline_main(output_file=baseline_file, num_requests=30)
        dist.barrier()

    with open(baseline_file, "r") as f:
        baseline_data = json.load(f)

    if QUICK_MODEL_NAME not in baseline_data:
        print(f"ERROR: '{QUICK_MODEL_NAME}' not found in baseline file.")
        return

    rank = dist.get_rank()
    last_rank = dist.get_world_size() - 1

    # Only use the first 4 baseline entries (one batch)
    model_baseline = baseline_data[QUICK_MODEL_NAME][:4]

    print("Initialising main service...")
    main = MainService(handle_output, verbose=True)

    print(f"> Adding model {QUICK_MODEL_NAME}")
    main.add_model(QUICK_MODEL_NAME, load_conv_next(), conv_next_rand_inputs(),
                   optimizer_class=TimeBasedShishaPipelineOptimizer,
                   rebalance_interval=4)

    requests: list[uuid.UUID] = []

    if rank == 0:
        for entry in model_baseline:
            x = torch.tensor(entry["input"])
            req_id = main.queue_work(QUICK_MODEL_NAME, x)
            requests.append(req_id)
            print(f"  > Queued request {req_id}")

        # Send request IDs to last rank
        if last_rank != 0:
            n = torch.tensor([len(requests)], dtype=torch.int)
            dist.send(n, dst=last_rank)
            if len(requests) > 0:
                t = uuids_to_tensor(requests, len(requests))
                dist.send(t, dst=last_rank)

    elif rank == last_rank:
        n = torch.zeros(1, dtype=torch.int)
        dist.recv(n, src=0)
        count = n.item()
        if count > 0:
            t = torch.zeros(count * 4, dtype=torch.int)
            dist.recv(t, src=0)
            decoded = tensor_to_uuids(t)
            requests = [u for u in decoded if u is not None]

    # Run pipeline (all ranks)
    main.run(exit_when_done=True)

    # Compare outputs (last rank only)
    if rank == last_rank:
        print("\nComparing outputs...")
        failed = 0
        for i, req_id in enumerate(requests):
            pipeline_output = quick_results[req_id]
            baseline_output = torch.tensor(model_baseline[i]["output"])

            if pipeline_output.shape != baseline_output.shape:
                print(f"  Request {i}: FAIL (shape mismatch: "
                      f"pipeline={list(pipeline_output.shape)}, baseline={list(baseline_output.shape)})")
                failed += 1
            elif torch.allclose(pipeline_output, baseline_output, atol=1e-6):
                print(f"  Request {i}: PASS")
            else:
                print(f"  Request {i}: FAIL (values differ)")
                failed += 1

        if failed == 0:
            print(f"\nQuick eval PASSED ({len(requests)} requests)")
        else:
            print(f"\nQuick eval FAILED ({failed}/{len(requests)} requests)")

    print(f"rank:{dist.get_rank()} exiting!")


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(device)
    dist.init_process_group(backend=backend)

    baseline = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE_FILE
    quick_evaluation_main(baseline_file=baseline)

    dist.destroy_process_group()
