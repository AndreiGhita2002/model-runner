import json
import os
import sys
import uuid
from typing import Any

import torch
import torch.distributed as dist

from model_runner import PipelineServer, uuids_to_tensor, tensor_to_uuids
from model_runner.pipeline_optimizer import ReactiveShishaOptimiser
from tests.testing_models import load_conv_next, conv_next_rand_inputs
from tests.util import generate_batch


QUICK_MODEL_NAME = "conv_next"
QUICK_NUM_REQUESTS = 4
QUICK_SEED = 37
QUICK_BATCH_SIZE = 32


quick_results: dict[uuid.UUID, Any] = {}
quick_timings: dict[uuid.UUID, dict | None] = {}


def handle_output(request_id: uuid.UUID, model_name: str, output: Any, timing: dict | None):
    quick_results[request_id] = output
    quick_timings[request_id] = timing
    print(f"Received output from {model_name} request: {request_id}. Shape: {output.shape}")


def quick_evaluation_main(baseline_file: str = None):
    """Run a single model with a small number of requests to verify the pipeline works.

    Generates inputs from seeds (no baseline file needed). If a baseline file
    with output hashes is provided, compares against it.
    """
    rank = dist.get_rank()
    last_rank = dist.get_world_size() - 1

    print("Initialising main service...")
    main = PipelineServer(handle_output, verbose=True)

    print(f"> Adding model {QUICK_MODEL_NAME}")
    main.add_model(QUICK_MODEL_NAME, load_conv_next(), conv_next_rand_inputs(),
                   optimizer_class=ReactiveShishaOptimiser,
                   rebalance_interval=4, async_optimization=False)

    requests: list[uuid.UUID] = []

    if rank == 0:
        for i in range(QUICK_NUM_REQUESTS):
            input_seed = QUICK_SEED + i
            x = generate_batch(conv_next_rand_inputs, QUICK_BATCH_SIZE, input_seed)
            req_id = main.queue_work(QUICK_MODEL_NAME, x)
            requests.append(req_id)
            print(f"  > Queued request {req_id} (seed={input_seed})")

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

    # Force rebalance: queue one more request and trigger a forced rebalance
    if rank == 0:
        print("\n  > Forcing rebalance for next forward pass...")
        main.force_rebalance(QUICK_MODEL_NAME)
        input_seed = QUICK_SEED + QUICK_NUM_REQUESTS
        x = generate_batch(conv_next_rand_inputs, QUICK_BATCH_SIZE, input_seed)
        req_id = main.queue_work(QUICK_MODEL_NAME, x)
        requests.append(req_id)
        print(f"  > Queued request {req_id} (seed={input_seed})")

        # Send the new request ID to last rank
        if last_rank != 0:
            t = uuids_to_tensor([req_id], 1)
            dist.send(t, dst=last_rank)
    elif rank == last_rank:
        t = torch.zeros(4, dtype=torch.int)
        dist.recv(t, src=0)
        decoded = tensor_to_uuids(t)
        requests.extend([u for u in decoded if u is not None])

    main.run(exit_when_done=True)

    # Check results (last rank only)
    if rank == last_rank:
        print(f"\nQuick eval completed: {len(requests)} requests")
        for i, req_id in enumerate(requests):
            output = quick_results[req_id]
            print(f"  Request {i} (seed={QUICK_SEED + i}): output shape {list(output.shape)}")

        # Optional baseline hash comparison
        if baseline_file is not None and os.path.exists(baseline_file):
            print(f"\nComparing against baseline: {baseline_file}")
            with open(baseline_file, "r") as f:
                raw = json.load(f)
            baseline_results = raw.get("results", raw)
            if QUICK_MODEL_NAME in baseline_results:
                baseline_batches = baseline_results[QUICK_MODEL_NAME].get("batches", [])
                import hashlib
                passed = 0
                failed = 0
                for i, req_id in enumerate(requests):
                    if i >= len(baseline_batches):
                        break
                    entry = baseline_batches[i]
                    if "output_hashes" not in entry:
                        continue
                    output = quick_results[req_id]
                    eval_hashes = [
                        hashlib.sha256(output[j].numpy().tobytes()).hexdigest()
                        for j in range(output.shape[0])
                    ]
                    if eval_hashes == entry["output_hashes"]:
                        print(f"  Request {i}: PASS")
                        passed += 1
                    else:
                        print(f"  Request {i}: FAIL (hash mismatch)")
                        failed += 1
                if passed + failed > 0:
                    print(f"  Hash comparison: {passed} pass, {failed} fail")
            else:
                print(f"  '{QUICK_MODEL_NAME}' not found in baseline")
        elif baseline_file is not None:
            print(f"\nBaseline file not found: {baseline_file}")

    print(f"rank:{dist.get_rank()} exiting!")


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(device)
    dist.init_process_group(backend=backend)

    baseline = sys.argv[1] if len(sys.argv) > 1 else None
    quick_evaluation_main(baseline_file=baseline)

    dist.destroy_process_group()
