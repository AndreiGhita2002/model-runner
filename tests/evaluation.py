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
from tests.testing_models import evaluation_models


def load_baseline(baseline_file: str):
    """Load baseline data from a JSON file, generating it if it doesn't exist.

    If the file is missing, rank 0 generates it via ``baseline_main``. A barrier
    ensures all ranks wait for generation to complete before reading.

    Args:
        baseline_file: Path to the baseline JSON file.

    Returns:
        Parsed JSON data as a dict.
    """
    if not os.path.exists(baseline_file):
        print(f"Baseline file not found, generating {baseline_file}...")
        if dist.get_rank() == 0:
            baseline_main(output_file=baseline_file)
        dist.barrier()  # Wait for rank 0 to finish writing

    with open(baseline_file, "r") as f:
        return json.load(f)


evaluation_results: dict[uuid.UUID, Any] = {}
"""Global dict storing pipeline outputs keyed by request UUID. Populated by ``handle_output``."""

evaluation_timings: dict[uuid.UUID, dict | None] = {}
"""Global dict storing pipeline wall-clock timings keyed by request UUID."""


def handle_output(request_id: uuid.UUID, model_name: str, output: Any, timing: dict | None):
    """Callback passed to ``MainService`` to collect pipeline outputs.

    Stores the output and timing in the global dicts for later comparison.

    Args:
        request_id: UUID of the completed request.
        model_name: Name of the model that produced the output.
        output: The output tensor from the pipeline.
        timing: Wall-clock timing dict with ``"start"`` and ``"end"`` keys, or ``None``.
    """
    evaluation_results[request_id] = output
    evaluation_timings[request_id] = timing
    print(f"Received output from {model_name} from request: {request_id}. Output shape: {output.shape}")


def evaluation_main(baseline_file: str = DEFAULT_BASELINE_FILE):
    """Run evaluation: queue baseline inputs through the pipeline and compare outputs.

    Loads baseline data, registers models, queues work (rank 0 only), runs the
    pipeline across all ranks, and compares outputs against baseline on the last rank.

    Args:
        baseline_file: Path to the baseline JSON file.
    """
    # Load baseline data
    baseline_data = load_baseline(baseline_file)

    rank = dist.get_rank()
    last_rank = dist.get_world_size() - 1
    requests: dict[str, list[uuid.UUID]] = dict()

    # Init main
    print("Initialising main service...")
    main = MainService(handle_output, verbose=True)

    # Adding models (all ranks)
    for model_name, load_model, rand_input in evaluation_models:
        print(f"> Adding model {model_name} with load function {load_model.__name__}")
        main.add_model(model_name, load_model(), rand_input(),
                       optimizer_class=TimeBasedShishaPipelineOptimizer,
                       rebalance_interval=4, n_microbatches=5, async_optimization=False)

    # Queue work (rank 0 only)
    if rank == 0:
        for model_name, _, _ in evaluation_models:
            requests[model_name] = list()
            for entry in baseline_data[model_name]:
                x = torch.tensor(entry["input"])
                req_id = main.queue_work(model_name, x)
                requests[model_name].append(req_id)
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
    elif rank == last_rank: # Unless there is a single rank in the world
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

    # Checking the work
    print("\nComparing outputs (only on the last rank)...")
    if rank == last_rank:
        failed_requests = []
        for model_name in requests:
            for i, req_id in enumerate(requests[model_name]):
                pipeline_output = evaluation_results[req_id]
                baseline_output = torch.tensor(baseline_data[model_name][i]["output"])

                if pipeline_output.shape != baseline_output.shape:
                    print(f"  [{model_name}] Request {req_id}: FAIL (shape mismatch)")
                    print(
                        f"    Pipeline shape: {list(pipeline_output.shape)}, Baseline shape: {list(baseline_output.shape)}")
                    # print(f"    Pipeline output: {pipeline_output}")
                    # print(f"    Baseline output: {baseline_output}")
                    failed_requests.append((model_name, req_id))
                elif torch.allclose(pipeline_output, baseline_output, atol=1e-6):
                    print(f"  [{model_name}] Request {req_id}: PASS")
                else:
                    print(f"  [{model_name}] Request {req_id}: FAIL")
                    failed_requests.append((model_name, req_id))

        if not failed_requests:
            print("\nAll outputs match baseline!")
        else:
            print(f"\n{len(failed_requests)} request(s) differ from baseline.")

        # Timing comparison — group requests by pipeline batch
        print("\n" + "=" * 60)
        print("Timing Comparison (pipeline vs baseline)")
        print("=" * 60)
        faster_count = 0
        slower_count = 0
        total_diff = 0.0
        n_batches = 0

        for model_name in requests:
            print(f"\n  [{model_name}]")

            # Group consecutive requests into batches using the shared timing object
            batches: list[list[tuple[int, uuid.UUID]]] = []
            last_timing_id = None
            for i, req_id in enumerate(requests[model_name]):
                tid = id(evaluation_timings.get(req_id))
                if tid != last_timing_id:
                    batches.append([])
                    last_timing_id = tid
                batches[-1].append((i, req_id))

            # Track per-batch data for rebalance improvement analysis
            # Each entry: (pipeline_fwd_raw, did_rebalance)
            batch_records: list[tuple[float, bool]] = []

            for batch_idx, batch in enumerate(batches):
                first_req_id = batch[0][1]
                pipeline_timing = evaluation_timings.get(first_req_id)
                if pipeline_timing is None:
                    print(f"    Batch {batch_idx}: timing unavailable")
                    continue

                fwd = pipeline_timing["forward"]
                reb = pipeline_timing["rebalance"]
                pipeline_fwd_raw = fwd["end"] - fwd["start"]
                pipeline_reb = reb["end"] - reb["start"]
                did_rebalance = reb["did_rebalance"]
                rebalanced = " (rebalanced)" if did_rebalance else ""

                batch_size = pipeline_timing.get("batch_size", len(batch))
                n_microbatches = pipeline_timing.get("n_microbatches", batch_size)
                is_padded = batch_size < n_microbatches
                padding_label = f" (padded {batch_size}/{n_microbatches})" if is_padded else ""

                # Scale pipeline time to account for wasted padding work
                pipeline_fwd = pipeline_fwd_raw * (batch_size / n_microbatches) if is_padded else pipeline_fwd_raw

                batch_records.append((pipeline_fwd_raw, did_rebalance))

                print(f"    Batch {batch_idx}{padding_label}:")
                baseline_total = 0.0
                for i, req_id in batch:
                    baseline_entry = baseline_data[model_name][i]
                    baseline_timing = baseline_entry.get("timing")
                    if baseline_timing is None:
                        print(f"      Request {i}: baseline timing unavailable")
                        continue
                    bl = baseline_timing["end"] - baseline_timing["start"]
                    baseline_total += bl
                    print(f"      Request {i}: baseline={bl:.4f}s")

                diff = pipeline_fwd - baseline_total
                n_batches += 1
                total_diff += diff
                if diff < 0:
                    faster_count += 1
                else:
                    slower_count += 1

                fwd_label = f"pipeline_fwd={pipeline_fwd:.4f}s"
                if is_padded:
                    fwd_label += f" (raw={pipeline_fwd_raw:.4f}s)"
                print(f"      {fwd_label}, "
                      f"rebalance={pipeline_reb:.4f}s{rebalanced}, "
                      f"baseline_total={baseline_total:.4f}s, "
                      f"diff={diff:+.4f}s")

            # Rebalance improvement analysis
            # Group batches into pipeline segments: a new segment starts after a
            # batch that triggered a rebalance (that batch still ran on the OLD
            # pipeline, so it belongs to the previous segment).
            if batch_records:
                segments: list[list[float]] = [[]]
                for fwd_time, did_reb in batch_records:
                    segments[-1].append(fwd_time)
                    if did_reb:
                        segments.append([])
                # Drop trailing empty segment if the last batch rebalanced
                if not segments[-1]:
                    segments.pop()

                if len(segments) >= 2:
                    first_avg = sum(segments[0]) / len(segments[0])
                    last_avg = sum(segments[-1]) / len(segments[-1])
                    improvement = first_avg - last_avg
                    pct = (improvement / first_avg) * 100 if first_avg > 0 else 0

                    print(f"\n    Rebalance improvement ({len(segments)} pipeline configs):")
                    for seg_idx, seg in enumerate(segments):
                        seg_avg = sum(seg) / len(seg)
                        print(f"      Config {seg_idx}: avg={seg_avg:.4f}s "
                              f"({len(seg)} batch{'es' if len(seg) != 1 else ''})")
                    print(f"      Initial avg: {first_avg:.4f}s -> Final avg: {last_avg:.4f}s "
                          f"({improvement:+.4f}s, {pct:+.1f}%)")
                else:
                    print(f"\n    Rebalance improvement: no rebalance occurred")

        if n_batches > 0:
            avg_diff = total_diff / n_batches
            print(f"\n  Summary ({n_batches} batches): {faster_count} faster, "
                  f"{slower_count} slower, avg diff={avg_diff:+.4f}s")
        else:
            print("\n  No timing data available for comparison.")

    print(f"rank:{dist.get_rank()} exiting!")


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(device)
    dist.init_process_group(backend=backend)

    baseline = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE_FILE
    evaluation_main(baseline_file=baseline)

    dist.destroy_process_group()
