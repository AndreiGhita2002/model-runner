import json
import os
import sys
from typing import Any

import torch
import torch.distributed as dist

from model_runner import MainService
from tests.testing_models import evaluation_models
from tests.baseline import baseline_main, DEFAULT_BASELINE_FILE


def load_baseline(baseline_file: str):
    if not os.path.exists(baseline_file):
        print(f"Baseline file not found, generating {baseline_file}...")
        baseline_main(output_file=baseline_file)

    with open(baseline_file, "r") as f:
        return json.load(f)


evaluation_results: dict[int, Any] = {}


def handle_output(request_id: int, model_name: str, output: Any):
    evaluation_results[request_id] = output
    print(f"Received output from {model_name} from request: {request_id}. Output shape: {output.shape}")


def evaluation_main(baseline_file: str = DEFAULT_BASELINE_FILE):
    # Load baseline data
    baseline_data = load_baseline(baseline_file)

    # Constants:
    requests: dict[str, list[int]] = dict()

    # Init main
    print("Initialising main service...")
    main = MainService(handle_output, verbose=True)

    # Adding models
    for model_name, load_model, rand_input in evaluation_models:
        print(f"> Adding model {model_name} with load function {load_model.__name__}")
        main.add_model(model_name, load_model(), rand_input(), model_output_is_static=True)

        # Adding work from baseline inputs
        requests[model_name] = list()
        for entry in baseline_data[model_name]:
            x = torch.tensor(entry["input"])
            req_id = main.queue_work(model_name, x)
            requests[model_name].append(req_id)
            print(f" > Work added with request id: {req_id}")

    # Running the main service:
    main.run(exit_when_done=True)

    # Checking the work
    print("\nComparing outputs (only on the last rank)...")
    if dist.get_rank() == dist.get_world_size() - 1:
        failed_requests = []
        for model_name in requests:
            for i, req_id in enumerate(requests[model_name]):
                pipeline_output = evaluation_results[req_id]
                baseline_output = torch.tensor(baseline_data[model_name][i]["output"])

                if pipeline_output.shape != baseline_output.shape:
                    print(f"  [{model_name}] Request {req_id}: FAIL (shape mismatch)")
                    print(
                        f"    Pipeline shape: {list(pipeline_output.shape)}, Baseline shape: {list(baseline_output.shape)}")
                    print(f"    Pipeline output: {pipeline_output}")
                    print(f"    Baseline output: {baseline_output}")
                    failed_requests.append((model_name, req_id))
                elif torch.allclose(pipeline_output, baseline_output, atol=1e-6):
                    print(f"  [{model_name}] Request {req_id}: PASS")
                else:
                    print(f"  [{model_name}] Request {req_id}: FAIL")
                    failed_requests.append((model_name, req_id))

        if not failed_requests:
            print("\nAll outputs match baseline!")
        else:
            print(f"\n{len(failed_requests)} request(s) differ from baseline:")
            for model_name, req_id in failed_requests:
                print(f"  - {model_name}: request {req_id}")

    print(f"rank:{dist.get_rank()} exiting!")


if __name__ == '__main__':
    device = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(device)
    dist.init_process_group(backend=backend)

    baseline = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE_FILE
    evaluation_main(baseline_file=baseline)

    dist.destroy_process_group()
