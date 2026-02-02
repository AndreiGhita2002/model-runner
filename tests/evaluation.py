import json
import os
import torch

from model_runner import MainService
from tests.testing_models import evaluation_models
from tests.baseline import baseline_main

BASELINE_FILE = "baseline_outputs.json"


def load_baseline():
    if not os.path.exists(BASELINE_FILE):
        print(f"Baseline file not found, generating {BASELINE_FILE}...")
        baseline_main(output_file=BASELINE_FILE)

    with open(BASELINE_FILE, "r") as f:
        return json.load(f)


def evaluation_main():
    # Load baseline data
    baseline_data = load_baseline()

    # Constants:
    requests: dict[str, list[int]] = dict()

    # Init main
    print("Initialising main service...")
    main = MainService(verbose=True)

    # Adding models
    for model_name, load_model, _ in evaluation_models:
        print(f"> Adding model {model_name} with load function {load_model.__name__}")
        main.add_model(model_name, load_model())

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
    print("\nComparing outputs...")
    failed_requests = []
    for model_name in requests:
        for i, req_id in enumerate(requests[model_name]):
            pipeline_output = main.get_work_results(req_id)
            baseline_output = torch.tensor(baseline_data[model_name][i]["output"])

            if pipeline_output.shape != baseline_output.shape:
                print(f"  [{model_name}] Request {req_id}: FAIL (shape mismatch)")
                print(f"    Pipeline shape: {list(pipeline_output.shape)}, Baseline shape: {list(baseline_output.shape)}")
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


if __name__ == '__main__':
    evaluation_main()
