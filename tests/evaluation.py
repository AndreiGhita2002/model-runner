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
    #TODO: compare pipeline outputs with baseline_data[model_name][i]["output"]

    #TODO: finish evaluation


if __name__ == '__main__':
    evaluation_main()
