import pprint
from typing import Any

import torch
from torch import nn

from src.timed_module import TimedModule, make_module_timed
from tests.conv_next import ConvNext
from tests.simple_net import SimpleNet


class MainService:
    models: dict[str, nn.Module] = {}

    def __init__(self, depth=2):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.depth = depth

        # Initialise models
        # These are some testing models, in the final deliverable it should input models from the user
        self.models['simple-net'] = make_module_timed(SimpleNet(self.device), device=self.device, depth=depth)
        self.models['conv-next'] = make_module_timed(ConvNext(self.device), device=self.device, depth=depth)

    def run_model(self, model_name: str, x: Any, randomise_input=False):
        model = self.models.get(model_name, None)

        if model is None:
            print("MainService.run_model: provided model_name does not correspond to any known model!\n"
                  " provided model_name: ", model_name)
            return None

        if randomise_input or x is None:
            if callable(model.rand_inputs):
                x = model.rand_inputs()
            if x is None or not callable(model.rand_inputs):
                return {'error': 'Input was not provided, or the model does not define rand_inputs function!'}

        if x is None:
            print("MainService.run_model: provided input is None!")

        # TODO: How should I split this?
        # pipe = pipeline(
        #     module=model,
        #     mb_args=(x,),
        #     # split_spec={
        #     #     "layers.1": SplitPoint.BEGINNING,
        #     # }
        # )
        # print("pipe:", pipe)

        return model.run()

    def get_logs(self):
        #todo
        # return self.logger.to_dict()
        l = {}
        for (model_name, model) in self.models.items():
            if isinstance(model, TimedModule):
                l[model_name] = model.get_logs()
            else:
                l[model_name] = None
        return l

    def get_model_names(self):
        return self.models.keys()


def test_main_service():
    # Configuration
    N_RUNS = 5  # Number of times to run each model
    RESULT_FILE = "results.txt"

    # Initialise service
    main = MainService()

    print("=" * 80)
    print("PyTorch Model Load Balancer - Testing Suite")
    print("=" * 80)
    print(f"Device: {main.device}")
    print(f"Depth: {main.depth}")
    print(f"Available models: {list(main.get_model_names())}")
    print(f"Running each model {N_RUNS} times...")
    print("=" * 80)

    # Store all results
    all_results = {}

    # Run each model N times
    for model_name in main.get_model_names():
        print(f"\n{'=' * 80}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 80}")

        model_results = []

        for run_idx in range(N_RUNS):
            print(f"\nRun {run_idx + 1}/{N_RUNS}...")

            # Run the model
            result = main.run_model(model_name, None, randomise_input=True)

            # Get logs after this run
            logs = main.get_logs()

            # Store results
            model_results.append({
                'run': run_idx + 1,
                'result': result,
                'logs': logs[model_name] if model_name in logs else None
            })

            # Print summary for this run
            print(f"  Result: {result}")
            if logs.get(model_name):
                print(f"  Timing info available: Yes")

        all_results[model_name] = model_results
        print(f"\nCompleted {N_RUNS} runs for {model_name}")

    # Print summary statistics
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Total runs: {len(results)}")
        successful_runs = sum(1 for r in results if r['result'] is not None and 'error' not in r['result'])
        print(f"  Successful runs: {successful_runs}/{len(results)}")

    # Write detailed results to the file
    print(f"\n{'=' * 80}")
    print(f"Writing detailed results to {RESULT_FILE}...")
    print(f"{'=' * 80}\n")

    with open(RESULT_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch Model Load Balancer - Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Device: {main.device}\n")
        f.write(f"Depth: {main.depth}\n")
        f.write(f"Number of runs per model: {N_RUNS}\n\n")

        for model_name, results in all_results.items():
            f.write("=" * 80 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write("=" * 80 + "\n\n")

            for result_data in results:
                f.write(f"--- Run {result_data['run']} ---\n")
                f.write(f"Result: {result_data['result']}\n\n")

                if result_data['logs']:
                    f.write("Logs:\n")
                    f.write(pprint.pformat(result_data['logs'], width=100))
                    f.write("\n\n")
                else:
                    f.write("No logs available\n\n")

            f.write("\n")

    print(f"Results written to {RESULT_FILE}")
    print("Testing complete!")

if __name__ == '__main__':
    test_main_service()
