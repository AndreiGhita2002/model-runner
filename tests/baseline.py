import json
import sys
import torch
from tests.testing_models import evaluation_models

DEFAULT_BASELINE_FILE = "baseline_outputs.json"


def baseline_main(num_requests: int = 5, seed: int = 37, output_file: str = DEFAULT_BASELINE_FILE):
    results = {}

    for model_name, load_model, rand_inputs in evaluation_models:
        print(f"Running baseline for model: {model_name}")

        # Load model in eval mode
        model = load_model()
        model.eval()

        results[model_name] = []

        for i in range(num_requests):
            # Set seed for reproducible inputs
            torch.manual_seed(seed + i)
            x = rand_inputs()

            # Run forward pass
            with torch.no_grad():
                output = model(x)

            # Store input/output pair
            results[model_name].append({
                "seed": seed + i,
                "input": x.tolist(),
                "output": output.tolist()
            })
            print(f"  > Request {i}: input shape {list(x.shape)}, output shape {list(output.shape)}")

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline outputs saved to {output_file}")
    return results


if __name__ == '__main__':
    output_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE_FILE
    baseline_main(output_file=output_file)
