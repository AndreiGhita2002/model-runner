import torch
from torch import nn

from model_splitter import ModelSplitter


def test_model_splitter():
    # Create a simple test model
    test_model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, 10)
    )

    # Test layer-based splitting
    print("=" * 80)
    print("Testing Layer-Based Splitting")
    print("=" * 80)
    splitter = ModelSplitter(num_stages=3, distribution_strategy="layer_based")
    split_spec = splitter.create_split_spec(test_model)
    print(splitter.pretty_split_info_str(split_spec))
    print("\nSplit spec:", split_spec)

    # Test computation-based splitting with mock timing data
    print("\n" + "=" * 80)
    print("Testing Computation-Based Splitting")
    print("=" * 80)
    timing_profile = {
        '0': 10.0,  # Conv2d
        '1': 1.0,  # ReLU
        '2': 20.0,  # Conv2d
        '3': 1.0,  # ReLU
        '4': 30.0,  # Conv2d
        '5': 1.0,  # ReLU
        '6': 5.0,  # AdaptiveAvgPool2d
        '7': 0.5,  # Flatten
        '8': 15.0,  # Linear
    }

    splitter = ModelSplitter(num_stages=3, distribution_strategy="computation_based")
    split_spec = splitter.create_split_spec(test_model, timing_profile=timing_profile)
    print(splitter.pretty_split_info_str(split_spec))
    print("\nSplit spec:", split_spec)

    # Test device placement
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("\n" + "=" * 80)
        print("Testing Device Placement")
        print("=" * 80)
        devices = [torch.device(f"cuda:{i}") for i in range(min(3, torch.cuda.device_count()))]
        print(f"Available devices: {devices}")

        # Create a fresh model for device placement
        test_model_2 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        splitter = ModelSplitter(num_stages=len(devices), distribution_strategy="layer_based")
        split_spec = splitter.create_split_spec(test_model_2)
        test_model_2 = splitter.apply_split_to_devices(test_model_2, split_spec, devices)
        print("\nDevice placement complete!")