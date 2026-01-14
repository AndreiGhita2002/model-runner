import pprint
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

import torch
from torch import nn

from main import MainService, DeviceManager, MultiDeviceWrapper
from model_splitter import ModelSplitter
from timed_module import TimedModule
from tests.conv_next import ConvNext
from tests.simple_net import SimpleNet


class MockDeviceProperties:
    """Mock CUDA device properties."""
    def __init__(self, name: str, total_memory: int = 8_000_000_000):
        self.name = name
        self.total_memory = total_memory
        self.major = 8
        self.minor = 0


class MockDeviceManager:
    """
    Simulated DeviceManager that doesn't require real CUDA devices.
    Creates fake torch.device objects for testing.
    """
    def __init__(self, num_devices: int = 3):
        self.devices: List[torch.device] = []
        self._num_devices = num_devices
        self._initialize_devices()

    def _initialize_devices(self):
        print(f"[MockDeviceManager] Simulating {self._num_devices} CUDA device(s)")
        for i in range(self._num_devices):
            # Use CPU device as stand-in (simulated)
            device = torch.device('cpu')
            print(f"  Simulated Device {i}: MockGPU-{i}")
            self.devices.append(device)

    def get_device(self, index: int = 0) -> torch.device:
        if index >= len(self.devices):
            raise IndexError(f"Device index {index} out of range.")
        return self.devices[index]

    def get_all_devices(self) -> List[torch.device]:
        return self.devices.copy()

    def num_devices(self) -> int:
        return len(self.devices)

    def get_device_memory_info(self, device_index: int = 0) -> Dict[str, float]:
        return {
            'allocated': 0.0,
            'reserved': 0.0,
            'total': 8.0
        }


class MockTimedModule(TimedModule):
    """
    Mock TimedModule that returns configurable timing profiles.
    Used to simulate different model execution times for testing rebalancing.
    """
    def __init__(self, module: nn.Module, timing_profile: Dict[str, float]):
        # Don't call super().__init__ to avoid device issues
        self.inner = module
        self.device = 'cpu'
        self.depth = 1
        self.wrapping_a_wrapper = False
        self._timing_profile = timing_profile
        self._logs = None
        self._update_logs()

    def set_timing_profile(self, timing_profile: Dict[str, float]):
        """Update the timing profile (simulates changed execution times)."""
        self._timing_profile = timing_profile
        self._update_logs()

    def _update_logs(self):
        """Build logs structure from timing profile."""
        children = []
        for name, elapsed in self._timing_profile.items():
            children.append({
                'module_name': name,
                'times': {'elapsed': elapsed},
                'children': []
            })
        self._logs = {
            'module_name': self.inner._get_name(),
            'times': {'elapsed': sum(self._timing_profile.values())},
            'children': children
        }

    def get_logs(self):
        return self._logs

    def run(self, x=None):
        # Simulated run - just return input or random tensor
        if x is None:
            x = torch.randn(1, 10)
        return self.inner(x) if hasattr(self.inner, 'forward') else x

    def rand_inputs(self):
        return torch.randn(1, 10)


class SimpleSequentialNet(nn.Sequential):
    """
    Simple sequential model for testing splitting.
    Has 6 layers that can be distributed across 3 devices.
    """
    def __init__(self):
        super().__init__(
            nn.Linear(10, 32),   # layer_0
            nn.ReLU(),          # layer_1
            nn.Linear(32, 64),  # layer_2
            nn.ReLU(),          # layer_3
            nn.Linear(64, 32),  # layer_4
            nn.Linear(32, 10),  # layer_5
        )

    def rand_inputs(self):
        return torch.randn(1, 10)

    def _get_name(self):
        return "SimpleSequentialNet"


def initialize_test_models(main: MainService):
    """Initialize test models."""
    print("Initializing MainService...")
    print(f"Creating models on primary device: {main.primary_device}")

    simple_net = SimpleNet(str(main.primary_device))
    conv_next = ConvNext(str(main.primary_device))

    main.add_model('simple_net', simple_net)
    main.add_model('conv-next', conv_next)

    print(f"Initialized {len(main.models)} models")

def test_main_service():
    N_RUNS = 5
    RESULT_FILE = "results.txt" # TODO print to file

    main = MainService(
        depth=2,
        use_multi_device=True,
        split_strategy="computation_based",
        verbose=True,
    )
    initialize_test_models(main)

    main.print_status()

    # Queue work
    for i in range(N_RUNS):
        for j, model_name in enumerate(main.models.keys()):
            req = j + i * len(main.models)
            main.queue_work(model_name, None, req)

    # Run the Main Service
    print("Running...")
    main.run(exit_when_done=True)

    # Work queue should be empty
    assert main.work_queue.empty()
    print("All work done!")

    # Extract the responses
    for i in range(N_RUNS):
        for j, model_name in enumerate(main.models.keys()):
            req = j + i * len(main.models)
            res = main.get_work_results(req)
            if res is None:
                print(f"[req:{req} ERR] Work results for {model_name}, run {i} failed!")
            else:
                print(f"[req:{req} OK] Work results for {model_name}, run {i} succeeded!")

    # Print outputs:
    # pprint.pprint(main.model_outputs)


def test_rebalance_models_simulated():
    """
    Test MainService.rebalance_models with simulated devices and timing profiles.

    This test:
    1. Creates a MainService with a MockDeviceManager (3 simulated devices)
    2. Adds a model with a MockTimedModule that provides controllable timing
    3. Simulates model runs with changing timing profiles
    4. Verifies that rebalancing occurs when timing changes significantly
    5. Prints split specs whenever rebalancing happens

    Assumptions:
    - We simulate 3 CUDA devices using CPU tensors
    - Timing profiles are manually set to control rebalancing behavior
    - The test verifies the splitting algorithm distributes layers based on timing
    """
    print("\n" + "=" * 80)
    print("TEST: test_rebalance_models_simulated")
    print("=" * 80)

    # === Setup: Create MainService with mocked DeviceManager ===
    print("\n[Setup] Creating MainService with MockDeviceManager (3 devices)...")

    # Create mock device manager
    mock_device_manager = MockDeviceManager(num_devices=3)

    # Create MainService without initializing (we'll set up manually)
    # Patch DeviceManager to use our mock
    with patch.object(MainService, '__init__', lambda self, **kwargs: None):
        main = MainService()

    # Manually initialize the MainService with mock components
    main.device_manager = mock_device_manager
    main.primary_device = mock_device_manager.get_device(0)
    main.depth = 2
    main.use_multi_device = True
    main.split_strategy = "computation_based"
    main.verbose = True
    main.num_stages = mock_device_manager.num_devices()
    main.rebalance_threshold = 0.10
    main.rebalance_timings = {}
    main.timing_profiles = {}
    main.models = {}
    main.multi_device_models = {}
    main.model_outputs = {}

    main.splitter = ModelSplitter(
        num_stages=main.num_stages,
        distribution_strategy=main.split_strategy
    )

    # === Create test model with initial timing profile ===
    print("\n[Setup] Creating test model with initial timing profile...")

    test_model = SimpleSequentialNet()

    # Initial timing: layers have equal timing (100ms each)
    # Expected split with 3 devices: [0,1] -> device 0, [2,3] -> device 1, [4,5] -> device 2
    initial_timing = {
        '0': 100.0,  # Linear
        '1': 100.0,  # ReLU
        '2': 100.0,  # Linear
        '3': 100.0,  # ReLU
        '4': 100.0,  # Linear
        '5': 100.0,  # Linear
    }

    mock_timed_model = MockTimedModule(test_model, initial_timing)
    main.models['test-model'] = mock_timed_model

    print(f"  Model layers: {list(initial_timing.keys())}")
    print(f"  Initial timing (equal): {initial_timing}")

    # === Create initial multi-device wrapper ===
    print("\n[Setup] Creating initial multi-device wrapper...")

    initial_split_spec = main.splitter.create_split_spec(
        test_model,
        timing_profile=initial_timing
    )
    print(f"  Initial split spec: {initial_split_spec}")

    wrapper = MultiDeviceWrapper(
        test_model,
        initial_split_spec,
        mock_device_manager.get_all_devices()
    )
    main.multi_device_models['test-model'] = wrapper
    main.timing_profiles['test-model'] = initial_timing

    # === Test 1: No rebalancing when timing unchanged ===
    print("\n" + "-" * 40)
    print("[Test 1] Timing unchanged - should NOT rebalance")
    print("-" * 40)

    old_split = wrapper.split_spec.copy()
    main.rebalance_models()

    if wrapper.split_spec == old_split:
        print("  PASS: Split spec unchanged (as expected)")
    else:
        print("  FAIL: Split spec changed unexpectedly!")
        print(f"    Old: {old_split}")
        print(f"    New: {wrapper.split_spec}")

    # === Test 2: Rebalancing when timing changes significantly ===
    print("\n" + "-" * 40)
    print("[Test 2] Timing changed significantly - SHOULD rebalance")
    print("-" * 40)

    # New timing: first layers are much slower (simulates workload change)
    # This should shift more layers to later devices
    changed_timing = {
        '0': 300.0,  # Linear - now 3x slower
        '1': 200.0,  # ReLU - now 2x slower
        '2': 50.0,   # Linear - faster
        '3': 50.0,   # ReLU - faster
        '4': 50.0,   # Linear - faster
        '5': 50.0,   # Linear - faster
    }

    print(f"  Changed timing (unbalanced): {changed_timing}")
    mock_timed_model.set_timing_profile(changed_timing)

    old_split = wrapper.split_spec.copy()
    print(f"  Split spec BEFORE rebalance: {old_split}")

    main.rebalance_models()

    print(f"  Split spec AFTER rebalance: {wrapper.split_spec}")

    if wrapper.split_spec != old_split:
        print("  PASS: Split spec changed (as expected)")
        # Verify the new split makes sense
        stage_times = {0: 0, 1: 0, 2: 0}
        for layer, stage in wrapper.split_spec.items():
            stage_times[stage] += changed_timing.get(layer, 0)
        print(f"  New stage times: {stage_times}")
    else:
        print("  FAIL: Split spec should have changed!")

    # === Test 3: Rebalancing again with different profile ===
    print("\n" + "-" * 40)
    print("[Test 3] Another timing change - verify continued rebalancing")
    print("-" * 40)

    # New timing: middle layers are slowest
    another_timing = {
        '0': 50.0,
        '1': 50.0,
        '2': 400.0,  # Much slower
        '3': 300.0,  # Much slower
        '4': 50.0,
        '5': 50.0,
    }

    print(f"  New timing (middle-heavy): {another_timing}")
    mock_timed_model.set_timing_profile(another_timing)

    old_split = wrapper.split_spec.copy()
    print(f"  Split spec BEFORE rebalance: {old_split}")

    main.rebalance_models()

    print(f"  Split spec AFTER rebalance: {wrapper.split_spec}")

    if wrapper.split_spec != old_split:
        print("  PASS: Split spec changed (as expected)")
        stage_times = {0: 0, 1: 0, 2: 0}
        for layer, stage in wrapper.split_spec.items():
            stage_times[stage] += another_timing.get(layer, 0)
        print(f"  New stage times: {stage_times}")
    else:
        print("  Note: Split spec unchanged (may be optimal already)")

    # === Print timing statistics ===
    print("\n" + "-" * 40)
    print("[Stats] Rebalancing timing data")
    print("-" * 40)

    if 'test-model' in main.rebalance_timings:
        timings = main.rebalance_timings['test-model']
        print(f"  Total rebalances: {len(timings['total'])}")
        for key in ['total', 'split_spec', 'apply_split']:
            if timings[key]:
                avg_ns = sum(timings[key]) / len(timings[key])
                print(f"  Avg {key} time: {avg_ns / 1e6:.3f}ms")
    else:
        print("  No rebalancing timing data recorded")

    print("\n" + "=" * 80)
    print("TEST COMPLETE: test_rebalance_models_simulated")
    print("=" * 80)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--simulated':
        test_rebalance_models_simulated()
    else:
        test_main_service()