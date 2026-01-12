import pprint
import queue
import time
from typing import Any, List, Dict, Optional

import torch
from torch import nn
from torch.multiprocessing.queue import Queue

from model_splitter import ModelSplitter, extract_timing_profile_from_logs
from timed_module import TimedModule, make_module_timed
from tests.conv_next import ConvNext
from tests.simple_net import SimpleNet


class DeviceManager:
    """Manages available CUDA devices for model distribution."""

    def __init__(self):
        self.devices: List[torch.device] = []
        self._initialize_devices()

    def _initialize_devices(self):
        """Detect and initialize all available CUDA devices."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This load balancer requires CUDA devices.")

        device_num = torch.cuda.device_count()
        print(f"Detected {device_num} CUDA device(s)")

        for i in range(device_num):
            device = torch.device(f"cuda:{i}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
            self.devices.append(device)

    def get_device(self, index: int = 0) -> torch.device:
        """Get device by index."""
        if index >= len(self.devices):
            raise IndexError(f"Device index {index} out of range. Only {len(self.devices)} devices available.")
        return self.devices[index]

    def get_all_devices(self) -> List[torch.device]:
        """Get all available devices."""
        return self.devices.copy()

    def num_devices(self) -> int:
        """Return a number of available devices."""
        return len(self.devices)

    def get_device_memory_info(self, device_index: int = 0) -> Dict[str, float]:
        """Get memory information for a specific device."""
        torch.cuda.set_device(device_index)
        return {
            'allocated': torch.cuda.memory_allocated(device_index) / 1e9,
            'reserved': torch.cuda.memory_reserved(device_index) / 1e9,
            'total': torch.cuda.get_device_properties(device_index).total_memory / 1e9
        }


class MultiDeviceWrapper(nn.Module):
    """
    Simple wrapper for models split across multiple devices.
    Handles forward passes with automatic device transfers.
    """

    def __init__(self, model: nn.Module, split_spec: Dict[str, int], devices: List[torch.device]):
        super().__init__()
        self.model = model
        self.split_spec = split_spec
        self.devices = devices
        self.layer_devices = self.build_layer_device_map()

    def build_layer_device_map(self) -> Dict[str, torch.device]:
        """Build mapping of layer names to devices."""
        layer_devices = {}
        for name, stage_idx in self.split_spec.items():
            layer_devices[name] = self.devices[stage_idx]
        return layer_devices

    def forward(self, x):
        """Forward pass with automatic device transfers."""
        current_device = x.device

        # TODO something here looks fishy

        # For Sequential models, iterate through children
        if isinstance(self.model, nn.Sequential):
            for name, module in self.model.named_children():
                target_device = self.layer_devices.get(name, current_device)
                if x.device != target_device:
                    x = x.to(target_device)
                    current_device = target_device
                x = module(x)
        else:
            x = self.model(x)

        return x

    def rand_inputs(self):
        """Pass through to the inner model's rand_inputs if available."""
        if hasattr(self.model, 'rand_inputs'):
            return self.model.rand_inputs()
        return None


class MainService:
    """
    Main service for load-balanced model inference across multiple GPUs.
    Uses PyTorch 2.x built-in pipelining (torch.distributed.pipelining) when available.
    """

    models: Dict[str, nn.Module] = {}
    multi_device_models: Dict[str, MultiDeviceWrapper] = {}
    # Work Queue: (request ID, model name, input data): Tuple[int, str, Any]
    work_queue: Queue = queue.Queue()
    # Model Outputs: request ID -> output # TODO: make sure this Dict is multi thread safe
    model_outputs: Dict[int, Any] = {}
    # Threshold equal to the minimum change in a models performance that triggers rebalancing
    rebalance_threshold = 0.10

    def __init__(self, depth=2, use_multi_device=True, split_strategy="computation_based"):
        """
        Args:
            depth: Depth for TimedModule profiling
            use_multi_device: Whether to use multi-device splitting
            split_strategy: Strategy for splitting models
        """
        self.device_manager = DeviceManager()
        self.primary_device = self.device_manager.get_device(0)

        self.depth = depth
        self.use_multi_device = use_multi_device
        self.split_strategy = split_strategy

        self.num_stages = self.device_manager.num_devices()
        self.splitter = ModelSplitter(
            num_stages=self.num_stages,
            distribution_strategy=split_strategy
        )

        self.timing_profiles: Dict[str, Dict[str, float]] = {}

        print("Initializing MainService...")
        self._initialize_test_models()

    def _initialize_test_models(self):
        """Initialize test models."""
        print(f"Creating models on primary device: {self.primary_device}")

        simple_net = SimpleNet(str(self.primary_device))
        conv_next = ConvNext(str(self.primary_device))

        self.models['simple-net'] = make_module_timed(
            simple_net,
            device=str(self.primary_device),
            depth=self.depth
        )
        self.models['conv-next'] = make_module_timed(
            conv_next,
            device=str(self.primary_device),
            depth=self.depth
        )

        print(f"Initialized {len(self.models)} models")

    def run(self, exit_when_done = False):
        # main loop
        while True:
            # check queue
            if not self.work_queue.empty():
                #run models
                (req_id, model_name, work) = self.work_queue.get(block=True)

                output = self.run_model(model_name=model_name, x=work)

                # output the output
                self.model_outputs[req_id] = output

                # rebalance the models
                self.rebalance_models()
            else:
                #TODO: make MainService.run more like a service
                # sleep for a bit
                # until the user requests another workload
                # or perhaps exit

                # For now, we exit when the queue is done
                if exit_when_done:
                    return

    def rebalance_models(self):
        """
        Rebalances the models in regard to timing data.

        This method collects timing information from the most recent model runs
        and re-distributes model layers across devices if the timing profile
        has changed significantly.

        Assumptions:
        - Models are wrapped in TimedModule and provide timing logs via get_logs()
        - Only CUDA devices are supported for multi-device distribution
        - Timing information from the most recent run is representative of future runs
        - Rebalancing is only performed for models that already have multi-device wrappers
        - A threshold of 10% change in timing distribution triggers rebalancing
        """
        #TODO record how long rebalancing takes

        if self.num_stages < 2:
            # No rebalancing needed with only one device
            return

        for model_name, model in self.models.items():
            # Skip models that don't have multi-device wrappers yet
            if model_name not in self.multi_device_models:
                continue

            # Extract timing information from the model's most recent run
            if not isinstance(model, TimedModule):
                continue

            logs = model.get_logs()
            if logs is None:
                continue

            new_profile = extract_timing_profile_from_logs(logs)
            if not new_profile:
                continue

            # Check if the timing profile has changed significantly
            old_profile = self.timing_profiles.get(model_name, {})
            if not self._should_rebalance(old_profile, new_profile):
                continue

            # Update the stored timing profile
            self.timing_profiles[model_name] = new_profile

            # Get the inner model from the existing wrapper
            wrapper = self.multi_device_models[model_name]
            inner_model = wrapper.model

            # Create a new split specification based on updated timing
            new_split_spec = self.splitter.create_split_spec(
                inner_model,
                timing_profile=new_profile
            )

            # Check if the split specification actually changed
            if new_split_spec == wrapper.split_spec:
                continue

            print(f"Rebalancing model: {model_name}")
            print(f"  Old split: {wrapper.split_spec}")
            print(f"  New split: {new_split_spec}")

            # Apply the new split to devices
            devices = self.device_manager.get_all_devices()
            inner_model = self.splitter.apply_split_to_devices(
                inner_model, new_split_spec, devices
            )

            # Update the wrapper with new split specification
            wrapper.split_spec = new_split_spec
            wrapper.layer_devices = wrapper.build_layer_device_map()

    def _should_rebalance(self, old_profile: Dict[str, float], new_profile: Dict[str, float]) -> bool:
        """
        Determine if rebalancing is needed based on timing profile changes.

        Returns True if timing distribution has changed by more than the threshold.

        Assumptions:
        - A 10% relative change in total layer time distribution warrants rebalancing
        - If no old profile exists, rebalancing should occur
        - Only layers present in both profiles are compared
        """
        threshold = self.rebalance_threshold

        if not old_profile:
            return True

        # Find common layers
        common_layers = set(old_profile.keys()) & set(new_profile.keys())
        if not common_layers:
            return True

        # Calculate total time for normalisation
        old_total = sum(old_profile.get(layer, 0) for layer in common_layers)
        new_total = sum(new_profile.get(layer, 0) for layer in common_layers)

        if old_total == 0 or new_total == 0:
            return True

        # Compare normalised timing distributions
        max_change = 0.0

        for layer in common_layers:
            old_ratio = old_profile.get(layer, 0) / old_total
            new_ratio = new_profile.get(layer, 0) / new_total
            change = abs(new_ratio - old_ratio)
            max_change = max(max_change, change)

        return max_change > threshold

    def profile_model(self, model_name: str, num_warmup: int = 2, num_profile: int = 5) -> Dict[str, float]:
        """
        Profile a model with junk data to get timing information for each layer.
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")

        print(f"\nProfiling model: {model_name}")
        print(f"  Warmup runs: {num_warmup}")
        print(f"  Profile runs: {num_profile}")

        # Warmup runs
        for i in range(num_warmup):
            x = model.rand_inputs()
            if x is not None:
                model.run(x)

        # Profile runs
        accumulated_profile = {}
        for i in range(num_profile):
            x = model.rand_inputs()
            if x is not None:
                model.run(x)
                logs = model.get_logs()
                profile = extract_timing_profile_from_logs(logs)
                for name, elapsed_time in profile.items():
                    accumulated_profile[name] = accumulated_profile.get(name, 0.0) + elapsed_time

        # Average the timings
        avg_profile = {
            name: elapsed_time / num_profile
            for name, elapsed_time in accumulated_profile.items()
        }

        self.timing_profiles[model_name] = avg_profile
        print(f"  Profiling complete. Found {len(avg_profile)} timed layers")
        return avg_profile

    def create_multi_device_model(self, model_name: str, force_reprofile: bool = False) -> Optional[MultiDeviceWrapper]:
        """
        Create a multi-device version of the model.
        Uses torch.distributed.pipelining if available, otherwise manual placement.
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")

        if self.num_stages < 2:
            print(f"Only {self.num_stages} device(s) available. Multi-device requires at least 2.")
            return None

        print(f"\nCreating multi-device model: {model_name}")

        # Get or create a timing profile
        if force_reprofile or model_name not in self.timing_profiles:
            timing_profile = self.profile_model(model_name)
        else:
            timing_profile = self.timing_profiles[model_name]
            print(f"  Using cached timing profile")

        # Get the inner model
        inner_model = model.inner if isinstance(model, TimedModule) else model

        # Create a split specification
        split_spec = self.splitter.create_split_spec(
            inner_model,
            timing_profile=timing_profile
        )

        print(f"  Split specification:")
        print(self.splitter.pretty_split_info_str(split_spec))

        try:
            devices = self.device_manager.get_all_devices()

            # Apply split to devices (manual placement always works)
            inner_model = self.splitter.apply_split_to_devices(inner_model, split_spec, devices)

            # Create wrapper
            wrapper = MultiDeviceWrapper(inner_model, split_spec, devices)

            self.multi_device_models[model_name] = wrapper
            print(f"  Multi-device model created successfully")

            return wrapper

        except Exception as e:
            print(f"  ERROR creating multi-device model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_model(self,
                  model_name: str,
                  x: Any = None,
                  randomise_input: bool = False,
                  use_multi_device: Optional[bool] = None):
        """
        Run a model with optional multi-device execution.
        """
        use_md = self.use_multi_device if use_multi_device is None else use_multi_device

        model = self.models.get(model_name, None)
        if model is None:
            print(f"MainService.run_model: model '{model_name}' not found!")
            return None

        if randomise_input or x is None:
            if callable(model.rand_inputs):
                x = model.rand_inputs()
            if x is None:
                return {'error': 'Input was not provided, or the model does not define rand_inputs function!'}

        # Run with multi-device if requested
        if use_md and self.num_stages >= 2:
            if model_name not in self.multi_device_models:
                self.create_multi_device_model(model_name)

            if model_name in self.multi_device_models:
                md_model = self.multi_device_models[model_name]
                try:
                    with torch.no_grad():
                        # TODO: why does this only send x to the first device?
                        #  surely it should be a device of the model
                        #  MultiDeviceWrapper.forward seems to move x to the correct device
                        first_device = self.device_manager.get_device(0)
                        # send data to the first device
                        if x.device != first_device:
                            x = x.to(first_device)
                        output = md_model(x)
                    return output

                except Exception as e:
                    print(f"Multi-device execution failed: {e}")
                    print("Falling back to standard execution")
                    import traceback
                    traceback.print_exc()

        # Run on a single device
        return model.run(x)

    def get_logs(self) -> Dict[str, Any]:
        """Get timing logs from all models."""
        logs = {}
        for model_name, model in self.models.items():
            if isinstance(model, TimedModule):
                logs[model_name] = model.get_logs()
            else:
                logs[model_name] = None
        return logs

    def get_model_names(self) -> List[str]:
        """Get a list of available model names."""
        return list(self.models.keys())

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices."""
        info = {
            'num_devices': self.device_manager.num_devices(),
            'devices': []
        }

        for i, device in enumerate(self.device_manager.get_all_devices()):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_properties(i).name,
                'memory': self.device_manager.get_device_memory_info(i)
            }
            info['devices'].append(device_info)

        return info

    def print_status(self):
        """Print current service status."""
        print("\n" + "=" * 80)
        print("MainService Status")
        print("=" * 80)
        print(f"Number of models: {len(self.models)}")
        print(f"Number of multi-device models: {len(self.multi_device_models)}")
        print(f"Multi-device mode: {'enabled' if self.use_multi_device else 'disabled'}")
        print(f"Split strategy: {self.split_strategy}")

        device_info = self.get_device_info()
        print(f"\nDevices ({device_info['num_devices']}):")
        for dev in device_info['devices']:
            print(f"  [{dev['index']}] {dev['name']}")
            mem = dev['memory']
            print(f"      Memory: {mem['allocated']:.2f}/{mem['total']:.2f} GB allocated")

        print("\nModels:")
        for name in self.get_model_names():
            has_profile = name in self.timing_profiles
            has_multi_device = name in self.multi_device_models
            print(f"  {name}: profiled={has_profile}, multi_device={has_multi_device}")

#TODO this should be in test_main.py
# it was temporarily moved here
def test_main_service():
    """Test function for MainService."""
    N_RUNS = 5
    RESULT_FILE = "results.txt"

    main = MainService(
        depth=2,
        use_multi_device=True,
        split_strategy="computation_based"
    )

    main.print_status()

    print("\n" + "=" * 80)
    print("PyTorch Model Load Balancer - Testing Suite")
    print("=" * 80)
    print(f"Running each model {N_RUNS} times...")

    all_results = {}

    # Profile and create multi-device models
    print("\n" + "=" * 80)
    print("Phase 1: Model Profiling")
    print("=" * 80)
    for model_name in main.get_model_names():
        main.profile_model(model_name, num_warmup=2, num_profile=3)
        if main.num_stages >= 2:
            main.create_multi_device_model(model_name)

    # Run each model N times
    print("\n" + "=" * 80)
    print("Phase 2: Model Execution")
    print("=" * 80)

    for model_name in main.get_model_names():
        print(f"\n{'=' * 80}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 80}")

        model_results = []

        for run_idx in range(N_RUNS):
            print(f"\nRun {run_idx + 1}/{N_RUNS}...")

            start_time = time.time()
            result = main.run_model(model_name, None, randomise_input=True)
            elapsed = time.time() - start_time

            logs = main.get_logs()

            model_results.append({
                'run': run_idx + 1,
                'result': result,
                'elapsed_time': elapsed,
                'logs': logs[model_name] if model_name in logs else None
            })

            print(f"  Result type: {type(result)}")
            print(f"  Elapsed time: {elapsed*1000:.2f}ms")
            if logs.get(model_name):
                print(f"  Timing info available: Yes")

        all_results[model_name] = model_results
        avg_time = sum(r['elapsed_time'] for r in model_results) / len(model_results)
        print(f"\nCompleted {N_RUNS} runs for {model_name}")
        print(f"Average execution time: {avg_time*1000:.2f}ms")

    # Print summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Total runs: {len(results)}")
        successful_runs = sum(1 for r in results if r['result'] is not None and not isinstance(r['result'], dict))
        print(f"  Successful runs: {successful_runs}/{len(results)}")
        avg_time = sum(r['elapsed_time'] for r in results) / len(results)
        print(f"  Average time: {avg_time*1000:.2f}ms")

    main.print_status()

    # Write results
    print(f"\n{'=' * 80}")
    print(f"Writing detailed results to {RESULT_FILE}...")
    print(f"{'=' * 80}\n")

    with open(RESULT_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch Model Load Balancer - Test Results\n")
        f.write("=" * 80 + "\n\n")

        device_info = main.get_device_info()
        f.write(f"Number of devices: {device_info['num_devices']}\n")
        for dev in device_info['devices']:
            f.write(f"  Device {dev['index']}: {dev['name']}\n")
        f.write(f"\nDepth: {main.depth}\n")
        f.write(f"Number of runs per model: {N_RUNS}\n")
        f.write(f"Multi-device mode: {main.use_multi_device}\n")
        f.write(f"Split strategy: {main.split_strategy}\n\n")

        for model_name, results in all_results.items():
            f.write("=" * 80 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write("=" * 80 + "\n\n")

            if model_name in main.timing_profiles:
                f.write("Timing Profile:\n")
                f.write(pprint.pformat(main.timing_profiles[model_name], width=100))
                f.write("\n\n")

            times = [r['elapsed_time'] for r in results]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            f.write(f"Execution Statistics:\n")
            f.write(f"  Average: {avg_time*1000:.2f}ms\n")
            f.write(f"  Min: {min_time*1000:.2f}ms\n")
            f.write(f"  Max: {max_time*1000:.2f}ms\n\n")

            for result_data in results:
                f.write(f"--- Run {result_data['run']} ---\n")
                f.write(f"Result type: {type(result_data['result'])}\n")
                f.write(f"Elapsed time: {result_data['elapsed_time']*1000:.2f}ms\n\n")

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