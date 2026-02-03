import queue
import threading
from typing import Any, List, Dict

import torch
from torch import nn
import torch.distributed as dist

from .adaptive_pipeline import AdaptivePipeline
from .device_manager import DeviceManager
from .timed_module import make_module_timed


class MainService:
    """
    Main service that manages adaptive pipelines.
    """

    #TODO: find a more appropriate name for this
    # maybe AdaptivePipelineRunner? PipelineRuntime?

    pipelines: dict[str, AdaptivePipeline] = {}
    # Work queues per model: model_name -> Queue of (request_id, input_data)
    work_by_model: Dict[str, queue.Queue] = {}
    next_req_id: int = 0 #TODO: switch to uuid
    # Model Outputs: request ID -> output (protected by _outputs_lock)
    model_outputs: dict[int, Any] = {}
    _outputs_lock: threading.Lock = threading.Lock()
    # Verbose logging flag
    verbose: bool = False

    def __init__(self, default_timing_depth: int = 3, verbose=False):
        """
        Args:
            default_timing_depth: Depth for TimedModule profiling
            verbose: Enable verbose logging output
        """
        self.default_timing_depth = default_timing_depth
        self.verbose = verbose

        self.device_manager = DeviceManager(verbose=verbose)
        self.primary_device = self.device_manager.get_device(0)

    def _log(self, msg: str):
        """Print message if verbose logging is enabled."""
        if self.verbose:
            print(msg)

    def add_model(self, model_name: str, model: nn.Module, example_input: Any, device=None, depth: int | None = None, **kwargs):
        """
        Add a model to the service.

        Args:
            model_name: Unique name for the model
            model: The PyTorch model to add
            example_input: An example input for the model
            device: Device to run the model on (default: primary device)
            depth: Depth for TimedModule profiling (default: self.default_timing_depth)
            **kwargs: Additional arguments passed to AdaptivePipeline:
                - pipeline_optimizer: Custom optimiser (default: GreedyPipelineOptimizer)
                - rebalance_interval: How often to check for rebalancing (default: 10)
                - rebalance_threshold: Minimum change to trigger rebalancing (default: 0.1)
                - n_microbatches: Number of microbatches for pipeline (default: 4)
                - initial_pipeline_config: Initial pipeline configuration
                - async_optimization: Use async optimisation (default: False)

        """
        if device is None:
            device = str(self.primary_device)

        depth = depth or self.default_timing_depth

        if self.pipelines.get(model_name, None) is not None:
            raise Exception(f"Pipeline with name {model_name} already exists!")

        timed_model = make_module_timed(
            model,
            device=device,
            depth=depth
        )

        # Pass verbose from self if not explicitly provided in kwargs
        if 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose

        self.pipelines[model_name] = AdaptivePipeline(
            timed_model,
            model_name,
            self.device_manager,
            example_input,
            **kwargs
        )
        self.work_by_model[model_name] = queue.Queue()

    def queue_work(self, model_name: str, x: Any, request_id: int | None = None) -> int:
        if model_name not in self.work_by_model:
            raise ValueError(f"Model '{model_name}' not found. Add it with add_model() first.")
        if request_id is None:
            request_id = self.next_req_id
            self.next_req_id += 1
        self.work_by_model[model_name].put((request_id, x))
        return request_id

    def get_work_results(self, request_id: int) -> Any | None:
        with self._outputs_lock:
            return self.model_outputs.get(request_id, None)

    def run(self, exit_when_done = False):
        self._log("MainService.run: starting main loop")
        rank = dist.get_rank() if dist.is_initialized() else 0

        # main loop
        while True:
            did_work = False

            # Process each model's work in microbatches
            for model_name, model_queue in self.work_by_model.items():
                pipeline = self.pipelines[model_name]
                n_microbatches = pipeline.n_microbatches

                # Only rank 0 checks the queue and prepares inputs
                if rank == 0:
                    has_work = torch.tensor([1 if not model_queue.empty() else 0], dtype=torch.int)
                else:
                    has_work = torch.tensor([0], dtype=torch.int)

                # Broadcast whether there's work from rank 0 to all ranks
                # This synchronises all ranks
                dist.broadcast(has_work, src=0)

                if has_work.item() == 0:
                    continue

                did_work = True
                req_ids = []
                batched_input = None

                if rank == 0:
                    # Collect up to n_microbatches items from this model's queue
                    work_items = []
                    while len(work_items) < n_microbatches and not model_queue.empty():
                        work_items.append(model_queue.get(block=False))

                    req_ids = [item[0] for item in work_items]
                    inputs = [item[1] for item in work_items]

                    # Pad batch if needed (scheduler expects exactly n_microbatches)
                    while len(inputs) < n_microbatches:
                        inputs.append(inputs[-1])  # Duplicate last input as padding

                    self._log(f"MainService.run: processing {len(work_items)} requests for model '{model_name}' (microbatch size: {n_microbatches})")

                    # Stack inputs into a batch tensor
                    batched_input = torch.stack(inputs)

                # All ranks must call forward together
                outputs = pipeline.forward(batched_input)

                # Only rank 0 stores outputs
                if rank == 0:
                    with self._outputs_lock:
                        for j, req_id in enumerate(req_ids):
                            self.model_outputs[req_id] = outputs[j]
                            self._log(f"MainService.run: completed request {req_id}")

            #TODO: make MainService.run more like a service
            # sleep for a bit
            # until the user requests another workload
            # or perhaps exit

            # For now, we exit when the queue is done
            if not did_work and exit_when_done:
                self._log("MainService.run: queue empty, exiting")
                return

    def get_logs(self) -> Dict[str, Any]:
        """Get timing logs from all pipelines."""
        logs = {}
        for model_name, pipeline in self.pipelines.items():
            logs[model_name] = pipeline.time_logs
        return logs

    def get_model_names(self) -> List[str]:
        """Get a list of available model names."""
        return list(self.pipelines.keys())

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
        print(f"Number of pipelines: {len(self.pipelines)}")

        device_info = self.get_device_info()
        print(f"\nDevices ({device_info['num_devices']}):")
        for dev in device_info['devices']:
            print(f"  [{dev['index']}] {dev['name']}")
            mem = dev['memory']
            print(f"      Memory: {mem['allocated']:.2f}/{mem['total']:.2f} GB allocated")

        print("\nPipelines:")
        for name, pipeline in self.pipelines.items():
            num_stages = len(pipeline.stages) if pipeline.stages else 0
            num_logs = len(pipeline.time_logs)
            print(f"  {name}: stages={num_stages}, time_logs={num_logs}")
