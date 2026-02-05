import queue
import struct
import uuid
from typing import Any, List, Dict, Callable

import torch
import torch.distributed as dist
from torch import nn

from .adaptive_pipeline import AdaptivePipeline
from .device_manager import DeviceManager
from .timed_module import make_module_timed


def uuids_to_tensor(uuids: list[uuid.UUID], pad_to: int) -> torch.Tensor:
    """Encode UUIDs as a flat tensor of int32 (4 ints per UUID), padded with zeros."""
    ints = []
    for u in uuids:
        ints.extend(struct.unpack('>4i', u.bytes))
    # Pad remaining slots with zeros (nil UUID = sentinel)
    ints.extend([0] * (pad_to - len(uuids)) * 4)
    return torch.tensor(ints, dtype=torch.int)


def tensor_to_uuids(t: torch.Tensor) -> list[uuid.UUID | None]:
    """Decode flat int32 tensor back to UUIDs. Returns None for nil UUID (sentinel)."""
    values = t.tolist()
    result = []
    for i in range(0, len(values), 4):
        chunk = values[i:i + 4]
        if all(v == 0 for v in chunk):
            result.append(None)
        else:
            raw = struct.pack('>4i', *chunk)
            result.append(uuid.UUID(bytes=raw))
    return result


class MainService:
    """
    Main service that manages adaptive pipelines.
    """

    #TODO: find a more appropriate name for this
    # maybe AdaptivePipelineRunner? PipelineRuntime? PipelineOrchestrator?

    def __init__(self, handle_output_fn: Callable[[uuid.UUID, str, Any], None], default_timing_depth: int = 3, verbose=False):
        """
        Args:
            handle_output_fn: Function for handling model outputs.
              Should be of type f(request_id: uuid.UUID, model_name: str, output: Any) -> None
            default_timing_depth: Depth for TimedModule profiling
            verbose: Enable verbose logging output
        """
        self.pipelines: dict[str, AdaptivePipeline] = {}
        self.work_by_model: Dict[str, queue.Queue] = {}
        self.handle_output_fn = handle_output_fn
        self.default_timing_depth = default_timing_depth
        self.verbose = verbose

        self.device_manager = DeviceManager(verbose=verbose)
        self.primary_device = self.device_manager.get_device(0)

    def _log(self, msg: str):
        """Print message if verbose logging is enabled."""
        if self.verbose:
            print(msg)

    def add_model(self, model_name: str, model: nn.Module, example_input: Any, model_output_is_static: bool, device=None, depth: int | None = None, **kwargs):
        """
        Add a model to the service.

        Args:
            model_name: Unique name for the model
            model: The PyTorch model to add
            example_input: An example input for the model
            model_output_is_static: does the model always output a tensor of the same size?
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
            model_output_is_static,
            **kwargs
        )
        self.work_by_model[model_name] = queue.Queue()

    def queue_work(self, model_name: str, x: Any) -> uuid.UUID:
        """Queue work for a model. Only call this on rank 0."""
        if dist.get_rank() != 0:
            raise RuntimeError("queue_work() must only be called on rank 0")
        if model_name not in self.work_by_model:
            raise ValueError(f"Model '{model_name}' not found. Add it with add_model() first.")
        request_id = uuid.uuid4()
        self.work_by_model[model_name].put((request_id, x))
        return request_id

    def _run_pipeline(self, model_name: str):
        pipeline = self.pipelines[model_name]
        model_queue = self.work_by_model[model_name]
        n_microbatches = pipeline.n_microbatches
        rank = dist.get_rank()
        last_rank = dist.get_world_size() - 1
        batched_input = None

        if rank == 0:
            # Collect up to n_microbatches items from this model's queue
            work_items: list[tuple[uuid.UUID, Any]] = []
            while len(work_items) < n_microbatches and not model_queue.empty():
                work_items.append(model_queue.get(block=False))

            req_ids = [item[0] for item in work_items]
            inputs = [item[1] for item in work_items]

            # Pad batch if needed (scheduler expects exactly n_microbatches)
            while len(inputs) < n_microbatches:
                inputs.append(inputs[-1])  # Duplicate last input as padding

            self._log(
                f"MainService.run: processing {len(work_items)} requests for model '{model_name}' (microbatch size: {n_microbatches})")

            # Concatenate inputs along batch dimension (inputs already have batch dim)
            batched_input = torch.cat(inputs, dim=0).contiguous()

            if dist.get_world_size() != 1:
                # Encode UUIDs as int32 tensor (4 ints per UUID, nil UUID = padding sentinel)
                t_req_ids = uuids_to_tensor(req_ids, n_microbatches)

        # All ranks must call forward together
        outputs = pipeline.forward(batched_input)

        # Send req_ids after forward so it doesn't interfere with the scheduler's P2P
        if rank == 0 and dist.get_world_size() != 1:
            dist.send(t_req_ids, dst=last_rank)

        # Output is only on the last rank
        # Which should handle the output with the user defined function
        if rank == last_rank:
            # Receive the request ids from the first rank
            if dist.get_world_size() != 1:
                t_req_ids = torch.zeros(n_microbatches * 4, dtype=torch.int)
                dist.recv(t_req_ids, src=0)
                req_ids = tensor_to_uuids(t_req_ids)

            for i, req_id in enumerate(req_ids):
                if req_id is None:
                    continue  # Skip padding entries (nil UUID sentinel)
                output = outputs[i]
                self.handle_output_fn(req_id, model_name, output)

    def run(self, exit_when_done = False):
        """The main loop of the service."""
        self._log("MainService.run: starting main loop")
        rank = dist.get_rank()
        while True:
            did_work = False

            # Process each model's work — drain all batches before moving to next model
            for model_name, model_queue in self.work_by_model.items():
                while True:
                    # Only rank 0 checks the queue
                    if rank == 0:
                        has_work = torch.tensor([1 if not model_queue.empty() else 0], dtype=torch.int)
                    else:
                        has_work = torch.tensor([0], dtype=torch.int)

                    # Broadcast whether there's work from rank 0 to all ranks
                    # This synchronises all ranks
                    dist.broadcast(has_work, src=0)

                    if has_work.item() == 0:
                        break

                    self._run_pipeline(model_name)
                    did_work = True

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
            device_info = {'index': i, 'device': str(device)}
            if device.type == 'cuda':
                device_info['name'] = torch.cuda.get_device_properties(i).name
                device_info['memory'] = self.device_manager.get_device_memory_info(i)
            else:
                device_info['name'] = 'CPU'
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
            if 'memory' in dev:
                mem = dev['memory']
                print(f"      Memory: {mem['allocated']:.2f}/{mem['total']:.2f} GB allocated")

        print("\nPipelines:")
        for name, pipeline in self.pipelines.items():
            num_stages = len(pipeline.stages) if pipeline.stages else 0
            num_logs = len(pipeline.time_logs)
            print(f"  {name}: stages={num_stages}, time_logs={num_logs}")
