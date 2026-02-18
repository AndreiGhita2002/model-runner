from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch import nn

from .adaptive_pipeline import AdaptivePipeline
from .device_manager import DeviceManager
from .pipeline_optimizer import PipelineOptimizer, GreedyPipelineOptimizer
from .timed_module import make_module_timed


@dataclass
class ForwardResult:
    """Result of a single pipeline forward pass.

    Attributes:
        outputs: Per-microbatch output tensors (rank 0 and last rank only; ``None`` elsewhere).
        timing: Timing dict with ``"forward"`` and ``"rebalance"`` sub-dicts
            (rank 0 and last rank only; ``None`` elsewhere).
        batch_size: Number of real (non-padded) inputs in the batch.
    """
    outputs: list[Any] | None
    timing: dict | None
    batch_size: int


class PipelineRunner:
    """Manages models and executes single pipeline batches across distributed ranks.

    Handles model registration (wrapping in ``TimedModule`` and ``AdaptivePipeline``),
    executing a single forward pass with automatic padding, and relaying outputs from
    the last rank back to rank 0.

    Requires ``torch.distributed`` to be initialised before use.
    """

    def __init__(self, default_timing_depth: int = 3, verbose: bool = False):
        """Initialise the runner.

        Args:
            default_timing_depth: Default depth for TimedModule profiling.
            verbose: Enable verbose logging to stdout.
        """
        self.pipelines: dict[str, AdaptivePipeline] = {}
        self.default_timing_depth = default_timing_depth
        self.verbose = verbose

        self.device_manager = DeviceManager(verbose=verbose)
        self.primary_device = self.device_manager.get_device(0)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def add_model(self, model_name: str, model: nn.Module, example_input: Any,
                  optimizer_class: type[PipelineOptimizer] = GreedyPipelineOptimizer,
                  device=None, depth: int | None = None, **kwargs):
        """Register a model and create its adaptive pipeline. Must be called on all ranks.

        The model is wrapped in a ``TimedModule`` for profiling and then handed to an
        ``AdaptivePipeline`` which manages stage splitting and rebalancing.

        Args:
            model_name: Unique name for the model.
            model: The PyTorch model. Caller is responsible for setting eval/train mode.
            example_input: A representative input tensor used to trace the pipeline.
            optimizer_class: Pipeline optimiser class. Defaults to ``GreedyPipelineOptimizer``.
            device: Device to run the model on (default: primary device).
            depth: Depth for TimedModule profiling (default: ``default_timing_depth``).
            **kwargs: Forwarded to ``AdaptivePipeline``.

        Raises:
            Exception: If a model with the same name is already registered.
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

        if 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose

        self.pipelines[model_name] = AdaptivePipeline(
            timed_model,
            model_name,
            self.device_manager,
            example_input,
            optimizer_class=optimizer_class,
            **kwargs,
        )

    def forward(self, model_name: str, inputs: list[Any] | None) -> ForwardResult:
        """Run one pipeline batch. Must be called on all ranks.

        Rank 0 provides a list of input tensors; other ranks pass ``None``.
        Inputs are padded to ``n_microbatches`` if needed. After the forward pass,
        outputs are relayed from the last rank to rank 0 via broadcast.

        Args:
            model_name: Name of a model registered with ``add_model``.
            inputs: List of input tensors on rank 0 (each with batch dim).
                ``None`` on other ranks.

        Returns:
            ``ForwardResult`` with outputs available on rank 0 and last rank.
        """
        pipeline = self.pipelines[model_name]
        n_microbatches = pipeline.n_microbatches
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        last_rank = world_size - 1
        batched_input = None
        batch_size = 0

        if rank == 0:
            batch_size = len(inputs)

            # Pad batch if needed (scheduler expects exactly n_microbatches)
            padded = list(inputs)
            while len(padded) < n_microbatches:
                padded.append(padded[-1])

            self._log(
                f"PipelineRunner.forward: processing {batch_size} inputs for model "
                f"'{model_name}' (microbatch size: {n_microbatches})")

            batched_input = torch.cat(padded, dim=0).contiguous()

        # All ranks must call forward together
        result = pipeline.forward(batched_input)
        outputs = result["output"]
        timing = result["timing"]

        # Relay outputs from last rank to rank 0
        if world_size == 1:
            if timing is not None:
                timing["batch_size"] = batch_size
                timing["n_microbatches"] = n_microbatches
            return ForwardResult(outputs=outputs, timing=timing, batch_size=batch_size)

        # Multi-rank: broadcast outputs and batch_size to all ranks
        if rank == last_rank:
            cpu_outputs = [o.detach().cpu() for o in outputs] if outputs is not None else outputs
            relay = [(cpu_outputs, timing, batch_size)]
        else:
            relay = [None]

        dist.broadcast_object_list(relay, src=last_rank)

        outputs, timing, batch_size = relay[0]

        if timing is not None:
            timing["batch_size"] = batch_size
            timing["n_microbatches"] = n_microbatches

        # Only rank 0 and last rank get the full result
        if rank == 0 or rank == last_rank:
            return ForwardResult(outputs=outputs, timing=timing, batch_size=batch_size)

        return ForwardResult(outputs=None, timing=None, batch_size=batch_size)

    def get_logs(self) -> Dict[str, Any]:
        """Return timing logs from all registered pipelines."""
        logs = {}
        for model_name, pipeline in self.pipelines.items():
            logs[model_name] = pipeline.time_logs
        return logs

    def get_model_names(self) -> List[str]:
        """Return the names of all registered models."""
        return list(self.pipelines.keys())

    def get_device_info(self) -> Dict[str, Any]:
        """Return information about available compute devices."""
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

    def force_rebalance(self, model_name: str):
        """Request a forced rebalance of the named pipeline.

        Args:
            model_name: Name of a model registered with ``add_model``.

        Raises:
            ValueError: If ``model_name`` has not been registered.
        """
        pipeline = self.pipelines.get(model_name)
        if pipeline is None:
            raise ValueError(f"Model '{model_name}' not found.")
        pipeline.request_force_rebalance()

    def print_status(self):
        """Print a summary of the runner to stdout."""
        print("\n" + "=" * 80)
        print("PipelineRunner Status")
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
            num_stages = pipeline.pipe.num_stages if pipeline.pipe else 0
            num_logs = len(pipeline.time_logs)
            print(f"  {name}: stages={num_stages}, time_logs={num_logs}")
