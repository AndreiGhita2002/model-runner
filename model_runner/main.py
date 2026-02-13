import queue
import struct
import threading
import uuid
from typing import Any, List, Dict, Callable

import torch
import torch.distributed as dist
from torch import nn

from .adaptive_pipeline import AdaptivePipeline
from .device_manager import DeviceManager
from .pipeline_optimizer import PipelineOptimizer, GreedyPipelineOptimizer
from .timed_module import make_module_timed


def uuids_to_tensor(uuids: list[uuid.UUID], pad_to: int) -> torch.Tensor:
    """Encode UUIDs as a flat int32 tensor for P2P transfer via torch.distributed.

    Each UUID is serialized as 4 signed int32 values (16 bytes). Remaining slots
    up to ``pad_to`` are filled with zeros (nil UUID), which acts as the padding sentinel.

    Args:
        uuids: UUIDs to encode. Length must be <= pad_to.
        pad_to: Total number of UUID slots in the output tensor.

    Returns:
        Flat int32 tensor of shape ``(pad_to * 4,)``.
    """
    ints = []
    for u in uuids:
        ints.extend(struct.unpack('>4i', u.bytes))
    # Pad remaining slots with zeros (nil UUID = sentinel)
    ints.extend([0] * (pad_to - len(uuids)) * 4)
    return torch.tensor(ints, dtype=torch.int)


def tensor_to_uuids(t: torch.Tensor) -> list[uuid.UUID | None]:
    """Decode a flat int32 tensor back to UUIDs. Inverse of ``uuids_to_tensor``.

    Args:
        t: Flat int32 tensor of shape ``(n * 4,)`` produced by ``uuids_to_tensor``.

    Returns:
        List of ``n`` entries. Each entry is a ``uuid.UUID``, or ``None`` for
        nil-UUID slots (the padding sentinel).
    """
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
    """Orchestrates adaptive pipeline-parallel inference across distributed ranks.

    Models are registered with ``add_model``, work is submitted via ``queue_work``
    (rank 0 only), and ``run`` drives the processing loop on all ranks. Completed
    outputs are delivered through the ``handle_output_fn`` callback on the last rank.

    Requires ``torch.distributed`` to be initialised before use.
    """

    #TODO(naming): find a more appropriate name for this
    # maybe AdaptivePipelineRunner? PipelineRuntime? PipelineOrchestrator?

    def __init__(self, handle_output_fn: Callable[[uuid.UUID, str, Any, dict | None], None], default_timing_depth: int = 3, verbose=False):
        """Initialise the service.

        Args:
            handle_output_fn: Callback invoked on the last rank when a request
                completes. Signature: ``(request_id: uuid.UUID, model_name: str, output: Any, timing: dict | None) -> None``.
            default_timing_depth: Default depth for TimedModule profiling.
            verbose: Enable verbose logging to stdout.
        """
        self.pipelines: dict[str, AdaptivePipeline] = {}
        self.work_by_model: Dict[str, queue.Queue] = {}
        self.handle_output_fn = handle_output_fn
        self.default_timing_depth = default_timing_depth
        self.verbose = verbose

        # Result storage for async Flask API (request_id -> (model_name, output, timing))
        self._results: dict[uuid.UUID, tuple[str, Any, dict | None]] = {}
        self._results_lock = threading.Lock()

        # Synchronization for cross-thread/process work submission
        self._work_available = threading.Condition()
        self._shutdown_requested = False

        self.device_manager = DeviceManager(verbose=verbose)
        self.primary_device = self.device_manager.get_device(0)

    def _log(self, msg: str):
        """Print a message to stdout if verbose logging is enabled.

        Args:
            msg: The message to print.
        """
        if self.verbose:
            print(msg)

    def add_model(self, model_name: str, model: nn.Module, example_input: Any, model_output_is_static: bool,
                  optimizer_class: type[PipelineOptimizer] = GreedyPipelineOptimizer,
                  device=None, depth: int | None = None, **kwargs):
        """Register a model and create its adaptive pipeline. Must be called on all ranks.

        The model is wrapped in a ``TimedModule`` for profiling and then handed to an
        ``AdaptivePipeline`` which manages stage splitting and rebalancing.

        Args:
            model_name: Unique name for the model. Used to reference it in ``queue_work``.
            model: The PyTorch model. Caller is responsible for setting eval/train mode.
            example_input: A representative input tensor used to trace the pipeline.
            model_output_is_static: Whether the model always produces the same output shape.
            optimizer_class: Pipeline optimiser class. Constructed internally by
                ``AdaptivePipeline`` with ``num_stages``, ``root_uuid``, and
                ``device_manager``. Defaults to ``GreedyPipelineOptimizer``.
            device: Device to run the model on (default: primary device).
            depth: Depth for TimedModule profiling (default: ``default_timing_depth``).
            **kwargs: Forwarded to ``AdaptivePipeline``:
                - optimizer_kwargs: Extra kwargs for the optimiser constructor
                - max_log_entries: Max timing measurements per module (default: 5)
                - n_microbatches: Number of microbatches for pipeline (default: 4)
                - initial_pipeline_config: Initial pipeline configuration
                - async_optimization: Use async optimisation (default: False)

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

        # Pass verbose from self if not explicitly provided in kwargs
        if 'verbose' not in kwargs:
            kwargs['verbose'] = self.verbose

        self.pipelines[model_name] = AdaptivePipeline(
            timed_model,
            model_name,
            self.device_manager,
            example_input,
            optimizer_class=optimizer_class,
            **kwargs
        )
        self.work_by_model[model_name] = queue.Queue()

    def queue_work(self, model_name: str, x: Any) -> uuid.UUID:
        """Submit an inference request for a model. Must only be called on rank 0.

        A UUID is generated internally and returned so the caller can correlate it
        with the output delivered later via ``handle_output_fn``.

        Args:
            model_name: Name of a model previously registered with ``add_model``.
            x: Input tensor (must include the batch dimension).

        Returns:
            The UUID assigned to this request.

        Raises:
            RuntimeError: If called on a rank other than 0.
            ValueError: If ``model_name`` has not been registered.
        """
        if dist.get_rank() != 0:
            raise RuntimeError("queue_work() must only be called on rank 0")
        if model_name not in self.work_by_model:
            raise ValueError(f"Model '{model_name}' not found. Add it with add_model() first.")
        request_id = uuid.uuid4()
        self.work_by_model[model_name].put((request_id, x))
        with self._work_available:
            self._work_available.notify()
        return request_id

    def _run_pipeline(self, model_name: str):
        """Execute one pipeline batch for the given model across all ranks.

        Rank 0 drains up to ``n_microbatches`` items from the model's work queue,
        pads if necessary, and runs the pipeline forward pass. Request IDs are sent
        to the last rank via P2P so it can dispatch outputs through ``handle_output_fn``.

        Args:
            model_name: Name of the model whose queue to process.
        """
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
        result = pipeline.forward(batched_input)
        outputs = result["output"]
        timing = result["timing"]

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
                self.handle_output_fn(req_id, model_name, output, timing)

        # Relay results back to rank 0 for the async Flask API
        world_size = dist.get_world_size()
        if world_size == 1:
            # Rank 0 IS the last rank — store results directly
            for i, req_id in enumerate(req_ids):
                if req_id is None:
                    continue
                output = outputs[i].detach().cpu()
                with self._results_lock:
                    self._results[req_id] = (model_name, output, timing)
        else:
            # Last rank broadcasts results to all ranks (including rank 0)
            if rank == last_rank:
                result_entries = []
                for i, req_id in enumerate(req_ids):
                    if req_id is None:
                        continue
                    result_entries.append((req_id, model_name, outputs[i].detach().cpu(), timing))
                broadcast_list = [result_entries]
            else:
                broadcast_list = [None]

            dist.broadcast_object_list(broadcast_list, src=last_rank)

            # Rank 0 stores the results
            if rank == 0:
                result_entries = broadcast_list[0]
                with self._results_lock:
                    for req_id, m_name, output, req_timing in result_entries:
                        self._results[req_id] = (m_name, output, req_timing)

    def run(self, exit_when_done=False):
        """Run the main processing loop. Must be called on all ranks.

        Continuously drains work queues for every registered model. All ranks
        participate in each pipeline forward pass (synchronized via broadcast).

        When no work is available and ``exit_when_done=False``, rank 0 waits on a
        condition variable that is signalled by ``queue_work()``. Other ranks
        synchronize via broadcast. Call ``shutdown()`` to request a graceful exit.

        Args:
            exit_when_done: If True, return once all queues are empty. If False
                (default), wait for new work or a shutdown signal.
        """
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

            # No work was processed this iteration
            if not did_work:
                if exit_when_done:
                    self._log("MainService.run: queue empty, exiting")
                    return

                # Check for shutdown and wait for new work (rank 0 only decides)
                if rank == 0:
                    with self._work_available:
                        # Check shutdown flag and whether any queue has work
                        should_exit = self._shutdown_requested
                        has_any_work = any(not q.empty() for q in self.work_by_model.values())

                        if not should_exit and not has_any_work:
                            # Wait for queue_work() or shutdown() to signal
                            self._work_available.wait()
                            should_exit = self._shutdown_requested

                    # Broadcast shutdown decision to all ranks
                    shutdown_tensor = torch.tensor([1 if should_exit else 0], dtype=torch.int)
                else:
                    shutdown_tensor = torch.tensor([0], dtype=torch.int)

                dist.broadcast(shutdown_tensor, src=0)

                if shutdown_tensor.item() == 1:
                    self._log("MainService.run: shutdown requested, exiting")
                    return

    def shutdown(self):
        """Request a graceful shutdown of the service.

        Signals the ``run()`` loop to exit after completing any in-progress work.
        Thread-safe; can be called from a different thread than the one running ``run()``.
        Only effective on rank 0 (other ranks follow via broadcast).
        """
        with self._work_available:
            self._shutdown_requested = True
            self._work_available.notify()
        self._log("MainService.shutdown: shutdown requested")

    def is_shutdown_requested(self) -> bool:
        """Check whether shutdown has been requested.

        Returns:
            True if ``shutdown()`` has been called.
        """
        return self._shutdown_requested

    def get_logs(self) -> Dict[str, Any]:
        """Return timing logs from all registered pipelines.

        Returns:
            Dict mapping model name to its pipeline's timing log entries.
        """
        logs = {}
        for model_name, pipeline in self.pipelines.items():
            logs[model_name] = pipeline.time_logs
        return logs

    def get_model_names(self) -> List[str]:
        """Return the names of all registered models.

        Returns:
            List of model name strings.
        """
        return list(self.pipelines.keys())

    def get_device_info(self) -> Dict[str, Any]:
        """Return information about available compute devices.

        Returns:
            Dict with keys ``num_devices`` (int) and ``devices`` (list of dicts).
            Each device dict contains ``index``, ``device``, ``name``, and
            optionally ``memory`` (with ``allocated`` and ``total`` in GB) for CUDA devices.
        """
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

    def get_result(self, request_id: uuid.UUID) -> tuple[str, Any, dict | None] | None:
        """Pop a completed result by request ID.

        Returns:
            ``(model_name, output, timing)`` if the result is ready, or ``None`` if
            the request is still pending. ``timing`` is a dict with ``"start"`` and
            ``"end"`` wall-clock timestamps, or ``None``.
        """
        with self._results_lock:
            return self._results.pop(request_id, None)

    def force_rebalance(self, model_name: str):
        """Request a forced rebalance of the named pipeline.

        Thread-safe: may be called from the Flask thread.

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
        """Print a summary of the service to stdout, including devices and pipeline info."""
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
