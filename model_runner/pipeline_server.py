import queue
import threading
import uuid
from typing import Any, List, Dict, Callable
from enum import Enum

import torch
import torch.distributed as dist
from torch import nn

from .pipeline_runner import PipelineRunner
from .pipeline_optimizer import PipelineOptimizer, GreedyPipelineOptimizer
from .util import uuids_to_tensor, tensor_to_uuids


class WorkType(Enum):
    Raw = 0
    GeneratorFn = 1


class PipelineServer:
    """Orchestrates adaptive pipeline-parallel inference across distributed ranks.

    Wraps a ``PipelineRunner`` and adds work queuing, request ID tracking,
    result storage, and the distributed run loop. Models are registered with
    ``add_model``, work is submitted via ``queue_work`` (rank 0 only), and
    ``run`` drives the processing loop on all ranks.

    Requires ``torch.distributed`` to be initialised before use.
    """

    def __init__(self, handle_output_fn: Callable[[uuid.UUID, str, Any, dict | None], None] | None = None,
                 default_timing_depth: int = 3, verbose=False):
        """Initialise the server.

        Args:
            handle_output_fn: Optional callback invoked on the last rank when a
                request completes. Signature:
                ``(request_id, model_name, output, timing) -> None``.
                If ``None`` (default), outputs are only stored internally for
                retrieval via ``get_result()`` (the Flask API path).
            default_timing_depth: Default depth for TimedModule profiling.
            verbose: Enable verbose logging to stdout.
        """
        self.runner = PipelineRunner(
            default_timing_depth=default_timing_depth,
            verbose=verbose,
        )
        self.handle_output_fn = handle_output_fn
        self.verbose = verbose

        self.work_by_model: Dict[str, queue.Queue] = {}

        # Result storage for async Flask API (request_id -> (model_name, output, timing))
        self._results: dict[uuid.UUID, tuple[str, Any, dict | None]] = {}
        self._results_lock = threading.Lock()

        # Synchronization for cross-thread/process work submission
        self._work_available = threading.Condition()
        self._shutdown_requested = False
        self._stop_continuous = False

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def add_model(self, model_name: str, model: nn.Module, example_input: Any,
                  optimizer_class: type[PipelineOptimizer] = GreedyPipelineOptimizer,
                  device=None, depth: int | None = None, **kwargs):
        """Register a model. Delegates to ``PipelineRunner.add_model`` and creates a work queue.

        Args:
            model_name: Unique name for the model. Used to reference it in ``queue_work``.
            model: The PyTorch model.
            example_input: A representative input tensor used to trace the pipeline.
            optimizer_class: Pipeline optimiser class. Defaults to ``GreedyPipelineOptimizer``.
            device: Device to run the model on (default: primary device).
            depth: Depth for TimedModule profiling.
            **kwargs: Forwarded to ``AdaptivePipeline``.
        """
        self.runner.add_model(model_name, model, example_input,
                              optimizer_class=optimizer_class,
                              device=device, depth=depth, **kwargs)
        self.work_by_model[model_name] = queue.Queue()

    def queue_work(self, model_name: str, x: Any, is_generator_fn: bool = False) -> uuid.UUID:
        """Submit an inference request for a model. Must only be called on rank 0.

        Args:
            model_name: Name of a model previously registered with ``add_model``.
            x: Input tensor (must include the batch dimension).
            is_generator_fn: (optional) True if x is a function that outputs the model input,
                rather than being the input itself. Default: False

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
        if is_generator_fn and not callable(x):
            raise TypeError("x must be callable when is_generator_fn=True")
        request_id = uuid.uuid4()
        work_type = WorkType.GeneratorFn if is_generator_fn else WorkType.Raw
        self.work_by_model[model_name].put((request_id, work_type, x))
        with self._work_available:
            self._work_available.notify()
        return request_id

    def _run_pipeline(self, model_name: str):
        """Execute one pipeline batch for the given model across all ranks.

        Rank 0 drains up to ``n_microbatches`` items from the model's work queue,
        calls ``runner.forward()``, then maps outputs to request IDs and dispatches
        them through ``handle_output_fn`` and result storage.

        Args:
            model_name: Name of the model whose queue to process.
        """
        pipeline = self.runner.pipelines[model_name]
        model_queue = self.work_by_model[model_name]
        n_microbatches = pipeline.n_microbatches
        rank = dist.get_rank()
        last_rank = dist.get_world_size() - 1

        inputs = None
        req_ids = []

        if rank == 0:
            # Collect up to n_microbatches items from this model's queue
            req_ids: list[uuid.UUID] = []
            inputs: list[Any] = []
            while len(inputs) < n_microbatches and not model_queue.empty():
                req_id, work_type, payload = model_queue.get(block=False)
                if work_type == WorkType.GeneratorFn:
                    inputs.append(payload())
                else:
                    inputs.append(payload)
                req_ids.append(req_id)

            self._log(
                f"PipelineServer.run: processing {len(inputs)} requests for model '{model_name}' "
                f"(microbatch size: {n_microbatches})")

            # Send request IDs to last rank via P2P (after forward, but prepare tensor now)
            if dist.get_world_size() != 1:
                t_req_ids = uuids_to_tensor(req_ids, n_microbatches)

        # All ranks call forward together — outputs are relayed to rank 0 inside forward()
        fwd_result = self.runner.forward(model_name, inputs)

        # Send req_ids after forward so it doesn't interfere with the scheduler's P2P
        if rank == 0 and dist.get_world_size() != 1:
            dist.send(t_req_ids, dst=last_rank)

        # Invoke handle_output_fn on the last rank
        if rank == last_rank:
            if dist.get_world_size() != 1:
                t_req_ids = torch.zeros(n_microbatches * 4, dtype=torch.int)
                dist.recv(t_req_ids, src=0)
                req_ids = tensor_to_uuids(t_req_ids)

            for i, req_id in enumerate(req_ids):
                if req_id is None:
                    continue
                if self.handle_output_fn is not None:
                    self.handle_output_fn(req_id, model_name, fwd_result.outputs[i], fwd_result.timing)

        # Store results on rank 0 (outputs already relayed by PipelineRunner.forward)
        if rank == 0:
            with self._results_lock:
                for i, req_id in enumerate(req_ids):
                    if req_id is None:
                        continue
                    self._results[req_id] = (model_name, fwd_result.outputs[i].detach().cpu(), fwd_result.timing)

    def run(self, exit_when_done=False):
        """Run the main processing loop. Must be called on all ranks.

        Continuously drains work queues for every registered model. All ranks
        participate in each pipeline forward pass (synchronised via broadcast).

        Args:
            exit_when_done: If True, return once all queues are empty. If False
                (default), wait for new work or a shutdown signal.
        """
        self._log("PipelineServer.run: starting main loop")
        rank = dist.get_rank()

        while True:
            did_work = False

            for model_name, model_queue in self.work_by_model.items():
                while True:
                    if rank == 0:
                        has_work = torch.tensor([1 if not model_queue.empty() else 0], dtype=torch.int)
                    else:
                        has_work = torch.tensor([0], dtype=torch.int)

                    dist.broadcast(has_work, src=0)

                    if has_work.item() == 0:
                        break

                    self._run_pipeline(model_name)
                    did_work = True

            if not did_work:
                if exit_when_done:
                    self._log("PipelineServer.run: queue empty, exiting")
                    return

                if rank == 0:
                    with self._work_available:
                        should_exit = self._shutdown_requested
                        has_any_work = any(not q.empty() for q in self.work_by_model.values())

                        if not should_exit and not has_any_work:
                            self._work_available.wait()
                            should_exit = self._shutdown_requested

                    shutdown_tensor = torch.tensor([1 if should_exit else 0], dtype=torch.int)
                else:
                    shutdown_tensor = torch.tensor([0], dtype=torch.int)

                dist.broadcast(shutdown_tensor, src=0)

                if shutdown_tensor.item() == 1:
                    self._log("PipelineServer.run: shutdown requested, exiting")
                    return

    def run_continuous(self, model_name: str, input_generator: Callable[[], Any],
                       handle_output_fn: Callable[[uuid.UUID, str, Any, dict | None], None] | None = None):
        """Run a single model continuously, generating inputs on the fly.

        Designed for interference experiments. Generates inputs inline (no queue),
        calls forward, invokes the callback, and repeats until ``stop_continuous()``
        is called. Must be called on all ranks.

        Args:
            model_name: Registered model to run.
            input_generator: Callable that returns one input tensor per call.
                Only called on rank 0.
            handle_output_fn: Optional callback invoked on the last rank per request.
                Overrides the server-level ``handle_output_fn`` if provided.
        """
        pipeline = self.runner.pipelines[model_name]
        n_microbatches = pipeline.n_microbatches
        rank = dist.get_rank()
        last_rank = dist.get_world_size() - 1
        output_fn = handle_output_fn or self.handle_output_fn
        self._stop_continuous = False

        self._log(f"PipelineServer.run_continuous: starting for model '{model_name}'")

        while True:
            # Check stop flag (broadcast from rank 0)
            if rank == 0:
                stop = torch.tensor([1 if self._stop_continuous else 0], dtype=torch.int)
            else:
                stop = torch.tensor([0], dtype=torch.int)
            dist.broadcast(stop, src=0)
            if stop.item() == 1:
                self._log("PipelineServer.run_continuous: stop requested, exiting")
                return

            # Generate inputs and request IDs on rank 0
            inputs = None
            req_ids = []
            t_req_ids = None

            if rank == 0:
                req_ids = [uuid.uuid4() for _ in range(n_microbatches)]
                inputs = [input_generator() for _ in range(n_microbatches)]

                if dist.get_world_size() != 1:
                    t_req_ids = uuids_to_tensor(req_ids, n_microbatches)

            # All ranks call forward together
            fwd_result = self.runner.forward(model_name, inputs)

            # Send req_ids after forward
            if rank == 0 and dist.get_world_size() != 1:
                dist.send(t_req_ids, dst=last_rank)

            # Handle outputs on last rank
            if rank == last_rank:
                if dist.get_world_size() != 1:
                    t_req_ids = torch.zeros(n_microbatches * 4, dtype=torch.int)
                    dist.recv(t_req_ids, src=0)
                    req_ids = tensor_to_uuids(t_req_ids)

                for i, req_id in enumerate(req_ids):
                    if req_id is None:
                        continue
                    if output_fn is not None:
                        output_fn(req_id, model_name, fwd_result.outputs[i], fwd_result.timing)

    def stop_continuous(self):
        """Signal ``run_continuous`` to stop after the current forward pass."""
        self._stop_continuous = True

    def shutdown(self):
        """Request a graceful shutdown of the server."""
        with self._work_available:
            self._shutdown_requested = True
            self._work_available.notify()
        self._log("PipelineServer.shutdown: shutdown requested")

    def is_shutdown_requested(self) -> bool:
        """Check whether shutdown has been requested."""
        return self._shutdown_requested

    def get_result(self, request_id: uuid.UUID) -> tuple[str, Any, dict | None] | None:
        """Pop a completed result by request ID.

        Returns:
            ``(model_name, output, timing)`` if the result is ready, or ``None`` if
            the request is still pending.
        """
        with self._results_lock:
            return self._results.pop(request_id, None)

    def get_results(self) -> dict[uuid.UUID, tuple[str, Any, dict | None]]:
        """Pop all completed results."""
        with self._results_lock:
            results = dict(self._results)
            self._results.clear()
            return results

    # ── Delegates to PipelineRunner ──────────────────────────────────────

    def get_logs(self) -> Dict[str, Any]:
        """Return timing logs from all registered pipelines."""
        return self.runner.get_logs()

    def get_model_names(self) -> List[str]:
        """Return the names of all registered models."""
        return self.runner.get_model_names()

    def get_device_info(self) -> Dict[str, Any]:
        """Return information about available compute devices."""
        return self.runner.get_device_info()

    def force_rebalance(self, model_name: str):
        """Request a forced rebalance of the named pipeline."""
        self.runner.force_rebalance(model_name)

    def print_status(self):
        """Print a summary of the server to stdout."""
        self.runner.print_status()
