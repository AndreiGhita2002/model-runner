import multiprocessing as mp
import queue
import threading
import time
import uuid
from typing import Optional, Any

import torch
import torch.fx as fx

from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe, SplitPoint, Pipe
from torch.distributed.pipelining._IR import annotate_split_points
import torch.distributed as dist
from torch.distributed.pipelining.schedules import PipelineScheduleSingle, PipelineScheduleMulti

from .timed_module import TimedModule, timed_module_registry, timed_module_hierarchy
from .device_manager import DeviceManager
from .pipeline_optimizer import PipelineOptimizer, PipelineConfig, TimeBasedShishaPipelineOptimizer

# The ATen op that pipe_split() compiles to in the FX graph
_aten_pipe_split = torch.ops.pippy._pipe_split.default


def _make_split_policy(keep_indices: set[int]):
    """Create a split_policy that keeps only pipe_split nodes at the given indices.

    The cached trace contains a pipe_split node before every child (except the first).
    This policy removes the ones not needed for the current split_spec.

    Args:
        keep_indices: Set of pipe_split indices to keep (0-based, in child order).
    """
    def policy(traced: fx.GraphModule) -> fx.GraphModule:
        pipe_splits = [
            node for node in traced.graph.nodes
            if node.op == "call_function" and node.target == _aten_pipe_split
        ]
        for i, node in enumerate(pipe_splits):
            if i not in keep_indices:
                traced.graph.erase_node(node)
        traced.graph.lint()
        traced.recompile()
        return traced
    return policy


class _ContiguousStageWrapper(torch.nn.Module):
    """Wraps a pipeline stage submodule to make its output contiguous.

    Required because PyTorch's P2P communication (isend) requires contiguous tensors.

    Args:
        module: The stage submodule to wrap.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """Run the wrapped module and return a contiguous output.

        Args:
            *args: Positional arguments forwarded to the wrapped module.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            The module's output, made contiguous if it is a tensor.
        """
        output = self.module(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return output.contiguous()
        return output


def _optimizer_process_worker(
    optimizer: PipelineOptimizer,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
):
    """Background worker that runs the pipeline optimiser in a separate process.

    Blocks on ``request_queue``, runs ``optimizer.optimize`` when a request arrives,
    and puts the result (a ``PipelineConfig`` or ``None``) on ``result_queue``.
    Exits when ``shutdown_event`` is set.

    Args:
        optimizer: The pipeline optimiser instance.
        request_queue: Queue of ``(time_logs, current_config)`` tuples to process.
        result_queue: Queue where results are placed (``PipelineConfig`` or ``None``).
        shutdown_event: Event signalling the worker to exit.
    """
    while not shutdown_event.is_set():
        try:
            # Non-blocking check with timeout to allow shutdown
            request = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        time_logs, current_config = request
        new_config = optimizer.optimize(time_logs, current_config)
        result_queue.put(new_config)


class AdaptivePipeline:
    """Manages a PyTorch pipeline with automatic stage rebalancing.

    Wraps a ``TimedModule`` in a ``ScheduleGPipe`` pipeline and periodically
    re-optimises the stage split based on collected timing data. Rebalancing
    can run synchronously (blocking) or asynchronously (in a background process).

    Requires ``torch.distributed`` to be initialised before use.
    """
    name: str
    current_config: Optional[PipelineConfig]
    pipe: Pipe | None
    scheduler: PipelineScheduleSingle | PipelineScheduleMulti | None

    # Maximum number of timing measurements to keep per module
    max_log_entries: int
    # Time logs
    time_logs: dict[uuid.UUID, list[float]]

    #TODO: sometimes models can genuinely not be split enough and world_size will always > num stages

    def __init__(
            self,
            model: TimedModule,
            model_name: str,
            device_manager: DeviceManager,
            example_input: Any,
            optimizer_class: type[PipelineOptimizer] = TimeBasedShishaPipelineOptimizer,
            max_log_entries: int = 20, #todo: used to be 5; should ideally be a running average or smth
            n_microbatches: int = 4,
            initial_pipeline_config: PipelineConfig | None = None,
            verbose: bool = False,
            async_optimization: bool = False,
            **kwargs
    ):
        """Initialise the adaptive pipeline and build the initial stage split.

        Args:
            model: A ``TimedModule``-wrapped model to run in the pipeline.
            model_name: Human-readable name (used for logging).
            device_manager: Provides device allocation for pipeline stages.
            example_input: Representative input tensor used to trace the pipeline graph.
                record the output shape (needed for pre-allocated receive buffers).
            optimizer_class: Class of the pipeline optimiser to use. Constructed
                internally with ``num_stages``, ``root_uuid``, and ``device_manager``.
                Defaults to ``GreedyPipelineOptimizer``.
            optimizer_kwargs: Extra keyword arguments forwarded to the optimiser
                constructor (e.g. ``alpha`` for ``TimeBasedShishaPipelineOptimizer``).
            max_log_entries: Maximum number of timing measurements to keep per module.
                Older entries are discarded when the cap is reached.
            n_microbatches: Number of microbatches per pipeline step. Clamped to at
                least ``world_size``.
            initial_pipeline_config: Explicit first split config. If None, a uniform
                split is generated automatically.
            verbose: Enable verbose logging to stdout.
            async_optimization: If True, run the optimiser in a background process
                so rebalancing does not block the forward path.
        """
        self.original_model = model
        self.name = model_name
        self.device_manager = device_manager
        self.example_input = example_input
        self.max_log_entries = max_log_entries
        self.time_logs = {}
        self.verbose = verbose
        self.async_optimization = async_optimization
        self.num_stages = dist.get_world_size()
        self.n_microbatches = n_microbatches if n_microbatches >= self.num_stages else self.num_stages

        self.pipeline_optimizer = optimizer_class(
            num_stages=self.num_stages,
            root_uuid=model.uuid,
            device_manager=device_manager,
            depth=model.depth,
            **kwargs,
        )

        # Current pipeline state
        self.current_config = None
        self.pipe = None
        self.stage = None
        self.scheduler = None

        # Cached FX trace: trace once with all possible split points,
        # then reuse on subsequent rebuilds with a split_policy that
        # removes unwanted pipe_split nodes.
        self._cached_export = None
        self._cached_children = None  # children list at time of caching (for invalidation)

        # Force-rebalance flag (set from Flask thread, read from main loop)
        self._force_rebalance: bool = False
        self._force_rebalance_lock = threading.Lock()

        # Async optimisation state (only used on rank 0)
        self._optimizer_process: Optional[mp.Process] = None
        self._request_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._shutdown_event: Optional[mp.Event] = None
        self._pending_optimization: bool = False

        if self.async_optimization and dist.get_rank() == 0:
            self._start_optimizer_process()

        # Initial pipeline setup — rank 0 computes, broadcasts to all
        if initial_pipeline_config is None:
            if dist.get_rank() == 0:
                initial_pipeline_config = self.pipeline_optimizer.initial_setup()
            config_list = [initial_pipeline_config]
            dist.broadcast_object_list(config_list, src=0)
            initial_pipeline_config = config_list[0]
        self.rebuild_pipeline(initial_pipeline_config)

    def _log(self, msg: str):
        """Print a message to stdout if verbose logging is enabled.

        Args:
            msg: The message to print.
        """
        if self.verbose:
            print(msg)

    def _start_optimizer_process(self):
        """Spawn the background optimiser process and its communication queues.

        TODO: The background process receives a fork/pickle *copy* of
        ``self.pipeline_optimizer``. If the main process later mutates the
        optimizer (e.g. ``generate_safe_config()`` changes depth/children, or
        ``should_rebalance`` updates internal counters), the background copy
        diverges.  This can cause the background optimizer to keep producing
        configs at the wrong depth after a fallback.  Fix options:
        - Restart the background process after any optimizer mutation.
        - Send updated optimizer state over the queue.
        - Move the optimizer entirely into the background process and proxy
          all calls through the queues.
        """
        self._request_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._shutdown_event = mp.Event()

        self._optimizer_process = mp.Process(
            target=_optimizer_process_worker,
            args=(
                self.pipeline_optimizer,
                self._request_queue,
                self._result_queue,
                self._shutdown_event,
            ),
            daemon=True,
        )
        self._optimizer_process.start()
        self._log(f"Started optimizer process (PID: {self._optimizer_process.pid})")

    def _stop_optimizer_process(self):
        """Signal the background optimiser to exit and wait for it to terminate."""
        if self._optimizer_process is not None:
            self._shutdown_event.set()
            self._optimizer_process.join(timeout=1.0)
            if self._optimizer_process.is_alive():
                self._optimizer_process.terminate()
            self._optimizer_process = None
            self._log("Stopped optimizer process")

    def _send_optimization_request(self):
        """Enqueue the current timing logs and config for background optimisation.

        No-op if an optimisation request is already in flight.
        """
        if not self._pending_optimization:
            self._request_queue.put((self.time_logs.copy(), self.current_config))
            self._pending_optimization = True
            self._log("Sent optimization request to background process")

    def _check_optimization_result(self) -> Optional[PipelineConfig]:
        """Poll the background optimiser for a completed result (non-blocking).

        Returns:
            A new ``PipelineConfig`` if the optimiser produced one, or ``None``
            if no result is ready or no rebalance was needed.
        """
        try:
            result = self._result_queue.get_nowait()
            self._pending_optimization = False
            return result
        except queue.Empty:
            return None

    def shutdown(self):
        """Release resources, including the background optimiser process on rank 0."""
        if dist.get_rank() == 0:
            self._stop_optimizer_process()

    def request_force_rebalance(self):
        """Request a forced rebalance on the next forward pass.

        Thread-safe: may be called from a Flask thread while the main loop
        runs ``forward()`` on the main thread.
        """
        with self._force_rebalance_lock:
            self._force_rebalance = True

    def forward(self, x: Any) -> dict[str, Any]:
        """Run one pipeline step across all ranks. Must be called on all ranks.

        Only rank 0 supplies the input; other ranks pass ``None``. After the
        forward pass, timing logs are updated and a rebalance check may trigger
        (synchronously or asynchronously depending on configuration).

        Args:
            x: Batched input tensor on rank 0 (concatenated microbatches).
                Ignored on other ranks.

        Returns:
            Dict with keys:
            - ``"output"``: On the last rank, list of per-microbatch output tensors;
              on other ranks, ``None``.
            - ``"timing"``: On the last rank, dict with ``"forward"`` and
              ``"rebalance"`` sub-dicts; on other ranks, ``None``.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        last_rank = world_size - 1
        self._log(f"rank:{rank} in forward")

        # Capture wall-clock start time on rank 0
        if rank == 0:
            forward_start = time.perf_counter()

        # Only the first rank gets the input
        with torch.no_grad():
            if rank == 0:
                output = self.scheduler.step(x)
            else:
                output = self.scheduler.step()

        # Capture forward end time on last rank immediately after the step,
        # before update_logs / P2P so it only measures actual computation.
        if rank == last_rank:
            forward_end = time.perf_counter()

        self.update_logs()

        # Send forward_start from rank 0 to last rank
        if world_size > 1:
            if rank == 0:
                t = torch.tensor([forward_start], dtype=torch.float64)
                dist.send(t, dst=last_rank)
            elif rank == last_rank:
                t = torch.tensor([0.0], dtype=torch.float64)
                dist.recv(t, src=0)
                forward_start = t.item()

        if self.async_optimization:
            rebalance_timing = self._forward_async_optimization()
        else:
            rebalance_timing = self._forward_sync_optimization()

        timing = None
        if rank == last_rank:
            timing = {
                "forward": {"start": forward_start, "end": forward_end},
                "rebalance": rebalance_timing,
            }

        return {"output": output, "timing": timing}

    def _forward_sync_optimization(self) -> dict[str, Any]:
        """Check for rebalance and apply it synchronously.

        Only rank 0 runs the optimiser; the decision and new config are
        broadcast so every rank takes the same path (no conditional collectives).

        Returns:
            Dict with ``"start"``, ``"end"`` wall-clock timestamps and
            ``"did_rebalance"`` boolean.
        """
        rebalance_start = time.perf_counter()

        # Rank 0 decides and computes; others receive via broadcast
        if dist.get_rank() == 0:
            # getting the value of force rebalance
            with self._force_rebalance_lock:
                force = self._force_rebalance
                self._force_rebalance = False
            # computing the new config
            new_config = self.pipeline_optimizer.optimize(
                self.time_logs, self.current_config, force_rebalance=force
            )
            config_list = [new_config]
        else:
            config_list = [None]

        dist.broadcast_object_list(config_list, src=0)
        new_config = config_list[0]

        did_rebalance = new_config is not None
        if did_rebalance:
            self._log("Sync optimization: rebalancing pipeline")
            self.time_logs = {}
            self.rebuild_pipeline(new_config)

        rebalance_end = time.perf_counter()
        at_optimum = self.pipeline_optimizer.at_optimum if dist.get_rank() == 0 else False
        return {"start": rebalance_start, "end": rebalance_end,
                "did_rebalance": did_rebalance, "at_optimum": at_optimum}

    def _forward_async_optimization(self) -> dict[str, Any]:
        """Submit timing data to the background optimiser and apply any ready result.

        Only rank 0 manages the background process; results are broadcast so
        every rank rebuilds together.

        Returns:
            Dict with ``"start"``, ``"end"`` wall-clock timestamps and
            ``"did_rebalance"`` boolean.
        """
        rebalance_start = time.perf_counter()

        # Only rank 0 sends requests and polls for results
        if dist.get_rank() == 0:
            with self._force_rebalance_lock:
                force = self._force_rebalance
                self._force_rebalance = False
            if force:
                # Bypass background process — optimise directly so it takes
                # effect on the very next forward pass.
                new_config = self.pipeline_optimizer.optimize(
                    self.time_logs, self.current_config, force_rebalance=True
                )
            else:
                if not self._pending_optimization:
                    self._send_optimization_request()
                new_config = self._check_optimization_result()
            config_list = [new_config]
        else:
            config_list = [None]

        dist.broadcast_object_list(config_list, src=0)
        new_config = config_list[0]

        did_rebalance = new_config is not None
        if did_rebalance:
            self._log("Async optimization: received new config, rebuilding pipeline")
            self.time_logs = {}
            self.rebuild_pipeline(new_config)

        rebalance_end = time.perf_counter()
        at_optimum = self.pipeline_optimizer.at_optimum if dist.get_rank() == 0 else False
        return {"start": rebalance_start, "end": rebalance_end,
                "did_rebalance": did_rebalance, "at_optimum": at_optimum}

    def _trace_and_cache(self):
        """Trace the model once with all possible split points and cache the result.

        Creates a split_spec with every child (except the first) as a BEGINNING
        split point, traces the model, and stores the ExportedProgram. Subsequent
        rebuilds use a split_policy to remove unwanted pipe_split nodes from
        the cached trace instead of retracing.
        """
        children = self.pipeline_optimizer.children

        # Build max split spec: every child except first gets a split point
        max_split_spec = {}
        for i in range(1, len(children)):
            path = self.pipeline_optimizer._uuid_to_path(children[i])
            max_split_spec[path] = SplitPoint.BEGINNING

        # Annotate all split points, trace, then clean up annotations
        self._cleanup_split_annotations()
        annotate_split_points(self.original_model, max_split_spec)
        self._cached_export = Pipe._trace_with_export(
            self.original_model, (self.example_input,)
        )
        self._cleanup_split_annotations()

        # Remember which children this cache was built for
        self._cached_children = list(children)

        self._log(f"[rank:{dist.get_rank()}] cached FX trace with {len(max_split_spec)} split points")

    def _build_pipe_from_cache(self, config: PipelineConfig) -> Pipe:
        """Build a Pipe from the cached trace using a split_policy.

        Determines which pipe_split nodes to keep based on the config's split_spec,
        then calls Pipe._from_traced with a policy that removes the rest.
        """
        children = self.pipeline_optimizer.children

        # Map child paths to their index in the children list
        child_path_to_idx = {}
        for i, child_uuid in enumerate(children):
            path = self.pipeline_optimizer._uuid_to_path(child_uuid)
            child_path_to_idx[path] = i

        # Determine which pipe_split indices to keep.
        # pipe_split #(i-1) corresponds to "before children[i]" (since children[0] has no split).
        keep_indices = set()
        for path, sp in config.split_spec.items():
            if sp == SplitPoint.BEGINNING and path in child_path_to_idx:
                child_idx = child_path_to_idx[path]
                if child_idx > 0:
                    keep_indices.add(child_idx - 1)

        policy = _make_split_policy(keep_indices)
        return Pipe._from_traced(
            self.original_model,
            self._cached_export,
            split_policy=policy,
        )

    def _cleanup_split_annotations(self):
        """Remove stale ``pipe_split()`` annotations left by previous ``pipeline()`` calls.

        ``annotate_split_points()`` (called inside ``pipeline()``) mutates each
        split-point module's ``forward()`` in-place, storing the original as
        ``_orig_forward``. Without cleanup, a rebuild with a *different*
        ``split_spec`` would accumulate old markers, producing more stages than
        intended.
        """
        for _, module in self.original_model.named_modules():
            if hasattr(module, '_orig_forward'):
                module.forward = module._orig_forward
                del module._orig_forward

    def rebuild_pipeline(self, config: PipelineConfig):
        """Tear down the current pipeline and rebuild it from a new config.

        Traces the model graph, splits it according to ``config.split_spec``,
        creates ``PipelineStage`` objects with the device mapping, and
        instantiates a new ``ScheduleGPipe`` scheduler.

        If the resulting stage count doesn't match ``world_size`` (e.g. because
        a split point landed inside a parallel branch that the tracer cannot
        partition), the optimizer is reconfigured to depth-1 (top-level children
        only) and the pipeline is rebuilt with a safe split.

        Args:
            config: The split specification and device mapping to apply.

        Raises:
            RuntimeError: If the resulting number of stages does not equal
                ``world_size`` even after the depth-1 fallback.
        """
        self._log(f"[rank:{dist.get_rank()}] rebuilding pipeline...")
        self.current_config = config

        t0 = time.perf_counter()

        # Invalidate cache if children have changed (e.g. depth reconfiguration)
        children = self.pipeline_optimizer.children
        if self._cached_export is not None and self._cached_children != list(children):
            self._log("Cache invalidated: children changed")
            self._cached_export = None

        # Trace and cache on first call (or after invalidation)
        if self._cached_export is None:
            self._trace_and_cache()

        t1 = time.perf_counter()

        # Build pipe from cached trace
        self.pipe = self._build_pipe_from_cache(config)
        t2 = time.perf_counter()

        # Validate: num_stages must equal world_size for pipeline parallelism
        world_size = dist.get_world_size()
        if self.pipe.num_stages != world_size:
            self._log(
                f"Warning: split produced {self.pipe.num_stages} stage(s) instead of "
                f"{world_size}. Falling back to depth-1 (top-level) split."
            )
            # Safe config changes children → invalidate cache
            config = self.pipeline_optimizer.generate_safe_config()
            self.current_config = config
            self._cached_export = None
            self._trace_and_cache()
            self.pipe = self._build_pipe_from_cache(config)
            t2 = time.perf_counter()

        if self.pipe.num_stages != world_size:
            raise RuntimeError(
                f"Pipeline has {self.pipe.num_stages} stages but world_size is {world_size}. "
                f"PyTorch pipeline parallelism requires num_stages == world_size. "
                f"Either adjust your model split or run with --nproc_per_node={self.pipe.num_stages}"
            )

        # Create only this rank's pipeline stage (not all stages)
        rank = dist.get_rank()
        assert rank in config.device_mapping, f"Rank {rank} not in device_mapping: {config.device_mapping}"

        stage_module = self.pipe.get_stage_module(rank)
        t3 = time.perf_counter()
        self.stage = PipelineStage(
            _ContiguousStageWrapper(stage_module),
            stage_index=rank,
            num_stages=self.pipe.num_stages,
            device=config.device_mapping[rank]
        )
        t4 = time.perf_counter()

        # Create scheduler
        self.scheduler = ScheduleGPipe(self.stage, n_microbatches=self.n_microbatches)
        t5 = time.perf_counter()

        # Cache which children belong to this rank's stage for update_logs()
        self._local_children = self._compute_local_stage_children(config)

        self._log(
            f"[rank:{rank}] rebuild timing: cache={t1-t0:.4f}s  "
            f"pipe_from_cache={t2-t1:.4f}s  PipelineStage={t4-t3:.4f}s  "
            f"ScheduleGPipe={t5-t4:.4f}s  total={t5-t0:.4f}s"
        )

    def _compute_local_stage_children(self, config: PipelineConfig) -> list[uuid.UUID]:
        """Return the child UUIDs that belong to this rank's pipeline stage."""
        children = self.pipeline_optimizer.children
        split_spec = config.split_spec

        rank = dist.get_rank()
        stages: list[list[uuid.UUID]] = [[]]

        for child_uuid in children:
            child = timed_module_registry.get(child_uuid)
            child_path = child.get_path() if child is not None else None
            if child_path in split_spec and split_spec[child_path] == SplitPoint.BEGINNING:
                stages.append([])
            stages[-1].append(child_uuid)

        if rank < len(stages):
            return stages[rank]
        return []

    def update_logs(self):
        """Collect timing data from each rank's local stage and combine across all ranks.

        Each rank only has valid timing for modules in its own pipeline stage.
        Local logs are gathered via ``dist.all_gather_object`` and merged into
        ``self.time_logs``.

        Returns:
            The updated ``time_logs`` dict (maps module UUID to list of durations).
        """
        local_logs = {}
        for child_uuid in self._local_children:
            child = timed_module_registry.get(child_uuid)
            if child is not None:
                child.get_logs(local_logs)

        all_local_logs = [None] * dist.get_world_size()
        dist.all_gather_object(all_local_logs, local_logs)

        for rank_logs in all_local_logs:
            if rank_logs is None:
                continue
            for mod_uuid, times in rank_logs.items():
                if mod_uuid in self.time_logs:
                    self.time_logs[mod_uuid].extend(times)
                else:
                    self.time_logs[mod_uuid] = list(times)

        # Cap each module's log to the most recent max_log_entries measurements
        if self.max_log_entries > 0:
            for mod_uuid in self.time_logs:
                entries = self.time_logs[mod_uuid]
                if len(entries) > self.max_log_entries:
                    self.time_logs[mod_uuid] = entries[-self.max_log_entries:]

        return self.time_logs
