import multiprocessing as mp
import queue
import time
import uuid
import warnings
from typing import Optional, Any

import torch

# Suppress PyTorch internal FutureWarning about LeafSpec deprecation
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=FutureWarning)

from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe, SplitPoint, Pipe
import torch.distributed as dist
from torch.distributed.pipelining.schedules import PipelineScheduleSingle, PipelineScheduleMulti

from .timed_module import TimedModule, timed_module_registry, timed_module_hierarchy
from .device_manager import DeviceManager
from .pipeline_optimizer import PipelineOptimizer, PipelineConfig, TimeBasedShishaPipelineOptimizer


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

        # Check if we should rebalance
        if optimizer.should_rebalance(time_logs, current_config):
            new_config = optimizer.optimize(time_logs, current_config)
            result_queue.put(new_config)
        else:
            # Signal that we checked but no rebalance needed
            result_queue.put(None)

def extract_shapes(obj):
    """Recursively extract tensor shapes from a nested structure.

    Args:
        obj: A tensor, or a nested tuple/list/dict of tensors.

    Returns:
        The same nested structure with tensors replaced by their ``torch.Size``,
        or ``None`` for non-tensor leaves.
    """
    if isinstance(obj, torch.Tensor):
        return obj.shape
    elif isinstance(obj, (tuple, list)):
        return type(obj)(extract_shapes(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: extract_shapes(v) for k, v in obj.items()}
    else:
        return None


class AdaptivePipeline:
    """Manages a PyTorch pipeline with automatic stage rebalancing.

    Wraps a ``TimedModule`` in a ``ScheduleGPipe`` pipeline and periodically
    re-optimises the stage split based on collected timing data. Rebalancing
    can run synchronously (blocking) or asynchronously (in a background process).

    Requires ``torch.distributed`` to be initialized before use.
    """
    name: str
    current_config: Optional[PipelineConfig]
    pipe: Pipe | None
    scheduler: PipelineScheduleSingle | PipelineScheduleMulti | None

    # Maximum number of timing measurements to keep per module
    max_log_entries: int
    # Time logs
    time_logs: dict[uuid.UUID, list[float]]

    def __init__(
            self,
            model: TimedModule,
            model_name: str,
            device_manager: DeviceManager,
            example_input: Any,
            optimizer_class: type[PipelineOptimizer] = TimeBasedShishaPipelineOptimizer,
            optimizer_kwargs: dict | None = None,
            max_log_entries: int = 5,
            n_microbatches: int = 4,
            initial_pipeline_config: PipelineConfig | None = None,
            verbose: bool = False,
            async_optimization: bool = False,
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
            **(optimizer_kwargs or {}),
        )

        # Current pipeline state
        self.current_config = None
        self.pipe = None
        self.stages = []
        self.scheduler = None

        # Force-rebalance flag (set from Flask thread, read from main loop)
        self._force_rebalance: bool = False

        # Async optimization state (only used on rank 0)
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
        """Spawn the background optimiser process and its communication queues."""
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
            - ``"timing"``: On the last rank, dict with ``"start"`` and ``"end"``
              wall-clock timestamps; on other ranks, ``None``.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        last_rank = world_size - 1
        self._log(f"rank:{rank} in forward")

        # Capture wall-clock start time on rank 0
        if rank == 0:
            start_time = time.time()

        # Only the first rank gets the input
        with torch.no_grad():
            if rank == 0:
                output = self.scheduler.step(x)
            else:
                output = self.scheduler.step()

        self.update_logs()

        # Send start_time from rank 0 to last rank
        if world_size > 1:
            if rank == 0:
                t = torch.tensor([start_time], dtype=torch.float64)
                dist.send(t, dst=last_rank)
            elif rank == last_rank:
                t = torch.tensor([0.0], dtype=torch.float64)
                dist.recv(t, src=0)
                start_time = t.item()

        # Capture end time on last rank
        timing = None
        if rank == last_rank:
            end_time = time.time()
            timing = {"start": start_time, "end": end_time}

        if self.async_optimization:
            self._forward_async_optimization()
        else:
            self._forward_sync_optimization()

        return {"output": output, "timing": timing}

    def _forward_sync_optimization(self):
        """Check for rebalance and apply it synchronously.

        Only rank 0 runs the optimiser; the decision and new config are
        broadcast so every rank takes the same path (no conditional collectives).
        """
        # Rank 0 decides and computes; others receive via broadcast
        if dist.get_rank() == 0:
            force = self._force_rebalance
            self._force_rebalance = False
            if force or self.pipeline_optimizer.should_rebalance(self.time_logs, self.current_config):
                new_config = self.pipeline_optimizer.optimize(self.time_logs, self.current_config)
            else:
                new_config = None
            config_list = [new_config]
        else:
            config_list = [None]

        dist.broadcast_object_list(config_list, src=0)
        new_config = config_list[0]

        if new_config is not None:
            self._log("Sync optimization: rebalancing pipeline")
            self.time_logs = {}
            self.rebuild_pipeline(new_config)

    def _forward_async_optimization(self):
        """Submit timing data to the background optimiser and apply any ready result.

        Only rank 0 manages the background process; results are broadcast so
        every rank rebuilds together.
        """
        # Only rank 0 sends requests and polls for results
        if dist.get_rank() == 0:
            force = self._force_rebalance
            self._force_rebalance = False
            if force:
                # Bypass background process — optimise directly so it takes
                # effect on the very next forward pass.
                new_config = self.pipeline_optimizer.optimize(self.time_logs, self.current_config)
            else:
                if not self._pending_optimization:
                    self._send_optimization_request()
                new_config = self._check_optimization_result()
            config_list = [new_config]
        else:
            config_list = [None]

        dist.broadcast_object_list(config_list, src=0)
        new_config = config_list[0]

        if new_config is not None:
            self._log("Async optimization: received new config, rebuilding pipeline")
            self.time_logs = {}
            self.rebuild_pipeline(new_config)

    def rebuild_pipeline(self, config: PipelineConfig):
        """Tear down the current pipeline and rebuild it from a new config.

        Traces the model graph, splits it according to ``config.split_spec``,
        creates ``PipelineStage`` objects with the device mapping, and
        instantiates a new ``ScheduleGPipe`` scheduler.

        Args:
            config: The split specification and device mapping to apply.

        Raises:
            RuntimeError: If the resulting number of stages does not equal ``world_size``.
        """
        self._log(f"[rank:{dist.get_rank()}] rebuilding pipeline...")
        self.current_config = config

        # Convert UUID-based split_spec to path-based for PyTorch
        path_split_spec = {}
        for module_uuid, split_point in config.split_spec.items():
            timed_module = timed_module_registry.get(module_uuid)
            if timed_module is not None:
                path_split_spec[timed_module.get_path()] = split_point

        # Create pipe
        self.pipe = pipeline(
            module=self.original_model,
            mb_args=(self.example_input,),
            split_spec=path_split_spec
        )

        # Validate: num_stages must equal world_size for pipeline parallelism
        world_size = dist.get_world_size()
        if self.pipe.num_stages != world_size:
            raise RuntimeError(
                f"Pipeline has {self.pipe.num_stages} stages but world_size is {world_size}. "
                f"PyTorch pipeline parallelism requires num_stages == world_size. "
                f"Either adjust your model split or run with --nproc_per_node={self.pipe.num_stages}"
            )

        # Create stages with device mapping
        self.stages = []
        for i in range(self.pipe.num_stages):
            # If `i` is not in device mapping, then it is incomplete and something went wrong
            assert i in config.device_mapping, f"Stage {i} not in device_mapping: {config.device_mapping}"

            # Wrap stage submodule to ensure outputs are contiguous for P2P communication
            stage_module = self.pipe.get_stage_module(i)
            stage = PipelineStage(
                _ContiguousStageWrapper(stage_module),
                stage_index=i,
                num_stages=self.pipe.num_stages,
                device=config.device_mapping[i]
            )
            self.stages.append(stage)

        # Create scheduler
        self.scheduler = ScheduleGPipe(self.stages[dist.get_rank()], n_microbatches=self.n_microbatches)

        # Cache which children belong to this rank's stage for update_logs()
        self._local_children = self._compute_local_stage_children(config)

    def _compute_local_stage_children(self, config: PipelineConfig) -> list[uuid.UUID]:
        """Return the child UUIDs that belong to this rank's pipeline stage."""
        children = timed_module_hierarchy[self.original_model.uuid]
        split_spec = config.split_spec

        rank = dist.get_rank()
        stages: list[list[uuid.UUID]] = [[]]

        for child_uuid in children:
            if child_uuid in split_spec and split_spec[child_uuid] == SplitPoint.BEGINNING:
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
