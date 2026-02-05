import queue
import uuid
import warnings
import multiprocessing as mp
from typing import Optional, Any

import torch

# Suppress PyTorch internal FutureWarning about LeafSpec deprecation
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=FutureWarning)

from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe, SplitPoint, Pipe
import torch.distributed as dist
from torch.distributed.pipelining.schedules import PipelineScheduleSingle, PipelineScheduleMulti

from .timed_module import TimedModule, timed_module_registry, timed_module_hierarchy
from .device_manager import DeviceManager
from .pipeline_optimizer import PipelineOptimizer, GreedyPipelineOptimizer, PipelineConfig


def _optimizer_process_worker(
    optimizer: PipelineOptimizer,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
):
    """
    Background worker process that runs the pipeline optimiser.

    Listens for optimisation requests and sends back new configs.
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
    if isinstance(obj, torch.Tensor):
        return obj.shape
    elif isinstance(obj, (tuple, list)):
        return type(obj)(extract_shapes(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: extract_shapes(v) for k, v in obj.items()}
    else:
        return None


class AdaptivePipeline:
    """Manages a pytorch pipeline, and rebalances it every interval."""
    name: str
    current_config: Optional[PipelineConfig]
    pipe: Pipe | None
    scheduler: PipelineScheduleSingle | PipelineScheduleMulti | None

    # How many times to rebalance the pipeline?
    rebalance_interval: int
    # Current batch index; set back to 0 when rebalance_interval is reached
    batch_i: int
    # Threshold equal to the minimum change in a models performance that triggers rebalancing
    rebalance_threshold: float
    # Time logs
    time_logs: dict[uuid.UUID, list[float]]

    #TODO: we need to know the size of the output, and only use it if the model is static

    def __init__(
            self,
            model: TimedModule,
            model_name: str,
            device_manager: DeviceManager,
            example_input: Any,
            model_output_is_static: bool,
            pipeline_optimizer: PipelineOptimizer = None,
            rebalance_interval: int = 10,
            rebalance_threshold: float = 0.1,
            n_microbatches: int = 4,
            initial_pipeline_config: PipelineConfig | None = None,
            verbose: bool = False,
            async_optimization: bool = False,
    ):
        self.original_model = model
        self.name = model_name
        self.device_manager = device_manager
        self.example_input = example_input
        self.model_output_is_static = model_output_is_static
        self.rebalance_interval = rebalance_interval
        self.rebalance_threshold = rebalance_threshold
        self.time_logs = {}
        self.batch_i = 0
        self.verbose = verbose
        self.async_optimization = async_optimization
        self.num_stages = dist.get_world_size()
        self.n_microbatches = n_microbatches if n_microbatches >= self.num_stages else self.num_stages

        if model_output_is_static:
            self._dummy_run()

        self.pipeline_optimizer = pipeline_optimizer if pipeline_optimizer else GreedyPipelineOptimizer(
            root_uuid=model.uuid,
            num_stages=self.num_stages,
            rebalance_threshold=rebalance_threshold,
        )

        # Current pipeline state
        self.current_config = None
        self.pipe = None
        self.stages = []
        self.scheduler = None

        # Async optimization state
        self._optimizer_process: Optional[mp.Process] = None
        self._request_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._shutdown_event: Optional[mp.Event] = None
        self._pending_optimization: bool = False

        if self.async_optimization:
            self._start_optimizer_process()

        # Initial pipeline setup
        if initial_pipeline_config is None:
            initial_pipeline_config = self._initial_pipeline_config()
        self.rebuild_pipeline(initial_pipeline_config)

    def _log(self, msg: str):
        """Print message if verbose logging is enabled."""
        if self.verbose:
            print(msg)

    def _start_optimizer_process(self):
        """Start the background optimiser process."""
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
        """Stop the background optimiser process."""
        if self._optimizer_process is not None:
            self._shutdown_event.set()
            self._optimizer_process.join(timeout=1.0)
            if self._optimizer_process.is_alive():
                self._optimizer_process.terminate()
            self._optimizer_process = None
            self._log("Stopped optimizer process")

    def _send_optimization_request(self):
        """Send current state to the optimiser process (non-blocking)."""
        if not self._pending_optimization:
            self._request_queue.put((self.time_logs.copy(), self.current_config))
            self._pending_optimization = True
            self._log("Sent optimization request to background process")

    def _check_optimization_result(self) -> Optional[PipelineConfig]:
        """Check if the optimiser has a result ready (non-blocking)."""
        try:
            result = self._result_queue.get_nowait()
            self._pending_optimization = False
            return result
        except queue.Empty:
            return None

    def shutdown(self):
        """Clean up resources."""
        self._stop_optimizer_process()

    def _initial_pipeline_config(self) -> PipelineConfig:
        """Generate initial balanced split."""

        # Making the split spec
        # SplitPoint.BEGINNING means start a stage before this one, so we cannot mark the first module with it
        # because the first module is already the start of a stage implicitly
        children_uuid = timed_module_hierarchy[self.original_model.uuid]
        step = max(len(children_uuid) // self.num_stages, 1)
        split_spec = {}
        current_stage_num = 1
        for i in range(step, len(children_uuid), step):
            # new split point
            u = children_uuid[i]
            split_spec[u] = SplitPoint.BEGINNING
            current_stage_num += 1
            # we have enough stages
            if current_stage_num == self.num_stages:
                break

        # Making the device mapping
        num_devices = self.device_manager.num_devices()
        device_mapping = {i: self.device_manager.get_device(i % num_devices) for i in range(len(split_spec)+1)}

        return PipelineConfig(split_spec=split_spec, device_mapping=device_mapping)

    def forward(self, x: Any) -> Any:
        rank = dist.get_rank()
        print(f"rank:{rank} in forward")

        # Only the first rank gets the input
        with torch.no_grad():
            if rank == 0:
                output = self.scheduler.step(x)
            else:
                output = self.scheduler.step()

        self.update_logs()  #todo: this probably does not work in a parallel context
        self.batch_i += 1

        if self.async_optimization:
            self._forward_async_optimization()
        else:
            self._forward_sync_optimization()

        return output

    def _forward_sync_optimization(self):
        """Synchronous optimization: blocks during rebalancing."""
        if self.batch_i != 0 and self.batch_i % self.rebalance_interval == 0:
            self.batch_i = 0
            if self.pipeline_optimizer.should_rebalance(self.time_logs, self.current_config):
                self._log("Sync optimization: rebalancing pipeline")
                new_config = self.pipeline_optimizer.optimize(self.time_logs, self.current_config)
                self.time_logs = {}
                dist.barrier()
                self.rebuild_pipeline(new_config)

    def _forward_async_optimization(self):
        """Async optimization: runs optimiser in background process."""
        # Check if we should send a new optimisation request
        if self.batch_i != 0 and self.batch_i % self.rebalance_interval == 0:
            self.batch_i = 0
            self._send_optimization_request()

        # Check if a result is ready (non-blocking)
        new_config = self._check_optimization_result()
        if new_config is not None:
            self._log("Async optimization: received new config, rebuilding pipeline")
            self.time_logs = {}
            dist.barrier()
            self.rebuild_pipeline(new_config)

    def rebuild_pipeline(self, config: PipelineConfig):
        """Create a new pipeline from config."""
        print(f"[rank:{dist.get_rank()}] rebuilding pipeline...")
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

            # Create the pipeline stage
            stage = PipelineStage(
                self.pipe.get_stage_module(i),
                stage_index=i,
                num_stages=self.pipe.num_stages,
                device=config.device_mapping[i]
            )
            self.stages.append(stage)

        # Create scheduler
        self.scheduler = ScheduleGPipe(self.stages[dist.get_rank()], n_microbatches=self.n_microbatches)

    def update_logs(self):
        """Updates self.time_logs and returns them."""
        return self.original_model.get_logs(self.time_logs)

    def get_output_size(self):
        """Returns the size of the output if the output is static, and if not it will return None.
           Tell the pipeline if the model is static or not with the `model_output_is_static` field."""
        if self.model_output_is_static:
            return self.output_size
        else:
            return None

    def _dummy_run(self):
        """Runs a dummy run of the model to set `output_size`.
           Only works before the pipeline has been built and if `model_output_is_static` is True."""
        if not self.model_output_is_static:
            raise ValueError("AdaptivePipeline._dummy_run() was called when model output is not static.")

        # TODO: you can also use the dummy run to get initial timing data

        output = self.original_model(self.example_input)
        self.output_size = extract_shapes(output)
