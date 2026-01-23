import uuid
from dataclasses import dataclass
from typing import Optional, Any

from torch.distributed.pipelining import pipeline, PipelineStage, ScheduleGPipe, SplitPoint
import torch.distributed as dist

from model_runner import TimedModule, DeviceManager, PipelineOptimizer, GreedyPipelineOptimizer


@dataclass
class PipelineConfig:
    split_spec: dict
    device_mapping: dict[int, str]


class AdaptivePipeline:
    """Manages a pytorch pipeline, and rebalances it every interval."""
    name: str
    current_config: Optional[PipelineConfig]

    # How many times to rebalance the pipeline?
    rebalance_interval: int
    # Current batch index; set back to 0 when rebalance_interval is reached
    batch_i: int
    # Threshold equal to the minimum change in a models performance that triggers rebalancing
    rebalance_threshold: int
    # Time logs
    time_logs: dict[uuid.UUID, list[float]]

    def __init__(
            self,
            model: TimedModule,
            model_name: str,
            device_manager: DeviceManager,
            pipeline_optimizer: PipelineOptimizer = None,
            rebalance_interval: int = 10,
            rebalance_threshold: int = 0.1,
            initial_pipeline_config: PipelineConfig | None = None,
            verbose: bool = False,
    ):
        self.original_model = model
        self.name = model_name
        self.device_manager = device_manager
        self.rebalance_interval = rebalance_interval
        self.rebalance_threshold = rebalance_threshold
        self.time_logs = {}
        self.batch_i = 0
        self.verbose = verbose

        self.pipeline_optimizer = pipeline_optimizer if pipeline_optimizer else GreedyPipelineOptimizer()

        # Current pipeline state
        self.current_config = None
        self.pipe = None
        self.stages = []
        self.scheduler = None

        # Initial pipeline setup
        if initial_pipeline_config is None:
            initial_pipeline_config = self._initial_pipeline_config()
        self.rebuild_pipeline(initial_pipeline_config)

    def _log(self, msg: str):
        """Print message if verbose logging is enabled."""
        if self.verbose:
            print(msg)

    def _initial_pipeline_config(self) -> PipelineConfig:
        """Generate initial balanced split."""
        children = list(self.original_model.named_children())
        num_devices = self.device_manager.num_devices()

        # Simple initial split: divide evenly across devices
        step = max(1, len(children) // num_devices)
        split_points = [children[i * step][0] for i in range(1, num_devices)]

        split_spec = {name: SplitPoint.BEGINNING for name in split_points}
        device_mapping = {i: self.device_manager.get_device(i) for i in range(num_devices)}

        return PipelineConfig(split_spec=split_spec, device_mapping=device_mapping)

    def forward(self, x: Any) -> Any:
        rank = dist.get_rank()

        # Only the first thread gets the input
        if rank == 0:
            output = self.scheduler.step(x)
        else:
            output = self.scheduler.step()

        self.update_logs()

        if self.batch_i != 0 and self.batch_i % self.rebalance_interval == 0:
            self.batch_i = 0
            if self.pipeline_optimizer.should_rebalance(self.time_logs, self.current_config):
                self.optimize_pipeline()

        return output #TODO: output before you rebalance

    def optimize_pipeline(self):
        """Runs the pipeline optimiser to generate a new pipeline config."""
        # TODO: make this distributed, in theory it should be possible

        # Create the new config
        new_config = self.pipeline_optimizer.optimize(self.time_logs, self.current_config)

        # Reset logs
        self.time_logs = {}

        # Rebuild pipeline
        self.rebuild_pipeline(new_config)

    def rebuild_pipeline(self, config: PipelineConfig):
        """Create a new pipeline from config."""
        self.current_config = config

        # Create pipe
        self.pipe = pipeline(
            module=self.original_model,
            mb_args=(self.original_model.rand_inputs(),),
            split_spec=config.split_spec
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

        # Create scheduler ??
        self.scheduler = ScheduleGPipe(self.stages[dist.get_rank()], n_microbatches=4)

    def update_logs(self):
        """Updates self.time_logs and returns them."""
        # TODO: does this actually work after the pipeline stuff?
        return self.original_model.get_logs(self.time_logs)
