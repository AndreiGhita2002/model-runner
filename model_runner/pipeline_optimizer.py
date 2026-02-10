import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from torch.distributed.pipelining import SplitPoint

from . import TimedModule, DeviceManager
from .timed_module import timed_module_hierarchy


@dataclass
class PipelineConfig:
    split_spec: dict
    device_mapping: dict[int, str]


class PipelineOptimizer(ABC):
    """Abstract base class for pipeline optimisers."""

    def __init__(self, num_stages: int, root_uuid: uuid.UUID, device_manager: DeviceManager):
        self.num_stages = num_stages
        self.device_manager = device_manager
        self.root_uuid = root_uuid

    @abstractmethod
    def optimize(self, time_logs: dict[uuid.UUID, list[float]], old_config: PipelineConfig) -> PipelineConfig:
        """
        Optimise pipeline configuration based on timing data.

        Args:
            time_logs: Dict mapping TimedModule UUIDs to lists of elapsed times
            old_config: The current pipeline configuration

        Returns:
            New PipelineConfig with optimised split points
        """
        pass

    @abstractmethod
    def should_rebalance(self, time_logs: dict[uuid.UUID, list[float]], current_config: PipelineConfig) -> bool:
        """
        Determine if rebalancing is needed based on timing data.

        Args:
            time_logs: Dict mapping TimedModule UUIDs to lists of elapsed times
            current_config: The current pipeline configuration

        Returns:
            True if the pipeline should be rebalanced
        """
        pass

    def initial_setup(self) -> PipelineConfig:
        """Generate a uniform initial split across all ranks.

        Divides the model's top-level children evenly into ``world_size`` stages
        and assigns each stage to a device via round-robin.

        Returns:
            A ``PipelineConfig`` with a balanced split spec and device mapping.
        """

        # Making the split spec
        # SplitPoint.BEGINNING means start a stage before this one, so we cannot mark the first module with it
        # because the first module is already the start of a stage implicitly

        children_uuid = timed_module_hierarchy[self.root_uuid]
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
        device_mapping = {i: self.device_manager.get_device(i % num_devices) for i in range(len(split_spec) + 1)}

        return PipelineConfig(split_spec=split_spec, device_mapping=device_mapping)


class GreedyPipelineOptimizer(PipelineOptimizer):
    """
    Pipeline optimiser using a greedy algorithm to balance computation time across stages.

    TODO: this optimizer is pretty dumb
    """

    def __init__(self, num_stages: int, root_uuid: uuid.UUID, device_manager: DeviceManager, rebalance_threshold: float = 0.1):
        super().__init__(num_stages, root_uuid, device_manager)
        self.rebalance_threshold = rebalance_threshold

    def should_rebalance(self, time_logs: dict[uuid.UUID, list[float]], current_config: PipelineConfig) -> bool:
        """
        Determine if rebalancing is needed based on timing profile changes.

        Compares the first and last elements from the current logs to detect drift.
        Returns True if timing distribution has changed by more than the threshold.
        """
        if not time_logs:
            return True

        # Build profiles from first and last elements of each time list
        first_profile = {}
        last_profile = {}

        for module_uuid, times in time_logs.items():
            if isinstance(times, list) and len(times) >= 2:
                first_profile[module_uuid] = times[0]
                last_profile[module_uuid] = times[-1]

        if not first_profile:
            return True

        # Calculate total time for normalisation
        first_total = sum(first_profile.values())
        last_total = sum(last_profile.values())

        if first_total == 0 or last_total == 0:
            return True

        # Compare normalised timing distributions
        max_change = 0.0

        for module_uuid in first_profile:
            first_ratio = first_profile[module_uuid] / first_total
            last_ratio = last_profile[module_uuid] / last_total
            change = abs(last_ratio - first_ratio)
            max_change = max(max_change, change)

        return max_change > self.rebalance_threshold

    def optimize(self, time_logs: dict[uuid.UUID, list[float]], old_config: PipelineConfig) -> PipelineConfig:
        """
        Optimise pipeline configuration based on timing data.
        Uses a greedy algorithm to balance computation time across stages.

        Args:
            time_logs: Dict mapping TimedModule UUIDs to lists of elapsed times
            old_config: The current pipeline configuration

        Returns:
            New PipelineConfig with optimized split points
        """
        if not time_logs:
            return old_config

        # Build list of (module_name, avg_time) tuples from time_logs
        layer_times = self._extract_layer_times(time_logs)

        if not layer_times:
            return old_config

        # Calculate target time per stage
        total_time = sum(t for _, t in layer_times)
        if total_time == 0:
            return old_config

        target_time_per_stage = total_time / self.num_stages

        # Greedily assign layers to stages
        stage_assignments = self._assign_layers_to_stages(layer_times, target_time_per_stage)

        # Build new split_spec: mark where stages change with SplitPoint.BEGINNING
        split_spec = self._build_split_spec(stage_assignments)

        # Preserve device mapping from old config
        device_mapping = old_config.device_mapping

        return PipelineConfig(split_spec=split_spec, device_mapping=device_mapping)

    def _extract_layer_times(self, time_logs: dict[uuid.UUID, list[float]]) -> List[Tuple[uuid.UUID, float]]:
        """
        Extract UUIDs and average times from time logs.

        Returns:
            List of (uuid, avg_time) tuples
        """
        layer_times = []

        for module_uuid, times in time_logs.items():
            # Calculate average time (or use latest if only one)
            if isinstance(times, list) and len(times) > 0:
                avg_time = sum(times) / len(times)
            elif isinstance(times, (int, float)):
                avg_time = float(times)
            else:
                avg_time = 0.0

            layer_times.append((module_uuid, avg_time))

        return layer_times

    def _assign_layers_to_stages(
        self,
        layer_times: List[Tuple[uuid.UUID, float]],
        target_time_per_stage: float
    ) -> List[Tuple[uuid.UUID, int]]:
        """
        Greedily assign layers to stages to balance computation time.
        Only works on the children of the root module (self.root_uuid).

        Returns:
            List of (uuid, stage_index) tuples
        """
        # Collecting the times of only the children of the root module
        children_ids = timed_module_hierarchy[self.root_uuid]
        children_times = []
        for layer_time in layer_times:
            # If module is a child of the root
            if layer_time[0] in children_ids:
                children_times.append(layer_time)

        print(children_times)

        # The assignment algorithm:
        assignments = []
        current_stage_time = 0.0
        current_stage = 0
        for i, (module_uuid, time) in enumerate(children_times):
            assignments.append((module_uuid, current_stage))
            current_stage_time += time

            # Move to next stage if we've exceeded target and not on last stage
            if (current_stage_time >= target_time_per_stage and
                current_stage < self.num_stages - 1 and
                i < len(children_times) - 1):

                current_stage += 1
                current_stage_time = 0.0

        return assignments

    def _build_split_spec(self, stage_assignments: List[Tuple[uuid.UUID, int]]) -> dict[uuid.UUID, SplitPoint]:
        """
        Build split_spec dict marking stage boundaries with SplitPoint.BEGINNING.

        Returns:
            Dict mapping UUIDs to SplitPoint.BEGINNING for stage boundaries
        """
        split_spec = {}

        prev_stage = 0
        for module_uuid, stage in stage_assignments:
            # Mark the beginning of a new stage (skip stage 0, as it's implicit)
            if stage > prev_stage:
                split_spec[module_uuid] = SplitPoint.BEGINNING
                prev_stage = stage

        return split_spec
