import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from torch.distributed.pipelining import SplitPoint

import torch

from .timed_module import timed_module_hierarchy, timed_module_registry
from .device_manager import DeviceManager


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

class TimeBasedShishaPipelineOptimizer(PipelineOptimizer):
    """
    Pipeline optimiser based on the Shisha paper (Soomro et al., PPAM 2022).

    Uses a two-phase approach:
    1. Seed Generation (Algorithm 1): merges children into balanced groups using
       parameter count as weight proxy, then assigns stages to devices via ranking.
    2. Online Tuning (Algorithm 2): iteratively moves children from the slowest
       stage toward lighter stages, with patience parameter alpha.
    """

    #TODO(naming): TimeBasedShishaPipelineOptimizer is an awful name

    def __init__(self, num_stages: int, root_uuid: uuid.UUID, device_manager: DeviceManager,
                 alpha: int = 10,
                 assignment_choice: str = "rank_w",
                 balance_strategy: str = "nearest_lightest_fep"):

        super().__init__(num_stages, root_uuid, device_manager)
        self.n = self.device_manager.num_devices()
        self.alpha = alpha
        self.assignment_choice = assignment_choice
        self.balance_strategy = balance_strategy

        # Persistent state for online tuning across optimize() calls
        self._gamma = 0                    # non-improving iteration counter
        self._best_throughput = 0.0        # best 1/max_stage_time seen
        self._is_seeded = False            # whether seed has been used at least once

    # ── Helpers ──────────────────────────────────────────────────────────

    def _children_to_stages(self, config: PipelineConfig) -> list[list[uuid.UUID]]:
        """Reconstruct which children are in which stage from the split_spec."""
        children = timed_module_hierarchy[self.root_uuid]
        stages: list[list[uuid.UUID]] = [[]]
        for child_uuid in children:
            if child_uuid in config.split_spec and config.split_spec[child_uuid] == SplitPoint.BEGINNING:
                stages.append([])
            stages[-1].append(child_uuid)
        return stages

    def _compute_stage_times(self, time_logs: dict[uuid.UUID, list[float]],
                             config: PipelineConfig) -> list[float]:
        """Compute per-stage times by summing average child times."""
        stages = self._children_to_stages(config)
        stage_times = []
        for stage in stages:
            total = 0.0
            for child_uuid in stage:
                times = time_logs.get(child_uuid, [])
                if times:
                    total += sum(times) / len(times)
            stage_times.append(total)
        return stage_times

    def _get_child_weight(self, child_uuid: uuid.UUID) -> int:
        """Return parameter count for a child module (proxy for computational weight)."""
        module = timed_module_registry.get(child_uuid)
        if module is None:
            return 1
        count = sum(p.numel() for p in module.inner().parameters())
        return count if count > 0 else 1

    def _rank_devices(self) -> list[torch.device]:
        """Sort devices in descending performance order (CUDA first, then CPU)."""
        devices = self.device_manager.get_all_devices()
        # CUDA devices first (lower index = faster convention), then CPU
        cuda_devs = sorted([d for d in devices if d.type == "cuda"],
                           key=lambda d: d.index if d.index is not None else 0)
        cpu_devs = [d for d in devices if d.type != "cuda"]
        return cuda_devs + cpu_devs

    # ── Seed Generation (Algorithm 1) ────────────────────────────────────

    def _seed_generation(self) -> PipelineConfig:
        children = list(timed_module_hierarchy[self.root_uuid])
        N = self.num_stages

        # Phase 1: Merge children into N balanced groups by weight
        groups: list[tuple[list[uuid.UUID], int]] = [
            ([c], self._get_child_weight(c)) for c in children
        ]

        while len(groups) > N:
            # Find lightest group
            min_idx = min(range(len(groups)), key=lambda i: groups[i][1])

            # Find lightest immediate neighbor
            neighbors = []
            if min_idx > 0:
                neighbors.append(min_idx - 1)
            if min_idx < len(groups) - 1:
                neighbors.append(min_idx + 1)
            merge_idx = min(neighbors, key=lambda i: groups[i][1])

            # Merge: ensure correct order (lower index first)
            lo, hi = sorted([min_idx, merge_idx])
            merged_uuids = groups[lo][0] + groups[hi][0]
            merged_weight = groups[lo][1] + groups[hi][1]
            groups[lo] = (merged_uuids, merged_weight)
            del groups[hi]

        # Phase 2: Assign devices to stages via ranking
        ranked_devices = self._rank_devices()

        if self.assignment_choice == "rank_w":
            # Heaviest stage → fastest device
            stage_order = sorted(range(len(groups)), key=lambda i: groups[i][1], reverse=True)
        elif self.assignment_choice == "rank_l":
            # Most children → slowest device (SEP)
            stage_order = sorted(range(len(groups)), key=lambda i: len(groups[i][0]), reverse=True)
        else:
            stage_order = list(range(len(groups)))

        num_devices = len(ranked_devices)
        stage_to_device: dict[int, torch.device] = {}
        for rank, stage_idx in enumerate(stage_order):
            stage_to_device[stage_idx] = ranked_devices[rank % num_devices]

        # Build PipelineConfig
        split_spec: dict[uuid.UUID, SplitPoint] = {}
        device_mapping: dict[int, torch.device] = {}
        for stage_idx, (uuids, _weight) in enumerate(groups):
            if stage_idx > 0:
                split_spec[uuids[0]] = SplitPoint.BEGINNING
            device_mapping[stage_idx] = stage_to_device[stage_idx]

        return PipelineConfig(split_spec=split_spec, device_mapping=device_mapping)

    # ── Online Tuning (Algorithm 2) ──────────────────────────────────────

    def _online_tuning(self, time_logs: dict[uuid.UUID, list[float]],
                       old_config: PipelineConfig) -> PipelineConfig:
        """One iteration of online tuning: move one child from the slowest stage."""
        stage_times = self._compute_stage_times(time_logs, old_config)
        stages = self._children_to_stages(old_config)

        slowest_idx = max(range(len(stage_times)), key=lambda i: stage_times[i])

        # Can't move if slowest stage has only 1 child
        if len(stages[slowest_idx]) <= 1:
            return old_config

        # Find target stage
        target_idx = self._find_target_stage(stage_times, stages, old_config, slowest_idx)

        if target_idx is None or target_idx == slowest_idx:
            return old_config

        # If target is non-adjacent, step toward it
        if target_idx < slowest_idx:
            adjacent_idx = slowest_idx - 1
        else:
            adjacent_idx = slowest_idx + 1

        # Move one child from slowest to adjacent (in the direction of target)
        if adjacent_idx < slowest_idx:
            # Move first child of slowest to end of adjacent (left neighbor)
            child_to_move = stages[slowest_idx][0]
            stages[adjacent_idx].append(child_to_move)
            stages[slowest_idx] = stages[slowest_idx][1:]
        else:
            # Move last child of slowest to beginning of adjacent (right neighbor)
            child_to_move = stages[slowest_idx][-1]
            stages[adjacent_idx].insert(0, child_to_move)
            stages[slowest_idx] = stages[slowest_idx][:-1]

        # Build new split_spec from modified stages
        new_split_spec: dict[uuid.UUID, SplitPoint] = {}
        for stage_idx, stage_uuids in enumerate(stages):
            if stage_idx > 0 and stage_uuids:
                new_split_spec[stage_uuids[0]] = SplitPoint.BEGINNING

        return PipelineConfig(split_spec=new_split_spec, device_mapping=old_config.device_mapping)

    def _find_target_stage(self, stage_times: list[float], stages: list[list[uuid.UUID]],
                           config: PipelineConfig, slowest_idx: int) -> int | None:
        """Find the target stage to move a child toward, per balance_strategy."""
        num_stages = len(stage_times)
        if num_stages <= 1:
            return None

        if self.balance_strategy == "nearest_lightest_fep":
            # Among all stages != slowest, find one with lowest time,
            # preferring CUDA ("FEP") stages, breaking ties by proximity.
            cuda_candidates = []
            all_candidates = []
            for i in range(num_stages):
                if i == slowest_idx:
                    continue
                device = config.device_mapping.get(i)
                is_cuda = device is not None and device.type == "cuda"
                proximity = abs(i - slowest_idx)
                entry = (stage_times[i], proximity, i)
                all_candidates.append(entry)
                if is_cuda:
                    cuda_candidates.append(entry)

            candidates = cuda_candidates if cuda_candidates else all_candidates
            if not candidates:
                return None
            # Sort by (time, proximity) and pick best
            candidates.sort(key=lambda e: (e[0], e[1]))
            return candidates[0][2]

        elif self.balance_strategy == "nearest_fep":
            # Nearest CUDA stage; fallback to nearest overall.
            cuda_candidates = []
            all_candidates = []
            for i in range(num_stages):
                if i == slowest_idx:
                    continue
                device = config.device_mapping.get(i)
                is_cuda = device is not None and device.type == "cuda"
                proximity = abs(i - slowest_idx)
                entry = (proximity, i)
                all_candidates.append(entry)
                if is_cuda:
                    cuda_candidates.append(entry)

            candidates = cuda_candidates if cuda_candidates else all_candidates
            if not candidates:
                return None
            candidates.sort()
            return candidates[0][1]

        else:
            # Default: lightest stage overall
            best_idx = None
            best_time = float('inf')
            for i in range(num_stages):
                if i == slowest_idx:
                    continue
                if stage_times[i] < best_time:
                    best_time = stage_times[i]
                    best_idx = i
            return best_idx

    # ── Public interface ─────────────────────────────────────────────────

    def initial_setup(self) -> PipelineConfig:
        config = self._seed_generation()
        self._is_seeded = True
        return config

    def should_rebalance(self, time_logs: dict[uuid.UUID, list[float]],
                         current_config: PipelineConfig) -> bool:
        if not time_logs:
            return True

        stage_times = self._compute_stage_times(time_logs, current_config)

        if any(t == 0 for t in stage_times):
            return True

        throughput = 1.0 / max(stage_times)

        if self._best_throughput == 0.0:
            self._best_throughput = throughput
            return True

        if throughput > self._best_throughput:
            self._best_throughput = throughput
            self._gamma = 0
            return True
        else:
            self._gamma += 1
            if self._gamma >= self.alpha:
                return False
            return True

    def optimize(self, time_logs: dict[uuid.UUID, list[float]],
                 old_config: PipelineConfig) -> PipelineConfig:
        return self._online_tuning(time_logs, old_config)
