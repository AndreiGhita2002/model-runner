import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from torch.distributed.pipelining import SplitPoint

import torch

from .timed_module import timed_module_hierarchy, timed_module_registry
from .device_manager import DeviceManager
from .util import gpipe_split_spec

def nth_largest_index(arr: list, n: int):
    """
    Returns the index of the nth largest element in arr (0-indexed).
    n=0 returns the index of the largest element, n=1 the second largest, etc.
    Uses the Quickselect algorithm — O(n) average time.
    """
    if not 0 <= n < len(arr):
        raise ValueError(f"n={n} is out of range for list of length {len(arr)}")

    def select(pairs, k):
        pivot_val = pairs[0][1]
        low  = [(i, v) for i, v in pairs if v <  pivot_val]
        mid  = [(i, v) for i, v in pairs if v == pivot_val]
        high = [(i, v) for i, v in pairs if v >  pivot_val]
        if k <= len(low):
            return select(low, k)
        elif k <= len(low) + len(mid):
            return mid[0][0]
        else:
            return select(high, k - len(low) - len(mid))

    return select(list(enumerate(arr)), len(arr) - n)


@dataclass
class PipelineConfig:
    split_spec: dict[str, SplitPoint]  # module path → SplitPoint
    device_mapping: dict[int, torch.device]

    def save(self, path: str | Path):
        """Save config to a JSON file."""
        data = {
            "split_spec": {k: v.name for k, v in self.split_spec.items()},
            "device_mapping": {str(k): str(v) for k, v in self.device_mapping.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        split_spec = {k: SplitPoint[v] for k, v in data["split_spec"].items()}
        device_mapping = {int(k): torch.device(v) for k, v in data["device_mapping"].items()}
        return cls(split_spec=split_spec, device_mapping=device_mapping)


#TODO: move the pipeline optimizer used for baselines into tests module maybe

class PipelineOptimizer(ABC):
    """Abstract base class for pipeline optimisers.

    Subclasses implement ``optimize`` which receives timing data and the current
    config, and returns a new ``PipelineConfig`` if rebalancing is warranted or
    ``None`` otherwise. The ``force_rebalance`` flag lets callers bypass the
    subclass's internal rebalance criteria.
    """

    def __init__(self, num_stages: int, root_uuid: uuid.UUID, device_manager: DeviceManager,
                 depth: int = 1):
        self.num_stages = num_stages
        self.device_manager = device_manager
        self.root_uuid = root_uuid
        self.depth = depth
        self.children = self._collect_leaf_children()

    def _collect_leaf_children(self) -> list[uuid.UUID]:
        """Collect leaf modules of the timed hierarchy via DFS.

        Leaves are modules with no children in the timed hierarchy — the
        finest-grained timing points available. This gives the optimizer
        maximum flexibility for splitting.

        DFS preserves insertion order (``named_children()`` order) which
        matches forward-pass order.

        Returns:
            List of leaf module UUIDs to use as pipeline split candidates.
        """
        leaves: list[uuid.UUID] = []

        def dfs(node_uuid: uuid.UUID, current_depth: int):
            children = timed_module_hierarchy.get(node_uuid, [])
            if not children or current_depth >= self.depth:
                leaves.append(node_uuid)
            else:
                for child in children:
                    dfs(child, current_depth + 1)

        for child in timed_module_hierarchy.get(self.root_uuid, []):
            dfs(child, 1)
        return leaves

    def reconfigure_depth(self, new_depth: int):
        """Re-collect leaf children at a different depth.

        Updates ``self.children`` to reflect the new depth. Useful when
        splitting at the current depth produces an invalid pipeline (e.g.
        split points landing inside parallel branches).

        Args:
            new_depth: The new depth for ``_collect_leaf_children``.
        """
        self.depth = new_depth
        self.children = self._collect_leaf_children()

    def generate_safe_config(self) -> PipelineConfig:
        """Fall back to depth-1 children and regenerate the initial config.

        Reconfigures the optimizer to use only top-level children (which are
        always on the model's sequential path) and returns a fresh initial
        split. Subsequent ``optimize`` calls will also use depth-1 children.

        Returns:
            A ``PipelineConfig`` using top-level module boundaries.
        """
        self.reconfigure_depth(1)
        return self.initial_setup()

    @property
    def at_optimum(self) -> bool:
        """Whether the optimizer believes it has found the best config."""
        return False

    @staticmethod
    def _uuid_to_path(module_uuid: uuid.UUID) -> str:
        """Convert a module UUID to its dot-separated path string."""
        module = timed_module_registry.get(module_uuid)
        if module is None:
            raise ValueError(f"Module UUID {module_uuid} not found in registry")
        return module.get_path()

    @abstractmethod
    def optimize(self, time_logs: dict[uuid.UUID, list[float]], old_config: PipelineConfig,
                 force_rebalance: bool = False) -> PipelineConfig | None:
        """
        Optimise pipeline configuration based on timing data.

        Subclasses implement their own rebalance criteria internally.
        Returns ``None`` if no rebalance is needed.

        Args:
            time_logs: Dict mapping TimedModule UUIDs to lists of elapsed times
            old_config: The current pipeline configuration
            force_rebalance: If True, skip the rebalance check and always optimise.

        Returns:
            New PipelineConfig if rebalancing was performed, or None otherwise.
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

        children_uuid = self.children
        step = max(len(children_uuid) // self.num_stages, 1)
        split_spec: dict[str, SplitPoint] = {}
        current_stage_num = 1
        for i in range(step, len(children_uuid), step):
            # new split point
            path = self._uuid_to_path(children_uuid[i])
            split_spec[path] = SplitPoint.BEGINNING
            current_stage_num += 1
            # we have enough stages
            if current_stage_num == self.num_stages:
                break

        # Making the device mapping
        num_devices = self.device_manager.num_devices()
        device_mapping = {i: self.device_manager.get_device(i % num_devices) for i in range(len(split_spec) + 1)}

        return PipelineConfig(split_spec=split_spec, device_mapping=device_mapping)


class StaticGPipeOptimizer(PipelineOptimizer):
    """Static GPipe optimizer — cost-balanced initial split, never rebalances.

    Uses parameter-count-based DP partitioning (GPipe paper Section 2.2) for
    the initial split and returns ``None`` from ``optimize()`` so the pipeline
    config never changes. Useful as a baseline to measure the impact of
    dynamic rebalancing.
    """

    def __init__(self, num_stages, root_uuid, device_manager, depth=1, **kwargs):
        # Accept and ignore extra kwargs (e.g. rebalance_interval) for compatibility
        super().__init__(num_stages, root_uuid, device_manager, depth=depth)

    def initial_setup(self) -> PipelineConfig:
        root = timed_module_registry.get(self.root_uuid)
        if root is None:
            # Fallback to base class uniform split
            return super().initial_setup()

        model = root._inner
        split_spec = gpipe_split_spec(model, self.num_stages, depth=self.depth)

        num_devices = self.device_manager.num_devices()
        device_mapping = {i: self.device_manager.get_device(i % num_devices)
                          for i in range(len(split_spec) + 1)}

        return PipelineConfig(split_spec=split_spec, device_mapping=device_mapping)

    def optimize(self, time_logs, old_config, force_rebalance=False):
        return None


class StaticConfigOptimizer(PipelineOptimizer):
    """Static optimizer that loads a pre-computed config from a file.

    Uses the saved split_spec/device_mapping as the initial setup and never
    rebalances. Used to skip exhaustive exploration when a known-good config
    already exists.
    """

    def __init__(self, num_stages, root_uuid, device_manager, depth=1,
                 config_path=None, **kwargs):
        super().__init__(num_stages, root_uuid, device_manager, depth=depth)
        if config_path is None:
            raise ValueError("StaticConfigOptimizer requires config_path")
        self._config = PipelineConfig.load(config_path)

    def initial_setup(self) -> PipelineConfig:
        return self._config

    def optimize(self, time_logs, old_config, force_rebalance=False):
        return None

    @property
    def at_optimum(self):
        return True


class GreedyPipelineOptimizer(PipelineOptimizer):
    """
    Pipeline optimiser using a greedy algorithm to balance computation time across stages.
    """

    def __init__(self, num_stages: int, root_uuid: uuid.UUID, device_manager: DeviceManager,
                 depth: int = 1, rebalance_threshold: float = 0.1, rebalance_interval: int = 10):
        super().__init__(num_stages, root_uuid, device_manager, depth=depth)
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_interval = rebalance_interval
        self._call_count = 0

    def _should_rebalance(self, time_logs: dict[uuid.UUID, list[float]], current_config: PipelineConfig) -> bool:
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

    def optimize(self, time_logs: dict[uuid.UUID, list[float]], old_config: PipelineConfig,
                 force_rebalance: bool = False) -> PipelineConfig | None:
        """
        Optimise pipeline configuration based on timing data.
        Uses a greedy algorithm to balance computation time across stages.

        Returns None if no rebalance is needed.
        """
        # Check rebalance interval gate
        if not force_rebalance:
            self._call_count += 1
            if self._call_count < self.rebalance_interval:
                return None
            self._call_count = 0
            if not self._should_rebalance(time_logs, old_config):
                return None

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
        children_ids = self.children
        children_times = []
        for layer_time in layer_times:
            # If module is a child of the root
            if layer_time[0] in children_ids:
                children_times.append(layer_time)


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

    def _build_split_spec(self, stage_assignments: List[Tuple[uuid.UUID, int]]) -> dict[str, SplitPoint]:
        """
        Build split_spec dict marking stage boundaries with SplitPoint.BEGINNING.

        Returns:
            Dict mapping module paths to SplitPoint.BEGINNING for stage boundaries
        """
        split_spec: dict[str, SplitPoint] = {}

        prev_stage = 0
        for module_uuid, stage in stage_assignments:
            # Mark the beginning of a new stage (skip stage 0, as it's implicit)
            if stage > prev_stage:
                split_spec[self._uuid_to_path(module_uuid)] = SplitPoint.BEGINNING
                prev_stage = stage

        return split_spec

class ReactiveShishaOptimiser(PipelineOptimizer):
    """
    Reactive pipeline optimiser based on the Shisha paper (Soomro et al., PPAM 2022).

    Extends Shisha with two-level exploration (deep_alpha × sibling_alpha),
    explicit optimum detection with revert-to-best, and time-based optimum
    escape that restarts exploration when the environment changes.

    Uses a two-phase approach:
    1. Seed Generation (Algorithm 1): merges children into balanced groups using
       parameter count as weight proxy, then assigns stages to devices via ranking.
    2. Online Tuning (Algorithm 2): iteratively moves children from the slowest
       stage toward lighter stages, with patience parameter alpha.
    """

    def __init__(self, num_stages: int, root_uuid: uuid.UUID, device_manager: DeviceManager,
                 depth: int = 1,
                 deep_alpha: int = 5,
                 sibling_alpha: int = 1,
                 assignment_choice: str = "rank_w",
                 rebalance_interval: int = 3,
                 tolerance: float = 0.04,
                 optimum_tolerance: float = 0.12,
                 optimum_escape_duration: float = 20,
                 verbose: bool = False,
                 **kwargs):
        """
        Args:
            num_stages: Number of pipeline stages.
            root_uuid: UUID of the root TimedModule.
            device_manager: DeviceManager instance for device enumeration.
            depth: Depth in the TimedModule hierarchy to collect leaf children at.
            deep_alpha: Non-improving iteration limit for a single stage direction.
                When deep_alpha consecutive iterations fail to improve throughput,
                the optimiser stops exploring the current stage and moves on to the
                next slowest stage (increments sibling_gamma).
            sibling_alpha: Number of different stages to try before giving up.
                After exhausting deep_alpha attempts on sibling_alpha different
                stages, the optimiser declares an optimum and reverts to the best config.
            assignment_choice: Strategy for assigning stages to devices.
                - ``"rank_w"``: Heaviest stage (by parameter count) goes to the fastest device.
                - ``"rank_l"``: Stage with the most children goes to the fastest device.
            rebalance_interval: Number of forward passes between rebalance attempts.
                None means no automatic rebalancing.
            tolerance: Fraction of the best throughput that a new config can be worse
                by without counting as a regression during exploration.
            optimum_tolerance: Fraction of the best throughput used as the tolerance
                band when at optimum. Should be larger than tolerance to avoid
                restarting exploration on noise.
            optimum_escape_duration: duration in second before we restart exploration
                from an optimum state
            verbose: If True, print debug logs during optimisation.
        """
        super().__init__(num_stages, root_uuid, device_manager, depth=depth)
        self.verbose = verbose
        self.tolerance = tolerance
        self.optimum_tolerance = optimum_tolerance
        self.deep_alpha = deep_alpha
        self.sibling_alpha = min(sibling_alpha, num_stages)
        self.assignment_choice = assignment_choice
        self.rebalance_interval = rebalance_interval
        self._call_count = 0

        # Persistent state for online tuning across optimize() calls
        self._deep_gamma = 0                    # non-improving iteration counter
        self._best_throughput = 0.0        # best 1/max_stage_time seen
        self._best_config: PipelineConfig = None
        self._at_optimum = False
        self._return_best = False
        self._sibling_gamma = 0
        self.optimum_escape = optimum_escape_duration
        self._optimum_escape_start: float | None = None
        self._now: float = 0.0

        # Caches
        self._stage_times_cache = None
        self._slowest_stage_times = None

    @property
    def at_optimum(self) -> bool:
        return self._at_optimum

    @property
    def optimizer_state(self) -> dict:
        """Return current internal state for logging."""
        return {
            "deep_gamma": self._deep_gamma,
            "sibling_gamma": self._sibling_gamma,
            "best_throughput": self._best_throughput,
            "optimum_escape_elapsed": self._now - self._optimum_escape_start if self._optimum_escape_start is not None else 0.0,
        }

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _reset_stage_caches(self):
        """Clear cached stage times so they are recomputed on the next call.

        Called after any config change (online tuning move, revert to best, etc.)
        since the stage layout has changed and cached times are stale.
        """
        self._stage_times_cache = None
        self._slowest_stage_times = None

    def _children_to_stages(self, config: PipelineConfig) -> list[list[uuid.UUID]]:
        """Reconstruct which children are in which stage from the split_spec."""
        children = self.children
        stages: list[list[uuid.UUID]] = [[]]
        for child_uuid in children:
            child_path = self._uuid_to_path(child_uuid)
            if child_path in config.split_spec and config.split_spec[child_path] == SplitPoint.BEGINNING:
                stages.append([])
            stages[-1].append(child_uuid)
        return stages

    def _compute_stage_times(self, time_logs: dict[str, list[float]],
                             config: PipelineConfig) -> tuple[list[float], float]:
        """Compute per-stage average times and slowest stage time by summing average child times.

        time_logs is keyed by module path strings (not UUIDs) since each distributed
        rank has its own TimedModule registry with different UUIDs for the same modules.
        """

        if self._stage_times_cache is None:
            stages = self._children_to_stages(config)
            slowest_time = 0
            stage_times = []
            for stage in stages:
                total = 0.0
                for child_uuid in stage:
                    path = self._uuid_to_path(child_uuid)
                    times = time_logs.get(path, [])
                    if times:
                        total += sum(times) / len(times)
                stage_times.append(total)
                # update max
                if total > slowest_time:
                    slowest_time = total

            self._stage_times_cache = stage_times
            self._slowest_stage_times = slowest_time

        return self._stage_times_cache, self._slowest_stage_times

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
        children = list(self.children)
        N = self.num_stages

        if len(children) < N:
            raise RuntimeError(
                f"Cannot create {N} pipeline stages from only {len(children)} "
                f"children. Increase TimedModule depth or reduce world_size."
            )

        # Phase 1: Merge children into N balanced groups by weight
        groups: list[tuple[list[uuid.UUID], int]] = [
            ([c], self._get_child_weight(c)) for c in children
        ]

        while len(groups) > N:
            # Find the lightest group
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
            # Most children → fastest device
            stage_order = sorted(range(len(groups)), key=lambda i: len(groups[i][0]), reverse=True)
        else:
            stage_order = list(range(len(groups)))

        num_devices = len(ranked_devices)
        stage_to_device: dict[int, torch.device] = {}
        for rank, stage_idx in enumerate(stage_order):
            stage_to_device[stage_idx] = ranked_devices[rank % num_devices]

        # Build PipelineConfig
        split_spec: dict[str, SplitPoint] = {}
        device_mapping: dict[int, torch.device] = {}
        for stage_idx, (uuids, _weight) in enumerate(groups):
            if stage_idx > 0:
                split_spec[self._uuid_to_path(uuids[0])] = SplitPoint.BEGINNING
            device_mapping[stage_idx] = stage_to_device[stage_idx]

        return PipelineConfig(split_spec=split_spec, device_mapping=device_mapping)

    # ── Online Tuning (Algorithm 2) ──────────────────────────────────────

    def _online_tuning(self, time_logs: dict[str, list[float]],
                       old_config: PipelineConfig) -> PipelineConfig | None:
        """One iteration of online tuning: move one child from the slowest stage."""
        stage_times, _ = self._compute_stage_times(time_logs, old_config)
        stages = self._children_to_stages(old_config)
        stage_sizes = [len(s) for s in stages]
        self._log(f"[DEBUG _online_tuning] stage_times={[f'{t:.4f}' for t in stage_times]} "
              f"stage_sizes={stage_sizes} sibling_gamma={self._sibling_gamma}")

        # Selecting the nth slowest stage, depending on the current value of sibling gamma
        selected_idx = nth_largest_index(stage_times, self._sibling_gamma)

        # Can't move if slowest stage has only 1 child
        if len(stages[selected_idx]) <= 1:
            self._log(f"[DEBUG _online_tuning] stage {selected_idx} has 1 child, skipping")
            self._sibling_gamma += 1 # move to the next stage
            self._deep_gamma = 0
            return None

        # Find target stage
        target_idx = self._find_target_stage(stage_times, selected_idx)

        if target_idx is None or target_idx == selected_idx:
            self._log(f"[DEBUG _online_tuning] no valid target (target={target_idx}), returning None")
            return None

        self._log(f"[DEBUG _online_tuning] moving child from stage {selected_idx} toward stage {target_idx}")

        # target is to the left of selected
        if target_idx < selected_idx:
            # step towards it
            adjacent_idx = selected_idx - 1
            # Move first child of slowest to end of adjacent (left neighbour)
            child_to_move = stages[selected_idx][0]
            stages[adjacent_idx].append(child_to_move)
            stages[selected_idx] = stages[selected_idx][1:]

        # target is to the right of selected
        else:
            # same but in reverse
            adjacent_idx = selected_idx + 1
            # Move last child of slowest to beginning of adjacent (right neighbour)
            child_to_move = stages[selected_idx][-1]
            stages[adjacent_idx].insert(0, child_to_move)
            stages[selected_idx] = stages[selected_idx][:-1]

        # Build new split_spec from modified stages
        new_split_spec: dict[str, SplitPoint] = {}
        for stage_idx, stage_uuids in enumerate(stages):
            if stage_idx > 0 and stage_uuids:
                new_split_spec[self._uuid_to_path(stage_uuids[0])] = SplitPoint.BEGINNING

        self._reset_stage_caches()

        return PipelineConfig(split_spec=new_split_spec, device_mapping=old_config.device_mapping)

    def _find_target_stage(self, stage_times: list[float],
                           slowest_idx: int) -> int | None:
        """Find the lightest stage to move a child toward, breaking ties by proximity."""
        num_stages = len(stage_times)
        if num_stages <= 1:
            return None

        best_idx = None
        best_key = (float('inf'), float('inf'))
        for i in range(num_stages):
            if i == slowest_idx:
                continue
            key = (stage_times[i], abs(i - slowest_idx))
            if key < best_key:
                best_key = key
                best_idx = i
        return best_idx

    def _tick_gamma(self) -> bool:
        """Increment exploration counters. Returns True to keep exploring, False if exhausted."""
        self._deep_gamma += 1
        if self._deep_gamma >= self.deep_alpha:
            self._deep_gamma = 0
            self._sibling_gamma += 1
            if self._sibling_gamma >= self.sibling_alpha:
                self._return_best = True
                self._at_optimum = True
                return False
        return True

    def _should_rebalance(self, time_logs: dict[uuid.UUID, list[float]],
                          current_config: PipelineConfig) -> bool:
        self._now = time.monotonic()

        if not time_logs:
            self._log("[DEBUG _should_rebalance] no time_logs, returning True")
            return True

        stage_times, slowest_stage_time = self._compute_stage_times(time_logs, current_config)

        # Debug: check UUID overlap
        if self.verbose:
            children_set = set(self.children)
            logs_set = set(time_logs.keys())
            overlap = children_set & logs_set
            self._log(f"[DEBUG _should_rebalance] children={len(children_set)} time_log_keys={len(logs_set)} "
                  f"overlap={len(overlap)} missing={len(children_set - logs_set)}")

        # Some stages have no timing data yet — can't make a decision
        if any(t == 0 for t in stage_times):
            self._log(f"[DEBUG _should_rebalance] zero stage time found: {stage_times}, returning True")
            return True

        # TODO: maybe make it accurate? change this to actual TP
        throughput = 1.0 / slowest_stage_time

        # First config after optimum always becomes best config
        if self._best_throughput == 0.0:
            self._best_throughput = throughput
            self._best_config = current_config
            self._log(f"[DEBUG _should_rebalance] first config, tp={throughput:.4f}, returning True")
            return True

        tol = self.optimum_tolerance if self._at_optimum else self.tolerance
        threshold = self._best_throughput - self._best_throughput * tol
        self._log(f"[DEBUG _should_rebalance] tp={throughput:.4f} best={self._best_throughput:.4f} "
              f"threshold={threshold:.4f} tol={tol} gamma_d={self._deep_gamma} gamma_s={self._sibling_gamma}")

        # Better config found; reset gamma
        if throughput > self._best_throughput:
            self._best_throughput = throughput
            self._best_config = current_config
            self._deep_gamma = 0
            self._sibling_gamma = 0
            self._log(f"[DEBUG _should_rebalance] better config, returning {not self._at_optimum}")
            # should we keep exploring? only if we are not at optimum
            return not self._at_optimum

        # Throughput is within tolerance
        elif throughput >= threshold:
            if self._at_optimum:
                self._optimum_escape_start = None  # things are fine, reset escape timer
            self._log(f"[DEBUG _should_rebalance] within tolerance, returning {not self._at_optimum}")
            return not self._at_optimum

        # Throughput is worse, so we keep trying to find the optimum
        else:
            # no longer at optimum if we were there
            # unless we haven't been worse for long enough
            if self._at_optimum:
                self._now = time.monotonic()
                if self._optimum_escape_start is None:
                    self._optimum_escape_start = self._now
                if self._now - self._optimum_escape_start < self.optimum_escape:
                    # not enough time has passed — stay at optimum
                    return False
                # enough time worse — leave optimum and restart exploration
                self._optimum_escape_start = None
                self._at_optimum = False
                self._best_throughput = 0.0
                self._best_config = None
                self._sibling_gamma = 0
                self._deep_gamma = 0

            return self._tick_gamma()

    # ── Public interface ─────────────────────────────────────────────────

    def initial_setup(self) -> PipelineConfig:
        return self._seed_generation()

    def optimize(self, time_logs: dict[uuid.UUID, list[float]],
                 old_config: PipelineConfig,
                 force_rebalance: bool = False) -> PipelineConfig | None:
        # Check rebalance interval
        if not force_rebalance and self.rebalance_interval is not None:
            self._call_count += 1
            if self._call_count < self.rebalance_interval:
                return None
            self._call_count = 0
        self._log(f"[DEBUG optimize] passed rebalance interval gate (call_count reset)")

        # Clear stale cached stage times so _should_rebalance sees fresh data
        self._reset_stage_caches()

        # Always call _should_rebalance to update internal state (_gamma, _best_throughput),
        # but only gate on its result when not forced.
        should_rebalance = self._should_rebalance(time_logs, old_config)

        if not force_rebalance:
            # If at optimum, no further exploration
            if self._at_optimum and self._return_best:
                self._return_best = False
                self._log("[DEBUG optimize] at optimum, returning best config")
                return self._best_config

            # Exit if we don't need to rebalance
            elif not should_rebalance:
                self._log("[DEBUG optimize] should_rebalance=False, returning None")
                return None

        # We need to rebalance
        self._log("[DEBUG optimize] calling _online_tuning")
        result = self._online_tuning(time_logs, old_config)
        self._log(f"[DEBUG optimize] _online_tuning returned {'new config' if result is not None else 'None'}")
        return result


class ExhaustiveShishaOptimizer(ReactiveShishaOptimiser):
    """Shisha variant that explores exhaustively then freezes at the best config.

    Explores all configurations (deep_alpha × sibling_alpha attempts) without
    any tolerance — every config is evaluated and the best throughput is tracked.
    Once exploration is complete (gamma reaches alpha), reverts to the best
    config found and stays there permanently. No optimum escape.

    Used for experiment run D: find the single best config, then measure how
    it holds up under interference without further adaptation.
    """

    def __init__(self, num_stages, root_uuid, device_manager, depth=1,
                 deep_alpha=5, sibling_alpha=2, total_alpha=50,
                 timeout=1000,
                 assignment_choice="rank_w",
                 rebalance_interval=3, verbose=False,
                 save_config_path=None, **kwargs):
        super().__init__(
            num_stages, root_uuid, device_manager, depth=depth,
            deep_alpha=deep_alpha, sibling_alpha=sibling_alpha,
            assignment_choice=assignment_choice,
            rebalance_interval=rebalance_interval,
            tolerance=0.0, optimum_tolerance=0.0,
            optimum_escape_duration=999999,
            stop_at_first_optimum=True,
            verbose=verbose,
        )
        self._save_config_path = save_config_path
        self.total_alpha = total_alpha
        self._total_gamma = 0
        self._truncated = False  # True if optimum reached via total_gamma or timeout
        self._timeout = timeout
        self._start_time = None

    def _freeze(self, reason: str, truncated: bool = False):
        """Freeze at best config found so far."""
        self._return_best = True
        self._at_optimum = True
        self._truncated = truncated
        self._log(f"[Exhaustive] {reason}, freezing at best tp={self._best_throughput:.4f}"
                  f"{' (truncated)' if truncated else ''}")
        if self._save_config_path and self._best_config is not None:
            self._best_config.save(self._save_config_path)
            self._log(f"[Exhaustive] saved config to {self._save_config_path}")

    def _should_rebalance(self, time_logs, current_config):
        """Explore exhaustively: no tolerance, just track best and count gamma."""
        self._now = time.monotonic()

        if not time_logs:
            return True

        stage_times, slowest_stage_time = self._compute_stage_times(time_logs, current_config)

        if any(t == 0 for t in stage_times):
            return True

        throughput = 1.0 / slowest_stage_time

        # First config becomes best — also starts the timer
        if self._best_throughput == 0.0:
            self._best_throughput = throughput
            self._best_config = current_config
            self._start_time = self._now
            self._log(f"[Exhaustive] first config, tp={throughput:.4f}")
            return True

        # Already at optimum — stay permanently
        if self._at_optimum:
            return False

        # Check timeout
        if self._now - self._start_time >= self._timeout:
            self._freeze(f"timeout after {self._timeout}s", truncated=True)
            return False

        # Count every non-first step toward total budget
        self._total_gamma += 1

        # Track the best config seen
        if throughput > self._best_throughput:
            self._best_throughput = throughput
            self._best_config = current_config
            self._deep_gamma = 0
            self._sibling_gamma = 0
            self._log(f"[Exhaustive] new best tp={throughput:.4f}, reset gamma "
                      f"(total={self._total_gamma}/{self.total_alpha})")
            return True

        # Count non-improving steps
        self._deep_gamma += 1
        self._log(f"[Exhaustive] tp={throughput:.4f} best={self._best_throughput:.4f} "
                  f"gamma_d={self._deep_gamma} gamma_s={self._sibling_gamma} "
                  f"total={self._total_gamma}/{self.total_alpha}")

        if self._deep_gamma >= self.deep_alpha:
            self._deep_gamma = 0
            self._sibling_gamma += 1

            if self._sibling_gamma >= self.sibling_alpha:
                self._freeze("exploration complete")
                return False

        # Total budget exhausted
        if self._total_gamma >= self.total_alpha:
            self._freeze(f"total budget exhausted ({self.total_alpha})", truncated=True)
            return False

        return True

    @property
    def optimizer_state(self) -> dict:
        state = super().optimizer_state
        state["total_gamma"] = self._total_gamma
        state["truncated"] = self._truncated
        return state


# Backward-compat alias
TimeBasedShishaPipelineOptimizer = ReactiveShishaOptimiser
