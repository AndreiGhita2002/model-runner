from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn


class ModelSplitter:
    """
    Automatically analyses and splits a model into pipeline stages.

    Split Strategy: What are we splitting?
    1. Natural guess: Split at modules that would be 'natural' split points. This is opinionated.
    2. Depth: split at depth N  #TODO: do we want this?
    3. All: Split at all the child modules of the model.

    Distribution Strategies: How are we splitting?
    1. Layer-based: Split at sequential container boundaries (Conv blocks, Transformer layers)
    2. Computation-based: Use timing profiles to create balanced stages
    3. Memory-based: Split to balance memory usage across devices
    """

    def __init__(self,
                 num_stages: int = 2,
                 split_strategy: str = "natural",
                 distribution_strategy: str = "layer_based"):
        """
        Args:
            num_stages: Number of pipeline stages to create (should match # of devices)
            split_strategy: "natural", "depth N" (where N is an integer), or "all"
            distribution_strategy: "layer_based", "computation_based", or "memory_based"
        """
        self.num_stages = num_stages
        self.split_strategy = split_strategy
        self.distribution_strategy = distribution_strategy

    def find_split_candidates(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        Find split candidates based on the configured split_strategy.

        Args:
            model: The model to analyse

        Returns:
            List of (name, module) tuples representing split candidates
        """
        if self.split_strategy == "natural":
            return self.natural_split_candidates(model)
        elif self.split_strategy == "all":
            return self.all_candidates(model)
        elif self.split_strategy.startswith("depth"):
            # Extract depth value from "depth N" format
            return self.depth_based_candidates(model)
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy}")

    def natural_split_candidates(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        Analyse the model structure to find 'natural' split points.
        Returns a list of (name, module) tuples representing potential split boundaries.

        This function makes assumptions about what is a 'natural' split point.
        """
        split_candidates = []

        # Check if the model has a sequential structure
        if isinstance(model, nn.Sequential):
            # For Sequential models, each child is a candidate
            for name, module in model.named_children():
                split_candidates.append((name, module))
        else:
            # TODO some recursion might be good here

            # For other models, look for common patterns
            for name, module in model.named_children():
                # Look for sequential containers, layer lists, or major blocks
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    # Add each child of the sequential/list
                    for sub_name, sub_module in module.named_children():
                        full_name = f"{name}.{sub_name}"
                        split_candidates.append((full_name, sub_module))
                elif self._is_block_module(module):
                    # This is likely a major block (ResNet block, Transformer layer, etc.)
                    split_candidates.append((name, module))

        return split_candidates

    def depth_based_candidates(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        Find split candidates at a specific depth in the module hierarchy.

        For models wrapped by TimedModule, use the timing_depth field if available.
        Otherwise, extracts depth from the split_strategy string ("depth N").

        Args:
            model: The model to analyze

        Returns:
            List of (name, module) tuples at the specified depth
        """
        # Determine target depth
        if hasattr(model, 'timing_depth'):
            # Use timing_depth from the TimedModule wrapper
            target_depth = model.timing_depth
        else:
            # Extract from split_strategy string "depth N"
            try:
                parts = self.split_strategy.split()
                if len(parts) == 2 and parts[0] == "depth":
                    target_depth = int(parts[1])
                else:
                    raise ValueError(f"Invalid depth format: {self.split_strategy}")
            except (ValueError, IndexError):
                raise ValueError(f"split_strategy must be 'depth N' where N is an integer, got: {self.split_strategy}")

        split_candidates = []

        def _collect_at_depth(module: nn.Module, name: str, current_depth: int):
            """Recursively collect modules at target depth."""
            if current_depth == target_depth:
                split_candidates.append((name, module))
                return

            # Recurse into children
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                _collect_at_depth(child_module, full_name, current_depth + 1)

        # Start recursion from depth 0
        _collect_at_depth(model, "", 0)

        return split_candidates

    def all_candidates(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        Return all child modules as split candidates.

        This includes every direct child of the model without any filtering.

        Args:
            model: The model to analyse

        Returns:
            List of (name, module) tuples for all children
        """
        split_candidates = []

        for name, module in model.named_children():
            split_candidates.append((name, module))

        return split_candidates

    def _is_block_module(self, module: nn.Module) -> bool:
        """
        Determine if a module is a 'block' that should be kept together.
        """
        # TODO: justify the block types defined in `_is_block_module`
        # Common block patterns in CNNs and Transformers
        block_types = (
            nn.Conv2d, nn.Conv1d, nn.Conv3d,
            nn.BatchNorm2d, nn.BatchNorm1d,
            nn.Linear,
        )

        # Check if the module contains these types
        for child in module.children():
            if isinstance(child, block_types):
                return True

        # Also check the class name for common patterns
        class_name = module.__class__.__name__.lower()
        block_patterns = ['block', 'layer', 'stage', 'encoder', 'decoder', 'attention']
        has_block_pattern = any(pattern in class_name for pattern in block_patterns)

        return has_block_pattern

    def split_layer_based(self, candidates: List[Tuple[str, nn.Module]]) -> Dict[str, int]:
        """
        Create split specification based on model layer structure.
        Tries to evenly distribute layers across stages.

        Args:
            candidates: List of (name, module) tuples to distribute

        Returns:
            Dict mapping layer names to stage indices.
        """
        split_spec = {}

        if not candidates:
            print("Warning: No split candidates found. Model may not be splittable.")
            return split_spec

        # Calculate how many layers per stage
        total_layers = len(candidates)
        layers_per_stage = max(1, total_layers // self.num_stages)

        # Assign layers to stages
        for i, (layer_name, _) in enumerate(candidates):
            stage_idx = min(i // layers_per_stage, self.num_stages - 1)
            split_spec[layer_name] = stage_idx

        return split_spec

    def split_computation_based(
        self,
        candidates: List[Tuple[str, nn.Module]],
        timing_profile: Optional[Dict[str, float]] = None
    ) -> Dict[str, int]:
        """
        Create a split specification based on computation time.
        Tries to balance computation across stages using timing data.

        Args:
            candidates: List of (name, module) tuples to distribute
            timing_profile: Dict mapping layer names to execution times

        Returns:
            Dict mapping layer names to stage indices
        """
        if timing_profile is None:
            print("Warning: No timing profile provided. Falling back to layer-based split.")
            return self.split_layer_based(candidates)

        split_spec = {}

        if not candidates:
            return split_spec

        # Build list of (name, time) tuples
        layer_times = []
        for name, _ in candidates:
            # Try to find timing info for this layer
            time = timing_profile.get(name, 0.0)
            # If exact match not found, try to find partial matches
            if time == 0.0:
                for profile_name, profile_time in timing_profile.items():
                    if name in profile_name or profile_name in name:
                        time = max(time, profile_time)
            layer_times.append((name, time))

        # Calculate target time per stage
        total_time = sum(t for _, t in layer_times)
        if total_time == 0:
            print("Warning: Total time is zero. Falling back to layer-based split.")
            return self.split_layer_based(candidates)

        target_time_per_stage = total_time / self.num_stages

        # Greedily assign layers to stages
        current_stage_time = 0.0
        current_stage = 0

        for i, (name, time) in enumerate(layer_times):
            split_spec[name] = current_stage
            current_stage_time += time

            # If we've exceeded the target, and we're not on the last stage
            if (current_stage_time >= target_time_per_stage and
                current_stage < self.num_stages - 1 and
                i < len(layer_times) - 1):  # Don't move to a new stage at the very end

                current_stage += 1
                current_stage_time = 0.0

        return split_spec

    def split_memory_based(
        self,
        candidates: List[Tuple[str, nn.Module]],
        memory_profile: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Create a split specification based on memory usage.
        Tries to balance memory consumption across stages.

        Args:
            candidates: List of (name, module) tuples to distribute
            memory_profile: Dict mapping layer names to memory usage in bytes

        Returns:
            Dict mapping layer names to stage indices
        """
        if memory_profile is None:
            # Estimate memory based on parameters from candidates
            memory_profile = {}
            for name, module in candidates:
                param_memory = sum(p.numel() * p.element_size() for p in module.parameters())
                if param_memory > 0:
                    memory_profile[name] = param_memory

        # Use the same logic as computation-based but with memory instead of time
        return self.split_computation_based(candidates, memory_profile)

    def create_split_spec(
        self,
        model: nn.Module,
        timing_profile: Optional[Dict[str, float]] = None,
        memory_profile: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Create a split specification based on the configured strategy.

        Returns:
            Dict mapping layer names to stage indices (0 to num_stages-1)
        """
        # Get candidates using the split_strategy
        candidates = self.find_split_candidates(model)

        # Apply distribution strategy to the candidates
        if self.distribution_strategy == "layer_based":
            return self.split_layer_based(candidates)
        elif self.distribution_strategy == "computation_based":
            return self.split_computation_based(candidates, timing_profile)
        elif self.distribution_strategy == "memory_based":
            return self.split_memory_based(candidates, memory_profile)
        else:
            raise ValueError(f"Unknown distribution_strategy: {self.distribution_strategy}")

    def pretty_split_info_str(self, split_spec: Dict[str, int]) -> str:
        """
        Return a human-readable description of the split.
        """
        if not split_spec:
            return "No splits (single stage)"

        # Group layers by stage
        stages: Dict[int, List[str]] = {}
        for layer_name, stage_idx in split_spec.items():
            if stage_idx not in stages:
                stages[stage_idx] = []
            stages[stage_idx].append(layer_name)

        info = f"Model split into {self.num_stages} stages:\n"
        for stage_idx in sorted(stages.keys()):
            layers = stages[stage_idx]
            info += f"  Stage {stage_idx}: {len(layers)} layers\n"
            for layer in layers[:3]:  # Show first 3
                info += f"    - {layer}\n"
            if len(layers) > 3:
                info += f"    ... and {len(layers) - 3} more\n"
        return info

    def apply_split_to_devices(
        self,
        model: nn.Module,
        split_spec: Dict[str, int],
        devices: List[torch.device]
    ) -> nn.Module:
        """
        Apply the split specification by moving layers to different devices.
        This is a simple device placement strategy for PyTorch 2.x.

        Args:
            model: The model to split
            split_spec: Dict mapping layer names to stage indices
            devices: List of devices (one per stage)

        Returns:
            The model with layers on appropriate devices
        """
        if len(devices) < self.num_stages:
            raise ValueError(f"Need {self.num_stages} devices but only {len(devices)} provided")

        # Move modules to their assigned devices
        for name, module in model.named_children():
            if name in split_spec:
                stage_idx = split_spec[name]
                device = devices[stage_idx]
                module.to(device)
                print(f"Moved {name} to {device} (stage {stage_idx})")

        return model


def extract_timing_profile_from_logs(logs: Dict) -> Dict[str, float]:
    """
    Extract timing information from TimedModule logs and flatten into a dict.

    Args:
        logs: Nested log structure from TimedModule.get_logs()

    Returns:
        Flat dict mapping module names to elapsed times
    """
    profile = {}

    def _extract_recursive(log_node: Dict, prefix: str = ""):
        if not isinstance(log_node, dict):
            return

        # Get module name and timing
        module_name = log_node.get('module_name', '')
        times = log_node.get('times', {})
        elapsed = times.get('elapsed', 0.0)

        # Build full name with prefix
        full_name = f"{prefix}.{module_name}" if prefix else module_name

        if elapsed > 0:
            profile[full_name] = elapsed
            profile[module_name] = elapsed  # Also store short name

        # Process children
        children = log_node.get('children', [])
        if isinstance(children, dict):
            for child_name, child_node in children.items():
                _extract_recursive(child_node, full_name)
        elif isinstance(children, list):
            for child_node in children:
                _extract_recursive(child_node, full_name)

    _extract_recursive(logs)
    return profile
