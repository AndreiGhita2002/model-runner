import time
import uuid
from sys import stderr
from typing import Any, Dict, List

import torch
import torch.nn as nn

try:
    from . import gpu_timer
    print("gpu_timer imported successfully!")
except ImportError:
    print("Error: 'gpu_timer' module not found.")
    gpu_timer = None


timed_module_registry: Dict[uuid.UUID, 'TimedModule'] = {}
timed_module_hierarchy: Dict[uuid.UUID, List[uuid.UUID]] = {}  # {uuid: [child_uuid, ...]}


class TimedModule(nn.Module):
    def __init__(self, module: nn.Module, device, depth=10, wrapping_a_wrapper=None, parent_uuid: uuid.UUID = None, module_path: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module = module
        self.device = device
        self.depth = depth
        self.module_path = module_path

        self.uuid = uuid.uuid4()
        self.parent_uuid = parent_uuid
        timed_module_registry[self.uuid] = self

        # Register in hierarchy
        timed_module_hierarchy[self.uuid] = []
        # Add self to parent's children list
        if parent_uuid is not None and parent_uuid in timed_module_hierarchy:
            timed_module_hierarchy[parent_uuid].append(self.uuid)

        if wrapping_a_wrapper is not None:
            self.wrapping_a_wrapper = wrapping_a_wrapper
        else:
            self.wrapping_a_wrapper = not isinstance(module, nn.Module)

    def get_last_elapsed_cycles(self):
        pass

    def get_logs(self, existing_logs: Dict[uuid.UUID, List[float]] | None = None) -> Dict[uuid.UUID, List[float]]:
        """
        Collect timing logs using the hierarchy registry.
        Works even after PyTorch pipeline splits the model.
        """
        if existing_logs is None:
            existing_logs = {}

        def recurse_get_logs(module_uuid: uuid.UUID, logs: Dict[uuid.UUID, List[float]]):
            module = timed_module_registry.get(module_uuid)
            if module is None:
                return

            # Get timing for this module
            elapsed = module.get_last_elapsed_cycles()
            if logs.get(module_uuid) is None:
                logs[module_uuid] = [elapsed]
            else:
                logs[module_uuid].append(elapsed)

            # Recurse through children using hierarchy
            for child_uuid in timed_module_hierarchy.get(module_uuid, []):
                recurse_get_logs(child_uuid, logs)

        recurse_get_logs(self.uuid, logs=existing_logs)
        return existing_logs

    def inner(self) -> nn.Module:
        if not self.wrapping_a_wrapper:
            return self._module
        else:
            return self._module.model

    def _get_name(self):
        return self.inner()._get_name()

    def get_path(self) -> str:
        """Get the full path of this module in the model hierarchy."""
        return self.module_path

    def rand_inputs(self) -> Any:
        if callable(self._module.rand_inputs):
            return self._module.rand_inputs()
        else:
            print("Inner module does not have a `rand_inputs` function!")
            return None

    def to(self, *args, **kwargs):
        """Override to() to update self.device when module is moved."""
        result = super().to(*args, **kwargs)
        # Update device from first parameter/buffer we can find
        for param in self.parameters():
            self.device = param.device
            break
        for buffer in self.buffers():
            self.device = buffer.device
            break
        return result


#==================
# CUDA GPU Timed Module
#==================
class CUDATimedModule(TimedModule):
    """
    A wrapper that times the forward pass of a module using on-device clock64() CUDA kernels.
    This measures raw GPU cycles as seen by the device.
    torch.compile friendly.
    """

    def __init__(self, module: nn.Module, device, depth=1, parent_uuid: uuid.UUID = None, module_path: str = ""):
        super().__init__(module, device, depth, parent_uuid=parent_uuid, module_path=module_path)

        if gpu_timer is None:
            raise ImportError("CUDA extension 'gpu_timer' is not built.")

        self.inner().to(device)

        self.time_buffer = torch.zeros(1, dtype=torch.int64, device=device)
        self.last_elapsed_cycles = 0
        self.timing_depth = depth

        # Wrap children as well
        if depth is None or depth > 0:
            for child_name, child in list(self.inner().named_children()):
                # Decrement the depth when we go deeper in the tree
                d = None if depth is None else (depth - 1)
                # Build child path
                child_path = f"{module_path}.{child_name}" if module_path else child_name
                wrapped_child = CUDATimedModule(child, device, depth=d, parent_uuid=self.uuid, module_path=child_path)
                setattr(self.inner(), child_name, wrapped_child)

    def forward(self, *args, **kwargs):
        # 1. Start the timer
        gpu_timer.start(self.time_buffer)
        # 2. Run the inner module
        output = self._module(*args, **kwargs)
        # 3. End the timer
        gpu_timer.end(self.time_buffer)

        return output

    def get_last_elapsed_cycles(self):
        """
        Will synchronise and then output elapsed cycles.
        Not torch.compile friendly.
        """
        # Sync the device where the buffer actually lives
        torch.cuda.synchronize(self.time_buffer.device)

        self.last_elapsed_cycles = self.time_buffer.item()
        return self.last_elapsed_cycles


#==================
# CPU Timed Module
#==================
class CPUTimedModule(TimedModule):
    """
    A wrapper that times the forward pass of a module on CPU.
    """

    def __init__(self, module: nn.Module, device, depth=1, parent_uuid: uuid.UUID = None, module_path: str = ""):
        print(UserWarning("CPUTimedModule does not have logging working!"), file=stderr)

        super().__init__(module, device, depth, parent_uuid=parent_uuid, module_path=module_path)

        self.inner().to(device)

        self.last_elapsed_time = 0.0
        self.timing_depth = depth

        # TODO: initialize CPU timing mechanism

        # Wrap children as well
        if depth is None or depth > 0:
            for child_name, child in list(self.inner().named_children()):
                # Decrement the depth when we go deeper in the tree
                d = None if depth is None else (depth - 1)
                # Build child path
                child_path = f"{module_path}.{child_name}" if module_path else child_name
                wrapped_child = CPUTimedModule(child, device, depth=d, parent_uuid=self.uuid, module_path=child_path)
                setattr(self.inner(), child_name, wrapped_child)

    def forward(self, *args, **kwargs):
        #TODO: figure out cpu timing:
        # - figure out how compilation actually happens
        # - find a way

        # 1. Start timer
        start = time.perf_counter_ns()
        # 2. Run inner module
        output = self._module(*args, **kwargs)
        # 3. End timer and store elapsed
        end = time.perf_counter_ns()
        self.last_elapsed_time = end - start

        return output

    def get_last_elapsed_cycles(self):
        """
        Return the last elapsed time.
        For CPU, this returns time in nanoseconds (not cycles).
        """
        return self.last_elapsed_time


#==================
def make_module_timed(module: nn.Module, device=None, depth=None) -> TimedModule:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda" or "cuda" in str(device):
        return CUDATimedModule(module, device, depth=depth)
    else:
        return CPUTimedModule(module, device, depth=depth)
