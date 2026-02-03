import threading
import time
import uuid
import weakref
from typing import Dict, List

import torch
import torch.nn as nn

try:
    from .cuda_timing_kernel import cuda_timing_kernel_cpp
    print("cuda_timing_kernel imported successfully!")
except ImportError:
    print("Error: 'cuda_timing_kernel' module not found.")
    cuda_timing_kernel_cpp = None


_registry_lock = threading.Lock()
# WeakValueDictionary automatically removes entries when TimedModule is garbage collected
timed_module_registry: weakref.WeakValueDictionary[uuid.UUID, 'TimedModule'] = weakref.WeakValueDictionary()
timed_module_hierarchy: Dict[uuid.UUID, List[uuid.UUID]] = {}  # {uuid: [child_uuid, ...]}


def _cleanup_hierarchy(module_uuid: uuid.UUID, parent_uuid: uuid.UUID | None):
    """Remove a module from the hierarchy when it's garbage collected."""
    with _registry_lock:
        # Remove this module's entry from hierarchy
        if module_uuid in timed_module_hierarchy:
            del timed_module_hierarchy[module_uuid]
        # Remove from parent's children list
        if parent_uuid is not None and parent_uuid in timed_module_hierarchy:
            try:
                timed_module_hierarchy[parent_uuid].remove(module_uuid)
            except ValueError:
                pass  # Already removed


class TimedModule(nn.Module):
    def __init__(self, module: nn.Module, device, depth=10, parent_uuid: uuid.UUID = None, module_path: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner = module
        self.device = device
        self.depth = depth
        self.module_path = module_path

        self.uuid = uuid.uuid4()
        self.parent_uuid = parent_uuid

        # Register in registry and hierarchy (thread-safe)
        with _registry_lock:
            timed_module_registry[self.uuid] = self
            timed_module_hierarchy[self.uuid] = []
            if parent_uuid is not None and parent_uuid in timed_module_hierarchy:
                timed_module_hierarchy[parent_uuid].append(self.uuid)

        # Set up weak reference callback for automatic cleanup
        # When this TimedModule is garbage collected, _cleanup_hierarchy will be called
        weakref.finalize(self, _cleanup_hierarchy, self.uuid, self.parent_uuid)

    def get_last_elapsed_cycles(self):
        pass

    def get_logs(self, existing_logs: Dict[uuid.UUID, List[float]] | None = None) -> Dict[uuid.UUID, List[float]]:
        """
        Collect timing logs using the hierarchy registry.
        Works even after PyTorch pipeline splits the model.
        Thread-safe.
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

        with _registry_lock:
            recurse_get_logs(self.uuid, logs=existing_logs)
        return existing_logs

    def inner(self) -> nn.Module:
        return self._inner

    def __getattr__(self, attr: str):
        """Delegate attribute access to inner module for attributes not found on TimedModule."""
        #TODO(tests): make a good unit test for this and check all cases
        # this is a very critical builtin
        try:
            # Let torch.nn.Module handle it
            return super().__getattr__(attr)
            #^^ might be bad for stuff like named_children; could create infinite recursion somewhere else
        except AttributeError:
            try:
                # Let self._inner try
                return getattr(self._inner, attr)
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' or {nn.Module.__name__} "
                                     f"objects have no attribute '{attr}'")

    def _get_name(self):
        return f"Timed_{self.inner()._get_name()}[uuid:{self.uuid}]"

    def get_path(self) -> str:
        """Get the full path of this module in the model hierarchy."""
        return self.module_path

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

        if cuda_timing_kernel_cpp is None:
            raise ImportError("CUDA extension 'cuda_timing_kernel' is not built.")

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
        cuda_timing_kernel_cpp.start(self.time_buffer)
        # 2. Run the inner module
        output = (self.inner())(*args, **kwargs)
        # 3. End the timer
        cuda_timing_kernel_cpp.end(self.time_buffer)

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

    def _get_name(self):
        return f"CUDA_{super()._get_name()}"


#==================
# CPU Timed Module
#==================
@torch.library.custom_op("model_runner::cpu_time_ns", mutates_args={"time_buffer"})
def cpu_time_ns(time_buffer: torch.Tensor) -> None:
    """Write current time in nanoseconds to the buffer."""
    time_buffer[0] = time.time_ns()

@cpu_time_ns.register_fake
def _(time_buffer: torch.Tensor) -> None:
    return None


class CPUTimedModule(TimedModule):
    """
    A wrapper that times the forward pass of a module on CPU.
    torch.compile friendly.
    """

    def __init__(self, module: nn.Module, device, depth=1, parent_uuid: uuid.UUID = None, module_path: str = ""):
        super().__init__(module, device, depth, parent_uuid=parent_uuid, module_path=module_path)

        self.inner().to(device)

        # Use buffers for timing (torch.compile friendly)
        self.register_buffer('start_buffer', torch.zeros(1, dtype=torch.int64))
        self.register_buffer('end_buffer', torch.zeros(1, dtype=torch.int64))
        self.timing_depth = depth

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
        # 1. Start timer (writes to buffer)
        cpu_time_ns(self.start_buffer)
        # 2. Run inner module
        output = (self.inner())(*args, **kwargs)
        # 3. End timer (writes to buffer)
        cpu_time_ns(self.end_buffer)

        return output

    def get_last_elapsed_cycles(self):
        """
        Return the last elapsed time.
        For CPU, this returns time in nanoseconds (not cycles).
        """
        return (self.end_buffer[0] - self.start_buffer[0]).item()

    def _get_name(self):
        return f"CPU_{super()._get_name()}"


#==================
def make_module_timed(module: nn.Module, device=None, depth=None) -> TimedModule:
    if device is None:
        if torch.cuda.is_available() and cuda_timing_kernel_cpp is not None:
            device = "cuda"
        else:
            device = "cpu"

    if device == "cuda" or "cuda" in str(device):
        return CUDATimedModule(module, device, depth=depth)
    else:
        return CPUTimedModule(module, device, depth=depth)
