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
    """Remove a module from the hierarchy when it is garbage collected.

    Registered as a weak-reference finalizer by ``TimedModule.__init__``.

    Args:
        module_uuid: UUID of the module being collected.
        parent_uuid: UUID of the parent module (or None for root).
    """
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
    """Base class for modules that measure their own forward-pass duration.

    Wraps an inner ``nn.Module``, assigns it a UUID, and registers it in the
    global ``timed_module_registry`` / ``timed_module_hierarchy``. Subclasses
    (``CUDATimedModule``, ``CPUTimedModule``) provide the actual timing logic.

    Attribute access falls through to the inner module, so the wrapper is
    largely transparent to the rest of the model.

    Design notes on ``__getattr__`` and module introspection:
        We override ``__getattr__`` to delegate attribute access to ``_inner`` when
        an attribute isn't found on the wrapper. This provides transparency for most
        use cases, but has limitations:

        - ``named_children()`` / ``children()`` return the wrapper's children (i.e.,
          ``_inner``), not the original model's children. We intentionally do NOT
          override these methods because:

          1. It would break the ``nn.Module`` contract — ``self._modules`` wouldn't
             match what ``named_children()`` returns, causing issues with code that
             mutates children via ``setattr``.

          2. PyTorch's ``pipeline()`` and split-spec matching rely on the actual
             registered module paths. Changing what ``named_children()`` returns
             would break path-based pipeline splitting.

          3. ``state_dict()`` keys are built from the module hierarchy. Mismatched
             introspection would cause checkpoint save/load issues.

          4. We'd need to consistently override ~8 methods (``named_modules``,
             ``modules``, ``named_parameters``, ``parameters``, etc.) to avoid
             inconsistent behavior.

        - To inspect the original model structure, use ``timed_module.inner().named_children()``.

        - Potential infinite recursion: if ``__getattr__`` is called before ``_inner``
          is assigned in ``__init__``, it would recurse. Currently safe because
          ``self._inner = module`` is set immediately after ``super().__init__()``.
    """

    def __init__(self, module: nn.Module, device, depth=10, parent_uuid: uuid.UUID = None, module_path: str = "", *args, **kwargs):
        """Wrap a module for timing and register it in the global hierarchy.

        Args:
            module: The ``nn.Module`` to wrap.
            device: Device the module runs on (used by subclasses for buffer placement).
            depth: How many levels of children to recursively wrap. ``None`` for unlimited.
            parent_uuid: UUID of the parent ``TimedModule`` (None for root).
            module_path: Dot-separated path of this module inside the model tree
                (e.g. ``"layer1.conv"``). Used to map back to PyTorch split specs.
            *args: Forwarded to ``nn.Module.__init__``.
            **kwargs: Forwarded to ``nn.Module.__init__``.
        """
        super().__init__(*args, **kwargs)
        # _inner needs to be initialised first
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
        """Return the duration of the most recent forward pass.

        Returns:
            Elapsed time. Units depend on the subclass (GPU cycles for CUDA,
            nanoseconds for CPU). Base implementation returns ``None``.
        """
        pass

    def get_logs(self, existing_logs: Dict[uuid.UUID, List[float]] | None = None) -> Dict[uuid.UUID, List[float]]:
        """Recursively collect timing data from this module and all descendants.

        Traverses the global hierarchy (not ``nn.Module.children``), so it works
        even after PyTorch pipeline splitting has moved submodules. Thread-safe.

        Args:
            existing_logs: Dict to append into. If None, a new dict is created.

        Returns:
            Dict mapping each module's UUID to a list of elapsed-time measurements.
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
        """Return the unwrapped inner module.

        Returns:
            The original ``nn.Module`` passed at construction.
        """
        return self._inner

    def __getattr__(self, attr: str):
        """Delegate attribute access to the inner module when not found on self.

        Lookup order: ``nn.Module.__getattr__`` -> ``self._inner``.

        Args:
            attr: Attribute name.

        Returns:
            The attribute value.

        Raises:
            AttributeError: If neither this module nor the inner module has the attribute.
        """
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
        """Return a human-readable name including the inner module's name and UUID.

        Returns:
            String of the form ``Timed_<InnerName>[uuid:<uuid>]``.
        """
        return f"Timed_{self.inner()._get_name()}[uuid:{self.uuid}]"

    def get_path(self) -> str:
        """Return the dot-separated path of this module in the model hierarchy.

        Returns:
            The ``module_path`` string set at construction (e.g. ``"layer1.conv"``).
        """
        return self.module_path

    def to(self, *args, **kwargs):
        """Move the module to a device/dtype and update ``self.device`` accordingly.

        Args:
            *args: Forwarded to ``nn.Module.to``.
            **kwargs: Forwarded to ``nn.Module.to``.

        Returns:
            ``self``.
        """
        result = super().to(*args, **kwargs)
        # Update device from first parameter or buffer we can find
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            try:
                self.device = next(self.buffers()).device
            except StopIteration:
                pass  # No parameters or buffers, keep existing device
        return result


#==================
# CUDA GPU Timed Module
#==================
class CUDATimedModule(TimedModule):
    """Times forward passes using on-device ``clock64()`` CUDA kernels.

    Measures raw GPU cycles. ``torch.compile`` friendly. Requires the
    ``cuda_timing_kernel`` C++ extension to be built.

    Children are recursively wrapped up to ``depth`` levels.
    """

    def __init__(self, module: nn.Module, device, depth=1, parent_uuid: uuid.UUID = None, module_path: str = ""):
        """Wrap a module for CUDA cycle-level timing.

        Args:
            module: The ``nn.Module`` to wrap.
            device: CUDA device to place timing buffers on.
            depth: Levels of children to recursively wrap (``None`` = unlimited).
            parent_uuid: UUID of the parent ``TimedModule`` (None for root).
            module_path: Dot-separated path in the model tree.

        Raises:
            ImportError: If the ``cuda_timing_kernel`` extension is not available.
        """
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
        """Run the inner module bracketed by CUDA timer kernels.

        Args:
            *args: Forwarded to the inner module.
            **kwargs: Forwarded to the inner module.

        Returns:
            The inner module's output.
        """
        # 1. Start the timer
        cuda_timing_kernel_cpp.start(self.time_buffer)
        # 2. Run the inner module
        output = (self.inner())(*args, **kwargs)
        # 3. End the timer
        cuda_timing_kernel_cpp.end(self.time_buffer)

        return output

    def get_last_elapsed_cycles(self):
        """Synchronize the CUDA device and return the elapsed GPU cycles.

        Not ``torch.compile`` friendly (calls ``torch.cuda.synchronize``).

        Returns:
            Elapsed GPU cycles (int64) for the most recent forward pass.
        """
        # Sync the device where the buffer actually lives
        torch.cuda.synchronize(self.time_buffer.device)

        self.last_elapsed_cycles = self.time_buffer.item()
        return self.last_elapsed_cycles

    def _get_name(self):
        """Return a display name prefixed with ``CUDA_``.

        Returns:
            String of the form ``CUDA_Timed_<InnerName>[uuid:<uuid>]``.
        """
        return f"CUDA_{super()._get_name()}"


#==================
# CPU Timed Module
#==================
@torch.library.custom_op("model_runner::cpu_time_ns", mutates_args={"time_buffer"})
def cpu_time_ns(time_buffer: torch.Tensor) -> None:
    """Write ``time.time_ns()`` into ``time_buffer[0]``. ``torch.compile`` friendly custom op.

    Args:
        time_buffer: A 1-element int64 tensor (mutated in-place).
    """
    time_buffer[0] = time.time_ns()

@cpu_time_ns.register_fake
def _(time_buffer: torch.Tensor) -> None:
    return None


class CPUTimedModule(TimedModule):
    """Times forward passes on CPU using ``time.time_ns()`` via a custom op.

    ``torch.compile`` friendly. Children are recursively wrapped up to ``depth`` levels.
    """

    def __init__(self, module: nn.Module, device, depth=1, parent_uuid: uuid.UUID = None, module_path: str = ""):
        """Wrap a module for CPU nanosecond-level timing.

        Args:
            module: The ``nn.Module`` to wrap.
            device: Device to place the module on (should be CPU).
            depth: Levels of children to recursively wrap (``None`` = unlimited).
            parent_uuid: UUID of the parent ``TimedModule`` (None for root).
            module_path: Dot-separated path in the model tree.
        """
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
        """Run the inner module bracketed by nanosecond timestamps.

        Args:
            *args: Forwarded to the inner module.
            **kwargs: Forwarded to the inner module.

        Returns:
            The inner module's output.
        """
        # 1. Start timer (writes to buffer)
        cpu_time_ns(self.start_buffer)
        # 2. Run inner module
        output = (self.inner())(*args, **kwargs)
        # 3. End timer (writes to buffer)
        cpu_time_ns(self.end_buffer)

        return output

    def get_last_elapsed_cycles(self):
        """Return the elapsed wall-clock time of the most recent forward pass.

        Returns:
            Elapsed time in nanoseconds (int).
        """
        return (self.end_buffer[0] - self.start_buffer[0]).item()

    def _get_name(self):
        """Return a display name prefixed with ``CPU_``.

        Returns:
            String of the form ``CPU_Timed_<InnerName>[uuid:<uuid>]``.
        """
        return f"CPU_{super()._get_name()}"


#==================
def make_module_timed(module: nn.Module, device=None, depth=None) -> TimedModule:
    """Factory that wraps a module in the appropriate ``TimedModule`` subclass.

    Selects ``CUDATimedModule`` when ``device`` is a CUDA device (and the
    extension is available), otherwise ``CPUTimedModule``.

    Args:
        module: The ``nn.Module`` to wrap.
        device: Target device string or ``torch.device``. If None, auto-detects
            (CUDA if available, else CPU).
        depth: How many levels of children to recursively wrap. ``None`` for unlimited.

    Returns:
        A ``CUDATimedModule`` or ``CPUTimedModule`` wrapping the module.
    """
    if device is None:
        if torch.cuda.is_available() and cuda_timing_kernel_cpp is not None:
            device = "cuda"
        else:
            device = "cpu"

    if device == "cuda" or "cuda" in str(device):
        return CUDATimedModule(module, device, depth=depth)
    else:
        return CPUTimedModule(module, device, depth=depth)
