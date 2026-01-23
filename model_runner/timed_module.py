import uuid
from typing import Any, Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd.profiler_util import FunctionEventAvg
from torch.profiler import profile, record_function, ProfilerActivity

try:
    from . import gpu_timer
    print("gpu_timer imported successfully!")
except ImportError:
    print("Error: 'gpu_timer' module not found.")
    # print("Please build the extension first by running: pip install .")
    # TODO: ^^ this print is outdated
    gpu_timer = None


timed_module_registry = {}

class TimedModule(nn.Module):
    def __init__(self, module: nn.Module, device, depth=10, wrapping_a_wrapper=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner = module
        self.device = device
        self.depth = depth

        self.uuid = uuid.uuid4()
        timed_module_registry[self.uuid] = self
        self.parent_stage = None

        if wrapping_a_wrapper is not None:
            self.wrapping_a_wrapper = wrapping_a_wrapper
        else:
            self.wrapping_a_wrapper = not isinstance(module, nn.Module)

    def run(self, x=None):
        pass

    def get_logs(self, logs: Dict[uuid.UUID, List[float]] | None = None) -> Dict[uuid.UUID, List[float]]:
        pass

    def _inner(self) -> nn.Module:
        if not self.wrapping_a_wrapper:
            return self.inner
        else:
            return self.inner.model

    def _get_name(self):
        return self._inner()._get_name()

    def rand_inputs(self) -> Any:
        if callable(self.inner.rand_inputs):
            return self.inner.rand_inputs()
        else:
            print("Inner module does not have a `rand_inputs` function!")
            return None

timed_module_registry: Dict[uuid.UUID, TimedModule]
timed_module_structure: Dict[uuid.UUID, dict] # TODO: might be useful to store a tree structure of the models

#==================
# CUDA GPU Timed Module
#==================
class CUDATimedModule(TimedModule):
    """
    A wrapper that times the forward pass of a module using on-device clock64() CUDA kernels.
    This measures raw GPU cycles as seen by the device.
    torch.compile friendly.
    """

    def __init__(self, module: nn.Module, device, depth=1):
        super().__init__(module, device, depth)

        if gpu_timer is None:
            raise ImportError("CUDA extension 'gpu_timer' is not built.")

        self._inner().to(device)

        self.time_buffer = torch.zeros(1, dtype=torch.int64, device=device)
        self.last_elapsed_cycles = 0
        self.timing_depth = depth

        # Wrap children as well
        if depth is None or depth > 0:
            for child_name, child in list(self._inner().named_children()):
                # Decrement the depth when we go deeper in the tree
                d = None if depth is None else (depth - 1)
                wrapped_child = CUDATimedModule(child, device, depth=d)
                setattr(self._inner(), child_name, wrapped_child)
                # TODO: test this

    def forward(self, *args, **kwargs):
        # 1. Start the timer
        gpu_timer.start(self.time_buffer)
        # 2. Run the inner module
        output = self.inner(*args, **kwargs)
        # 3. End the timer
        gpu_timer.end(self.time_buffer)

        return output

    def get_last_elapsed_cycles(self):
        """
        Will synchronise and then output elapsed cycles.
        Not torch.compile friendly.
        """
        torch.cuda.synchronize()

        self.last_elapsed_cycles = self.time_buffer.item()
        return self.last_elapsed_cycles

    def get_logs(self, existing_logs: Dict[uuid.UUID, List[float]] | None = None) -> Dict[uuid.UUID, List[float]]:
        if existing_logs is None:
            existing_logs = {}

        def recurse_get_logs(module: TimedModule, logs: Dict[uuid.UUID, List[float]]):
            if logs.get(module.uuid, None) is None:
                logs[module.uuid] = module.get_last_elapsed_cycles()
            else:
                logs[module.uuid].append(module.get_last_elapsed_cycles())

            if module.depth > 0:
                for child_name, child in module._inner().named_children():
                    if isinstance(child, TimedModule):
                        child.recurse_get_logs(logs)

        recurse_get_logs(self, logs=existing_logs)
        return existing_logs

    def run(self, x=None) -> Any:
        if x is None:
            x = self.rand_inputs()

        with torch.no_grad():
            return self(x)


#==================
# CPU Timed Module
#==================
def _recursive_profiler_logs(prof_times: dict[str, Any], module: nn.Module, depth):
    module_name = module._get_name()
    logs = {
        'module_name': module_name,
        'times': {
            'elapsed': prof_times[module_name],
        },
        'children': [],
    }

    if depth > 0:
        for child_name, child in list(module.named_children()):
            if child_name in prof_times.keys():
                logs['children']['child_name'] = _recursive_profiler_logs(prof_times, child, depth - 1)

    return logs


def _make_time_lookup(events: list[FunctionEventAvg]):
    lookup = defaultdict(float)

    print("events:\n")
    for e in events:
        print(e)
        # 1. Grab the operation / module name
        name = getattr(e, "key", None) or getattr(e, "name", None) or getattr(e, "function_name", None)
        if name is None:
            continue

        # 2. Skip raw ATen operators and parameter‑access events
        if name.startswith("aten::") or name.endswith("weight"):
            # Look in its call stack for the python module name
            module_name = None
            for frame in e.stack:
                if ".forward" in frame.function:
                    # The function string is like 'SimpleNet.forward'
                    module_name = frame.function.split(".")[0]
                    break
            key = module_name
            if module_name is None:
                continue
        else:
            key = name

        # 4. Grab the per‑event CPU time
        time = getattr(e, "self_cpu_time_total", None)
        if time is None:
            time = getattr(e, "self_cpu_time", 0.0)

        lookup[key] += time

    return dict(lookup)


def _build_log_tree(module, lookup, parent_path=''):
    module_name = module._get_name()
    full_key = f'{parent_path}.{module_name}' if parent_path else module_name

    elapsed = lookup.get(full_key, 0.0)

    log = {
        'module_name': module_name,
        'times': {'elapsed': elapsed},
        'children': []
    }

    for child_name, child in module.named_children():
        # child_path = 'SimpleNet.0.conv1', 'SimpleNet.1.relu', …
        child_path = f'{full_key}.{child_name}'
        log['children'].append(
            _build_log_tree(child, lookup, parent_path=full_key)
        )

    return log


class CPUTimedModule(TimedModule):
    """
    A wrapper that times the forward pass of a module using the pytorch profiler.
    Should be torch.compile friendly.
    """

    def __init__(self, module: nn.Module, device, depth=10):
        super().__init__(module, device, depth)

        raise UserWarning("CPUTimedModule does not have logging working!")
        #TODO finish CPUTimedModule. Remaining issues:
        # 1. torch.compile mangles all the module names
        # 2. self.profiler.key_averages does not give us the stack events, despite doing everything it wanted of me

        # self.inner = torch.compile(self.inner)

        self._has_run = False

        self.profiler = profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,
        )

    def get_logs(self):
        if not self._has_run:
            return None

        lookup = _make_time_lookup(self.profiler.key_averages())

        print("lookup: \n", lookup)

        root_path = ''
        logs = _build_log_tree(self.inner, lookup, parent_path=root_path)

        return logs

    def run(self, x=None):
        if x is None:
            x = self.rand_inputs()

        self._has_run = True

        with self.profiler:
            with record_function(self.inner._get_name()):
                with torch.no_grad():
                    model_output = self.inner(x)

        print("Profiler Output:")
        print(self.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return model_output


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
