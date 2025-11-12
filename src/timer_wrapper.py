from typing import Any

import torch
import torch.nn as nn

try:
    import gpu_timer_cpp
except ImportError:
    print("Error: 'gpu_timer_cpp' module not found.")
    print("Please build the extension first by running: pip install .")
    gpu_timer_cpp = None
    # TODO should fall back to CPU logging on Mac


class TimerWrapper(nn.Module):
    """
    A wrapper that times the forward pass of a module using on-device clock64() CUDA kernels.
    This measures raw GPU cycles as seen by the device.
    torch.compile friendly.
    """

    def __init__(self, module):
        super().__init__()

        if gpu_timer_cpp is None:
            raise ImportError("CUDA extension 'gpu_timer_cpp' is not built.")

        self.inner = module

        self.time_buffer = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.last_elapsed_cycles = 0

    def forward(self, *args, **kwargs):
        # 1. Start the timer
        gpu_timer_cpp.start(self.time_buffer)
        # 2. Run the inner module
        output = self.inner(*args, **kwargs)
        # 3. End the timer
        gpu_timer_cpp.end(self.time_buffer)

        return output

    def get_last_elapsed_cycles(self):
        """
        Will synchronise and then output elapsed cycles.
        Not torch.compile friendly.
        """
        torch.cuda.synchronize()

        self.last_elapsed_cycles = self.time_buffer.item()
        return self.last_elapsed_cycles

    # For Debug:
    def rand_inputs(self) -> Any:
        if callable(self.inner.rand_inputs):
            return self.inner.rand_inputs()
        else:
            print("Inner module does not have a `rand_inputs` function!")
            return None
