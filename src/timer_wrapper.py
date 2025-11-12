from typing import Any

import torch
import torch.nn as nn

try:
    import gpu_timer_cpp
    GPU_TIMER_MODULE_AVAILABLE = True
    print("Successfully imported 'gpu_timer_cpp' CUDA extension.")
except ImportError:
    print("Warning: 'gpu_timer_cpp' module not built or not found.")
    print("Clock64TimerWrapper will not time operations on CUDA.")
    gpu_timer_cpp = None
    GPU_TIMER_MODULE_AVAILABLE = False


class TimerWrapper(nn.Module):
    """
    A wrapper that times the forward pass of a module using on-device clock64() CUDA kernels.
    This measures raw GPU cycles as seen by the device.
    torch.compile friendly.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

        # Will be initialised on the first forward pass
        self.timer_mode = "disabled"  # 'cuda', 'mps', 'cpu'
        self.last_measurement = 0
        self.time_buffer = None # For CUDA
        self.start_event_mps = None # For MPS
        self.end_event_mps = None # For MPS
        self._warned_cpu = False

    def _init_timer(self, device):
        """
        Called on the first forward pass to set up the correct
        timer (CUDA or MPS) for the module's device.
        """
        if self.timer_mode != "disabled":  # Already initialised
            return

        if device.type == 'cuda' and GPU_TIMER_MODULE_AVAILABLE:
            print(f"ModuleTimerWrapper: Initializing CUDA timer (clock64) for device: {device}")
            self.time_buffer = torch.zeros(1, dtype=torch.int64, device=device)
            self.timer_mode = "cuda"

        elif device.type == 'mps':
            print(f"ModuleTimerWrapper: Initializing MPS timer (Events) for device: {device}")
            # apparently `enable_timing=True` is not used for mps.Event
            self.start_event_mps = torch.mps.Event(enable_timing=True)
            self.end_event_mps = torch.mps.Event(enable_timing=True)
            self.timer_mode = "mps"

        elif device.type == 'cpu':
            if not self._warned_cpu:
                print("ModuleTimerWrapper: CPU device detected. Timing is disabled.")
                self._warned_cpu = True
            self.timer_mode = "cpu"

        else:
            if GPU_TIMER_MODULE_AVAILABLE:
                print(f"ModuleTimerWrapper: Device {device.type} unsupported. Timing disabled.")
            else:
                print(f"ModuleTimerWrapper: CUDA extension not found. Timing disabled for device {device.type}.")
            self.timer_mode = "disabled"

    def forward(self, *args, **kwargs):
        input_device = None
        if args and isinstance(args[0], torch.Tensor):
            input_device = args[0].device

        if input_device is None:
            # On CPU just run the module
            print("TimerWrapper.module is on the CPU!")
            # TODO you could have classic CPU only logging with
            return self.module(*args, **kwargs)

        # Initialise if this is the first pass
        self._init_timer(input_device)

        # 1. Start the timer
        if self.timer_mode == "cuda":
            gpu_timer_cpp.start(self.time_buffer)
        elif self.timer_mode == "mps":
            self.start_event_mps.record()

        # 2. Run the inner module
        output = self.module(*args, **kwargs)

        # 3. End the timer
        if self.timer_mode == "cuda":
            gpu_timer_cpp.end(self.time_buffer)
        elif self.timer_mode == "mps":
            self.end_event_mps.record()

        return output

    def get_last_elapsed_cycles(self):
        """
        Synchronises the device stream and returns the last
        measurement.

        - On CUDA: Returns elapsed GPU CYCLES (int).
        - On MPS: Returns elapsed WALL TIME (milliseconds, float).
        - On CPU: Returns 0. # TODO implement timing on CPU
        """
        if self.timer_mode == "cuda":
            torch.cuda.synchronize()
            self.last_measurement = self.time_buffer.item()
            print(f"(Measurement type: GPU Cycles)")
            return self.last_measurement

        elif self.timer_mode == "mps":
            self.end_event_mps.synchronize()

            self.start_event_mps.synchronize() # Wait for the start kernel
            self.end_event_mps.synchronize()   # Wait for the end kernel

            elapsed = self.start_event_mps.elapsed_time(self.end_event_mps)

            # Sticking to the compile-friendly approach:
            # We can't get a time, but we can synchronise.
            print("MPS: Synchronized stream. `elapsed_time` not supported.")
            print("(Measurement type: N/A for MPS, returning 0)")
            print(elapsed)
            self.last_measurement = elapsed # Cannot be measured this way
            return self.last_measurement

        else: # 'cpu' or 'disabled'
            if self.timer_mode == 'cpu':
                print("Timer is disabled (CPU). Returning 0.")
            return 0

    # For Debug:
    def rand_inputs(self) -> Any:
        if callable(self.inner.rand_inputs):
            return self.inner.rand_inputs()
        else:
            print("Inner module does not have a `rand_inputs` function!")
            return None


if __name__ == "__main__":

    # --- Test on CUDA (if available) ---
    if torch.cuda.is_available() and GPU_TIMER_MODULE_AVAILABLE:
        print("\n" + "=" * 30)
        print("--- Testing Eager Mode (CUDA) ---")
        print("=" * 30)

        my_module_cuda = nn.Sequential(
            nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024)
        ).cuda()

        timed_module_cuda = TimerWrapper(my_module_cuda)
        x_cuda = torch.randn(512, 1024, device="cuda")

        print("Running first forward pass (warmup)...")
        _ = timed_module_cuda(x_cuda)
        print(f"Result: {timed_module_cuda.get_last_measurement()}")

        print("\nRunning second forward pass...")
        _ = timed_module_cuda(x_cuda)
        print(f"Result: {timed_module_cuda.get_last_measurement()}")

        print("\n" + "=" * 30)
        print("--- Testing torch.compile Mode (CUDA) ---")
        print("=" * 30)

        my_module_cuda_compiled = torch.compile(my_module_cuda)
        compiled_timed_module_cuda = TimerWrapper(my_module_cuda_compiled)

        print("Running first compiled forward pass (warmup)...")
        _ = compiled_timed_module_cuda(x_cuda)
        print(f"Result (compiled): {compiled_timed_module_cuda.get_last_measurement()}")

        print("\nRunning second compiled forward pass...")
        _ = compiled_timed_module_cuda(x_cuda)
        print(f"Result (compiled): {compiled_timed_module_cuda.get_last_measurement()}")

    else:
        print("\nSkipping CUDA tests (CUDA not available or extension not built).")

    # --- Test on MPS (if available) ---
    if torch.backends.mps.is_available():
        print("\n" + "=" * 30)
        print("--- Testing Eager Mode (MPS) ---")
        print("=" * 30)

        my_module_mps = nn.Sequential(
            nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024)
        ).to("mps")

        timed_module_mps = TimerWrapper(my_module_mps)
        x_mps = torch.randn(512, 1024, device="mps")

        print("Running first forward pass (warmup) on MPS...")
        _ = timed_module_mps(x_mps)
        print(f"Result: {timed_module_mps.last_measurement}")

        print("\nRunning second forward pass on MPS...")
        _ = timed_module_mps(x_mps)
        print(f"Result: {timed_module_mps.last_measurement}")

    else:
        print("\nSkipping MPS tests (MPS not available).")
