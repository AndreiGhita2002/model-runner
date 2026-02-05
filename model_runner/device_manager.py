from typing import List, Dict

import torch


class DeviceManager:
    """Discovers and manages available compute devices (CPU and CUDA GPUs).

    Devices are indexed starting at 0 (always CPU), followed by any CUDA GPUs.
    """

    def __init__(self, verbose: bool = False):
        """Detect all available devices and store them.

        Args:
            verbose: If True, print discovered device information to stdout.
        """
        self.devices: List[torch.device] = []
        self.verbose = verbose
        self._initialize_devices()

    def _log(self, msg: str):
        """Print a message to stdout if verbose logging is enabled.

        Args:
            msg: The message to print.
        """
        if self.verbose:
            print(msg)

    def _initialize_devices(self):
        """Populate ``self.devices`` with CPU (always first) and any CUDA GPUs."""
        # Always add CPU
        cpu_device = torch.device("cpu")
        self.devices.append(cpu_device)
        self._log("Detected devices:")
        self._log("  Device 0: CPU")

        # Add CUDA devices if available
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            for i in range(cuda_count):
                device = torch.device(f"cuda:{i}")
                props = torch.cuda.get_device_properties(i)
                idx = len(self.devices)
                self._log(f"  Device {idx}: {props.name}")
                self._log(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
                self._log(f"    Compute capability: {props.major}.{props.minor}")
                self.devices.append(device)

        self._log(f"Total: {len(self.devices)} device(s)")

    def has_cuda(self) -> bool:
        """Check whether any CUDA devices were detected.

        Returns:
            True if at least one CUDA device is available.
        """
        return any(d.type == "cuda" for d in self.devices)

    def get_cuda_devices(self) -> List[torch.device]:
        """Return all detected CUDA devices.

        Returns:
            List of CUDA ``torch.device`` objects (empty if none available).
        """
        return [d for d in self.devices if d.type == "cuda"]

    def get_cpu_device(self) -> torch.device:
        """Return the CPU device.

        Returns:
            ``torch.device("cpu")``.
        """
        return torch.device("cpu")

    def get_device(self, index: int = 0) -> torch.device:
        """Return a device by its index (0 = CPU, 1+ = CUDA GPUs).

        Args:
            index: Zero-based device index.

        Returns:
            The ``torch.device`` at the given index.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if index >= len(self.devices):
            raise IndexError(f"Device index {index} out of range. Only {len(self.devices)} devices available.")
        return self.devices[index]

    def get_all_devices(self) -> List[torch.device]:
        """Return a copy of the full device list (CPU first, then CUDA).

        Returns:
            List of all ``torch.device`` objects.
        """
        return self.devices.copy()

    def num_devices(self) -> int:
        """Return the total number of available devices (CPU + CUDA).

        Returns:
            Device count.
        """
        return len(self.devices)

    def num_cuda_devices(self) -> int:
        """Return the number of CUDA devices.

        Returns:
            CUDA device count (0 if CUDA is unavailable).
        """
        return len(self.get_cuda_devices())

    def get_device_memory_info(self, device_index: int = 0) -> Dict[str, float]:
        """Return memory usage for a device.

        For CUDA devices, reports actual GPU memory. For CPU, uses ``psutil``
        if available, otherwise returns zeros.

        Args:
            device_index: Zero-based device index.

        Returns:
            Dict with ``allocated``, ``reserved``, and ``total`` in GB.
        """
        device = self.get_device(device_index)

        if device.type == "cuda":
            cuda_index = device.index if device.index is not None else 0
            return {
                'allocated': torch.cuda.memory_allocated(cuda_index) / 1e9,
                'reserved': torch.cuda.memory_reserved(cuda_index) / 1e9,
                'total': torch.cuda.get_device_properties(cuda_index).total_memory / 1e9
            }
        else:
            # CPU memory info (approximate using system memory)
            try:
                import psutil
                mem = psutil.virtual_memory()
                return {
                    'allocated': (mem.total - mem.available) / 1e9,
                    'reserved': 0.0,
                    'total': mem.total / 1e9
                }
            except ImportError:
                # psutil not available, return placeholder values
                return {
                    'allocated': 0.0,
                    'reserved': 0.0,
                    'total': 0.0
                }
