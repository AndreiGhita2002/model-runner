from typing import List, Dict

import torch


class DeviceManager:
    """Manages all available devices (CPU and CUDA) for model distribution."""

    def __init__(self, verbose: bool = False):
        """Initialise the device manager with all available devices."""
        self.devices: List[torch.device] = []
        self.verbose = verbose
        self._initialize_devices()

    def _log(self, msg: str):
        """Print message if verbose logging is enabled."""
        if self.verbose:
            print(msg)

    def _initialize_devices(self):
        """Detect and initialise all available devices (CPU + CUDA)."""
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
        """Return True if CUDA devices are available."""
        return any(d.type == "cuda" for d in self.devices)

    def get_cuda_devices(self) -> List[torch.device]:
        """Get all CUDA devices."""
        return [d for d in self.devices if d.type == "cuda"]

    def get_cpu_device(self) -> torch.device:
        """Get the CPU device."""
        return torch.device("cpu")

    def get_device(self, index: int = 0) -> torch.device:
        """Get device by index."""
        if index >= len(self.devices):
            raise IndexError(f"Device index {index} out of range. Only {len(self.devices)} devices available.")
        return self.devices[index]

    def get_all_devices(self) -> List[torch.device]:
        """Get all available devices."""
        return self.devices.copy()

    def num_devices(self) -> int:
        """Return the total number of available devices."""
        return len(self.devices)

    def num_cuda_devices(self) -> int:
        """Return the number of CUDA devices."""
        return len(self.get_cuda_devices())

    def get_device_memory_info(self, device_index: int = 0) -> Dict[str, float]:
        """Get memory information for a specific device."""
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
