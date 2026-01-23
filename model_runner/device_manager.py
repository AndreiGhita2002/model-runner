from typing import List, Dict

import torch


class DeviceManager:
    """Manages available CUDA devices for model distribution."""

    # TODO: add support for CPUs

    def __init__(self):
        self.devices: List[torch.device] = []
        self._initialize_devices()

    def _initialize_devices(self):
        """Detect and initialize all available CUDA devices."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This load balancer requires CUDA devices.")

        device_num = torch.cuda.device_count()
        print(f"Detected {device_num} CUDA device(s)")

        for i in range(device_num):
            device = torch.device(f"cuda:{i}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    PyTorch name: {str(device)}")
            self.devices.append(device)

    def get_device(self, index: int = 0) -> torch.device:
        """Get device by index."""
        if index >= len(self.devices):
            raise IndexError(f"Device index {index} out of range. Only {len(self.devices)} devices available.")
        return self.devices[index]

    def get_all_devices(self) -> List[torch.device]:
        """Get all available devices."""
        return self.devices.copy()

    def num_devices(self) -> int:
        """Return a number of available devices."""
        return len(self.devices)

    def get_device_memory_info(self, device_index: int = 0) -> Dict[str, float]:
        """Get memory information for a specific device."""
        torch.cuda.set_device(device_index)
        return {
            'allocated': torch.cuda.memory_allocated(device_index) / 1e9,
            'reserved': torch.cuda.memory_reserved(device_index) / 1e9,
            'total': torch.cuda.get_device_properties(device_index).total_memory / 1e9
        }
