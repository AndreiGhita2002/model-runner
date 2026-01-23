import queue
from typing import Any, List, Dict

import torch
from torch import nn
from torch.multiprocessing.queue import Queue

from . import AdaptivePipeline, DeviceManager
from .timed_module import make_module_timed


class MainService:
    """
    Main service that manages adaptive pipelines.
    """

    pipelines: dict[str, AdaptivePipeline]
    # Work Queue: (request ID, model name, input data): Tuple[int, str, Any]
    work_queue: Queue = queue.Queue()
    # Model Outputs: request ID -> output # TODO: make sure this Dict is multi thread safe
    model_outputs: Dict[int, Any] = {}
    # Verbose logging flag
    verbose: bool = False

    def __init__(self, depth=2, verbose=False):
        """
        Args:
            depth: Depth for TimedModule profiling
            verbose: Enable verbose logging output
        """
        self.device_manager = DeviceManager()
        self.primary_device = self.device_manager.get_device(0)

        self.depth = depth
        self.verbose = verbose

    def _log(self, msg: str):
        """Print message if verbose logging is enabled."""
        if self.verbose:
            print(msg)

    def add_model(self, model_name: str, model: nn.Module, device=None, depth: int | None=None):
        # TODO: add **kwargs here ^^ and pass them to pipeline and timed module
        if device is None:
            device = str(self.primary_device)

        depth = depth or self.depth

        if self.pipelines.get(model_name, None) is not None:
            raise Exception(f"Pipeline with name {model_name} already exists!")

        timed_model = make_module_timed(
            model,
            device=device,
            depth=depth
        )

        self.pipelines[model_name] = AdaptivePipeline(
            timed_model,
            model_name,
            device_manager=self.device_manager,
            verbose=self.verbose,
        )

    def queue_work(self, model_name: str, x: Any, request_id: int):
        self.work_queue.put((request_id, model_name, x))

    def get_work_results(self, request_id: int) -> Any | None:
        return self.model_outputs.get(request_id, None)

    def run(self, exit_when_done = False):
        self._log("MainService.run: starting main loop")
        # main loop
        while True:
            # check queue
            if not self.work_queue.empty():
                #run models
                (req_id, model_name, work) = self.work_queue.get(block=True)
                self._log(f"MainService.run: processing request {req_id} for model '{model_name}'")

                # TODO: make this async
                output = self.pipelines[model_name].forward(work)
                self._log(f"MainService.run: completed request {req_id}, output type: {type(output).__name__}")

                # output the output
                self.model_outputs[req_id] = output
            else:
                #TODO: make MainService.run more like a service
                # sleep for a bit
                # until the user requests another workload
                # or perhaps exit

                # For now, we exit when the queue is done
                if exit_when_done:
                    self._log("MainService.run: queue empty, exiting")
                    return

    def get_logs(self) -> Dict[str, Any]:
        """Get timing logs from all pipelines."""
        logs = {}
        for model_name, pipeline in self.pipelines.items():
            logs[model_name] = pipeline.time_logs
        return logs

    def get_model_names(self) -> List[str]:
        """Get a list of available model names."""
        return list(self.pipelines.keys())

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices."""
        info = {
            'num_devices': self.device_manager.num_devices(),
            'devices': []
        }

        for i, device in enumerate(self.device_manager.get_all_devices()):
            device_info = {
                'index': i,
                'name': torch.cuda.get_device_properties(i).name,
                'memory': self.device_manager.get_device_memory_info(i)
            }
            info['devices'].append(device_info)

        return info

    def print_status(self):
        """Print current service status."""
        print("\n" + "=" * 80)
        print("MainService Status")
        print("=" * 80)
        print(f"Number of pipelines: {len(self.pipelines)}")

        device_info = self.get_device_info()
        print(f"\nDevices ({device_info['num_devices']}):")
        for dev in device_info['devices']:
            print(f"  [{dev['index']}] {dev['name']}")
            mem = dev['memory']
            print(f"      Memory: {mem['allocated']:.2f}/{mem['total']:.2f} GB allocated")

        print("\nPipelines:")
        for name, pipeline in self.pipelines.items():
            num_stages = len(pipeline.stages) if pipeline.stages else 0
            num_logs = len(pipeline.time_logs)
            print(f"  {name}: stages={num_stages}, time_logs={num_logs}")
