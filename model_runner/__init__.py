# model_runner package
from .pipeline_server import PipelineServer
from .util import uuids_to_tensor, tensor_to_uuids
from .pipeline_runner import PipelineRunner, ForwardResult
from .timed_module import TimedModule, make_module_timed, timed_module_registry, timed_module_hierarchy
from .pipeline_optimizer import (
    PipelineOptimizer, GreedyPipelineOptimizer, StaticGPipeOptimizer,
    StaticConfigOptimizer,
    ReactiveShishaOptimiser, ExhaustiveShishaOptimizer, PipelineConfig,
    TimeBasedShishaPipelineOptimizer,  # backward-compat alias
)
from .adaptive_pipeline import AdaptivePipeline
from .device_manager import DeviceManager
from .flask_app import create_flask_app
import warnings

# Suppress PyTorch internal FutureWarning about LeafSpec deprecation
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=FutureWarning)

__all__ = [
    "PipelineServer",
    "PipelineRunner",
    "ForwardResult",
    "uuids_to_tensor",
    "tensor_to_uuids",
    "DeviceManager",
    "PipelineOptimizer",
    "GreedyPipelineOptimizer",
    "StaticGPipeOptimizer",
    "StaticConfigOptimizer",
    "ReactiveShishaOptimiser",
    "ExhaustiveShishaOptimizer",
    "TimeBasedShishaPipelineOptimizer",  # backward-compat alias
    "AdaptivePipeline",
    "PipelineConfig",
    "TimedModule",
    "make_module_timed",
    "timed_module_registry",
    "timed_module_hierarchy",
    "create_flask_app",
]
