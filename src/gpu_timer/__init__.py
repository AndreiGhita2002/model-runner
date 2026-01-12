# src/gpu_timer/__init__.py
from torch.utils.cpp_extension import load
from pathlib import Path

_here = Path(__file__).parent

gpu_timer_cpp = load(
    name='gpu_timer_cpp',
    sources=[
        str(_here / 'gpu_timer_binding.cpp'),
        str(_here / 'gpu_timer_kernels.cu'),
    ],
    verbose=True
)

start = gpu_timer_cpp.start
end = gpu_timer_cpp.end
