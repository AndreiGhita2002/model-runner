# src/gpu_timer/__init__.py
import torch.cuda
from torch.utils.cpp_extension import load
from pathlib import Path

#TODO: this should be renamed something to cuda_timer (make cuda part of the name)

_here = Path(__file__).parent

if torch.cuda.is_available():
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
else:
    print("gpu_timer cannot be built because PyTorch CUDA is not available.")
