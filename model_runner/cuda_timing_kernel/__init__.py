# model_runner/cuda_timing_kernel/__init__.py
import torch.cuda
from torch.utils.cpp_extension import load
from pathlib import Path

_here = Path(__file__).parent

cuda_timing_kernel_cpp = None

if torch.cuda.is_available():
    try:
        cuda_timing_kernel_cpp = load(
            name='cuda_timing_kernel_cpp',
            sources=[
                str(_here / 'cuda_timing_kernel_binding.cpp'),
                str(_here / 'cuda_timing_kernel.cu'),
            ],
            verbose=False
        )

        start = cuda_timing_kernel_cpp.start
        end = cuda_timing_kernel_cpp.end
    except Exception:
        cuda_timing_kernel_cpp = None
