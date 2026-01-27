# model_runner/cuda_timing_kernel/__init__.py
import torch.cuda
from torch.utils.cpp_extension import load
from pathlib import Path

_here = Path(__file__).parent

if torch.cuda.is_available():
    cuda_timing_kernel_cpp = load(
        name='cuda_timing_kernel_cpp',
        sources=[
            str(_here / 'cuda_timing_kernel_binding.cpp'),
            str(_here / 'cuda_timing_kernel.cu'),
        ],
        verbose=True
    )

    start = cuda_timing_kernel_cpp.start
    end = cuda_timing_kernel_cpp.end
else:
    print("cuda_timing_kernel cannot be built because PyTorch CUDA is not available.")
