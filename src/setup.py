from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.cuda import is_available as is_cuda_available

ext_modules = []

if is_cuda_available():
    print("CUDA available. Building CUDA extension 'gpu_timer_cpp'.")
    ext_modules.append(
        CUDAExtension('gpu_timer_cpp', [
            'gpu_timer_binding.cpp',
            'gpu_timer_kernels.cu',
        ])
    )
else:
    print("CUDA not available. Skipping build of 'gpu_timer_cpp'.")
    print("The 'gpu_timer_cpp' module will not be available.")

setup(
    name='gpu_timer_cpp',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    })
