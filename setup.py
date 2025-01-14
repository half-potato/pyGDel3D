from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='gdel3d',
    ext_modules=[
        CUDAExtension(
            name='gdel3d',
            sources=['src/main.cpp', 
                    'src/cuda_kernels.cu'],
            include_dirs=['include'],
            extra_compile_args={'cxx': ['-O2'],
                              'nvcc': ['-O2']},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
