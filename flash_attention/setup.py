from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='flash_attention',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'flash_attention.flash_attention_cuda',
            ['flash_attention/flash_attention.cpp', 'flash_attention_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math', '-gencode', 'arch=compute_70,code=sm_70',
                         '-gencode', 'arch=compute_75,code=sm_75',
                         '-gencode', 'arch=compute_80,code=sm_80',
                         '-gencode', 'arch=compute_86,code=sm_86'],
            }
        ),
    ],
    packages=['flash_attention'],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    install_requires=['torch'],
    python_requires='>=3.10',
)