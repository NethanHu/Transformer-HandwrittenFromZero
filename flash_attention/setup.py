from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention',
    ext_modules=[
        CUDAExtension(
            'flash_attention_cuda',
            ['flash_attention/flash_attention.cpp', 'flash_attention_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math',
                        '-gencode', 'arch=compute_70,code=sm_70',
                        '-gencode', 'arch=compute_80,code=sm_80'],
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=['flash_attention'],
)
