from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sum_square_cuda',
    ext_modules=[
        CUDAExtension('sum_square_cuda', [
            'sum_square_cuda.cpp',
            'sum_square_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
})
