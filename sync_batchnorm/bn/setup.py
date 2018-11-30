from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bn_cuda',
    ext_modules=[
        CUDAExtension('bn_cuda', [
            'bn_cuda.cpp',
            'bn_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
})
