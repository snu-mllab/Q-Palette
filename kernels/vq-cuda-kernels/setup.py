import os
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="vq_pack_gemm",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="vq_pack_gemm", 
            sources=["src/bindings.cpp", "src/gemm.cu", "src/gemm_routines.cu"],
            extra_compile_args={
                'cxx': ["-O3", "-DENABLE_BF16", "-std=c++17"],
                'nvcc': [
                    '-lineinfo', "-std=c++17", "-keep", "-O3", 
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_HALF2_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "-gencode=arch=compute_89,code=sm_89",
                ]
            },
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
)
