from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="tcq_kernels_cuda",
    ext_modules=[
        cpp_extension.CUDAExtension(name="tcq_kernels",
                      sources=[
                          "src/wrapper.cpp", "src/inference.cu",
                          "src/qtip_torch.cu"
                      ],
                      extra_compile_args={
                          "cxx":
                          ["-O3", "--fast-math", "-lineinfo", "-std=c++17"],
                          "nvcc": [
                              "-O3", "--use_fast_math", "-lineinfo", "-keep",
                              "-std=c++17", "--ptxas-options=-v",
                              "-gencode=arch=compute_89,code=sm_89",  # Ada RTX 6000, RTX 4090, etc. 
                          ],
                      })
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
)
