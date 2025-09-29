#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("pack_gemm", &pack_gemm, "PACK GEMM");
	m.def("pack_dequant", &pack_dequant, "PACK DEQUANT");
}
