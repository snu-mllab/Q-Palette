#ifndef GEMV_CUH
#define GEMV_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>

#include <torch/extension.h>
#include <cuda_runtime.h>

void pack_gemm(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
);

void pack_dequant(
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int w_bits
);
#endif 
