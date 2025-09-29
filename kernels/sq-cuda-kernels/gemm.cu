#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "gemm.h"
#include "gemm_routines.h"
#include "datatype.h"

////////////////////////////////////////////////////////////////////////////////
//                                     ANYPREC
////////////////////////////////////////////////////////////////////////////////

void pack_gemm_templated(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    uint32_t M = input.size(0);
    uint32_t N = output.size(2);
    uint32_t K = input.size(2);

    pack_matmul(
        (half*)input.data_ptr<c10::Half>(),
        (half*)output.data_ptr<c10::Half>(),
        (uint32_t*)qweight.data_ptr<int>(),
        (half*)lut.data_ptr<c10::Half>(),
        M, N, K,
        bitwidth,
        stream
    );
}

void pack_gemm_stream(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    TORCH_CHECK(bitwidth >= 2 && bitwidth <= 8, "Bitwidth must be between 2 and 8.");
    TORCH_CHECK(input.scalar_type() == lut.scalar_type() && input.scalar_type() == output.scalar_type(), 
                "Mismatched data types between input, lut, and output tensors.");
    TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight tensor must be of type int.");
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, hidden_size).");
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, hidden_size).");

    TORCH_CHECK(lut.dim() == 1 && lut.size(0) == (1 << bitwidth),
    "lut tensor must be of shape (2 ** bitwidth,). Expected (", (1 << bitwidth), "), got (", lut.size(0), ").");

    // qweight is of shape (bitwidth, output_feat, input_feat / 32)
    TORCH_CHECK(qweight.dim() == 2 && qweight.size(0) == output.size(2) && qweight.size(1) == bitwidth * input.size(2) / 32,
    "qweight tensor must be of shape (output_feat, bitwidth * input_feat / 32). Expected (", output.size(2), ", ", bitwidth * input.size(2) / 32, "), got (", qweight.size(0), ", ", qweight.size(1), ").");

    // Check that sequence length is 1
    TORCH_CHECK(input.size(1) == 1, "Only sequence length of 1 is supported.");
    TORCH_CHECK(output.size(1) == 1, "Only sequence length of 1 is supported.");

    // Check that input and output are both on GPU
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output tensors must be on GPU.");

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(qweight.is_contiguous(), "qweight tensor must be contiguous.");
    TORCH_CHECK(lut.is_contiguous(), "lut tensor must be contiguous.");

    auto dtype = input.scalar_type();
    if (dtype == at::kFloat) {
        TORCH_CHECK(false, "SQ GEMM does not support float data type. Please use half.");
    } else if (dtype == at::kHalf) {
        pack_gemm_templated(input, output, qweight, lut, bitwidth, stream);
    } else if (dtype == at::kBFloat16) {
        TORCH_CHECK(false, "SQ GEMM does not support bfloat16 data type. Please use half.");
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

void pack_gemm(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pack_gemm_stream(input, output, qweight, lut, bitwidth, stream);
}

void pack_dequant_templated(
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int w_bits,
    cudaStream_t stream
) {
    TORCH_CHECK(qweight.ndimension() == 2 && qweight.dtype() == torch::kInt && lut.dtype() == torch::kHalf);
    TORCH_CHECK(qweight.device() == lut.device() && qweight.is_cuda());
    TORCH_CHECK(w_bits >= 2 && w_bits <= 8);
    const int N = qweight.size(0);
    const int K = qweight.size(1) * 32 / w_bits;

    pack_dequant_kbit(
        (uint32_t *)qweight.data_ptr<int>(),
        N, K,
        (half *)lut.data_ptr<c10::Half>(),
        (half *)output.data_ptr<c10::Half>(),
        w_bits,
        stream
    );
}

void pack_dequant(
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int w_bits
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pack_dequant_templated(output, qweight, lut, w_bits, stream);
}