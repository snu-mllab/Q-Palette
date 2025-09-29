#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>
#include <torch/extension.h>
#include "gemm_routines.cu"
#include "gemm_routines.h"

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)          do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (false)
using namespace torch::indexing;

template <int maxm, int nbits, int vec_sz, int code_n, typename codeT, int recons_n>
__host__ static void vq_pack_gemm(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut
) {
    CHECK_INPUT(output);
    TORCH_CHECK(output.ndimension() == 3);
    TORCH_CHECK(output.scalar_type() == torch::kHalf);

    CHECK_INPUT(input);
    TORCH_CHECK(input.ndimension() == 3);
    TORCH_CHECK(input.scalar_type() == torch::kHalf);
    
    // Check that sequence length is 1
    TORCH_CHECK(input.size(1) == 1, "Only sequence length of 1 is supported.");
    TORCH_CHECK(output.size(1) == 1, "Only sequence length of 1 is supported.");

    CHECK_INPUT(qweight);
    if constexpr (std::is_same_v<codeT, uint16_t>) {
        TORCH_CHECK(qweight.ndimension() == 2 && qweight.dtype() == torch::kUInt16 && lut.dtype() == torch::kHalf);
    } else if constexpr (std::is_same_v<codeT, uint32_t>) {
        TORCH_CHECK(qweight.ndimension() == 2 && qweight.dtype() == torch::kUInt32 && lut.dtype() == torch::kHalf);
    } else if constexpr (std::is_same_v<codeT, uint64_t>) {
        TORCH_CHECK(qweight.ndimension() == 2 && qweight.dtype() == torch::kUInt64 && lut.dtype() == torch::kHalf);
    }

    uint32_t N = qweight.size(0);
    uint32_t codeT_sz = sizeof(codeT) * 8;
    uint32_t K = qweight.size(1) * codeT_sz / nbits * vec_sz;
    uint32_t M = input.numel() / K;

    TORCH_CHECK(M == maxm, "M must be equal to maxm.");
    TORCH_CHECK(input.device() == qweight.device() && input.device() == lut.device() && input.is_cuda());

    vq_pack_gemm_ptr<maxm, nbits, vec_sz, code_n, codeT, recons_n>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<codeT*>(qweight.data_ptr<codeT>()),
        reinterpret_cast<half*>(lut.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
}

template <int nbits, int vec_sz, int code_n, typename codeT, int recons_n>
__host__ static void vq_pack_dequant(
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut
) {
    CHECK_INPUT(output);
    TORCH_CHECK(output.ndimension() == 2);
    TORCH_CHECK(output.scalar_type() == torch::kHalf);

    CHECK_INPUT(qweight);
    if constexpr (std::is_same_v<codeT, uint16_t>) {
        TORCH_CHECK(qweight.ndimension() == 2 && qweight.dtype() == torch::kUInt16 && lut.dtype() == torch::kHalf);
    } else if constexpr (std::is_same_v<codeT, uint32_t>) {
        TORCH_CHECK(qweight.ndimension() == 2 && qweight.dtype() == torch::kUInt32 && lut.dtype() == torch::kHalf);
    } else if constexpr (std::is_same_v<codeT, uint64_t>) {
        TORCH_CHECK(qweight.ndimension() == 2 && qweight.dtype() == torch::kUInt64 && lut.dtype() == torch::kHalf);
    }
    uint32_t N = qweight.size(0);
    uint32_t codeT_sz = sizeof(codeT) * 8;
    uint32_t K = qweight.size(1) * codeT_sz / nbits * vec_sz;

    vq_pack_dequant_ptr<nbits, vec_sz, code_n, codeT, recons_n>(
        reinterpret_cast<codeT*>(qweight.data_ptr<codeT>()),
        N, K,
        reinterpret_cast<half*>(lut.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        at::cuda::getCurrentCUDAStream()
    );
}



__host__ extern void vq_pack_dequant_3_2_3_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<3, 2, 3, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_3_2_3_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 3, 2, 3, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_3_2_3_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 3, 2, 3, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_3_2_3_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 3, 2, 3, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_3_2_3_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 3, 2, 3, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_3_2_6_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<3, 2, 6, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_3_2_6_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 3, 2, 6, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_3_2_6_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 3, 2, 6, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_3_2_6_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 3, 2, 6, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_3_2_6_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 3, 2, 6, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_4_2_2_32_16(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<4, 2, 2, uint32_t, 16>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_4_2_2_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 4, 2, 2, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_4_2_2_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 4, 2, 2, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_4_2_2_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 4, 2, 2, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_4_2_2_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 4, 2, 2, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_4_2_4_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<4, 2, 4, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_4_2_4_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 4, 2, 4, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_4_2_4_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 4, 2, 4, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_4_2_4_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 4, 2, 4, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_4_2_4_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 4, 2, 4, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_4_2_8_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<4, 2, 8, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_4_2_8_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 4, 2, 8, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_4_2_8_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 4, 2, 8, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_4_2_8_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 4, 2, 8, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_4_2_8_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 4, 2, 8, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_5_2_5_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<5, 2, 5, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_5_2_5_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 5, 2, 5, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_5_2_5_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 5, 2, 5, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_5_2_5_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 5, 2, 5, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_5_2_5_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 5, 2, 5, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_5_2_10_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<5, 2, 10, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_5_2_10_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 5, 2, 10, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_5_2_10_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 5, 2, 10, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_5_2_10_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 5, 2, 10, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_5_2_10_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 5, 2, 10, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_6_2_3_32_16(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<6, 2, 3, uint32_t, 16>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_6_2_3_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 6, 2, 3, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_6_2_3_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 6, 2, 3, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_6_2_3_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 6, 2, 3, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_6_2_3_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 6, 2, 3, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_6_2_6_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<6, 2, 6, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_6_2_6_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 6, 2, 6, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_6_2_6_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 6, 2, 6, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_6_2_6_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 6, 2, 6, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_6_2_6_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 6, 2, 6, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_6_2_12_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<6, 2, 12, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_6_2_12_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 6, 2, 12, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_6_2_12_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 6, 2, 12, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_6_2_12_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 6, 2, 12, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_6_2_12_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 6, 2, 12, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_7_2_7_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<7, 2, 7, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_7_2_7_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 7, 2, 7, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_7_2_7_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 7, 2, 7, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_7_2_7_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 7, 2, 7, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_7_2_7_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 7, 2, 7, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_7_2_14_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<7, 2, 14, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_7_2_14_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 7, 2, 14, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_7_2_14_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 7, 2, 14, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_7_2_14_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 7, 2, 14, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_7_2_14_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 7, 2, 14, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_8_2_4_32_16(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<8, 2, 4, uint32_t, 16>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_8_2_4_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 8, 2, 4, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_8_2_4_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 8, 2, 4, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_8_2_4_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 8, 2, 4, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_8_2_4_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 8, 2, 4, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_8_2_8_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<8, 2, 8, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_8_2_8_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 8, 2, 8, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_8_2_8_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 8, 2, 8, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_8_2_8_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 8, 2, 8, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_8_2_8_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 8, 2, 8, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_8_2_16_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<8, 2, 16, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_8_2_16_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 8, 2, 16, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_8_2_16_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 8, 2, 16, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_8_2_16_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 8, 2, 16, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_8_2_16_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 8, 2, 16, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_9_2_9_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<9, 2, 9, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_9_2_9_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 9, 2, 9, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_9_2_9_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 9, 2, 9, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_9_2_9_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 9, 2, 9, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_9_2_9_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 9, 2, 9, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_9_2_18_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<9, 2, 18, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_9_2_18_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 9, 2, 18, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_9_2_18_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 9, 2, 18, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_9_2_18_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 9, 2, 18, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_9_2_18_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 9, 2, 18, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_10_2_5_32_16(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<10, 2, 5, uint32_t, 16>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_10_2_5_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 10, 2, 5, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_10_2_5_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 10, 2, 5, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_10_2_5_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 10, 2, 5, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_10_2_5_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 10, 2, 5, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_10_2_10_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<10, 2, 10, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_10_2_10_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 10, 2, 10, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_10_2_10_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 10, 2, 10, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_10_2_10_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 10, 2, 10, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_10_2_10_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 10, 2, 10, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_10_2_20_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<10, 2, 20, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_10_2_20_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 10, 2, 20, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_10_2_20_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 10, 2, 20, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_10_2_20_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 10, 2, 20, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_10_2_20_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 10, 2, 20, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_11_2_11_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<11, 2, 11, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_11_2_11_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 11, 2, 11, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_11_2_11_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 11, 2, 11, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_11_2_11_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 11, 2, 11, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_11_2_11_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 11, 2, 11, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_11_2_22_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<11, 2, 22, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_11_2_22_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 11, 2, 22, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_11_2_22_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 11, 2, 22, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_11_2_22_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 11, 2, 22, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_11_2_22_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 11, 2, 22, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_12_2_6_32_16(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<12, 2, 6, uint32_t, 16>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_12_2_6_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 12, 2, 6, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_12_2_6_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 12, 2, 6, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_12_2_6_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 12, 2, 6, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_12_2_6_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 12, 2, 6, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_12_2_12_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<12, 2, 12, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_12_2_12_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 12, 2, 12, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_12_2_12_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 12, 2, 12, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_12_2_12_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 12, 2, 12, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_12_2_12_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 12, 2, 12, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_12_2_24_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<12, 2, 24, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_12_2_24_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 12, 2, 24, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_12_2_24_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 12, 2, 24, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_12_2_24_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 12, 2, 24, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_12_2_24_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 12, 2, 24, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_6_4_3_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<6, 4, 3, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_6_4_3_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 6, 4, 3, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_6_4_3_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 6, 4, 3, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_6_4_3_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 6, 4, 3, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_6_4_3_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 6, 4, 3, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_6_4_6_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<6, 4, 6, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_6_4_6_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 6, 4, 6, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_6_4_6_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 6, 4, 6, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_6_4_6_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 6, 4, 6, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_6_4_6_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 6, 4, 6, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_7_4_7_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<7, 4, 7, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_7_4_7_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 7, 4, 7, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_7_4_7_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 7, 4, 7, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_7_4_7_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 7, 4, 7, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_7_4_7_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 7, 4, 7, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_8_4_2_32_16(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<8, 4, 2, uint32_t, 16>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_8_4_2_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 8, 4, 2, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_8_4_2_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 8, 4, 2, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_8_4_2_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 8, 4, 2, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_8_4_2_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 8, 4, 2, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_8_4_4_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<8, 4, 4, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_8_4_4_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 8, 4, 4, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_8_4_4_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 8, 4, 4, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_8_4_4_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 8, 4, 4, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_8_4_4_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 8, 4, 4, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_8_4_8_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<8, 4, 8, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_8_4_8_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 8, 4, 8, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_8_4_8_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 8, 4, 8, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_8_4_8_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 8, 4, 8, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_8_4_8_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 8, 4, 8, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_9_4_9_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<9, 4, 9, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_9_4_9_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 9, 4, 9, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_9_4_9_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 9, 4, 9, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_9_4_9_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 9, 4, 9, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_9_4_9_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 9, 4, 9, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_10_4_5_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<10, 4, 5, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_10_4_5_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 10, 4, 5, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_10_4_5_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 10, 4, 5, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_10_4_5_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 10, 4, 5, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_10_4_5_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 10, 4, 5, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_10_4_10_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<10, 4, 10, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_10_4_10_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 10, 4, 10, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_10_4_10_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 10, 4, 10, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_10_4_10_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 10, 4, 10, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_10_4_10_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 10, 4, 10, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_11_4_11_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<11, 4, 11, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_11_4_11_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 11, 4, 11, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_11_4_11_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 11, 4, 11, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_11_4_11_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 11, 4, 11, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_11_4_11_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 11, 4, 11, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_12_4_3_32_16(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<12, 4, 3, uint32_t, 16>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_12_4_3_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 12, 4, 3, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_12_4_3_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 12, 4, 3, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_12_4_3_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 12, 4, 3, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_12_4_3_32_16(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 12, 4, 3, uint32_t, 16>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_12_4_6_32_32(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<12, 4, 6, uint32_t, 32>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_12_4_6_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 12, 4, 6, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_12_4_6_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 12, 4, 6, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_12_4_6_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 12, 4, 6, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_12_4_6_32_32(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 12, 4, 6, uint32_t, 32>(in, out, qweight, lut);}
__host__ extern void vq_pack_dequant_12_4_12_32_64(torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_dequant<12, 4, 12, uint32_t, 64>(out, qweight, lut);}
__host__ extern void vq_pack_gemm_1_12_4_12_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<1, 12, 4, 12, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_2_12_4_12_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<2, 12, 4, 12, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_4_12_4_12_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<4, 12, 4, 12, uint32_t, 64>(in, out, qweight, lut);}
__host__ extern void vq_pack_gemm_8_12_4_12_32_64(torch::Tensor in, torch::Tensor out, torch::Tensor qweight, torch::Tensor lut){vq_pack_gemm<8, 12, 4, 12, uint32_t, 64>(in, out, qweight, lut);}