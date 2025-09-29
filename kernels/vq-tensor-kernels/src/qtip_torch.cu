
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/types.h>
#include <torch/extension.h>
#include "inference.cu"

using namespace torch::indexing;

template <uint32_t R, uint32_t M, uint32_t N, uint32_t K, Q_MODE Q, LUT_TYPE L, LOAD_TYPE LD>
__host__ static void decompress_gemm(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2);
    TORCH_CHECK(out.scalar_type() == torch::kFloat32);

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 1);
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16);

    size_t m = out.size(1);
    size_t n = out.size(0);
    size_t k = x.size(1);

    TORCH_CHECK(m == M);
    TORCH_CHECK(n == N);
    TORCH_CHECK(k == K);
    TORCH_CHECK(compressed.numel() * 32 * (1 << ((int)L)) == R * m * k);
    TORCH_CHECK(x.size(0) == n);
    if constexpr((Q == Q_MODE::PER_TENSOR)) {
        TORCH_CHECK(codebook.size(0) == (1<<R) * (1 << ((int)L)));
    } else if constexpr(Q == Q_MODE::PER_COLUMN) {
        TORCH_CHECK(codebook.size(0) == (1<<R) * m * (1 << ((int)L)));
    }

    at::DeviceGuard guard(x.device());
    
    decompress_gemm_ptr<R, M, N, K, Q, L, LD>(
            reinterpret_cast<float *>(out.data_ptr<float>()),
            reinterpret_cast<const uint32_t *>(compressed.data_ptr<int32_t>()),
            reinterpret_cast<const half2 *>(x.data_ptr<c10::Half>()),
            reinterpret_cast<const half *>(codebook.data_ptr<c10::Half>()),
            at::cuda::getCurrentCUDAStream()
    );
}

template <uint32_t R, Q_MODE Q, LUT_TYPE L, LOAD_TYPE LD>
__host__ static void decompress(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
){
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2);
    TORCH_CHECK(out.scalar_type() == torch::kFloat16);

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 1);
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16);

    size_t m = out.size(0);
    size_t k = out.size(1);

    TORCH_CHECK(compressed.numel() * 32 * (1 << ((int)L)) == R * m * k);
    if constexpr((Q == Q_MODE::PER_TENSOR)) {
        TORCH_CHECK(codebook.size(0) == (1<<R) * (1 << ((int)L)));
    } else if constexpr(Q == Q_MODE::PER_COLUMN) {
        TORCH_CHECK(codebook.size(0) == (1<<R) * m * (1 << ((int)L)));
    }

    decompress_ptr<R, Q, L, LD>(
            reinterpret_cast<half2 *>(out.data_ptr<c10::Half>()),
            reinterpret_cast<const uint32_t *>(compressed.data_ptr<int32_t>()),
            reinterpret_cast<const half *>(codebook.data_ptr<c10::Half>()),
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(k),
            at::cuda::getCurrentCUDAStream()
    );
}

__host__ extern void decompress_2_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<2U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_2_256_1_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_2_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_4_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_8_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_1_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_2_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_4_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_8_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_1_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_2_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_4_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_8_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_3_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<3U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_3_256_1_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_2_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_4_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_8_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_1_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_2_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_4_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_8_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_1_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_2_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_4_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_8_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_4_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<4U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_4_256_1_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_2_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_4_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_8_256_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_1_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_2_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_4_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_8_53248_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_1_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_2_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_4_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_8_16384_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_14336_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_1_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_2_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_4_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_8_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_11008_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_4096_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_1024_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_28672_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_8192_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_13824_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_1_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_2_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_4_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_8_5120_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_13568_sq_dup(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::DUP>(out, compressed, x, codebook);
}

__host__ extern void decompress_2_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<2U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_2_256_1_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_2_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_4_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_8_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_1_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_2_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_4_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_8_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_3_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<3U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_3_256_1_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_2_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_4_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_8_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_1_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_2_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_4_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_8_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_4_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<4U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_4_256_1_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_2_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_4_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_8_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_1_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_2_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_4_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_8_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_5_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<5U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_5_256_1_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_256_2_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_256_4_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_256_8_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_53248_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_53248_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_53248_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_53248_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_1_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_2_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_4_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_8_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_1_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_2_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_4_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_8_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_14336_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_14336_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_14336_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_14336_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_6144_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_6144_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_6144_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_6144_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_512_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_512_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_512_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_512_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5504_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5504_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5504_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5504_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_11008_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_11008_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_11008_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_11008_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_22016_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_22016_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_22016_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_22016_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_12288_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_12288_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_12288_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_12288_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_57344_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_57344_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_57344_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_57344_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_1_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_2_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_4_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_8_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_1_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_2_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_4_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_8_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_1_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_2_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_4_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_8_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13824_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13824_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13824_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13824_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27648_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27648_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27648_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27648_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_15360_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_15360_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_15360_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_15360_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13568_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13568_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13568_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13568_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27136_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27136_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27136_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27136_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_1_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_2_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_4_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_8_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_6_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<6U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_6_256_1_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_256_2_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_256_4_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_256_8_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_53248_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_53248_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_53248_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_53248_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_1_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_2_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_4_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_8_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_1_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_2_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_4_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_8_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_14336_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_14336_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_14336_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_14336_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_6144_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_6144_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_6144_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_6144_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_512_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_512_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_512_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_512_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5504_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5504_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5504_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5504_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_11008_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_11008_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_11008_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_11008_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_22016_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_22016_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_22016_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_22016_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_12288_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_12288_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_12288_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_12288_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_57344_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_57344_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_57344_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_57344_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_1_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_2_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_4_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_8_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_1_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_2_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_4_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_8_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_1_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_2_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_4_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_8_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13824_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13824_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13824_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13824_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27648_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27648_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27648_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27648_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_15360_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_15360_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_15360_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_15360_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13568_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13568_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13568_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13568_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27136_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27136_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27136_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27136_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_1_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_2_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_4_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_8_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_7_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<7U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_7_256_1_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_256_2_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_256_4_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_256_8_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_53248_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_53248_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_53248_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_53248_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_1_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_2_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_4_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_8_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_1_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_2_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_4_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_8_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_14336_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_14336_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_14336_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_14336_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_6144_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_6144_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_6144_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_6144_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_512_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_512_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_512_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_512_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5504_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5504_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5504_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5504_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_11008_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_11008_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_11008_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_11008_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_22016_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_22016_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_22016_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_22016_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_12288_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_12288_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_12288_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_12288_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_57344_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_57344_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_57344_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_57344_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_1_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_2_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_4_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_8_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_1_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_2_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_4_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_8_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_1_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_2_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_4_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_8_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13824_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13824_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13824_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13824_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27648_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27648_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27648_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27648_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_15360_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_15360_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_15360_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_15360_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13568_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13568_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13568_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13568_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27136_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27136_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27136_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27136_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_1_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_2_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_4_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_8_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_8_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<8U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_8_256_1_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_256_2_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_256_4_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_256_8_256_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_53248_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_53248_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_53248_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_53248_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_1_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_2_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_4_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_8_53248_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_1_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_2_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_4_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_8_16384_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_1_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_2_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_4_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_8_14336_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_14336_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_14336_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_14336_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_14336_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_6144_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_6144_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_6144_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_6144_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_512_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_512_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_512_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_512_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_1_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_2_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_4_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_8_11008_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5504_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5504_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5504_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5504_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_11008_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_11008_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_11008_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_11008_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_22016_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_22016_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_22016_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_22016_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_12288_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_12288_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_12288_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_12288_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_1_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_2_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_4_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_8_4096_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_57344_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_57344_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_57344_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_57344_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_1_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_2_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_4_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_8_1024_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_1_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_2_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_4_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_8_28672_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_1_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_2_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_4_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_8_8192_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_1_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_2_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_4_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_8_13824_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13824_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13824_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13824_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13824_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27648_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27648_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27648_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27648_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_15360_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_15360_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_15360_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_15360_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13568_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13568_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13568_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13568_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27136_1_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27136_2_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27136_4_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27136_8_5120_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_1_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_2_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_4_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_8_13568_sq(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::SQ_LUT, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_2_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<2U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_2_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_2_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<2U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_3_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<3U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_3_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_3_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<3U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_4_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<4U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_4_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_4_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<4U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_5_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<5U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_5_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_5_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<5U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_6_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<6U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_6_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_6_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<6U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_7_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<7U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_7_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_7_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<7U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_8_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<8U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_8_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_8_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<8U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_9_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<9U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_9_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_9_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<9U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_10_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<10U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_10_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_10_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<10U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_11_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<11U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_11_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_11_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<11U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_12_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {
    decompress<12U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, codebook);
}

__host__ extern void decompress_gemm_12_256_1_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 256U, 1U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_256_2_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 256U, 2U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_256_4_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 256U, 4U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_256_8_256_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 256U, 8U, 256U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_53248_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 53248U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_53248_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 53248U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_53248_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 53248U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_53248_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 53248U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_16384_1_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 16384U, 1U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_16384_2_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 16384U, 2U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_16384_4_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 16384U, 4U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_16384_8_53248_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 16384U, 8U, 53248U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_16384_1_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 16384U, 1U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_16384_2_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 16384U, 2U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_16384_4_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 16384U, 4U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_16384_8_16384_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 16384U, 8U, 16384U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_1_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 1U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_2_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 2U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_4_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 4U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_8_14336_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 8U, 14336U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_14336_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 14336U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_14336_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 14336U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_14336_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 14336U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_14336_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 14336U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_28672_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 28672U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_28672_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 28672U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_28672_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 28672U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_28672_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 28672U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_6144_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 6144U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_6144_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 6144U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_6144_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 6144U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_6144_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 6144U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_512_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 512U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_512_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 512U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_512_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 512U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_512_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 512U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_2048_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 2048U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_2048_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 2048U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_2048_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 2048U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_2048_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 2048U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_2048_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 2048U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_2048_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 2048U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_2048_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 2048U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_2048_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 2048U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_1_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 1U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_2_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 2U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_4_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 4U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_4096_8_11008_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 4096U, 8U, 11008U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5504_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5504U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5504_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5504U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5504_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5504U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5504_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5504U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_11008_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 11008U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_11008_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 11008U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_11008_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 11008U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_11008_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 11008U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_22016_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 22016U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_22016_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 22016U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_22016_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 22016U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_22016_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 22016U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_12288_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 12288U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_12288_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 12288U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_12288_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 12288U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_12288_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 12288U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_1_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 1U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_2_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 2U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_4_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 4U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_8_4096_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 8U, 4096U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_10240_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 10240U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_10240_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 10240U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_10240_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 10240U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_10240_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 10240U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_57344_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 57344U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_57344_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 57344U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_57344_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 57344U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_57344_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 57344U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_1_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 1U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_2_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 2U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_4_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 4U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_8_1024_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 8U, 1024U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_1_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 1U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_2_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 2U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_4_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 4U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_8192_8_28672_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 8192U, 8U, 28672U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_28672_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 28672U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_28672_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 28672U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_28672_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 28672U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_28672_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 28672U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_1_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 1U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_2_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 2U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_4_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 4U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_1024_8_8192_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 1024U, 8U, 8192U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_1_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 1U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_2_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 2U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_4_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 4U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_8_13824_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 8U, 13824U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_13824_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 13824U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_13824_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 13824U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_13824_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 13824U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_13824_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 13824U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_27648_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 27648U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_27648_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 27648U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_27648_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 27648U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_27648_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 27648U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_10240_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 10240U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_10240_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 10240U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_10240_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 10240U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_10240_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 10240U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_15360_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 15360U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_15360_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 15360U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_15360_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 15360U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_15360_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 15360U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_13568_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 13568U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_13568_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 13568U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_13568_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 13568U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_13568_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 13568U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_27136_1_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 27136U, 1U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_27136_2_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 27136U, 2U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_27136_4_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 27136U, 4U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_27136_8_5120_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 27136U, 8U, 5120U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_1_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 1U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_2_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 2U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_4_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 4U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}

__host__ extern void decompress_gemm_12_5120_8_13568_vq2(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {
    decompress_gemm<12U, 5120U, 8U, 13568U, Q_MODE::PER_TENSOR, LUT_TYPE::VQ_LUT_2, LOAD_TYPE::ONCE>(out, compressed, x, codebook);
}
