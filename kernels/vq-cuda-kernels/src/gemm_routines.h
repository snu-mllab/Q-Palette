#pragma once

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

enum class AccumType {
    HalfAccum,  // Variation #1
    FloatAccumHmul,  // Variation #2
    FloatAccumFmul // Variation #3
};
#define gpuErrchk(ans)      do { gpuAssert((ans), __FILE__, __LINE__); } while (false)

__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}
#define NROWS 4
#define DIV_ROUND_UP(x, y) (((x)+(y)-1)/(y))

template <int maxm, int nbits, int vec_sz, int code_n, typename codeT, int recons_n>
__host__ static void vq_pack_gemm_ptr(
        const half *in,            // [M, K]
        half *out,           // [M, N]
        const codeT *qweight,       // [w_bits, N, K/32]
        const half *lut,           // [out_size, num_centroids]
        uint32_t M,              // batch size
        uint32_t N,              // output size
        uint32_t K,              // input size
        CUstream_st *stream
);


template <int nbits, int vec_sz, int code_n, typename codeT, int recons_n>
__host__ static void vq_pack_dequant_ptr(
    const codeT *qweight,
    uint32_t N, uint32_t K,
    const half *lut, 
    half *weight,
    CUstream_st *stream
);
