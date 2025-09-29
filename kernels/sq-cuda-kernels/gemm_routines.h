#ifndef ANYPREC_CUH
#define ANYPREC_CUH

#include <cstdint>
#include "datatype.h"
#include "typetraits.h"

void pack_matmul(
        half *in,            // [M, K]
        half *out,           // [M, N]
        uint32_t *qweight,       // [w_bits, N, K/32]
        half *lut,           // [out_size, num_centroids]
        uint32_t M,              // batch size
        uint32_t N,              // output size
        uint32_t K,              // input size
        int w_bits,               // weight bits
        cudaStream_t stream
);


void pack_dequant_kbit(
    const uint32_t *qweight,
    const uint32_t N, const uint32_t K,
    const half *lut, half *weight,
    int w_bits,
    cudaStream_t stream
);

#endif // ANYPREC_CUH