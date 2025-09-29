#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>
#include "gemm_routines.h"

template <int nbits, int vec_sz, int code_n, typename codeT>
__device__ __forceinline__ void vq_pack_dequant_routine(const codeT Bcode[], half2 B_row[], half shC[]);

// nbits: 3	vec_sz: 2	code_n: 3	avg_bits:  1.500	SMEM_sz: 0KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<3, 2, 3, uint32_t>(const uint32_t code[3], half2 recons[32], half SMEM[16]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x7];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 3) & 0x7];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 6) & 0x7];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 9) & 0x7];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 12) & 0x7];
    recons[5] = ((half2 *)SMEM)[(tmp0 >> 15) & 0x7];
    recons[6] = ((half2 *)SMEM)[(tmp0 >> 18) & 0x7];
    recons[7] = ((half2 *)SMEM)[(tmp0 >> 21) & 0x7];
    recons[8] = ((half2 *)SMEM)[(tmp0 >> 24) & 0x7];
    recons[9] = ((half2 *)SMEM)[(tmp0 >> 27) & 0x7];
    recons[10] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x4)];
    recons[11] = ((half2 *)SMEM)[(tmp1 >> 1) & 0x7];
    recons[12] = ((half2 *)SMEM)[(tmp1 >> 4) & 0x7];
    recons[13] = ((half2 *)SMEM)[(tmp1 >> 7) & 0x7];
    recons[14] = ((half2 *)SMEM)[(tmp1 >> 10) & 0x7];
    recons[15] = ((half2 *)SMEM)[(tmp1 >> 13) & 0x7];
    recons[16] = ((half2 *)SMEM)[(tmp1 >> 16) & 0x7];
    recons[17] = ((half2 *)SMEM)[(tmp1 >> 19) & 0x7];
    recons[18] = ((half2 *)SMEM)[(tmp1 >> 22) & 0x7];
    recons[19] = ((half2 *)SMEM)[(tmp1 >> 25) & 0x7];
    recons[20] = ((half2 *)SMEM)[(tmp1 >> 28) & 0x7];
    recons[21] = ((half2 *)SMEM)[((tmp1 >> 31) & 0x1) | ((tmp2 << 1) & 0x6)];
    recons[22] = ((half2 *)SMEM)[(tmp2 >> 2) & 0x7];
    recons[23] = ((half2 *)SMEM)[(tmp2 >> 5) & 0x7];
    recons[24] = ((half2 *)SMEM)[(tmp2 >> 8) & 0x7];
    recons[25] = ((half2 *)SMEM)[(tmp2 >> 11) & 0x7];
    recons[26] = ((half2 *)SMEM)[(tmp2 >> 14) & 0x7];
    recons[27] = ((half2 *)SMEM)[(tmp2 >> 17) & 0x7];
    recons[28] = ((half2 *)SMEM)[(tmp2 >> 20) & 0x7];
    recons[29] = ((half2 *)SMEM)[(tmp2 >> 23) & 0x7];
    recons[30] = ((half2 *)SMEM)[(tmp2 >> 26) & 0x7];
    recons[31] = ((half2 *)SMEM)[(tmp2 >> 29) & 0x7];
}

// nbits: 3	vec_sz: 2	code_n: 6	avg_bits:  1.500	SMEM_sz: 0KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<3, 2, 6, uint32_t>(const uint32_t code[6], half2 recons[64], half SMEM[16]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x7];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 3) & 0x7];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 6) & 0x7];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 9) & 0x7];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 12) & 0x7];
    recons[5] = ((half2 *)SMEM)[(tmp0 >> 15) & 0x7];
    recons[6] = ((half2 *)SMEM)[(tmp0 >> 18) & 0x7];
    recons[7] = ((half2 *)SMEM)[(tmp0 >> 21) & 0x7];
    recons[8] = ((half2 *)SMEM)[(tmp0 >> 24) & 0x7];
    recons[9] = ((half2 *)SMEM)[(tmp0 >> 27) & 0x7];
    recons[10] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x4)];
    recons[11] = ((half2 *)SMEM)[(tmp1 >> 1) & 0x7];
    recons[12] = ((half2 *)SMEM)[(tmp1 >> 4) & 0x7];
    recons[13] = ((half2 *)SMEM)[(tmp1 >> 7) & 0x7];
    recons[14] = ((half2 *)SMEM)[(tmp1 >> 10) & 0x7];
    recons[15] = ((half2 *)SMEM)[(tmp1 >> 13) & 0x7];
    recons[16] = ((half2 *)SMEM)[(tmp1 >> 16) & 0x7];
    recons[17] = ((half2 *)SMEM)[(tmp1 >> 19) & 0x7];
    recons[18] = ((half2 *)SMEM)[(tmp1 >> 22) & 0x7];
    recons[19] = ((half2 *)SMEM)[(tmp1 >> 25) & 0x7];
    recons[20] = ((half2 *)SMEM)[(tmp1 >> 28) & 0x7];
    recons[21] = ((half2 *)SMEM)[((tmp1 >> 31) & 0x1) | ((tmp2 << 1) & 0x6)];
    recons[22] = ((half2 *)SMEM)[(tmp2 >> 2) & 0x7];
    recons[23] = ((half2 *)SMEM)[(tmp2 >> 5) & 0x7];
    recons[24] = ((half2 *)SMEM)[(tmp2 >> 8) & 0x7];
    recons[25] = ((half2 *)SMEM)[(tmp2 >> 11) & 0x7];
    recons[26] = ((half2 *)SMEM)[(tmp2 >> 14) & 0x7];
    recons[27] = ((half2 *)SMEM)[(tmp2 >> 17) & 0x7];
    recons[28] = ((half2 *)SMEM)[(tmp2 >> 20) & 0x7];
    recons[29] = ((half2 *)SMEM)[(tmp2 >> 23) & 0x7];
    recons[30] = ((half2 *)SMEM)[(tmp2 >> 26) & 0x7];
    recons[31] = ((half2 *)SMEM)[(tmp2 >> 29) & 0x7];
    recons[32] = ((half2 *)SMEM)[(tmp3) & 0x7];
    recons[33] = ((half2 *)SMEM)[(tmp3 >> 3) & 0x7];
    recons[34] = ((half2 *)SMEM)[(tmp3 >> 6) & 0x7];
    recons[35] = ((half2 *)SMEM)[(tmp3 >> 9) & 0x7];
    recons[36] = ((half2 *)SMEM)[(tmp3 >> 12) & 0x7];
    recons[37] = ((half2 *)SMEM)[(tmp3 >> 15) & 0x7];
    recons[38] = ((half2 *)SMEM)[(tmp3 >> 18) & 0x7];
    recons[39] = ((half2 *)SMEM)[(tmp3 >> 21) & 0x7];
    recons[40] = ((half2 *)SMEM)[(tmp3 >> 24) & 0x7];
    recons[41] = ((half2 *)SMEM)[(tmp3 >> 27) & 0x7];
    recons[42] = ((half2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x4)];
    recons[43] = ((half2 *)SMEM)[(tmp4 >> 1) & 0x7];
    recons[44] = ((half2 *)SMEM)[(tmp4 >> 4) & 0x7];
    recons[45] = ((half2 *)SMEM)[(tmp4 >> 7) & 0x7];
    recons[46] = ((half2 *)SMEM)[(tmp4 >> 10) & 0x7];
    recons[47] = ((half2 *)SMEM)[(tmp4 >> 13) & 0x7];
    recons[48] = ((half2 *)SMEM)[(tmp4 >> 16) & 0x7];
    recons[49] = ((half2 *)SMEM)[(tmp4 >> 19) & 0x7];
    recons[50] = ((half2 *)SMEM)[(tmp4 >> 22) & 0x7];
    recons[51] = ((half2 *)SMEM)[(tmp4 >> 25) & 0x7];
    recons[52] = ((half2 *)SMEM)[(tmp4 >> 28) & 0x7];
    recons[53] = ((half2 *)SMEM)[((tmp4 >> 31) & 0x1) | ((tmp5 << 1) & 0x6)];
    recons[54] = ((half2 *)SMEM)[(tmp5 >> 2) & 0x7];
    recons[55] = ((half2 *)SMEM)[(tmp5 >> 5) & 0x7];
    recons[56] = ((half2 *)SMEM)[(tmp5 >> 8) & 0x7];
    recons[57] = ((half2 *)SMEM)[(tmp5 >> 11) & 0x7];
    recons[58] = ((half2 *)SMEM)[(tmp5 >> 14) & 0x7];
    recons[59] = ((half2 *)SMEM)[(tmp5 >> 17) & 0x7];
    recons[60] = ((half2 *)SMEM)[(tmp5 >> 20) & 0x7];
    recons[61] = ((half2 *)SMEM)[(tmp5 >> 23) & 0x7];
    recons[62] = ((half2 *)SMEM)[(tmp5 >> 26) & 0x7];
    recons[63] = ((half2 *)SMEM)[(tmp5 >> 29) & 0x7];
}

// nbits: 4	vec_sz: 2	code_n: 2	avg_bits:  2.000	SMEM_sz: 0KB	recons_n: 16
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<4, 2, 2, uint32_t>(const uint32_t code[2], half2 recons[16], half SMEM[32]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xf];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 4) & 0xf];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 8) & 0xf];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 12) & 0xf];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 16) & 0xf];
    recons[5] = ((half2 *)SMEM)[(tmp0 >> 20) & 0xf];
    recons[6] = ((half2 *)SMEM)[(tmp0 >> 24) & 0xf];
    recons[7] = ((half2 *)SMEM)[(tmp0 >> 28) & 0xf];
    recons[8] = ((half2 *)SMEM)[(tmp1) & 0xf];
    recons[9] = ((half2 *)SMEM)[(tmp1 >> 4) & 0xf];
    recons[10] = ((half2 *)SMEM)[(tmp1 >> 8) & 0xf];
    recons[11] = ((half2 *)SMEM)[(tmp1 >> 12) & 0xf];
    recons[12] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xf];
    recons[13] = ((half2 *)SMEM)[(tmp1 >> 20) & 0xf];
    recons[14] = ((half2 *)SMEM)[(tmp1 >> 24) & 0xf];
    recons[15] = ((half2 *)SMEM)[(tmp1 >> 28) & 0xf];
}

// nbits: 4	vec_sz: 2	code_n: 4	avg_bits:  2.000	SMEM_sz: 0KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<4, 2, 4, uint32_t>(const uint32_t code[4], half2 recons[32], half SMEM[32]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xf];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 4) & 0xf];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 8) & 0xf];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 12) & 0xf];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 16) & 0xf];
    recons[5] = ((half2 *)SMEM)[(tmp0 >> 20) & 0xf];
    recons[6] = ((half2 *)SMEM)[(tmp0 >> 24) & 0xf];
    recons[7] = ((half2 *)SMEM)[(tmp0 >> 28) & 0xf];
    recons[8] = ((half2 *)SMEM)[(tmp1) & 0xf];
    recons[9] = ((half2 *)SMEM)[(tmp1 >> 4) & 0xf];
    recons[10] = ((half2 *)SMEM)[(tmp1 >> 8) & 0xf];
    recons[11] = ((half2 *)SMEM)[(tmp1 >> 12) & 0xf];
    recons[12] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xf];
    recons[13] = ((half2 *)SMEM)[(tmp1 >> 20) & 0xf];
    recons[14] = ((half2 *)SMEM)[(tmp1 >> 24) & 0xf];
    recons[15] = ((half2 *)SMEM)[(tmp1 >> 28) & 0xf];
    recons[16] = ((half2 *)SMEM)[(tmp2) & 0xf];
    recons[17] = ((half2 *)SMEM)[(tmp2 >> 4) & 0xf];
    recons[18] = ((half2 *)SMEM)[(tmp2 >> 8) & 0xf];
    recons[19] = ((half2 *)SMEM)[(tmp2 >> 12) & 0xf];
    recons[20] = ((half2 *)SMEM)[(tmp2 >> 16) & 0xf];
    recons[21] = ((half2 *)SMEM)[(tmp2 >> 20) & 0xf];
    recons[22] = ((half2 *)SMEM)[(tmp2 >> 24) & 0xf];
    recons[23] = ((half2 *)SMEM)[(tmp2 >> 28) & 0xf];
    recons[24] = ((half2 *)SMEM)[(tmp3) & 0xf];
    recons[25] = ((half2 *)SMEM)[(tmp3 >> 4) & 0xf];
    recons[26] = ((half2 *)SMEM)[(tmp3 >> 8) & 0xf];
    recons[27] = ((half2 *)SMEM)[(tmp3 >> 12) & 0xf];
    recons[28] = ((half2 *)SMEM)[(tmp3 >> 16) & 0xf];
    recons[29] = ((half2 *)SMEM)[(tmp3 >> 20) & 0xf];
    recons[30] = ((half2 *)SMEM)[(tmp3 >> 24) & 0xf];
    recons[31] = ((half2 *)SMEM)[(tmp3 >> 28) & 0xf];
}

// nbits: 4	vec_sz: 2	code_n: 8	avg_bits:  2.000	SMEM_sz: 0KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<4, 2, 8, uint32_t>(const uint32_t code[8], half2 recons[64], half SMEM[32]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xf];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 4) & 0xf];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 8) & 0xf];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 12) & 0xf];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 16) & 0xf];
    recons[5] = ((half2 *)SMEM)[(tmp0 >> 20) & 0xf];
    recons[6] = ((half2 *)SMEM)[(tmp0 >> 24) & 0xf];
    recons[7] = ((half2 *)SMEM)[(tmp0 >> 28) & 0xf];
    recons[8] = ((half2 *)SMEM)[(tmp1) & 0xf];
    recons[9] = ((half2 *)SMEM)[(tmp1 >> 4) & 0xf];
    recons[10] = ((half2 *)SMEM)[(tmp1 >> 8) & 0xf];
    recons[11] = ((half2 *)SMEM)[(tmp1 >> 12) & 0xf];
    recons[12] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xf];
    recons[13] = ((half2 *)SMEM)[(tmp1 >> 20) & 0xf];
    recons[14] = ((half2 *)SMEM)[(tmp1 >> 24) & 0xf];
    recons[15] = ((half2 *)SMEM)[(tmp1 >> 28) & 0xf];
    recons[16] = ((half2 *)SMEM)[(tmp2) & 0xf];
    recons[17] = ((half2 *)SMEM)[(tmp2 >> 4) & 0xf];
    recons[18] = ((half2 *)SMEM)[(tmp2 >> 8) & 0xf];
    recons[19] = ((half2 *)SMEM)[(tmp2 >> 12) & 0xf];
    recons[20] = ((half2 *)SMEM)[(tmp2 >> 16) & 0xf];
    recons[21] = ((half2 *)SMEM)[(tmp2 >> 20) & 0xf];
    recons[22] = ((half2 *)SMEM)[(tmp2 >> 24) & 0xf];
    recons[23] = ((half2 *)SMEM)[(tmp2 >> 28) & 0xf];
    recons[24] = ((half2 *)SMEM)[(tmp3) & 0xf];
    recons[25] = ((half2 *)SMEM)[(tmp3 >> 4) & 0xf];
    recons[26] = ((half2 *)SMEM)[(tmp3 >> 8) & 0xf];
    recons[27] = ((half2 *)SMEM)[(tmp3 >> 12) & 0xf];
    recons[28] = ((half2 *)SMEM)[(tmp3 >> 16) & 0xf];
    recons[29] = ((half2 *)SMEM)[(tmp3 >> 20) & 0xf];
    recons[30] = ((half2 *)SMEM)[(tmp3 >> 24) & 0xf];
    recons[31] = ((half2 *)SMEM)[(tmp3 >> 28) & 0xf];
    recons[32] = ((half2 *)SMEM)[(tmp4) & 0xf];
    recons[33] = ((half2 *)SMEM)[(tmp4 >> 4) & 0xf];
    recons[34] = ((half2 *)SMEM)[(tmp4 >> 8) & 0xf];
    recons[35] = ((half2 *)SMEM)[(tmp4 >> 12) & 0xf];
    recons[36] = ((half2 *)SMEM)[(tmp4 >> 16) & 0xf];
    recons[37] = ((half2 *)SMEM)[(tmp4 >> 20) & 0xf];
    recons[38] = ((half2 *)SMEM)[(tmp4 >> 24) & 0xf];
    recons[39] = ((half2 *)SMEM)[(tmp4 >> 28) & 0xf];
    recons[40] = ((half2 *)SMEM)[(tmp5) & 0xf];
    recons[41] = ((half2 *)SMEM)[(tmp5 >> 4) & 0xf];
    recons[42] = ((half2 *)SMEM)[(tmp5 >> 8) & 0xf];
    recons[43] = ((half2 *)SMEM)[(tmp5 >> 12) & 0xf];
    recons[44] = ((half2 *)SMEM)[(tmp5 >> 16) & 0xf];
    recons[45] = ((half2 *)SMEM)[(tmp5 >> 20) & 0xf];
    recons[46] = ((half2 *)SMEM)[(tmp5 >> 24) & 0xf];
    recons[47] = ((half2 *)SMEM)[(tmp5 >> 28) & 0xf];
    recons[48] = ((half2 *)SMEM)[(tmp6) & 0xf];
    recons[49] = ((half2 *)SMEM)[(tmp6 >> 4) & 0xf];
    recons[50] = ((half2 *)SMEM)[(tmp6 >> 8) & 0xf];
    recons[51] = ((half2 *)SMEM)[(tmp6 >> 12) & 0xf];
    recons[52] = ((half2 *)SMEM)[(tmp6 >> 16) & 0xf];
    recons[53] = ((half2 *)SMEM)[(tmp6 >> 20) & 0xf];
    recons[54] = ((half2 *)SMEM)[(tmp6 >> 24) & 0xf];
    recons[55] = ((half2 *)SMEM)[(tmp6 >> 28) & 0xf];
    recons[56] = ((half2 *)SMEM)[(tmp7) & 0xf];
    recons[57] = ((half2 *)SMEM)[(tmp7 >> 4) & 0xf];
    recons[58] = ((half2 *)SMEM)[(tmp7 >> 8) & 0xf];
    recons[59] = ((half2 *)SMEM)[(tmp7 >> 12) & 0xf];
    recons[60] = ((half2 *)SMEM)[(tmp7 >> 16) & 0xf];
    recons[61] = ((half2 *)SMEM)[(tmp7 >> 20) & 0xf];
    recons[62] = ((half2 *)SMEM)[(tmp7 >> 24) & 0xf];
    recons[63] = ((half2 *)SMEM)[(tmp7 >> 28) & 0xf];
}

// nbits: 5	vec_sz: 2	code_n: 5	avg_bits:  2.500	SMEM_sz: 0KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<5, 2, 5, uint32_t>(const uint32_t code[5], half2 recons[32], half SMEM[64]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x1f];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 5) & 0x1f];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 10) & 0x1f];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 15) & 0x1f];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 20) & 0x1f];
    recons[5] = ((half2 *)SMEM)[(tmp0 >> 25) & 0x1f];
    recons[6] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x1c)];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 3) & 0x1f];
    recons[8] = ((half2 *)SMEM)[(tmp1 >> 8) & 0x1f];
    recons[9] = ((half2 *)SMEM)[(tmp1 >> 13) & 0x1f];
    recons[10] = ((half2 *)SMEM)[(tmp1 >> 18) & 0x1f];
    recons[11] = ((half2 *)SMEM)[(tmp1 >> 23) & 0x1f];
    recons[12] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x10)];
    recons[13] = ((half2 *)SMEM)[(tmp2 >> 1) & 0x1f];
    recons[14] = ((half2 *)SMEM)[(tmp2 >> 6) & 0x1f];
    recons[15] = ((half2 *)SMEM)[(tmp2 >> 11) & 0x1f];
    recons[16] = ((half2 *)SMEM)[(tmp2 >> 16) & 0x1f];
    recons[17] = ((half2 *)SMEM)[(tmp2 >> 21) & 0x1f];
    recons[18] = ((half2 *)SMEM)[(tmp2 >> 26) & 0x1f];
    recons[19] = ((half2 *)SMEM)[((tmp2 >> 31) & 0x1) | ((tmp3 << 1) & 0x1e)];
    recons[20] = ((half2 *)SMEM)[(tmp3 >> 4) & 0x1f];
    recons[21] = ((half2 *)SMEM)[(tmp3 >> 9) & 0x1f];
    recons[22] = ((half2 *)SMEM)[(tmp3 >> 14) & 0x1f];
    recons[23] = ((half2 *)SMEM)[(tmp3 >> 19) & 0x1f];
    recons[24] = ((half2 *)SMEM)[(tmp3 >> 24) & 0x1f];
    recons[25] = ((half2 *)SMEM)[((tmp3 >> 29) & 0x7) | ((tmp4 << 3) & 0x18)];
    recons[26] = ((half2 *)SMEM)[(tmp4 >> 2) & 0x1f];
    recons[27] = ((half2 *)SMEM)[(tmp4 >> 7) & 0x1f];
    recons[28] = ((half2 *)SMEM)[(tmp4 >> 12) & 0x1f];
    recons[29] = ((half2 *)SMEM)[(tmp4 >> 17) & 0x1f];
    recons[30] = ((half2 *)SMEM)[(tmp4 >> 22) & 0x1f];
    recons[31] = ((half2 *)SMEM)[(tmp4 >> 27) & 0x1f];
}

// nbits: 5	vec_sz: 2	code_n: 10	avg_bits:  2.500	SMEM_sz: 0KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<5, 2, 10, uint32_t>(const uint32_t code[10], half2 recons[64], half SMEM[64]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x1f];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 5) & 0x1f];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 10) & 0x1f];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 15) & 0x1f];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 20) & 0x1f];
    recons[5] = ((half2 *)SMEM)[(tmp0 >> 25) & 0x1f];
    recons[6] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x1c)];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 3) & 0x1f];
    recons[8] = ((half2 *)SMEM)[(tmp1 >> 8) & 0x1f];
    recons[9] = ((half2 *)SMEM)[(tmp1 >> 13) & 0x1f];
    recons[10] = ((half2 *)SMEM)[(tmp1 >> 18) & 0x1f];
    recons[11] = ((half2 *)SMEM)[(tmp1 >> 23) & 0x1f];
    recons[12] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x10)];
    recons[13] = ((half2 *)SMEM)[(tmp2 >> 1) & 0x1f];
    recons[14] = ((half2 *)SMEM)[(tmp2 >> 6) & 0x1f];
    recons[15] = ((half2 *)SMEM)[(tmp2 >> 11) & 0x1f];
    recons[16] = ((half2 *)SMEM)[(tmp2 >> 16) & 0x1f];
    recons[17] = ((half2 *)SMEM)[(tmp2 >> 21) & 0x1f];
    recons[18] = ((half2 *)SMEM)[(tmp2 >> 26) & 0x1f];
    recons[19] = ((half2 *)SMEM)[((tmp2 >> 31) & 0x1) | ((tmp3 << 1) & 0x1e)];
    recons[20] = ((half2 *)SMEM)[(tmp3 >> 4) & 0x1f];
    recons[21] = ((half2 *)SMEM)[(tmp3 >> 9) & 0x1f];
    recons[22] = ((half2 *)SMEM)[(tmp3 >> 14) & 0x1f];
    recons[23] = ((half2 *)SMEM)[(tmp3 >> 19) & 0x1f];
    recons[24] = ((half2 *)SMEM)[(tmp3 >> 24) & 0x1f];
    recons[25] = ((half2 *)SMEM)[((tmp3 >> 29) & 0x7) | ((tmp4 << 3) & 0x18)];
    recons[26] = ((half2 *)SMEM)[(tmp4 >> 2) & 0x1f];
    recons[27] = ((half2 *)SMEM)[(tmp4 >> 7) & 0x1f];
    recons[28] = ((half2 *)SMEM)[(tmp4 >> 12) & 0x1f];
    recons[29] = ((half2 *)SMEM)[(tmp4 >> 17) & 0x1f];
    recons[30] = ((half2 *)SMEM)[(tmp4 >> 22) & 0x1f];
    recons[31] = ((half2 *)SMEM)[(tmp4 >> 27) & 0x1f];
    recons[32] = ((half2 *)SMEM)[(tmp5) & 0x1f];
    recons[33] = ((half2 *)SMEM)[(tmp5 >> 5) & 0x1f];
    recons[34] = ((half2 *)SMEM)[(tmp5 >> 10) & 0x1f];
    recons[35] = ((half2 *)SMEM)[(tmp5 >> 15) & 0x1f];
    recons[36] = ((half2 *)SMEM)[(tmp5 >> 20) & 0x1f];
    recons[37] = ((half2 *)SMEM)[(tmp5 >> 25) & 0x1f];
    recons[38] = ((half2 *)SMEM)[((tmp5 >> 30) & 0x3) | ((tmp6 << 2) & 0x1c)];
    recons[39] = ((half2 *)SMEM)[(tmp6 >> 3) & 0x1f];
    recons[40] = ((half2 *)SMEM)[(tmp6 >> 8) & 0x1f];
    recons[41] = ((half2 *)SMEM)[(tmp6 >> 13) & 0x1f];
    recons[42] = ((half2 *)SMEM)[(tmp6 >> 18) & 0x1f];
    recons[43] = ((half2 *)SMEM)[(tmp6 >> 23) & 0x1f];
    recons[44] = ((half2 *)SMEM)[((tmp6 >> 28) & 0xf) | ((tmp7 << 4) & 0x10)];
    recons[45] = ((half2 *)SMEM)[(tmp7 >> 1) & 0x1f];
    recons[46] = ((half2 *)SMEM)[(tmp7 >> 6) & 0x1f];
    recons[47] = ((half2 *)SMEM)[(tmp7 >> 11) & 0x1f];
    recons[48] = ((half2 *)SMEM)[(tmp7 >> 16) & 0x1f];
    recons[49] = ((half2 *)SMEM)[(tmp7 >> 21) & 0x1f];
    recons[50] = ((half2 *)SMEM)[(tmp7 >> 26) & 0x1f];
    recons[51] = ((half2 *)SMEM)[((tmp7 >> 31) & 0x1) | ((tmp8 << 1) & 0x1e)];
    recons[52] = ((half2 *)SMEM)[(tmp8 >> 4) & 0x1f];
    recons[53] = ((half2 *)SMEM)[(tmp8 >> 9) & 0x1f];
    recons[54] = ((half2 *)SMEM)[(tmp8 >> 14) & 0x1f];
    recons[55] = ((half2 *)SMEM)[(tmp8 >> 19) & 0x1f];
    recons[56] = ((half2 *)SMEM)[(tmp8 >> 24) & 0x1f];
    recons[57] = ((half2 *)SMEM)[((tmp8 >> 29) & 0x7) | ((tmp9 << 3) & 0x18)];
    recons[58] = ((half2 *)SMEM)[(tmp9 >> 2) & 0x1f];
    recons[59] = ((half2 *)SMEM)[(tmp9 >> 7) & 0x1f];
    recons[60] = ((half2 *)SMEM)[(tmp9 >> 12) & 0x1f];
    recons[61] = ((half2 *)SMEM)[(tmp9 >> 17) & 0x1f];
    recons[62] = ((half2 *)SMEM)[(tmp9 >> 22) & 0x1f];
    recons[63] = ((half2 *)SMEM)[(tmp9 >> 27) & 0x1f];
}

// nbits: 6	vec_sz: 2	code_n: 3	avg_bits:  3.000	SMEM_sz: 0KB	recons_n: 16
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<6, 2, 3, uint32_t>(const uint32_t code[3], half2 recons[16], half SMEM[128]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x3f];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 6) & 0x3f];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 12) & 0x3f];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 18) & 0x3f];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 24) & 0x3f];
    recons[5] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3c)];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 4) & 0x3f];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 10) & 0x3f];
    recons[8] = ((half2 *)SMEM)[(tmp1 >> 16) & 0x3f];
    recons[9] = ((half2 *)SMEM)[(tmp1 >> 22) & 0x3f];
    recons[10] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x30)];
    recons[11] = ((half2 *)SMEM)[(tmp2 >> 2) & 0x3f];
    recons[12] = ((half2 *)SMEM)[(tmp2 >> 8) & 0x3f];
    recons[13] = ((half2 *)SMEM)[(tmp2 >> 14) & 0x3f];
    recons[14] = ((half2 *)SMEM)[(tmp2 >> 20) & 0x3f];
    recons[15] = ((half2 *)SMEM)[(tmp2 >> 26) & 0x3f];
}

// nbits: 6	vec_sz: 2	code_n: 6	avg_bits:  3.000	SMEM_sz: 0KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<6, 2, 6, uint32_t>(const uint32_t code[6], half2 recons[32], half SMEM[128]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x3f];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 6) & 0x3f];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 12) & 0x3f];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 18) & 0x3f];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 24) & 0x3f];
    recons[5] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3c)];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 4) & 0x3f];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 10) & 0x3f];
    recons[8] = ((half2 *)SMEM)[(tmp1 >> 16) & 0x3f];
    recons[9] = ((half2 *)SMEM)[(tmp1 >> 22) & 0x3f];
    recons[10] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x30)];
    recons[11] = ((half2 *)SMEM)[(tmp2 >> 2) & 0x3f];
    recons[12] = ((half2 *)SMEM)[(tmp2 >> 8) & 0x3f];
    recons[13] = ((half2 *)SMEM)[(tmp2 >> 14) & 0x3f];
    recons[14] = ((half2 *)SMEM)[(tmp2 >> 20) & 0x3f];
    recons[15] = ((half2 *)SMEM)[(tmp2 >> 26) & 0x3f];
    recons[16] = ((half2 *)SMEM)[(tmp3) & 0x3f];
    recons[17] = ((half2 *)SMEM)[(tmp3 >> 6) & 0x3f];
    recons[18] = ((half2 *)SMEM)[(tmp3 >> 12) & 0x3f];
    recons[19] = ((half2 *)SMEM)[(tmp3 >> 18) & 0x3f];
    recons[20] = ((half2 *)SMEM)[(tmp3 >> 24) & 0x3f];
    recons[21] = ((half2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x3c)];
    recons[22] = ((half2 *)SMEM)[(tmp4 >> 4) & 0x3f];
    recons[23] = ((half2 *)SMEM)[(tmp4 >> 10) & 0x3f];
    recons[24] = ((half2 *)SMEM)[(tmp4 >> 16) & 0x3f];
    recons[25] = ((half2 *)SMEM)[(tmp4 >> 22) & 0x3f];
    recons[26] = ((half2 *)SMEM)[((tmp4 >> 28) & 0xf) | ((tmp5 << 4) & 0x30)];
    recons[27] = ((half2 *)SMEM)[(tmp5 >> 2) & 0x3f];
    recons[28] = ((half2 *)SMEM)[(tmp5 >> 8) & 0x3f];
    recons[29] = ((half2 *)SMEM)[(tmp5 >> 14) & 0x3f];
    recons[30] = ((half2 *)SMEM)[(tmp5 >> 20) & 0x3f];
    recons[31] = ((half2 *)SMEM)[(tmp5 >> 26) & 0x3f];
}

// nbits: 6	vec_sz: 2	code_n: 12	avg_bits:  3.000	SMEM_sz: 0KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<6, 2, 12, uint32_t>(const uint32_t code[12], half2 recons[64], half SMEM[128]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x3f];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 6) & 0x3f];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 12) & 0x3f];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 18) & 0x3f];
    recons[4] = ((half2 *)SMEM)[(tmp0 >> 24) & 0x3f];
    recons[5] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3c)];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 4) & 0x3f];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 10) & 0x3f];
    recons[8] = ((half2 *)SMEM)[(tmp1 >> 16) & 0x3f];
    recons[9] = ((half2 *)SMEM)[(tmp1 >> 22) & 0x3f];
    recons[10] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x30)];
    recons[11] = ((half2 *)SMEM)[(tmp2 >> 2) & 0x3f];
    recons[12] = ((half2 *)SMEM)[(tmp2 >> 8) & 0x3f];
    recons[13] = ((half2 *)SMEM)[(tmp2 >> 14) & 0x3f];
    recons[14] = ((half2 *)SMEM)[(tmp2 >> 20) & 0x3f];
    recons[15] = ((half2 *)SMEM)[(tmp2 >> 26) & 0x3f];
    recons[16] = ((half2 *)SMEM)[(tmp3) & 0x3f];
    recons[17] = ((half2 *)SMEM)[(tmp3 >> 6) & 0x3f];
    recons[18] = ((half2 *)SMEM)[(tmp3 >> 12) & 0x3f];
    recons[19] = ((half2 *)SMEM)[(tmp3 >> 18) & 0x3f];
    recons[20] = ((half2 *)SMEM)[(tmp3 >> 24) & 0x3f];
    recons[21] = ((half2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x3c)];
    recons[22] = ((half2 *)SMEM)[(tmp4 >> 4) & 0x3f];
    recons[23] = ((half2 *)SMEM)[(tmp4 >> 10) & 0x3f];
    recons[24] = ((half2 *)SMEM)[(tmp4 >> 16) & 0x3f];
    recons[25] = ((half2 *)SMEM)[(tmp4 >> 22) & 0x3f];
    recons[26] = ((half2 *)SMEM)[((tmp4 >> 28) & 0xf) | ((tmp5 << 4) & 0x30)];
    recons[27] = ((half2 *)SMEM)[(tmp5 >> 2) & 0x3f];
    recons[28] = ((half2 *)SMEM)[(tmp5 >> 8) & 0x3f];
    recons[29] = ((half2 *)SMEM)[(tmp5 >> 14) & 0x3f];
    recons[30] = ((half2 *)SMEM)[(tmp5 >> 20) & 0x3f];
    recons[31] = ((half2 *)SMEM)[(tmp5 >> 26) & 0x3f];
    recons[32] = ((half2 *)SMEM)[(tmp6) & 0x3f];
    recons[33] = ((half2 *)SMEM)[(tmp6 >> 6) & 0x3f];
    recons[34] = ((half2 *)SMEM)[(tmp6 >> 12) & 0x3f];
    recons[35] = ((half2 *)SMEM)[(tmp6 >> 18) & 0x3f];
    recons[36] = ((half2 *)SMEM)[(tmp6 >> 24) & 0x3f];
    recons[37] = ((half2 *)SMEM)[((tmp6 >> 30) & 0x3) | ((tmp7 << 2) & 0x3c)];
    recons[38] = ((half2 *)SMEM)[(tmp7 >> 4) & 0x3f];
    recons[39] = ((half2 *)SMEM)[(tmp7 >> 10) & 0x3f];
    recons[40] = ((half2 *)SMEM)[(tmp7 >> 16) & 0x3f];
    recons[41] = ((half2 *)SMEM)[(tmp7 >> 22) & 0x3f];
    recons[42] = ((half2 *)SMEM)[((tmp7 >> 28) & 0xf) | ((tmp8 << 4) & 0x30)];
    recons[43] = ((half2 *)SMEM)[(tmp8 >> 2) & 0x3f];
    recons[44] = ((half2 *)SMEM)[(tmp8 >> 8) & 0x3f];
    recons[45] = ((half2 *)SMEM)[(tmp8 >> 14) & 0x3f];
    recons[46] = ((half2 *)SMEM)[(tmp8 >> 20) & 0x3f];
    recons[47] = ((half2 *)SMEM)[(tmp8 >> 26) & 0x3f];
    recons[48] = ((half2 *)SMEM)[(tmp9) & 0x3f];
    recons[49] = ((half2 *)SMEM)[(tmp9 >> 6) & 0x3f];
    recons[50] = ((half2 *)SMEM)[(tmp9 >> 12) & 0x3f];
    recons[51] = ((half2 *)SMEM)[(tmp9 >> 18) & 0x3f];
    recons[52] = ((half2 *)SMEM)[(tmp9 >> 24) & 0x3f];
    recons[53] = ((half2 *)SMEM)[((tmp9 >> 30) & 0x3) | ((tmp10 << 2) & 0x3c)];
    recons[54] = ((half2 *)SMEM)[(tmp10 >> 4) & 0x3f];
    recons[55] = ((half2 *)SMEM)[(tmp10 >> 10) & 0x3f];
    recons[56] = ((half2 *)SMEM)[(tmp10 >> 16) & 0x3f];
    recons[57] = ((half2 *)SMEM)[(tmp10 >> 22) & 0x3f];
    recons[58] = ((half2 *)SMEM)[((tmp10 >> 28) & 0xf) | ((tmp11 << 4) & 0x30)];
    recons[59] = ((half2 *)SMEM)[(tmp11 >> 2) & 0x3f];
    recons[60] = ((half2 *)SMEM)[(tmp11 >> 8) & 0x3f];
    recons[61] = ((half2 *)SMEM)[(tmp11 >> 14) & 0x3f];
    recons[62] = ((half2 *)SMEM)[(tmp11 >> 20) & 0x3f];
    recons[63] = ((half2 *)SMEM)[(tmp11 >> 26) & 0x3f];
}

// nbits: 7	vec_sz: 2	code_n: 7	avg_bits:  3.500	SMEM_sz: 0KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<7, 2, 7, uint32_t>(const uint32_t code[7], half2 recons[32], half SMEM[256]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x7f];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 7) & 0x7f];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 14) & 0x7f];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 21) & 0x7f];
    recons[4] = ((half2 *)SMEM)[((tmp0 >> 28) & 0xf) | ((tmp1 << 4) & 0x70)];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 3) & 0x7f];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 10) & 0x7f];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 17) & 0x7f];
    recons[8] = ((half2 *)SMEM)[(tmp1 >> 24) & 0x7f];
    recons[9] = ((half2 *)SMEM)[((tmp1 >> 31) & 0x1) | ((tmp2 << 1) & 0x7e)];
    recons[10] = ((half2 *)SMEM)[(tmp2 >> 6) & 0x7f];
    recons[11] = ((half2 *)SMEM)[(tmp2 >> 13) & 0x7f];
    recons[12] = ((half2 *)SMEM)[(tmp2 >> 20) & 0x7f];
    recons[13] = ((half2 *)SMEM)[((tmp2 >> 27) & 0x1f) | ((tmp3 << 5) & 0x60)];
    recons[14] = ((half2 *)SMEM)[(tmp3 >> 2) & 0x7f];
    recons[15] = ((half2 *)SMEM)[(tmp3 >> 9) & 0x7f];
    recons[16] = ((half2 *)SMEM)[(tmp3 >> 16) & 0x7f];
    recons[17] = ((half2 *)SMEM)[(tmp3 >> 23) & 0x7f];
    recons[18] = ((half2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x7c)];
    recons[19] = ((half2 *)SMEM)[(tmp4 >> 5) & 0x7f];
    recons[20] = ((half2 *)SMEM)[(tmp4 >> 12) & 0x7f];
    recons[21] = ((half2 *)SMEM)[(tmp4 >> 19) & 0x7f];
    recons[22] = ((half2 *)SMEM)[((tmp4 >> 26) & 0x3f) | ((tmp5 << 6) & 0x40)];
    recons[23] = ((half2 *)SMEM)[(tmp5 >> 1) & 0x7f];
    recons[24] = ((half2 *)SMEM)[(tmp5 >> 8) & 0x7f];
    recons[25] = ((half2 *)SMEM)[(tmp5 >> 15) & 0x7f];
    recons[26] = ((half2 *)SMEM)[(tmp5 >> 22) & 0x7f];
    recons[27] = ((half2 *)SMEM)[((tmp5 >> 29) & 0x7) | ((tmp6 << 3) & 0x78)];
    recons[28] = ((half2 *)SMEM)[(tmp6 >> 4) & 0x7f];
    recons[29] = ((half2 *)SMEM)[(tmp6 >> 11) & 0x7f];
    recons[30] = ((half2 *)SMEM)[(tmp6 >> 18) & 0x7f];
    recons[31] = ((half2 *)SMEM)[(tmp6 >> 25) & 0x7f];
}

// nbits: 7	vec_sz: 2	code_n: 14	avg_bits:  3.500	SMEM_sz: 0KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<7, 2, 14, uint32_t>(const uint32_t code[14], half2 recons[64], half SMEM[256]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    uint32_t tmp12 = code[12];
    uint32_t tmp13 = code[13];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x7f];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 7) & 0x7f];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 14) & 0x7f];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 21) & 0x7f];
    recons[4] = ((half2 *)SMEM)[((tmp0 >> 28) & 0xf) | ((tmp1 << 4) & 0x70)];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 3) & 0x7f];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 10) & 0x7f];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 17) & 0x7f];
    recons[8] = ((half2 *)SMEM)[(tmp1 >> 24) & 0x7f];
    recons[9] = ((half2 *)SMEM)[((tmp1 >> 31) & 0x1) | ((tmp2 << 1) & 0x7e)];
    recons[10] = ((half2 *)SMEM)[(tmp2 >> 6) & 0x7f];
    recons[11] = ((half2 *)SMEM)[(tmp2 >> 13) & 0x7f];
    recons[12] = ((half2 *)SMEM)[(tmp2 >> 20) & 0x7f];
    recons[13] = ((half2 *)SMEM)[((tmp2 >> 27) & 0x1f) | ((tmp3 << 5) & 0x60)];
    recons[14] = ((half2 *)SMEM)[(tmp3 >> 2) & 0x7f];
    recons[15] = ((half2 *)SMEM)[(tmp3 >> 9) & 0x7f];
    recons[16] = ((half2 *)SMEM)[(tmp3 >> 16) & 0x7f];
    recons[17] = ((half2 *)SMEM)[(tmp3 >> 23) & 0x7f];
    recons[18] = ((half2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x7c)];
    recons[19] = ((half2 *)SMEM)[(tmp4 >> 5) & 0x7f];
    recons[20] = ((half2 *)SMEM)[(tmp4 >> 12) & 0x7f];
    recons[21] = ((half2 *)SMEM)[(tmp4 >> 19) & 0x7f];
    recons[22] = ((half2 *)SMEM)[((tmp4 >> 26) & 0x3f) | ((tmp5 << 6) & 0x40)];
    recons[23] = ((half2 *)SMEM)[(tmp5 >> 1) & 0x7f];
    recons[24] = ((half2 *)SMEM)[(tmp5 >> 8) & 0x7f];
    recons[25] = ((half2 *)SMEM)[(tmp5 >> 15) & 0x7f];
    recons[26] = ((half2 *)SMEM)[(tmp5 >> 22) & 0x7f];
    recons[27] = ((half2 *)SMEM)[((tmp5 >> 29) & 0x7) | ((tmp6 << 3) & 0x78)];
    recons[28] = ((half2 *)SMEM)[(tmp6 >> 4) & 0x7f];
    recons[29] = ((half2 *)SMEM)[(tmp6 >> 11) & 0x7f];
    recons[30] = ((half2 *)SMEM)[(tmp6 >> 18) & 0x7f];
    recons[31] = ((half2 *)SMEM)[(tmp6 >> 25) & 0x7f];
    recons[32] = ((half2 *)SMEM)[(tmp7) & 0x7f];
    recons[33] = ((half2 *)SMEM)[(tmp7 >> 7) & 0x7f];
    recons[34] = ((half2 *)SMEM)[(tmp7 >> 14) & 0x7f];
    recons[35] = ((half2 *)SMEM)[(tmp7 >> 21) & 0x7f];
    recons[36] = ((half2 *)SMEM)[((tmp7 >> 28) & 0xf) | ((tmp8 << 4) & 0x70)];
    recons[37] = ((half2 *)SMEM)[(tmp8 >> 3) & 0x7f];
    recons[38] = ((half2 *)SMEM)[(tmp8 >> 10) & 0x7f];
    recons[39] = ((half2 *)SMEM)[(tmp8 >> 17) & 0x7f];
    recons[40] = ((half2 *)SMEM)[(tmp8 >> 24) & 0x7f];
    recons[41] = ((half2 *)SMEM)[((tmp8 >> 31) & 0x1) | ((tmp9 << 1) & 0x7e)];
    recons[42] = ((half2 *)SMEM)[(tmp9 >> 6) & 0x7f];
    recons[43] = ((half2 *)SMEM)[(tmp9 >> 13) & 0x7f];
    recons[44] = ((half2 *)SMEM)[(tmp9 >> 20) & 0x7f];
    recons[45] = ((half2 *)SMEM)[((tmp9 >> 27) & 0x1f) | ((tmp10 << 5) & 0x60)];
    recons[46] = ((half2 *)SMEM)[(tmp10 >> 2) & 0x7f];
    recons[47] = ((half2 *)SMEM)[(tmp10 >> 9) & 0x7f];
    recons[48] = ((half2 *)SMEM)[(tmp10 >> 16) & 0x7f];
    recons[49] = ((half2 *)SMEM)[(tmp10 >> 23) & 0x7f];
    recons[50] = ((half2 *)SMEM)[((tmp10 >> 30) & 0x3) | ((tmp11 << 2) & 0x7c)];
    recons[51] = ((half2 *)SMEM)[(tmp11 >> 5) & 0x7f];
    recons[52] = ((half2 *)SMEM)[(tmp11 >> 12) & 0x7f];
    recons[53] = ((half2 *)SMEM)[(tmp11 >> 19) & 0x7f];
    recons[54] = ((half2 *)SMEM)[((tmp11 >> 26) & 0x3f) | ((tmp12 << 6) & 0x40)];
    recons[55] = ((half2 *)SMEM)[(tmp12 >> 1) & 0x7f];
    recons[56] = ((half2 *)SMEM)[(tmp12 >> 8) & 0x7f];
    recons[57] = ((half2 *)SMEM)[(tmp12 >> 15) & 0x7f];
    recons[58] = ((half2 *)SMEM)[(tmp12 >> 22) & 0x7f];
    recons[59] = ((half2 *)SMEM)[((tmp12 >> 29) & 0x7) | ((tmp13 << 3) & 0x78)];
    recons[60] = ((half2 *)SMEM)[(tmp13 >> 4) & 0x7f];
    recons[61] = ((half2 *)SMEM)[(tmp13 >> 11) & 0x7f];
    recons[62] = ((half2 *)SMEM)[(tmp13 >> 18) & 0x7f];
    recons[63] = ((half2 *)SMEM)[(tmp13 >> 25) & 0x7f];
}

// nbits: 8	vec_sz: 2	code_n: 4	avg_bits:  4.000	SMEM_sz: 0KB	recons_n: 16
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<8, 2, 4, uint32_t>(const uint32_t code[4], half2 recons[16], half SMEM[512]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 8) & 0xff];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 16) & 0xff];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 24) & 0xff];
    recons[4] = ((half2 *)SMEM)[(tmp1) & 0xff];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 8) & 0xff];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xff];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 24) & 0xff];
    recons[8] = ((half2 *)SMEM)[(tmp2) & 0xff];
    recons[9] = ((half2 *)SMEM)[(tmp2 >> 8) & 0xff];
    recons[10] = ((half2 *)SMEM)[(tmp2 >> 16) & 0xff];
    recons[11] = ((half2 *)SMEM)[(tmp2 >> 24) & 0xff];
    recons[12] = ((half2 *)SMEM)[(tmp3) & 0xff];
    recons[13] = ((half2 *)SMEM)[(tmp3 >> 8) & 0xff];
    recons[14] = ((half2 *)SMEM)[(tmp3 >> 16) & 0xff];
    recons[15] = ((half2 *)SMEM)[(tmp3 >> 24) & 0xff];
}

// nbits: 8	vec_sz: 2	code_n: 8	avg_bits:  4.000	SMEM_sz: 0KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<8, 2, 8, uint32_t>(const uint32_t code[8], half2 recons[32], half SMEM[512]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 8) & 0xff];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 16) & 0xff];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 24) & 0xff];
    recons[4] = ((half2 *)SMEM)[(tmp1) & 0xff];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 8) & 0xff];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xff];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 24) & 0xff];
    recons[8] = ((half2 *)SMEM)[(tmp2) & 0xff];
    recons[9] = ((half2 *)SMEM)[(tmp2 >> 8) & 0xff];
    recons[10] = ((half2 *)SMEM)[(tmp2 >> 16) & 0xff];
    recons[11] = ((half2 *)SMEM)[(tmp2 >> 24) & 0xff];
    recons[12] = ((half2 *)SMEM)[(tmp3) & 0xff];
    recons[13] = ((half2 *)SMEM)[(tmp3 >> 8) & 0xff];
    recons[14] = ((half2 *)SMEM)[(tmp3 >> 16) & 0xff];
    recons[15] = ((half2 *)SMEM)[(tmp3 >> 24) & 0xff];
    recons[16] = ((half2 *)SMEM)[(tmp4) & 0xff];
    recons[17] = ((half2 *)SMEM)[(tmp4 >> 8) & 0xff];
    recons[18] = ((half2 *)SMEM)[(tmp4 >> 16) & 0xff];
    recons[19] = ((half2 *)SMEM)[(tmp4 >> 24) & 0xff];
    recons[20] = ((half2 *)SMEM)[(tmp5) & 0xff];
    recons[21] = ((half2 *)SMEM)[(tmp5 >> 8) & 0xff];
    recons[22] = ((half2 *)SMEM)[(tmp5 >> 16) & 0xff];
    recons[23] = ((half2 *)SMEM)[(tmp5 >> 24) & 0xff];
    recons[24] = ((half2 *)SMEM)[(tmp6) & 0xff];
    recons[25] = ((half2 *)SMEM)[(tmp6 >> 8) & 0xff];
    recons[26] = ((half2 *)SMEM)[(tmp6 >> 16) & 0xff];
    recons[27] = ((half2 *)SMEM)[(tmp6 >> 24) & 0xff];
    recons[28] = ((half2 *)SMEM)[(tmp7) & 0xff];
    recons[29] = ((half2 *)SMEM)[(tmp7 >> 8) & 0xff];
    recons[30] = ((half2 *)SMEM)[(tmp7 >> 16) & 0xff];
    recons[31] = ((half2 *)SMEM)[(tmp7 >> 24) & 0xff];
}

// nbits: 8	vec_sz: 2	code_n: 16	avg_bits:  4.000	SMEM_sz: 0KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<8, 2, 16, uint32_t>(const uint32_t code[16], half2 recons[64], half SMEM[512]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    uint32_t tmp12 = code[12];
    uint32_t tmp13 = code[13];
    uint32_t tmp14 = code[14];
    uint32_t tmp15 = code[15];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 8) & 0xff];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 16) & 0xff];
    recons[3] = ((half2 *)SMEM)[(tmp0 >> 24) & 0xff];
    recons[4] = ((half2 *)SMEM)[(tmp1) & 0xff];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 8) & 0xff];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xff];
    recons[7] = ((half2 *)SMEM)[(tmp1 >> 24) & 0xff];
    recons[8] = ((half2 *)SMEM)[(tmp2) & 0xff];
    recons[9] = ((half2 *)SMEM)[(tmp2 >> 8) & 0xff];
    recons[10] = ((half2 *)SMEM)[(tmp2 >> 16) & 0xff];
    recons[11] = ((half2 *)SMEM)[(tmp2 >> 24) & 0xff];
    recons[12] = ((half2 *)SMEM)[(tmp3) & 0xff];
    recons[13] = ((half2 *)SMEM)[(tmp3 >> 8) & 0xff];
    recons[14] = ((half2 *)SMEM)[(tmp3 >> 16) & 0xff];
    recons[15] = ((half2 *)SMEM)[(tmp3 >> 24) & 0xff];
    recons[16] = ((half2 *)SMEM)[(tmp4) & 0xff];
    recons[17] = ((half2 *)SMEM)[(tmp4 >> 8) & 0xff];
    recons[18] = ((half2 *)SMEM)[(tmp4 >> 16) & 0xff];
    recons[19] = ((half2 *)SMEM)[(tmp4 >> 24) & 0xff];
    recons[20] = ((half2 *)SMEM)[(tmp5) & 0xff];
    recons[21] = ((half2 *)SMEM)[(tmp5 >> 8) & 0xff];
    recons[22] = ((half2 *)SMEM)[(tmp5 >> 16) & 0xff];
    recons[23] = ((half2 *)SMEM)[(tmp5 >> 24) & 0xff];
    recons[24] = ((half2 *)SMEM)[(tmp6) & 0xff];
    recons[25] = ((half2 *)SMEM)[(tmp6 >> 8) & 0xff];
    recons[26] = ((half2 *)SMEM)[(tmp6 >> 16) & 0xff];
    recons[27] = ((half2 *)SMEM)[(tmp6 >> 24) & 0xff];
    recons[28] = ((half2 *)SMEM)[(tmp7) & 0xff];
    recons[29] = ((half2 *)SMEM)[(tmp7 >> 8) & 0xff];
    recons[30] = ((half2 *)SMEM)[(tmp7 >> 16) & 0xff];
    recons[31] = ((half2 *)SMEM)[(tmp7 >> 24) & 0xff];
    recons[32] = ((half2 *)SMEM)[(tmp8) & 0xff];
    recons[33] = ((half2 *)SMEM)[(tmp8 >> 8) & 0xff];
    recons[34] = ((half2 *)SMEM)[(tmp8 >> 16) & 0xff];
    recons[35] = ((half2 *)SMEM)[(tmp8 >> 24) & 0xff];
    recons[36] = ((half2 *)SMEM)[(tmp9) & 0xff];
    recons[37] = ((half2 *)SMEM)[(tmp9 >> 8) & 0xff];
    recons[38] = ((half2 *)SMEM)[(tmp9 >> 16) & 0xff];
    recons[39] = ((half2 *)SMEM)[(tmp9 >> 24) & 0xff];
    recons[40] = ((half2 *)SMEM)[(tmp10) & 0xff];
    recons[41] = ((half2 *)SMEM)[(tmp10 >> 8) & 0xff];
    recons[42] = ((half2 *)SMEM)[(tmp10 >> 16) & 0xff];
    recons[43] = ((half2 *)SMEM)[(tmp10 >> 24) & 0xff];
    recons[44] = ((half2 *)SMEM)[(tmp11) & 0xff];
    recons[45] = ((half2 *)SMEM)[(tmp11 >> 8) & 0xff];
    recons[46] = ((half2 *)SMEM)[(tmp11 >> 16) & 0xff];
    recons[47] = ((half2 *)SMEM)[(tmp11 >> 24) & 0xff];
    recons[48] = ((half2 *)SMEM)[(tmp12) & 0xff];
    recons[49] = ((half2 *)SMEM)[(tmp12 >> 8) & 0xff];
    recons[50] = ((half2 *)SMEM)[(tmp12 >> 16) & 0xff];
    recons[51] = ((half2 *)SMEM)[(tmp12 >> 24) & 0xff];
    recons[52] = ((half2 *)SMEM)[(tmp13) & 0xff];
    recons[53] = ((half2 *)SMEM)[(tmp13 >> 8) & 0xff];
    recons[54] = ((half2 *)SMEM)[(tmp13 >> 16) & 0xff];
    recons[55] = ((half2 *)SMEM)[(tmp13 >> 24) & 0xff];
    recons[56] = ((half2 *)SMEM)[(tmp14) & 0xff];
    recons[57] = ((half2 *)SMEM)[(tmp14 >> 8) & 0xff];
    recons[58] = ((half2 *)SMEM)[(tmp14 >> 16) & 0xff];
    recons[59] = ((half2 *)SMEM)[(tmp14 >> 24) & 0xff];
    recons[60] = ((half2 *)SMEM)[(tmp15) & 0xff];
    recons[61] = ((half2 *)SMEM)[(tmp15 >> 8) & 0xff];
    recons[62] = ((half2 *)SMEM)[(tmp15 >> 16) & 0xff];
    recons[63] = ((half2 *)SMEM)[(tmp15 >> 24) & 0xff];
}

// nbits: 9	vec_sz: 2	code_n: 9	avg_bits:  4.500	SMEM_sz: 1KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<9, 2, 9, uint32_t>(const uint32_t code[9], half2 recons[32], half SMEM[1024]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x1ff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 9) & 0x1ff];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 18) & 0x1ff];
    recons[3] = ((half2 *)SMEM)[((tmp0 >> 27) & 0x1f) | ((tmp1 << 5) & 0x1e0)];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 4) & 0x1ff];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 13) & 0x1ff];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 22) & 0x1ff];
    recons[7] = ((half2 *)SMEM)[((tmp1 >> 31) & 0x1) | ((tmp2 << 1) & 0x1fe)];
    recons[8] = ((half2 *)SMEM)[(tmp2 >> 8) & 0x1ff];
    recons[9] = ((half2 *)SMEM)[(tmp2 >> 17) & 0x1ff];
    recons[10] = ((half2 *)SMEM)[((tmp2 >> 26) & 0x3f) | ((tmp3 << 6) & 0x1c0)];
    recons[11] = ((half2 *)SMEM)[(tmp3 >> 3) & 0x1ff];
    recons[12] = ((half2 *)SMEM)[(tmp3 >> 12) & 0x1ff];
    recons[13] = ((half2 *)SMEM)[(tmp3 >> 21) & 0x1ff];
    recons[14] = ((half2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x1fc)];
    recons[15] = ((half2 *)SMEM)[(tmp4 >> 7) & 0x1ff];
    recons[16] = ((half2 *)SMEM)[(tmp4 >> 16) & 0x1ff];
    recons[17] = ((half2 *)SMEM)[((tmp4 >> 25) & 0x7f) | ((tmp5 << 7) & 0x180)];
    recons[18] = ((half2 *)SMEM)[(tmp5 >> 2) & 0x1ff];
    recons[19] = ((half2 *)SMEM)[(tmp5 >> 11) & 0x1ff];
    recons[20] = ((half2 *)SMEM)[(tmp5 >> 20) & 0x1ff];
    recons[21] = ((half2 *)SMEM)[((tmp5 >> 29) & 0x7) | ((tmp6 << 3) & 0x1f8)];
    recons[22] = ((half2 *)SMEM)[(tmp6 >> 6) & 0x1ff];
    recons[23] = ((half2 *)SMEM)[(tmp6 >> 15) & 0x1ff];
    recons[24] = ((half2 *)SMEM)[((tmp6 >> 24) & 0xff) | ((tmp7 << 8) & 0x100)];
    recons[25] = ((half2 *)SMEM)[(tmp7 >> 1) & 0x1ff];
    recons[26] = ((half2 *)SMEM)[(tmp7 >> 10) & 0x1ff];
    recons[27] = ((half2 *)SMEM)[(tmp7 >> 19) & 0x1ff];
    recons[28] = ((half2 *)SMEM)[((tmp7 >> 28) & 0xf) | ((tmp8 << 4) & 0x1f0)];
    recons[29] = ((half2 *)SMEM)[(tmp8 >> 5) & 0x1ff];
    recons[30] = ((half2 *)SMEM)[(tmp8 >> 14) & 0x1ff];
    recons[31] = ((half2 *)SMEM)[(tmp8 >> 23) & 0x1ff];
}

// nbits: 9	vec_sz: 2	code_n: 18	avg_bits:  4.500	SMEM_sz: 1KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<9, 2, 18, uint32_t>(const uint32_t code[18], half2 recons[64], half SMEM[1024]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    uint32_t tmp12 = code[12];
    uint32_t tmp13 = code[13];
    uint32_t tmp14 = code[14];
    uint32_t tmp15 = code[15];
    uint32_t tmp16 = code[16];
    uint32_t tmp17 = code[17];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x1ff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 9) & 0x1ff];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 18) & 0x1ff];
    recons[3] = ((half2 *)SMEM)[((tmp0 >> 27) & 0x1f) | ((tmp1 << 5) & 0x1e0)];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 4) & 0x1ff];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 13) & 0x1ff];
    recons[6] = ((half2 *)SMEM)[(tmp1 >> 22) & 0x1ff];
    recons[7] = ((half2 *)SMEM)[((tmp1 >> 31) & 0x1) | ((tmp2 << 1) & 0x1fe)];
    recons[8] = ((half2 *)SMEM)[(tmp2 >> 8) & 0x1ff];
    recons[9] = ((half2 *)SMEM)[(tmp2 >> 17) & 0x1ff];
    recons[10] = ((half2 *)SMEM)[((tmp2 >> 26) & 0x3f) | ((tmp3 << 6) & 0x1c0)];
    recons[11] = ((half2 *)SMEM)[(tmp3 >> 3) & 0x1ff];
    recons[12] = ((half2 *)SMEM)[(tmp3 >> 12) & 0x1ff];
    recons[13] = ((half2 *)SMEM)[(tmp3 >> 21) & 0x1ff];
    recons[14] = ((half2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x1fc)];
    recons[15] = ((half2 *)SMEM)[(tmp4 >> 7) & 0x1ff];
    recons[16] = ((half2 *)SMEM)[(tmp4 >> 16) & 0x1ff];
    recons[17] = ((half2 *)SMEM)[((tmp4 >> 25) & 0x7f) | ((tmp5 << 7) & 0x180)];
    recons[18] = ((half2 *)SMEM)[(tmp5 >> 2) & 0x1ff];
    recons[19] = ((half2 *)SMEM)[(tmp5 >> 11) & 0x1ff];
    recons[20] = ((half2 *)SMEM)[(tmp5 >> 20) & 0x1ff];
    recons[21] = ((half2 *)SMEM)[((tmp5 >> 29) & 0x7) | ((tmp6 << 3) & 0x1f8)];
    recons[22] = ((half2 *)SMEM)[(tmp6 >> 6) & 0x1ff];
    recons[23] = ((half2 *)SMEM)[(tmp6 >> 15) & 0x1ff];
    recons[24] = ((half2 *)SMEM)[((tmp6 >> 24) & 0xff) | ((tmp7 << 8) & 0x100)];
    recons[25] = ((half2 *)SMEM)[(tmp7 >> 1) & 0x1ff];
    recons[26] = ((half2 *)SMEM)[(tmp7 >> 10) & 0x1ff];
    recons[27] = ((half2 *)SMEM)[(tmp7 >> 19) & 0x1ff];
    recons[28] = ((half2 *)SMEM)[((tmp7 >> 28) & 0xf) | ((tmp8 << 4) & 0x1f0)];
    recons[29] = ((half2 *)SMEM)[(tmp8 >> 5) & 0x1ff];
    recons[30] = ((half2 *)SMEM)[(tmp8 >> 14) & 0x1ff];
    recons[31] = ((half2 *)SMEM)[(tmp8 >> 23) & 0x1ff];
    recons[32] = ((half2 *)SMEM)[(tmp9) & 0x1ff];
    recons[33] = ((half2 *)SMEM)[(tmp9 >> 9) & 0x1ff];
    recons[34] = ((half2 *)SMEM)[(tmp9 >> 18) & 0x1ff];
    recons[35] = ((half2 *)SMEM)[((tmp9 >> 27) & 0x1f) | ((tmp10 << 5) & 0x1e0)];
    recons[36] = ((half2 *)SMEM)[(tmp10 >> 4) & 0x1ff];
    recons[37] = ((half2 *)SMEM)[(tmp10 >> 13) & 0x1ff];
    recons[38] = ((half2 *)SMEM)[(tmp10 >> 22) & 0x1ff];
    recons[39] = ((half2 *)SMEM)[((tmp10 >> 31) & 0x1) | ((tmp11 << 1) & 0x1fe)];
    recons[40] = ((half2 *)SMEM)[(tmp11 >> 8) & 0x1ff];
    recons[41] = ((half2 *)SMEM)[(tmp11 >> 17) & 0x1ff];
    recons[42] = ((half2 *)SMEM)[((tmp11 >> 26) & 0x3f) | ((tmp12 << 6) & 0x1c0)];
    recons[43] = ((half2 *)SMEM)[(tmp12 >> 3) & 0x1ff];
    recons[44] = ((half2 *)SMEM)[(tmp12 >> 12) & 0x1ff];
    recons[45] = ((half2 *)SMEM)[(tmp12 >> 21) & 0x1ff];
    recons[46] = ((half2 *)SMEM)[((tmp12 >> 30) & 0x3) | ((tmp13 << 2) & 0x1fc)];
    recons[47] = ((half2 *)SMEM)[(tmp13 >> 7) & 0x1ff];
    recons[48] = ((half2 *)SMEM)[(tmp13 >> 16) & 0x1ff];
    recons[49] = ((half2 *)SMEM)[((tmp13 >> 25) & 0x7f) | ((tmp14 << 7) & 0x180)];
    recons[50] = ((half2 *)SMEM)[(tmp14 >> 2) & 0x1ff];
    recons[51] = ((half2 *)SMEM)[(tmp14 >> 11) & 0x1ff];
    recons[52] = ((half2 *)SMEM)[(tmp14 >> 20) & 0x1ff];
    recons[53] = ((half2 *)SMEM)[((tmp14 >> 29) & 0x7) | ((tmp15 << 3) & 0x1f8)];
    recons[54] = ((half2 *)SMEM)[(tmp15 >> 6) & 0x1ff];
    recons[55] = ((half2 *)SMEM)[(tmp15 >> 15) & 0x1ff];
    recons[56] = ((half2 *)SMEM)[((tmp15 >> 24) & 0xff) | ((tmp16 << 8) & 0x100)];
    recons[57] = ((half2 *)SMEM)[(tmp16 >> 1) & 0x1ff];
    recons[58] = ((half2 *)SMEM)[(tmp16 >> 10) & 0x1ff];
    recons[59] = ((half2 *)SMEM)[(tmp16 >> 19) & 0x1ff];
    recons[60] = ((half2 *)SMEM)[((tmp16 >> 28) & 0xf) | ((tmp17 << 4) & 0x1f0)];
    recons[61] = ((half2 *)SMEM)[(tmp17 >> 5) & 0x1ff];
    recons[62] = ((half2 *)SMEM)[(tmp17 >> 14) & 0x1ff];
    recons[63] = ((half2 *)SMEM)[(tmp17 >> 23) & 0x1ff];
}

// nbits: 10	vec_sz: 2	code_n: 5	avg_bits:  5.000	SMEM_sz: 2KB	recons_n: 16
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<10, 2, 5, uint32_t>(const uint32_t code[5], half2 recons[16], half SMEM[2048]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x3ff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 10) & 0x3ff];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 20) & 0x3ff];
    recons[3] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3fc)];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 8) & 0x3ff];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 18) & 0x3ff];
    recons[6] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x3f0)];
    recons[7] = ((half2 *)SMEM)[(tmp2 >> 6) & 0x3ff];
    recons[8] = ((half2 *)SMEM)[(tmp2 >> 16) & 0x3ff];
    recons[9] = ((half2 *)SMEM)[((tmp2 >> 26) & 0x3f) | ((tmp3 << 6) & 0x3c0)];
    recons[10] = ((half2 *)SMEM)[(tmp3 >> 4) & 0x3ff];
    recons[11] = ((half2 *)SMEM)[(tmp3 >> 14) & 0x3ff];
    recons[12] = ((half2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0x300)];
    recons[13] = ((half2 *)SMEM)[(tmp4 >> 2) & 0x3ff];
    recons[14] = ((half2 *)SMEM)[(tmp4 >> 12) & 0x3ff];
    recons[15] = ((half2 *)SMEM)[(tmp4 >> 22) & 0x3ff];
}

// nbits: 10	vec_sz: 2	code_n: 10	avg_bits:  5.000	SMEM_sz: 2KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<10, 2, 10, uint32_t>(const uint32_t code[10], half2 recons[32], half SMEM[2048]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x3ff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 10) & 0x3ff];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 20) & 0x3ff];
    recons[3] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3fc)];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 8) & 0x3ff];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 18) & 0x3ff];
    recons[6] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x3f0)];
    recons[7] = ((half2 *)SMEM)[(tmp2 >> 6) & 0x3ff];
    recons[8] = ((half2 *)SMEM)[(tmp2 >> 16) & 0x3ff];
    recons[9] = ((half2 *)SMEM)[((tmp2 >> 26) & 0x3f) | ((tmp3 << 6) & 0x3c0)];
    recons[10] = ((half2 *)SMEM)[(tmp3 >> 4) & 0x3ff];
    recons[11] = ((half2 *)SMEM)[(tmp3 >> 14) & 0x3ff];
    recons[12] = ((half2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0x300)];
    recons[13] = ((half2 *)SMEM)[(tmp4 >> 2) & 0x3ff];
    recons[14] = ((half2 *)SMEM)[(tmp4 >> 12) & 0x3ff];
    recons[15] = ((half2 *)SMEM)[(tmp4 >> 22) & 0x3ff];
    recons[16] = ((half2 *)SMEM)[(tmp5) & 0x3ff];
    recons[17] = ((half2 *)SMEM)[(tmp5 >> 10) & 0x3ff];
    recons[18] = ((half2 *)SMEM)[(tmp5 >> 20) & 0x3ff];
    recons[19] = ((half2 *)SMEM)[((tmp5 >> 30) & 0x3) | ((tmp6 << 2) & 0x3fc)];
    recons[20] = ((half2 *)SMEM)[(tmp6 >> 8) & 0x3ff];
    recons[21] = ((half2 *)SMEM)[(tmp6 >> 18) & 0x3ff];
    recons[22] = ((half2 *)SMEM)[((tmp6 >> 28) & 0xf) | ((tmp7 << 4) & 0x3f0)];
    recons[23] = ((half2 *)SMEM)[(tmp7 >> 6) & 0x3ff];
    recons[24] = ((half2 *)SMEM)[(tmp7 >> 16) & 0x3ff];
    recons[25] = ((half2 *)SMEM)[((tmp7 >> 26) & 0x3f) | ((tmp8 << 6) & 0x3c0)];
    recons[26] = ((half2 *)SMEM)[(tmp8 >> 4) & 0x3ff];
    recons[27] = ((half2 *)SMEM)[(tmp8 >> 14) & 0x3ff];
    recons[28] = ((half2 *)SMEM)[((tmp8 >> 24) & 0xff) | ((tmp9 << 8) & 0x300)];
    recons[29] = ((half2 *)SMEM)[(tmp9 >> 2) & 0x3ff];
    recons[30] = ((half2 *)SMEM)[(tmp9 >> 12) & 0x3ff];
    recons[31] = ((half2 *)SMEM)[(tmp9 >> 22) & 0x3ff];
}

// nbits: 10	vec_sz: 2	code_n: 20	avg_bits:  5.000	SMEM_sz: 2KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<10, 2, 20, uint32_t>(const uint32_t code[20], half2 recons[64], half SMEM[2048]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    uint32_t tmp12 = code[12];
    uint32_t tmp13 = code[13];
    uint32_t tmp14 = code[14];
    uint32_t tmp15 = code[15];
    uint32_t tmp16 = code[16];
    uint32_t tmp17 = code[17];
    uint32_t tmp18 = code[18];
    uint32_t tmp19 = code[19];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x3ff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 10) & 0x3ff];
    recons[2] = ((half2 *)SMEM)[(tmp0 >> 20) & 0x3ff];
    recons[3] = ((half2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3fc)];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 8) & 0x3ff];
    recons[5] = ((half2 *)SMEM)[(tmp1 >> 18) & 0x3ff];
    recons[6] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x3f0)];
    recons[7] = ((half2 *)SMEM)[(tmp2 >> 6) & 0x3ff];
    recons[8] = ((half2 *)SMEM)[(tmp2 >> 16) & 0x3ff];
    recons[9] = ((half2 *)SMEM)[((tmp2 >> 26) & 0x3f) | ((tmp3 << 6) & 0x3c0)];
    recons[10] = ((half2 *)SMEM)[(tmp3 >> 4) & 0x3ff];
    recons[11] = ((half2 *)SMEM)[(tmp3 >> 14) & 0x3ff];
    recons[12] = ((half2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0x300)];
    recons[13] = ((half2 *)SMEM)[(tmp4 >> 2) & 0x3ff];
    recons[14] = ((half2 *)SMEM)[(tmp4 >> 12) & 0x3ff];
    recons[15] = ((half2 *)SMEM)[(tmp4 >> 22) & 0x3ff];
    recons[16] = ((half2 *)SMEM)[(tmp5) & 0x3ff];
    recons[17] = ((half2 *)SMEM)[(tmp5 >> 10) & 0x3ff];
    recons[18] = ((half2 *)SMEM)[(tmp5 >> 20) & 0x3ff];
    recons[19] = ((half2 *)SMEM)[((tmp5 >> 30) & 0x3) | ((tmp6 << 2) & 0x3fc)];
    recons[20] = ((half2 *)SMEM)[(tmp6 >> 8) & 0x3ff];
    recons[21] = ((half2 *)SMEM)[(tmp6 >> 18) & 0x3ff];
    recons[22] = ((half2 *)SMEM)[((tmp6 >> 28) & 0xf) | ((tmp7 << 4) & 0x3f0)];
    recons[23] = ((half2 *)SMEM)[(tmp7 >> 6) & 0x3ff];
    recons[24] = ((half2 *)SMEM)[(tmp7 >> 16) & 0x3ff];
    recons[25] = ((half2 *)SMEM)[((tmp7 >> 26) & 0x3f) | ((tmp8 << 6) & 0x3c0)];
    recons[26] = ((half2 *)SMEM)[(tmp8 >> 4) & 0x3ff];
    recons[27] = ((half2 *)SMEM)[(tmp8 >> 14) & 0x3ff];
    recons[28] = ((half2 *)SMEM)[((tmp8 >> 24) & 0xff) | ((tmp9 << 8) & 0x300)];
    recons[29] = ((half2 *)SMEM)[(tmp9 >> 2) & 0x3ff];
    recons[30] = ((half2 *)SMEM)[(tmp9 >> 12) & 0x3ff];
    recons[31] = ((half2 *)SMEM)[(tmp9 >> 22) & 0x3ff];
    recons[32] = ((half2 *)SMEM)[(tmp10) & 0x3ff];
    recons[33] = ((half2 *)SMEM)[(tmp10 >> 10) & 0x3ff];
    recons[34] = ((half2 *)SMEM)[(tmp10 >> 20) & 0x3ff];
    recons[35] = ((half2 *)SMEM)[((tmp10 >> 30) & 0x3) | ((tmp11 << 2) & 0x3fc)];
    recons[36] = ((half2 *)SMEM)[(tmp11 >> 8) & 0x3ff];
    recons[37] = ((half2 *)SMEM)[(tmp11 >> 18) & 0x3ff];
    recons[38] = ((half2 *)SMEM)[((tmp11 >> 28) & 0xf) | ((tmp12 << 4) & 0x3f0)];
    recons[39] = ((half2 *)SMEM)[(tmp12 >> 6) & 0x3ff];
    recons[40] = ((half2 *)SMEM)[(tmp12 >> 16) & 0x3ff];
    recons[41] = ((half2 *)SMEM)[((tmp12 >> 26) & 0x3f) | ((tmp13 << 6) & 0x3c0)];
    recons[42] = ((half2 *)SMEM)[(tmp13 >> 4) & 0x3ff];
    recons[43] = ((half2 *)SMEM)[(tmp13 >> 14) & 0x3ff];
    recons[44] = ((half2 *)SMEM)[((tmp13 >> 24) & 0xff) | ((tmp14 << 8) & 0x300)];
    recons[45] = ((half2 *)SMEM)[(tmp14 >> 2) & 0x3ff];
    recons[46] = ((half2 *)SMEM)[(tmp14 >> 12) & 0x3ff];
    recons[47] = ((half2 *)SMEM)[(tmp14 >> 22) & 0x3ff];
    recons[48] = ((half2 *)SMEM)[(tmp15) & 0x3ff];
    recons[49] = ((half2 *)SMEM)[(tmp15 >> 10) & 0x3ff];
    recons[50] = ((half2 *)SMEM)[(tmp15 >> 20) & 0x3ff];
    recons[51] = ((half2 *)SMEM)[((tmp15 >> 30) & 0x3) | ((tmp16 << 2) & 0x3fc)];
    recons[52] = ((half2 *)SMEM)[(tmp16 >> 8) & 0x3ff];
    recons[53] = ((half2 *)SMEM)[(tmp16 >> 18) & 0x3ff];
    recons[54] = ((half2 *)SMEM)[((tmp16 >> 28) & 0xf) | ((tmp17 << 4) & 0x3f0)];
    recons[55] = ((half2 *)SMEM)[(tmp17 >> 6) & 0x3ff];
    recons[56] = ((half2 *)SMEM)[(tmp17 >> 16) & 0x3ff];
    recons[57] = ((half2 *)SMEM)[((tmp17 >> 26) & 0x3f) | ((tmp18 << 6) & 0x3c0)];
    recons[58] = ((half2 *)SMEM)[(tmp18 >> 4) & 0x3ff];
    recons[59] = ((half2 *)SMEM)[(tmp18 >> 14) & 0x3ff];
    recons[60] = ((half2 *)SMEM)[((tmp18 >> 24) & 0xff) | ((tmp19 << 8) & 0x300)];
    recons[61] = ((half2 *)SMEM)[(tmp19 >> 2) & 0x3ff];
    recons[62] = ((half2 *)SMEM)[(tmp19 >> 12) & 0x3ff];
    recons[63] = ((half2 *)SMEM)[(tmp19 >> 22) & 0x3ff];
}

// nbits: 11	vec_sz: 2	code_n: 11	avg_bits:  5.500	SMEM_sz: 4KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<11, 2, 11, uint32_t>(const uint32_t code[11], half2 recons[32], half SMEM[4096]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x7ff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 11) & 0x7ff];
    recons[2] = ((half2 *)SMEM)[((tmp0 >> 22) & 0x3ff) | ((tmp1 << 10) & 0x400)];
    recons[3] = ((half2 *)SMEM)[(tmp1 >> 1) & 0x7ff];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 12) & 0x7ff];
    recons[5] = ((half2 *)SMEM)[((tmp1 >> 23) & 0x1ff) | ((tmp2 << 9) & 0x600)];
    recons[6] = ((half2 *)SMEM)[(tmp2 >> 2) & 0x7ff];
    recons[7] = ((half2 *)SMEM)[(tmp2 >> 13) & 0x7ff];
    recons[8] = ((half2 *)SMEM)[((tmp2 >> 24) & 0xff) | ((tmp3 << 8) & 0x700)];
    recons[9] = ((half2 *)SMEM)[(tmp3 >> 3) & 0x7ff];
    recons[10] = ((half2 *)SMEM)[(tmp3 >> 14) & 0x7ff];
    recons[11] = ((half2 *)SMEM)[((tmp3 >> 25) & 0x7f) | ((tmp4 << 7) & 0x780)];
    recons[12] = ((half2 *)SMEM)[(tmp4 >> 4) & 0x7ff];
    recons[13] = ((half2 *)SMEM)[(tmp4 >> 15) & 0x7ff];
    recons[14] = ((half2 *)SMEM)[((tmp4 >> 26) & 0x3f) | ((tmp5 << 6) & 0x7c0)];
    recons[15] = ((half2 *)SMEM)[(tmp5 >> 5) & 0x7ff];
    recons[16] = ((half2 *)SMEM)[(tmp5 >> 16) & 0x7ff];
    recons[17] = ((half2 *)SMEM)[((tmp5 >> 27) & 0x1f) | ((tmp6 << 5) & 0x7e0)];
    recons[18] = ((half2 *)SMEM)[(tmp6 >> 6) & 0x7ff];
    recons[19] = ((half2 *)SMEM)[(tmp6 >> 17) & 0x7ff];
    recons[20] = ((half2 *)SMEM)[((tmp6 >> 28) & 0xf) | ((tmp7 << 4) & 0x7f0)];
    recons[21] = ((half2 *)SMEM)[(tmp7 >> 7) & 0x7ff];
    recons[22] = ((half2 *)SMEM)[(tmp7 >> 18) & 0x7ff];
    recons[23] = ((half2 *)SMEM)[((tmp7 >> 29) & 0x7) | ((tmp8 << 3) & 0x7f8)];
    recons[24] = ((half2 *)SMEM)[(tmp8 >> 8) & 0x7ff];
    recons[25] = ((half2 *)SMEM)[(tmp8 >> 19) & 0x7ff];
    recons[26] = ((half2 *)SMEM)[((tmp8 >> 30) & 0x3) | ((tmp9 << 2) & 0x7fc)];
    recons[27] = ((half2 *)SMEM)[(tmp9 >> 9) & 0x7ff];
    recons[28] = ((half2 *)SMEM)[(tmp9 >> 20) & 0x7ff];
    recons[29] = ((half2 *)SMEM)[((tmp9 >> 31) & 0x1) | ((tmp10 << 1) & 0x7fe)];
    recons[30] = ((half2 *)SMEM)[(tmp10 >> 10) & 0x7ff];
    recons[31] = ((half2 *)SMEM)[(tmp10 >> 21) & 0x7ff];
}

// nbits: 11	vec_sz: 2	code_n: 22	avg_bits:  5.500	SMEM_sz: 4KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<11, 2, 22, uint32_t>(const uint32_t code[22], half2 recons[64], half SMEM[4096]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    uint32_t tmp12 = code[12];
    uint32_t tmp13 = code[13];
    uint32_t tmp14 = code[14];
    uint32_t tmp15 = code[15];
    uint32_t tmp16 = code[16];
    uint32_t tmp17 = code[17];
    uint32_t tmp18 = code[18];
    uint32_t tmp19 = code[19];
    uint32_t tmp20 = code[20];
    uint32_t tmp21 = code[21];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0x7ff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 11) & 0x7ff];
    recons[2] = ((half2 *)SMEM)[((tmp0 >> 22) & 0x3ff) | ((tmp1 << 10) & 0x400)];
    recons[3] = ((half2 *)SMEM)[(tmp1 >> 1) & 0x7ff];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 12) & 0x7ff];
    recons[5] = ((half2 *)SMEM)[((tmp1 >> 23) & 0x1ff) | ((tmp2 << 9) & 0x600)];
    recons[6] = ((half2 *)SMEM)[(tmp2 >> 2) & 0x7ff];
    recons[7] = ((half2 *)SMEM)[(tmp2 >> 13) & 0x7ff];
    recons[8] = ((half2 *)SMEM)[((tmp2 >> 24) & 0xff) | ((tmp3 << 8) & 0x700)];
    recons[9] = ((half2 *)SMEM)[(tmp3 >> 3) & 0x7ff];
    recons[10] = ((half2 *)SMEM)[(tmp3 >> 14) & 0x7ff];
    recons[11] = ((half2 *)SMEM)[((tmp3 >> 25) & 0x7f) | ((tmp4 << 7) & 0x780)];
    recons[12] = ((half2 *)SMEM)[(tmp4 >> 4) & 0x7ff];
    recons[13] = ((half2 *)SMEM)[(tmp4 >> 15) & 0x7ff];
    recons[14] = ((half2 *)SMEM)[((tmp4 >> 26) & 0x3f) | ((tmp5 << 6) & 0x7c0)];
    recons[15] = ((half2 *)SMEM)[(tmp5 >> 5) & 0x7ff];
    recons[16] = ((half2 *)SMEM)[(tmp5 >> 16) & 0x7ff];
    recons[17] = ((half2 *)SMEM)[((tmp5 >> 27) & 0x1f) | ((tmp6 << 5) & 0x7e0)];
    recons[18] = ((half2 *)SMEM)[(tmp6 >> 6) & 0x7ff];
    recons[19] = ((half2 *)SMEM)[(tmp6 >> 17) & 0x7ff];
    recons[20] = ((half2 *)SMEM)[((tmp6 >> 28) & 0xf) | ((tmp7 << 4) & 0x7f0)];
    recons[21] = ((half2 *)SMEM)[(tmp7 >> 7) & 0x7ff];
    recons[22] = ((half2 *)SMEM)[(tmp7 >> 18) & 0x7ff];
    recons[23] = ((half2 *)SMEM)[((tmp7 >> 29) & 0x7) | ((tmp8 << 3) & 0x7f8)];
    recons[24] = ((half2 *)SMEM)[(tmp8 >> 8) & 0x7ff];
    recons[25] = ((half2 *)SMEM)[(tmp8 >> 19) & 0x7ff];
    recons[26] = ((half2 *)SMEM)[((tmp8 >> 30) & 0x3) | ((tmp9 << 2) & 0x7fc)];
    recons[27] = ((half2 *)SMEM)[(tmp9 >> 9) & 0x7ff];
    recons[28] = ((half2 *)SMEM)[(tmp9 >> 20) & 0x7ff];
    recons[29] = ((half2 *)SMEM)[((tmp9 >> 31) & 0x1) | ((tmp10 << 1) & 0x7fe)];
    recons[30] = ((half2 *)SMEM)[(tmp10 >> 10) & 0x7ff];
    recons[31] = ((half2 *)SMEM)[(tmp10 >> 21) & 0x7ff];
    recons[32] = ((half2 *)SMEM)[(tmp11) & 0x7ff];
    recons[33] = ((half2 *)SMEM)[(tmp11 >> 11) & 0x7ff];
    recons[34] = ((half2 *)SMEM)[((tmp11 >> 22) & 0x3ff) | ((tmp12 << 10) & 0x400)];
    recons[35] = ((half2 *)SMEM)[(tmp12 >> 1) & 0x7ff];
    recons[36] = ((half2 *)SMEM)[(tmp12 >> 12) & 0x7ff];
    recons[37] = ((half2 *)SMEM)[((tmp12 >> 23) & 0x1ff) | ((tmp13 << 9) & 0x600)];
    recons[38] = ((half2 *)SMEM)[(tmp13 >> 2) & 0x7ff];
    recons[39] = ((half2 *)SMEM)[(tmp13 >> 13) & 0x7ff];
    recons[40] = ((half2 *)SMEM)[((tmp13 >> 24) & 0xff) | ((tmp14 << 8) & 0x700)];
    recons[41] = ((half2 *)SMEM)[(tmp14 >> 3) & 0x7ff];
    recons[42] = ((half2 *)SMEM)[(tmp14 >> 14) & 0x7ff];
    recons[43] = ((half2 *)SMEM)[((tmp14 >> 25) & 0x7f) | ((tmp15 << 7) & 0x780)];
    recons[44] = ((half2 *)SMEM)[(tmp15 >> 4) & 0x7ff];
    recons[45] = ((half2 *)SMEM)[(tmp15 >> 15) & 0x7ff];
    recons[46] = ((half2 *)SMEM)[((tmp15 >> 26) & 0x3f) | ((tmp16 << 6) & 0x7c0)];
    recons[47] = ((half2 *)SMEM)[(tmp16 >> 5) & 0x7ff];
    recons[48] = ((half2 *)SMEM)[(tmp16 >> 16) & 0x7ff];
    recons[49] = ((half2 *)SMEM)[((tmp16 >> 27) & 0x1f) | ((tmp17 << 5) & 0x7e0)];
    recons[50] = ((half2 *)SMEM)[(tmp17 >> 6) & 0x7ff];
    recons[51] = ((half2 *)SMEM)[(tmp17 >> 17) & 0x7ff];
    recons[52] = ((half2 *)SMEM)[((tmp17 >> 28) & 0xf) | ((tmp18 << 4) & 0x7f0)];
    recons[53] = ((half2 *)SMEM)[(tmp18 >> 7) & 0x7ff];
    recons[54] = ((half2 *)SMEM)[(tmp18 >> 18) & 0x7ff];
    recons[55] = ((half2 *)SMEM)[((tmp18 >> 29) & 0x7) | ((tmp19 << 3) & 0x7f8)];
    recons[56] = ((half2 *)SMEM)[(tmp19 >> 8) & 0x7ff];
    recons[57] = ((half2 *)SMEM)[(tmp19 >> 19) & 0x7ff];
    recons[58] = ((half2 *)SMEM)[((tmp19 >> 30) & 0x3) | ((tmp20 << 2) & 0x7fc)];
    recons[59] = ((half2 *)SMEM)[(tmp20 >> 9) & 0x7ff];
    recons[60] = ((half2 *)SMEM)[(tmp20 >> 20) & 0x7ff];
    recons[61] = ((half2 *)SMEM)[((tmp20 >> 31) & 0x1) | ((tmp21 << 1) & 0x7fe)];
    recons[62] = ((half2 *)SMEM)[(tmp21 >> 10) & 0x7ff];
    recons[63] = ((half2 *)SMEM)[(tmp21 >> 21) & 0x7ff];
}

// nbits: 12	vec_sz: 2	code_n: 6	avg_bits:  6.000	SMEM_sz: 8KB	recons_n: 16
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<12, 2, 6, uint32_t>(const uint32_t code[6], half2 recons[16], half SMEM[8192]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xfff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 12) & 0xfff];
    recons[2] = ((half2 *)SMEM)[((tmp0 >> 24) & 0xff) | ((tmp1 << 8) & 0xf00)];
    recons[3] = ((half2 *)SMEM)[(tmp1 >> 4) & 0xfff];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xfff];
    recons[5] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0xff0)];
    recons[6] = ((half2 *)SMEM)[(tmp2 >> 8) & 0xfff];
    recons[7] = ((half2 *)SMEM)[(tmp2 >> 20) & 0xfff];
    recons[8] = ((half2 *)SMEM)[(tmp3) & 0xfff];
    recons[9] = ((half2 *)SMEM)[(tmp3 >> 12) & 0xfff];
    recons[10] = ((half2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0xf00)];
    recons[11] = ((half2 *)SMEM)[(tmp4 >> 4) & 0xfff];
    recons[12] = ((half2 *)SMEM)[(tmp4 >> 16) & 0xfff];
    recons[13] = ((half2 *)SMEM)[((tmp4 >> 28) & 0xf) | ((tmp5 << 4) & 0xff0)];
    recons[14] = ((half2 *)SMEM)[(tmp5 >> 8) & 0xfff];
    recons[15] = ((half2 *)SMEM)[(tmp5 >> 20) & 0xfff];
}

// nbits: 12	vec_sz: 2	code_n: 12	avg_bits:  6.000	SMEM_sz: 8KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<12, 2, 12, uint32_t>(const uint32_t code[12], half2 recons[32], half SMEM[8192]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xfff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 12) & 0xfff];
    recons[2] = ((half2 *)SMEM)[((tmp0 >> 24) & 0xff) | ((tmp1 << 8) & 0xf00)];
    recons[3] = ((half2 *)SMEM)[(tmp1 >> 4) & 0xfff];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xfff];
    recons[5] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0xff0)];
    recons[6] = ((half2 *)SMEM)[(tmp2 >> 8) & 0xfff];
    recons[7] = ((half2 *)SMEM)[(tmp2 >> 20) & 0xfff];
    recons[8] = ((half2 *)SMEM)[(tmp3) & 0xfff];
    recons[9] = ((half2 *)SMEM)[(tmp3 >> 12) & 0xfff];
    recons[10] = ((half2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0xf00)];
    recons[11] = ((half2 *)SMEM)[(tmp4 >> 4) & 0xfff];
    recons[12] = ((half2 *)SMEM)[(tmp4 >> 16) & 0xfff];
    recons[13] = ((half2 *)SMEM)[((tmp4 >> 28) & 0xf) | ((tmp5 << 4) & 0xff0)];
    recons[14] = ((half2 *)SMEM)[(tmp5 >> 8) & 0xfff];
    recons[15] = ((half2 *)SMEM)[(tmp5 >> 20) & 0xfff];
    recons[16] = ((half2 *)SMEM)[(tmp6) & 0xfff];
    recons[17] = ((half2 *)SMEM)[(tmp6 >> 12) & 0xfff];
    recons[18] = ((half2 *)SMEM)[((tmp6 >> 24) & 0xff) | ((tmp7 << 8) & 0xf00)];
    recons[19] = ((half2 *)SMEM)[(tmp7 >> 4) & 0xfff];
    recons[20] = ((half2 *)SMEM)[(tmp7 >> 16) & 0xfff];
    recons[21] = ((half2 *)SMEM)[((tmp7 >> 28) & 0xf) | ((tmp8 << 4) & 0xff0)];
    recons[22] = ((half2 *)SMEM)[(tmp8 >> 8) & 0xfff];
    recons[23] = ((half2 *)SMEM)[(tmp8 >> 20) & 0xfff];
    recons[24] = ((half2 *)SMEM)[(tmp9) & 0xfff];
    recons[25] = ((half2 *)SMEM)[(tmp9 >> 12) & 0xfff];
    recons[26] = ((half2 *)SMEM)[((tmp9 >> 24) & 0xff) | ((tmp10 << 8) & 0xf00)];
    recons[27] = ((half2 *)SMEM)[(tmp10 >> 4) & 0xfff];
    recons[28] = ((half2 *)SMEM)[(tmp10 >> 16) & 0xfff];
    recons[29] = ((half2 *)SMEM)[((tmp10 >> 28) & 0xf) | ((tmp11 << 4) & 0xff0)];
    recons[30] = ((half2 *)SMEM)[(tmp11 >> 8) & 0xfff];
    recons[31] = ((half2 *)SMEM)[(tmp11 >> 20) & 0xfff];
}

// nbits: 12	vec_sz: 2	code_n: 24	avg_bits:  6.000	SMEM_sz: 8KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<12, 2, 24, uint32_t>(const uint32_t code[24], half2 recons[64], half SMEM[8192]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    uint32_t tmp12 = code[12];
    uint32_t tmp13 = code[13];
    uint32_t tmp14 = code[14];
    uint32_t tmp15 = code[15];
    uint32_t tmp16 = code[16];
    uint32_t tmp17 = code[17];
    uint32_t tmp18 = code[18];
    uint32_t tmp19 = code[19];
    uint32_t tmp20 = code[20];
    uint32_t tmp21 = code[21];
    uint32_t tmp22 = code[22];
    uint32_t tmp23 = code[23];
    recons[0] = ((half2 *)SMEM)[(tmp0) & 0xfff];
    recons[1] = ((half2 *)SMEM)[(tmp0 >> 12) & 0xfff];
    recons[2] = ((half2 *)SMEM)[((tmp0 >> 24) & 0xff) | ((tmp1 << 8) & 0xf00)];
    recons[3] = ((half2 *)SMEM)[(tmp1 >> 4) & 0xfff];
    recons[4] = ((half2 *)SMEM)[(tmp1 >> 16) & 0xfff];
    recons[5] = ((half2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0xff0)];
    recons[6] = ((half2 *)SMEM)[(tmp2 >> 8) & 0xfff];
    recons[7] = ((half2 *)SMEM)[(tmp2 >> 20) & 0xfff];
    recons[8] = ((half2 *)SMEM)[(tmp3) & 0xfff];
    recons[9] = ((half2 *)SMEM)[(tmp3 >> 12) & 0xfff];
    recons[10] = ((half2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0xf00)];
    recons[11] = ((half2 *)SMEM)[(tmp4 >> 4) & 0xfff];
    recons[12] = ((half2 *)SMEM)[(tmp4 >> 16) & 0xfff];
    recons[13] = ((half2 *)SMEM)[((tmp4 >> 28) & 0xf) | ((tmp5 << 4) & 0xff0)];
    recons[14] = ((half2 *)SMEM)[(tmp5 >> 8) & 0xfff];
    recons[15] = ((half2 *)SMEM)[(tmp5 >> 20) & 0xfff];
    recons[16] = ((half2 *)SMEM)[(tmp6) & 0xfff];
    recons[17] = ((half2 *)SMEM)[(tmp6 >> 12) & 0xfff];
    recons[18] = ((half2 *)SMEM)[((tmp6 >> 24) & 0xff) | ((tmp7 << 8) & 0xf00)];
    recons[19] = ((half2 *)SMEM)[(tmp7 >> 4) & 0xfff];
    recons[20] = ((half2 *)SMEM)[(tmp7 >> 16) & 0xfff];
    recons[21] = ((half2 *)SMEM)[((tmp7 >> 28) & 0xf) | ((tmp8 << 4) & 0xff0)];
    recons[22] = ((half2 *)SMEM)[(tmp8 >> 8) & 0xfff];
    recons[23] = ((half2 *)SMEM)[(tmp8 >> 20) & 0xfff];
    recons[24] = ((half2 *)SMEM)[(tmp9) & 0xfff];
    recons[25] = ((half2 *)SMEM)[(tmp9 >> 12) & 0xfff];
    recons[26] = ((half2 *)SMEM)[((tmp9 >> 24) & 0xff) | ((tmp10 << 8) & 0xf00)];
    recons[27] = ((half2 *)SMEM)[(tmp10 >> 4) & 0xfff];
    recons[28] = ((half2 *)SMEM)[(tmp10 >> 16) & 0xfff];
    recons[29] = ((half2 *)SMEM)[((tmp10 >> 28) & 0xf) | ((tmp11 << 4) & 0xff0)];
    recons[30] = ((half2 *)SMEM)[(tmp11 >> 8) & 0xfff];
    recons[31] = ((half2 *)SMEM)[(tmp11 >> 20) & 0xfff];
    recons[32] = ((half2 *)SMEM)[(tmp12) & 0xfff];
    recons[33] = ((half2 *)SMEM)[(tmp12 >> 12) & 0xfff];
    recons[34] = ((half2 *)SMEM)[((tmp12 >> 24) & 0xff) | ((tmp13 << 8) & 0xf00)];
    recons[35] = ((half2 *)SMEM)[(tmp13 >> 4) & 0xfff];
    recons[36] = ((half2 *)SMEM)[(tmp13 >> 16) & 0xfff];
    recons[37] = ((half2 *)SMEM)[((tmp13 >> 28) & 0xf) | ((tmp14 << 4) & 0xff0)];
    recons[38] = ((half2 *)SMEM)[(tmp14 >> 8) & 0xfff];
    recons[39] = ((half2 *)SMEM)[(tmp14 >> 20) & 0xfff];
    recons[40] = ((half2 *)SMEM)[(tmp15) & 0xfff];
    recons[41] = ((half2 *)SMEM)[(tmp15 >> 12) & 0xfff];
    recons[42] = ((half2 *)SMEM)[((tmp15 >> 24) & 0xff) | ((tmp16 << 8) & 0xf00)];
    recons[43] = ((half2 *)SMEM)[(tmp16 >> 4) & 0xfff];
    recons[44] = ((half2 *)SMEM)[(tmp16 >> 16) & 0xfff];
    recons[45] = ((half2 *)SMEM)[((tmp16 >> 28) & 0xf) | ((tmp17 << 4) & 0xff0)];
    recons[46] = ((half2 *)SMEM)[(tmp17 >> 8) & 0xfff];
    recons[47] = ((half2 *)SMEM)[(tmp17 >> 20) & 0xfff];
    recons[48] = ((half2 *)SMEM)[(tmp18) & 0xfff];
    recons[49] = ((half2 *)SMEM)[(tmp18 >> 12) & 0xfff];
    recons[50] = ((half2 *)SMEM)[((tmp18 >> 24) & 0xff) | ((tmp19 << 8) & 0xf00)];
    recons[51] = ((half2 *)SMEM)[(tmp19 >> 4) & 0xfff];
    recons[52] = ((half2 *)SMEM)[(tmp19 >> 16) & 0xfff];
    recons[53] = ((half2 *)SMEM)[((tmp19 >> 28) & 0xf) | ((tmp20 << 4) & 0xff0)];
    recons[54] = ((half2 *)SMEM)[(tmp20 >> 8) & 0xfff];
    recons[55] = ((half2 *)SMEM)[(tmp20 >> 20) & 0xfff];
    recons[56] = ((half2 *)SMEM)[(tmp21) & 0xfff];
    recons[57] = ((half2 *)SMEM)[(tmp21 >> 12) & 0xfff];
    recons[58] = ((half2 *)SMEM)[((tmp21 >> 24) & 0xff) | ((tmp22 << 8) & 0xf00)];
    recons[59] = ((half2 *)SMEM)[(tmp22 >> 4) & 0xfff];
    recons[60] = ((half2 *)SMEM)[(tmp22 >> 16) & 0xfff];
    recons[61] = ((half2 *)SMEM)[((tmp22 >> 28) & 0xf) | ((tmp23 << 4) & 0xff0)];
    recons[62] = ((half2 *)SMEM)[(tmp23 >> 8) & 0xfff];
    recons[63] = ((half2 *)SMEM)[(tmp23 >> 20) & 0xfff];
}

// nbits: 6	vec_sz: 4	code_n: 3	avg_bits:  1.500	SMEM_sz: 0KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<6, 4, 3, uint32_t>(const uint32_t code[3], half2 recons[32], half SMEM[256]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0x3f];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 6) & 0x3f];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 12) & 0x3f];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp0 >> 18) & 0x3f];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp0 >> 24) & 0x3f];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3c)];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp1 >> 4) & 0x3f];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp1 >> 10) & 0x3f];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp1 >> 16) & 0x3f];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[(tmp1 >> 22) & 0x3f];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x30)];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp2 >> 2) & 0x3f];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp2 >> 8) & 0x3f];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[(tmp2 >> 14) & 0x3f];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp2 >> 20) & 0x3f];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp2 >> 26) & 0x3f];
}

// nbits: 6	vec_sz: 4	code_n: 6	avg_bits:  1.500	SMEM_sz: 0KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<6, 4, 6, uint32_t>(const uint32_t code[6], half2 recons[64], half SMEM[256]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0x3f];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 6) & 0x3f];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 12) & 0x3f];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp0 >> 18) & 0x3f];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp0 >> 24) & 0x3f];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3c)];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp1 >> 4) & 0x3f];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp1 >> 10) & 0x3f];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp1 >> 16) & 0x3f];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[(tmp1 >> 22) & 0x3f];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x30)];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp2 >> 2) & 0x3f];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp2 >> 8) & 0x3f];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[(tmp2 >> 14) & 0x3f];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp2 >> 20) & 0x3f];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp2 >> 26) & 0x3f];
    ((float2 *)recons)[16] = ((float2 *)SMEM)[(tmp3) & 0x3f];
    ((float2 *)recons)[17] = ((float2 *)SMEM)[(tmp3 >> 6) & 0x3f];
    ((float2 *)recons)[18] = ((float2 *)SMEM)[(tmp3 >> 12) & 0x3f];
    ((float2 *)recons)[19] = ((float2 *)SMEM)[(tmp3 >> 18) & 0x3f];
    ((float2 *)recons)[20] = ((float2 *)SMEM)[(tmp3 >> 24) & 0x3f];
    ((float2 *)recons)[21] = ((float2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x3c)];
    ((float2 *)recons)[22] = ((float2 *)SMEM)[(tmp4 >> 4) & 0x3f];
    ((float2 *)recons)[23] = ((float2 *)SMEM)[(tmp4 >> 10) & 0x3f];
    ((float2 *)recons)[24] = ((float2 *)SMEM)[(tmp4 >> 16) & 0x3f];
    ((float2 *)recons)[25] = ((float2 *)SMEM)[(tmp4 >> 22) & 0x3f];
    ((float2 *)recons)[26] = ((float2 *)SMEM)[((tmp4 >> 28) & 0xf) | ((tmp5 << 4) & 0x30)];
    ((float2 *)recons)[27] = ((float2 *)SMEM)[(tmp5 >> 2) & 0x3f];
    ((float2 *)recons)[28] = ((float2 *)SMEM)[(tmp5 >> 8) & 0x3f];
    ((float2 *)recons)[29] = ((float2 *)SMEM)[(tmp5 >> 14) & 0x3f];
    ((float2 *)recons)[30] = ((float2 *)SMEM)[(tmp5 >> 20) & 0x3f];
    ((float2 *)recons)[31] = ((float2 *)SMEM)[(tmp5 >> 26) & 0x3f];
}

// nbits: 7	vec_sz: 4	code_n: 7	avg_bits:  1.750	SMEM_sz: 0KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<7, 4, 7, uint32_t>(const uint32_t code[7], half2 recons[64], half SMEM[512]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0x7f];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 7) & 0x7f];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 14) & 0x7f];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp0 >> 21) & 0x7f];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[((tmp0 >> 28) & 0xf) | ((tmp1 << 4) & 0x70)];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[(tmp1 >> 3) & 0x7f];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp1 >> 10) & 0x7f];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp1 >> 17) & 0x7f];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp1 >> 24) & 0x7f];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[((tmp1 >> 31) & 0x1) | ((tmp2 << 1) & 0x7e)];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[(tmp2 >> 6) & 0x7f];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp2 >> 13) & 0x7f];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp2 >> 20) & 0x7f];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[((tmp2 >> 27) & 0x1f) | ((tmp3 << 5) & 0x60)];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp3 >> 2) & 0x7f];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp3 >> 9) & 0x7f];
    ((float2 *)recons)[16] = ((float2 *)SMEM)[(tmp3 >> 16) & 0x7f];
    ((float2 *)recons)[17] = ((float2 *)SMEM)[(tmp3 >> 23) & 0x7f];
    ((float2 *)recons)[18] = ((float2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x7c)];
    ((float2 *)recons)[19] = ((float2 *)SMEM)[(tmp4 >> 5) & 0x7f];
    ((float2 *)recons)[20] = ((float2 *)SMEM)[(tmp4 >> 12) & 0x7f];
    ((float2 *)recons)[21] = ((float2 *)SMEM)[(tmp4 >> 19) & 0x7f];
    ((float2 *)recons)[22] = ((float2 *)SMEM)[((tmp4 >> 26) & 0x3f) | ((tmp5 << 6) & 0x40)];
    ((float2 *)recons)[23] = ((float2 *)SMEM)[(tmp5 >> 1) & 0x7f];
    ((float2 *)recons)[24] = ((float2 *)SMEM)[(tmp5 >> 8) & 0x7f];
    ((float2 *)recons)[25] = ((float2 *)SMEM)[(tmp5 >> 15) & 0x7f];
    ((float2 *)recons)[26] = ((float2 *)SMEM)[(tmp5 >> 22) & 0x7f];
    ((float2 *)recons)[27] = ((float2 *)SMEM)[((tmp5 >> 29) & 0x7) | ((tmp6 << 3) & 0x78)];
    ((float2 *)recons)[28] = ((float2 *)SMEM)[(tmp6 >> 4) & 0x7f];
    ((float2 *)recons)[29] = ((float2 *)SMEM)[(tmp6 >> 11) & 0x7f];
    ((float2 *)recons)[30] = ((float2 *)SMEM)[(tmp6 >> 18) & 0x7f];
    ((float2 *)recons)[31] = ((float2 *)SMEM)[(tmp6 >> 25) & 0x7f];
}

// nbits: 8	vec_sz: 4	code_n: 2	avg_bits:  2.000	SMEM_sz: 1KB	recons_n: 16
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<8, 4, 2, uint32_t>(const uint32_t code[2], half2 recons[16], half SMEM[1024]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0xff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 8) & 0xff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 16) & 0xff];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp0 >> 24) & 0xff];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1) & 0xff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[(tmp1 >> 8) & 0xff];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp1 >> 16) & 0xff];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp1 >> 24) & 0xff];
}

// nbits: 8	vec_sz: 4	code_n: 4	avg_bits:  2.000	SMEM_sz: 1KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<8, 4, 4, uint32_t>(const uint32_t code[4], half2 recons[32], half SMEM[1024]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0xff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 8) & 0xff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 16) & 0xff];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp0 >> 24) & 0xff];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1) & 0xff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[(tmp1 >> 8) & 0xff];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp1 >> 16) & 0xff];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp1 >> 24) & 0xff];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp2) & 0xff];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[(tmp2 >> 8) & 0xff];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[(tmp2 >> 16) & 0xff];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp2 >> 24) & 0xff];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp3) & 0xff];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[(tmp3 >> 8) & 0xff];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp3 >> 16) & 0xff];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp3 >> 24) & 0xff];
}

// nbits: 8	vec_sz: 4	code_n: 8	avg_bits:  2.000	SMEM_sz: 1KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<8, 4, 8, uint32_t>(const uint32_t code[8], half2 recons[64], half SMEM[1024]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0xff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 8) & 0xff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 16) & 0xff];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp0 >> 24) & 0xff];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1) & 0xff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[(tmp1 >> 8) & 0xff];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp1 >> 16) & 0xff];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp1 >> 24) & 0xff];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp2) & 0xff];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[(tmp2 >> 8) & 0xff];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[(tmp2 >> 16) & 0xff];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp2 >> 24) & 0xff];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp3) & 0xff];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[(tmp3 >> 8) & 0xff];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp3 >> 16) & 0xff];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp3 >> 24) & 0xff];
    ((float2 *)recons)[16] = ((float2 *)SMEM)[(tmp4) & 0xff];
    ((float2 *)recons)[17] = ((float2 *)SMEM)[(tmp4 >> 8) & 0xff];
    ((float2 *)recons)[18] = ((float2 *)SMEM)[(tmp4 >> 16) & 0xff];
    ((float2 *)recons)[19] = ((float2 *)SMEM)[(tmp4 >> 24) & 0xff];
    ((float2 *)recons)[20] = ((float2 *)SMEM)[(tmp5) & 0xff];
    ((float2 *)recons)[21] = ((float2 *)SMEM)[(tmp5 >> 8) & 0xff];
    ((float2 *)recons)[22] = ((float2 *)SMEM)[(tmp5 >> 16) & 0xff];
    ((float2 *)recons)[23] = ((float2 *)SMEM)[(tmp5 >> 24) & 0xff];
    ((float2 *)recons)[24] = ((float2 *)SMEM)[(tmp6) & 0xff];
    ((float2 *)recons)[25] = ((float2 *)SMEM)[(tmp6 >> 8) & 0xff];
    ((float2 *)recons)[26] = ((float2 *)SMEM)[(tmp6 >> 16) & 0xff];
    ((float2 *)recons)[27] = ((float2 *)SMEM)[(tmp6 >> 24) & 0xff];
    ((float2 *)recons)[28] = ((float2 *)SMEM)[(tmp7) & 0xff];
    ((float2 *)recons)[29] = ((float2 *)SMEM)[(tmp7 >> 8) & 0xff];
    ((float2 *)recons)[30] = ((float2 *)SMEM)[(tmp7 >> 16) & 0xff];
    ((float2 *)recons)[31] = ((float2 *)SMEM)[(tmp7 >> 24) & 0xff];
}

// nbits: 9	vec_sz: 4	code_n: 9	avg_bits:  2.250	SMEM_sz: 2KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<9, 4, 9, uint32_t>(const uint32_t code[9], half2 recons[64], half SMEM[2048]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0x1ff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 9) & 0x1ff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 18) & 0x1ff];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[((tmp0 >> 27) & 0x1f) | ((tmp1 << 5) & 0x1e0)];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1 >> 4) & 0x1ff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[(tmp1 >> 13) & 0x1ff];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp1 >> 22) & 0x1ff];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[((tmp1 >> 31) & 0x1) | ((tmp2 << 1) & 0x1fe)];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp2 >> 8) & 0x1ff];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[(tmp2 >> 17) & 0x1ff];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[((tmp2 >> 26) & 0x3f) | ((tmp3 << 6) & 0x1c0)];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp3 >> 3) & 0x1ff];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp3 >> 12) & 0x1ff];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[(tmp3 >> 21) & 0x1ff];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[((tmp3 >> 30) & 0x3) | ((tmp4 << 2) & 0x1fc)];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp4 >> 7) & 0x1ff];
    ((float2 *)recons)[16] = ((float2 *)SMEM)[(tmp4 >> 16) & 0x1ff];
    ((float2 *)recons)[17] = ((float2 *)SMEM)[((tmp4 >> 25) & 0x7f) | ((tmp5 << 7) & 0x180)];
    ((float2 *)recons)[18] = ((float2 *)SMEM)[(tmp5 >> 2) & 0x1ff];
    ((float2 *)recons)[19] = ((float2 *)SMEM)[(tmp5 >> 11) & 0x1ff];
    ((float2 *)recons)[20] = ((float2 *)SMEM)[(tmp5 >> 20) & 0x1ff];
    ((float2 *)recons)[21] = ((float2 *)SMEM)[((tmp5 >> 29) & 0x7) | ((tmp6 << 3) & 0x1f8)];
    ((float2 *)recons)[22] = ((float2 *)SMEM)[(tmp6 >> 6) & 0x1ff];
    ((float2 *)recons)[23] = ((float2 *)SMEM)[(tmp6 >> 15) & 0x1ff];
    ((float2 *)recons)[24] = ((float2 *)SMEM)[((tmp6 >> 24) & 0xff) | ((tmp7 << 8) & 0x100)];
    ((float2 *)recons)[25] = ((float2 *)SMEM)[(tmp7 >> 1) & 0x1ff];
    ((float2 *)recons)[26] = ((float2 *)SMEM)[(tmp7 >> 10) & 0x1ff];
    ((float2 *)recons)[27] = ((float2 *)SMEM)[(tmp7 >> 19) & 0x1ff];
    ((float2 *)recons)[28] = ((float2 *)SMEM)[((tmp7 >> 28) & 0xf) | ((tmp8 << 4) & 0x1f0)];
    ((float2 *)recons)[29] = ((float2 *)SMEM)[(tmp8 >> 5) & 0x1ff];
    ((float2 *)recons)[30] = ((float2 *)SMEM)[(tmp8 >> 14) & 0x1ff];
    ((float2 *)recons)[31] = ((float2 *)SMEM)[(tmp8 >> 23) & 0x1ff];
}

// nbits: 10	vec_sz: 4	code_n: 5	avg_bits:  2.500	SMEM_sz: 4KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<10, 4, 5, uint32_t>(const uint32_t code[5], half2 recons[32], half SMEM[4096]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0x3ff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 10) & 0x3ff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 20) & 0x3ff];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3fc)];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1 >> 8) & 0x3ff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[(tmp1 >> 18) & 0x3ff];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x3f0)];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp2 >> 6) & 0x3ff];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp2 >> 16) & 0x3ff];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[((tmp2 >> 26) & 0x3f) | ((tmp3 << 6) & 0x3c0)];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[(tmp3 >> 4) & 0x3ff];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp3 >> 14) & 0x3ff];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0x300)];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[(tmp4 >> 2) & 0x3ff];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp4 >> 12) & 0x3ff];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp4 >> 22) & 0x3ff];
}

// nbits: 10	vec_sz: 4	code_n: 10	avg_bits:  2.500	SMEM_sz: 4KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<10, 4, 10, uint32_t>(const uint32_t code[10], half2 recons[64], half SMEM[4096]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0x3ff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 10) & 0x3ff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[(tmp0 >> 20) & 0x3ff];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3fc)];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1 >> 8) & 0x3ff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[(tmp1 >> 18) & 0x3ff];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0x3f0)];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp2 >> 6) & 0x3ff];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp2 >> 16) & 0x3ff];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[((tmp2 >> 26) & 0x3f) | ((tmp3 << 6) & 0x3c0)];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[(tmp3 >> 4) & 0x3ff];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp3 >> 14) & 0x3ff];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0x300)];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[(tmp4 >> 2) & 0x3ff];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp4 >> 12) & 0x3ff];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp4 >> 22) & 0x3ff];
    ((float2 *)recons)[16] = ((float2 *)SMEM)[(tmp5) & 0x3ff];
    ((float2 *)recons)[17] = ((float2 *)SMEM)[(tmp5 >> 10) & 0x3ff];
    ((float2 *)recons)[18] = ((float2 *)SMEM)[(tmp5 >> 20) & 0x3ff];
    ((float2 *)recons)[19] = ((float2 *)SMEM)[((tmp5 >> 30) & 0x3) | ((tmp6 << 2) & 0x3fc)];
    ((float2 *)recons)[20] = ((float2 *)SMEM)[(tmp6 >> 8) & 0x3ff];
    ((float2 *)recons)[21] = ((float2 *)SMEM)[(tmp6 >> 18) & 0x3ff];
    ((float2 *)recons)[22] = ((float2 *)SMEM)[((tmp6 >> 28) & 0xf) | ((tmp7 << 4) & 0x3f0)];
    ((float2 *)recons)[23] = ((float2 *)SMEM)[(tmp7 >> 6) & 0x3ff];
    ((float2 *)recons)[24] = ((float2 *)SMEM)[(tmp7 >> 16) & 0x3ff];
    ((float2 *)recons)[25] = ((float2 *)SMEM)[((tmp7 >> 26) & 0x3f) | ((tmp8 << 6) & 0x3c0)];
    ((float2 *)recons)[26] = ((float2 *)SMEM)[(tmp8 >> 4) & 0x3ff];
    ((float2 *)recons)[27] = ((float2 *)SMEM)[(tmp8 >> 14) & 0x3ff];
    ((float2 *)recons)[28] = ((float2 *)SMEM)[((tmp8 >> 24) & 0xff) | ((tmp9 << 8) & 0x300)];
    ((float2 *)recons)[29] = ((float2 *)SMEM)[(tmp9 >> 2) & 0x3ff];
    ((float2 *)recons)[30] = ((float2 *)SMEM)[(tmp9 >> 12) & 0x3ff];
    ((float2 *)recons)[31] = ((float2 *)SMEM)[(tmp9 >> 22) & 0x3ff];
}

// nbits: 11	vec_sz: 4	code_n: 11	avg_bits:  2.750	SMEM_sz: 8KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<11, 4, 11, uint32_t>(const uint32_t code[11], half2 recons[64], half SMEM[8192]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0x7ff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 11) & 0x7ff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[((tmp0 >> 22) & 0x3ff) | ((tmp1 << 10) & 0x400)];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp1 >> 1) & 0x7ff];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1 >> 12) & 0x7ff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[((tmp1 >> 23) & 0x1ff) | ((tmp2 << 9) & 0x600)];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp2 >> 2) & 0x7ff];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp2 >> 13) & 0x7ff];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[((tmp2 >> 24) & 0xff) | ((tmp3 << 8) & 0x700)];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[(tmp3 >> 3) & 0x7ff];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[(tmp3 >> 14) & 0x7ff];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[((tmp3 >> 25) & 0x7f) | ((tmp4 << 7) & 0x780)];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp4 >> 4) & 0x7ff];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[(tmp4 >> 15) & 0x7ff];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[((tmp4 >> 26) & 0x3f) | ((tmp5 << 6) & 0x7c0)];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp5 >> 5) & 0x7ff];
    ((float2 *)recons)[16] = ((float2 *)SMEM)[(tmp5 >> 16) & 0x7ff];
    ((float2 *)recons)[17] = ((float2 *)SMEM)[((tmp5 >> 27) & 0x1f) | ((tmp6 << 5) & 0x7e0)];
    ((float2 *)recons)[18] = ((float2 *)SMEM)[(tmp6 >> 6) & 0x7ff];
    ((float2 *)recons)[19] = ((float2 *)SMEM)[(tmp6 >> 17) & 0x7ff];
    ((float2 *)recons)[20] = ((float2 *)SMEM)[((tmp6 >> 28) & 0xf) | ((tmp7 << 4) & 0x7f0)];
    ((float2 *)recons)[21] = ((float2 *)SMEM)[(tmp7 >> 7) & 0x7ff];
    ((float2 *)recons)[22] = ((float2 *)SMEM)[(tmp7 >> 18) & 0x7ff];
    ((float2 *)recons)[23] = ((float2 *)SMEM)[((tmp7 >> 29) & 0x7) | ((tmp8 << 3) & 0x7f8)];
    ((float2 *)recons)[24] = ((float2 *)SMEM)[(tmp8 >> 8) & 0x7ff];
    ((float2 *)recons)[25] = ((float2 *)SMEM)[(tmp8 >> 19) & 0x7ff];
    ((float2 *)recons)[26] = ((float2 *)SMEM)[((tmp8 >> 30) & 0x3) | ((tmp9 << 2) & 0x7fc)];
    ((float2 *)recons)[27] = ((float2 *)SMEM)[(tmp9 >> 9) & 0x7ff];
    ((float2 *)recons)[28] = ((float2 *)SMEM)[(tmp9 >> 20) & 0x7ff];
    ((float2 *)recons)[29] = ((float2 *)SMEM)[((tmp9 >> 31) & 0x1) | ((tmp10 << 1) & 0x7fe)];
    ((float2 *)recons)[30] = ((float2 *)SMEM)[(tmp10 >> 10) & 0x7ff];
    ((float2 *)recons)[31] = ((float2 *)SMEM)[(tmp10 >> 21) & 0x7ff];
}

// nbits: 12	vec_sz: 4	code_n: 3	avg_bits:  3.000	SMEM_sz: 16KB	recons_n: 16
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<12, 4, 3, uint32_t>(const uint32_t code[3], half2 recons[16], half SMEM[16384]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0xfff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 12) & 0xfff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[((tmp0 >> 24) & 0xff) | ((tmp1 << 8) & 0xf00)];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp1 >> 4) & 0xfff];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1 >> 16) & 0xfff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0xff0)];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp2 >> 8) & 0xfff];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp2 >> 20) & 0xfff];
}

// nbits: 12	vec_sz: 4	code_n: 6	avg_bits:  3.000	SMEM_sz: 16KB	recons_n: 32
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<12, 4, 6, uint32_t>(const uint32_t code[6], half2 recons[32], half SMEM[16384]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0xfff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 12) & 0xfff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[((tmp0 >> 24) & 0xff) | ((tmp1 << 8) & 0xf00)];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp1 >> 4) & 0xfff];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1 >> 16) & 0xfff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0xff0)];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp2 >> 8) & 0xfff];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp2 >> 20) & 0xfff];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp3) & 0xfff];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[(tmp3 >> 12) & 0xfff];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0xf00)];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp4 >> 4) & 0xfff];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp4 >> 16) & 0xfff];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[((tmp4 >> 28) & 0xf) | ((tmp5 << 4) & 0xff0)];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp5 >> 8) & 0xfff];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp5 >> 20) & 0xfff];
}

// nbits: 12	vec_sz: 4	code_n: 12	avg_bits:  3.000	SMEM_sz: 16KB	recons_n: 64
template <>
__device__ __forceinline__ void vq_pack_dequant_routine<12, 4, 12, uint32_t>(const uint32_t code[12], half2 recons[64], half SMEM[16384]){
    uint32_t tmp0 = code[0];
    uint32_t tmp1 = code[1];
    uint32_t tmp2 = code[2];
    uint32_t tmp3 = code[3];
    uint32_t tmp4 = code[4];
    uint32_t tmp5 = code[5];
    uint32_t tmp6 = code[6];
    uint32_t tmp7 = code[7];
    uint32_t tmp8 = code[8];
    uint32_t tmp9 = code[9];
    uint32_t tmp10 = code[10];
    uint32_t tmp11 = code[11];
    ((float2 *)recons)[0] = ((float2 *)SMEM)[(tmp0) & 0xfff];
    ((float2 *)recons)[1] = ((float2 *)SMEM)[(tmp0 >> 12) & 0xfff];
    ((float2 *)recons)[2] = ((float2 *)SMEM)[((tmp0 >> 24) & 0xff) | ((tmp1 << 8) & 0xf00)];
    ((float2 *)recons)[3] = ((float2 *)SMEM)[(tmp1 >> 4) & 0xfff];
    ((float2 *)recons)[4] = ((float2 *)SMEM)[(tmp1 >> 16) & 0xfff];
    ((float2 *)recons)[5] = ((float2 *)SMEM)[((tmp1 >> 28) & 0xf) | ((tmp2 << 4) & 0xff0)];
    ((float2 *)recons)[6] = ((float2 *)SMEM)[(tmp2 >> 8) & 0xfff];
    ((float2 *)recons)[7] = ((float2 *)SMEM)[(tmp2 >> 20) & 0xfff];
    ((float2 *)recons)[8] = ((float2 *)SMEM)[(tmp3) & 0xfff];
    ((float2 *)recons)[9] = ((float2 *)SMEM)[(tmp3 >> 12) & 0xfff];
    ((float2 *)recons)[10] = ((float2 *)SMEM)[((tmp3 >> 24) & 0xff) | ((tmp4 << 8) & 0xf00)];
    ((float2 *)recons)[11] = ((float2 *)SMEM)[(tmp4 >> 4) & 0xfff];
    ((float2 *)recons)[12] = ((float2 *)SMEM)[(tmp4 >> 16) & 0xfff];
    ((float2 *)recons)[13] = ((float2 *)SMEM)[((tmp4 >> 28) & 0xf) | ((tmp5 << 4) & 0xff0)];
    ((float2 *)recons)[14] = ((float2 *)SMEM)[(tmp5 >> 8) & 0xfff];
    ((float2 *)recons)[15] = ((float2 *)SMEM)[(tmp5 >> 20) & 0xfff];
    ((float2 *)recons)[16] = ((float2 *)SMEM)[(tmp6) & 0xfff];
    ((float2 *)recons)[17] = ((float2 *)SMEM)[(tmp6 >> 12) & 0xfff];
    ((float2 *)recons)[18] = ((float2 *)SMEM)[((tmp6 >> 24) & 0xff) | ((tmp7 << 8) & 0xf00)];
    ((float2 *)recons)[19] = ((float2 *)SMEM)[(tmp7 >> 4) & 0xfff];
    ((float2 *)recons)[20] = ((float2 *)SMEM)[(tmp7 >> 16) & 0xfff];
    ((float2 *)recons)[21] = ((float2 *)SMEM)[((tmp7 >> 28) & 0xf) | ((tmp8 << 4) & 0xff0)];
    ((float2 *)recons)[22] = ((float2 *)SMEM)[(tmp8 >> 8) & 0xfff];
    ((float2 *)recons)[23] = ((float2 *)SMEM)[(tmp8 >> 20) & 0xfff];
    ((float2 *)recons)[24] = ((float2 *)SMEM)[(tmp9) & 0xfff];
    ((float2 *)recons)[25] = ((float2 *)SMEM)[(tmp9 >> 12) & 0xfff];
    ((float2 *)recons)[26] = ((float2 *)SMEM)[((tmp9 >> 24) & 0xff) | ((tmp10 << 8) & 0xf00)];
    ((float2 *)recons)[27] = ((float2 *)SMEM)[(tmp10 >> 4) & 0xfff];
    ((float2 *)recons)[28] = ((float2 *)SMEM)[(tmp10 >> 16) & 0xfff];
    ((float2 *)recons)[29] = ((float2 *)SMEM)[((tmp10 >> 28) & 0xf) | ((tmp11 << 4) & 0xff0)];
    ((float2 *)recons)[30] = ((float2 *)SMEM)[(tmp11 >> 8) & 0xfff];
    ((float2 *)recons)[31] = ((float2 *)SMEM)[(tmp11 >> 20) & 0xfff];
}

// 3	2	3	32	32
// 3	2	6	32	64
// 4	2	2	32	16
// 4	2	4	32	32
// 4	2	8	32	64
// 5	2	5	32	32
// 5	2	10	32	64
// 6	2	3	32	16
// 6	2	6	32	32
// 6	2	12	32	64
// 7	2	7	32	32
// 7	2	14	32	64
// 8	2	4	32	16
// 8	2	8	32	32
// 8	2	16	32	64
// 9	2	9	32	32
// 9	2	18	32	64
// 10	2	5	32	16
// 10	2	10	32	32
// 10	2	20	32	64
// 11	2	11	32	32
// 11	2	22	32	64
// 12	2	6	32	16
// 12	2	12	32	32
// 12	2	24	32	64
// 6	4	3	32	32
// 6	4	6	32	64
// 7	4	7	32	64
// 8	4	2	32	16
// 8	4	4	32	32
// 8	4	8	32	64
// 9	4	9	32	64
// 10	4	5	32	32
// 10	4	10	32	64
// 11	4	11	32	64
// 12	4	3	32	16
// 12	4	6	32	32
// 12	4	12	32	64

template <int nbits, int vec_sz, int code_n, typename codeT, int recons_n>
__global__ void vq_pack_dequant_kbit_store(uint32_t K, const codeT *Bcode, const half *lut, half *O) {
    constexpr int shC_siz = (1 << nbits) * vec_sz;
    // codeT size
    constexpr int codeT_sz = sizeof(codeT) * 8;
    extern __shared__ half shC[];
    const int totalThreads = blockDim.x * blockDim.y;
    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    // load 16 bytes at once
    if constexpr (shC_siz < 8){
        if (tid < shC_siz) {
            shC[tid] = lut[tid];
        }
    } else {
        for (int i = tid; i < shC_siz / 8; i += totalThreads)
        {
            ((float4 *)shC)[i] = ((float4 *)lut)[i];
        }
    }
    __syncthreads();
    int n_idx_base = blockIdx.x * NROWS + threadIdx.y;
    const int K_iter = DIV_ROUND_UP(K, recons_n * 2 * blockDim.x);
    const int K_iter_last = K / (recons_n * 2 * blockDim.x);
    const int recons_n_div4 = recons_n / 4;
    const int Bcode_row_sz = K * code_n / (recons_n * 2);
    __half2 B_row[recons_n];
    codeT Bcode_row[code_n];

    int eff_warp_size = blockDim.x;
    #pragma unroll
    for (int k = 0; k < K_iter; k++) {
        if (k == K_iter_last) {
            eff_warp_size = (K % (recons_n * 2 * blockDim.x)) / (recons_n * 2);
            if (threadIdx.x >= eff_warp_size) break;
        }
        const int k_code = k * code_n * blockDim.x + threadIdx.x;
        const int k_val = k * recons_n_div4 * blockDim.x + threadIdx.x;
        const int n_idx = n_idx_base;

        // Load B_code_row
        #pragma unroll
        for (int j = 0; j < code_n; j++) {
            const int k_code_idx = k_code + j * eff_warp_size;
            Bcode_row[j] = Bcode[n_idx * Bcode_row_sz + k_code_idx];
        }

        // Load B_row
        vq_pack_dequant_routine<nbits, vec_sz, code_n, codeT>(Bcode_row, B_row, shC);
        
        // Save B_row to O
        #pragma unroll
        for (int j = 0; j < recons_n_div4; j++) {
            const int k_val_idx = k_val + j * eff_warp_size;
            ((float4 *)O)[n_idx * (K / 8) + k_val_idx] = ((float4 *)B_row)[j];
        }
    }
}


/* warp-wide sum with tree-reduction */
__device__ __forceinline__ half warp_reduce_sum(
        half sum
) {
    #pragma unroll
    for (int i = 4; i >= 0; i--)
        sum += __shfl_down_sync(0xffffffff, sum, 1 << i);
    return sum;
}

template <int maxm, int nbits, int vec_sz, int code_n, typename codeT, int recons_n, AccumType mode>
__global__ void vq_pack_gemm_fp16(uint32_t M, uint32_t N, uint32_t K,
                                const half *A, const codeT *Bcode, const half *lut, half *C){
    /*
        A, B, C: row-major
        A : M x K
        B : N x K 
            Bcode : N x (K * nbits / 32) packed code
            lut : 1 << nbits 
        C : M x N, C = alpha * A * B.T + beta * C
        assume that M is small
        GridDim N / (nrows x multi_row)
        BlockDim 32 x nrows
    */
    constexpr int shC_siz = (1 << nbits) * vec_sz;
    // codeT size
    constexpr int codeT_sz = sizeof(codeT) * 8;

    extern __shared__ half shC[];
    const int totalThreads = blockDim.x * blockDim.y;
    const int tid          = threadIdx.y * blockDim.x + threadIdx.x;
    // load 16 bytes at once
    if constexpr (shC_siz < 8){
        if (tid < shC_siz) {
            shC[tid] = lut[tid];
        }
    } else {
        for (int i = tid; i < shC_siz / 8; i += totalThreads)
        {
            ((float4 *)shC)[i] = ((float4 *)lut)[i];
        }
    }
    __syncthreads();

    const int multi_row = (maxm == 1 ? 1 : 4);
    int n_idx_base = blockIdx.x * NROWS * multi_row + threadIdx.y;
    const int K_iter = DIV_ROUND_UP(K, (2 * recons_n) * blockDim.x);
    const int K_iter_last = K / (recons_n * 2 * blockDim.x);
    const int recons_n_div4 = recons_n / 4;
    const int Bcode_row_sz = K * code_n / (recons_n * 2);

    __half2 B_row[recons_n];
    codeT Bcode_row[code_n];
    using partial_t = std::conditional_t<
        (mode == AccumType::HalfAccum),
        __half,
        float
    >;
    using sum_t = std::conditional_t<
        (mode == AccumType::HalfAccum),
        half2,
        float2
    >;
    partial_t partial_sum[maxm * multi_row] = {};
    sum_t sum_ = {};
    int eff_warp_size = blockDim.x;

    #pragma unroll
    for (int k=0; k<K_iter; k++){
        if (k == K_iter_last){
            eff_warp_size = (K % (recons_n * 2 * blockDim.x)) / (recons_n * 2);
            if (threadIdx.x >= eff_warp_size) break;
        }
        #pragma unroll
        for (int h=0; h<multi_row; h++){
            const int k_code = k * code_n * blockDim.x + threadIdx.x;
            const int k_val = k * recons_n_div4 * blockDim.x + threadIdx.x;
            const int n_idx = n_idx_base + h * NROWS;

            // load B_code_row.
            // Bcode : N x (K * nbits / 32)
            // Bcode_row : B_code[n_idx * (K*nbits/32) + k * nbits * blockDim.x + threadIdx.x + blockDim.x * j]
            #pragma unroll
            for (int j = 0; j < code_n; j++) {
                const int k_code_idx = k_code + j * eff_warp_size;
                Bcode_row[j] = Bcode[n_idx * Bcode_row_sz + k_code_idx];
            }

            // load B_row 1 x K (1 x 32 half per each thread)
            vq_pack_dequant_routine<nbits, vec_sz, code_n, codeT>(Bcode_row, B_row, shC);
            // accumulate
            // K-axis : k * 32 * warp_size ~ (k+1) * 32 * warp_size
            // M-axis : 0 ~ maxm
            // N-axis : n_idx_base + 0 * multi_row  ~ n_idx_base + (nrows-1) * multi_row
            #pragma unroll
            for (int i=0; i<maxm; i++){
                if constexpr (mode == AccumType::HalfAccum){
                    sum_ = make_half2(__float2half(0.0f), __float2half(0.0f));
                } else {
                    sum_ = make_float2(0.0f, 0.0f);
                }
                #pragma unroll
                for (int j=0; j<recons_n_div4; j++){
                    const int k_val_idx = k_val + j * eff_warp_size;
                    float4 A_row = ((float4 *)A)[i * (K / 8) + k_val_idx];
                    half2 * A_row_half2 = (half2 *) &A_row;

                    #pragma unroll
                    for (int l=0; l<4; l++){
                        if constexpr (mode == AccumType::HalfAccum){
                            sum_ = __hfma2(A_row_half2[l], B_row[4*j+l], sum_);
                        } else if constexpr (mode == AccumType::FloatAccumHmul){
                            sum_.x = sum_.x + __half2float(__hmul(A_row_half2[l].x, B_row[4*j+l].x));
                            sum_.y = sum_.y + __half2float(__hmul(A_row_half2[l].y, B_row[4*j+l].y));
                        } else if constexpr (mode == AccumType::FloatAccumFmul){
                            sum_.x = sum_.x + __half2float(A_row_half2[l].x) * __half2float(B_row[4*j+l].x);
                            sum_.y = sum_.y + __half2float(A_row_half2[l].y) * __half2float(B_row[4*j+l].y);
                        }
                    }
                }
                if constexpr (mode == AccumType::HalfAccum){
                    partial_sum[i + h * maxm] = __hadd(partial_sum[i + h * maxm], __hadd(sum_.x, sum_.y));
                } else {
                    partial_sum[i + h * maxm] = partial_sum[i + h * maxm] + sum_.x + sum_.y;
                }
            }
        }
    }
    // write
    #pragma unroll
    for (int i=0; i<maxm*multi_row; i++){
        partial_sum[i] = warp_reduce_sum(partial_sum[i]);
    }
    if (threadIdx.x == 0){
        #pragma unroll
        for (int h=0; h<multi_row; h++){
            int n_idx = n_idx_base + h * NROWS;
            #pragma unroll
            for (int i=0; i<maxm; i++){
                int m_idx = i * N + n_idx;
                if constexpr (mode == AccumType::HalfAccum){    
                    C[m_idx] = partial_sum[i + h * maxm];
                } else {
                    C[m_idx] = __float2half(partial_sum[i + h * maxm]);
                }
            }
        }
    }
}

template <int maxm, int nbits, int vec_sz, int code_n, typename codeT, int recons_n, AccumType mode>
void ensure_vq_pack_gemm_fp16_AttributeSetOnce() {
    static bool done = false;
    if (!done) {
        done = true;

        using KernelFnType = decltype(&vq_pack_gemm_fp16<maxm, nbits, vec_sz, code_n, codeT, recons_n, mode>);
        KernelFnType kernelFn = &vq_pack_gemm_fp16<maxm, nbits, vec_sz, code_n, codeT, recons_n, mode>;

        int shC_siz = (1 << nbits) * vec_sz;
        size_t maxSharedBytes = shC_siz * 2;

        // If maxSharedBytes is greater than 48KB, set the attribute
        if (maxSharedBytes > 48 * 1024) {
            cudaError_t err = cudaFuncSetAttribute(
                kernelFn,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxSharedBytes
            );
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaFuncSetAttribute failed: %s\n",
                        cudaGetErrorString(err));
            }
        }
    }
}

template <int maxm, int nbits, int vec_sz, int code_n, typename codeT, int recons_n>
__host__ static void vq_pack_gemm_ptr(
    const half *in,
    half *out,
    const codeT *qweight,
    const half *lut,
    uint32_t M, uint32_t N, uint32_t K,
    CUstream_st *stream) {
    const size_t shC_bytes = (1 << nbits) * vec_sz * 2; //  half = 2 bytes
    assert(M == maxm);
    const int multi_row = (M == 1 ? 1 : 4);
    const int num_ksplit = 1;
    dim3 grid(N/(NROWS * multi_row)); dim3 block(32, NROWS, num_ksplit);

    ensure_vq_pack_gemm_fp16_AttributeSetOnce<maxm, nbits, vec_sz, code_n, codeT, recons_n, AccumType::HalfAccum>();
    vq_pack_gemm_fp16<maxm, nbits, vec_sz, code_n, codeT, recons_n, AccumType::HalfAccum><<<grid, block, shC_bytes, stream>>>(
        M, N, K,
        in,
        qweight,
        lut,
        out
    );
    gpuErrchk(cudaPeekAtLastError());
}

template <int nbits, int vec_sz, int code_n, typename codeT, int recons_n>
void ensure_vq_pack_dequant_kbit_store_AttributeSetOnce() {
    static bool done = false;
    if (!done) {
        done = true;

        using KernelFnType = decltype(&vq_pack_dequant_kbit_store<nbits, vec_sz, code_n, codeT, recons_n>);
        KernelFnType kernelFn = &vq_pack_dequant_kbit_store<nbits, vec_sz, code_n, codeT, recons_n>;

        int shC_siz = (1 << nbits) * vec_sz;
        size_t maxSharedBytes = shC_siz * 2;

        if (maxSharedBytes > 48 * 1024) {
            cudaError_t err = cudaFuncSetAttribute(
                kernelFn,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                maxSharedBytes
            );

            if (err != cudaSuccess) {
                fprintf(stderr, "cudaFuncSetAttribute failed: %s\n",
                        cudaGetErrorString(err));
            }
        }
    }
}
template <int nbits, int vec_sz, int code_n, typename codeT, int recons_n>
__host__ static void vq_pack_dequant_ptr(
    const codeT *qweight,
    uint32_t N, uint32_t K,
    const half *lut,
    half *weight,
    CUstream_st *stream
    ) {
    const size_t shC_bytes = (1 << nbits) * vec_sz * 2; //  half = 2 bytes

    ensure_vq_pack_dequant_kbit_store_AttributeSetOnce<nbits, vec_sz, code_n, codeT, recons_n>();

    dim3 grid(N/NROWS), block(32, NROWS);
    vq_pack_dequant_kbit_store<nbits, vec_sz, code_n, codeT, recons_n><<<grid, block, shC_bytes, stream>>>(
        K,
        qweight,
        lut,
        weight
    );
    gpuErrchk(cudaPeekAtLastError());
}
