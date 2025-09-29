#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>
#include "gemm_routines.h"
#include "typetraits.h"
#include "datatype.h"
enum class AccumType {
    HalfAccum,  // Variation #1
    FloatAccumHmul,  // Variation #2
    FloatAccumFmul // Variation #3
};

#define ANYPREC_NUM_ROWS 4
#define DIV_ROUND_UP(x, y) (((x)+(y)-1)/(y))

template <int nbits, int multi_bits>
__device__ __forceinline__ void pack_dequant(const uint32_t Bcode_row[], half2 B_row[], half shC[]);

template <>
__device__ __forceinline__ void pack_dequant<1, 1>(const uint32_t Bcode_row[1], half2 B_row[16], half shC[2]) {
    uint32_t tmp0 = Bcode_row[0];
    B_row[0] = make_half2(shC[(tmp0 >> 0) & 0x1], shC[(tmp0 >> 1) & 0x1]);
    B_row[1] = make_half2(shC[(tmp0 >> 2) & 0x1], shC[(tmp0 >> 3) & 0x1]);
    B_row[2] = make_half2(shC[(tmp0 >> 4) & 0x1], shC[(tmp0 >> 5) & 0x1]);
    B_row[3] = make_half2(shC[(tmp0 >> 6) & 0x1], shC[(tmp0 >> 7) & 0x1]);
    B_row[4] = make_half2(shC[(tmp0 >> 8) & 0x1], shC[(tmp0 >> 9) & 0x1]);
    B_row[5] = make_half2(shC[(tmp0 >> 10) & 0x1], shC[(tmp0 >> 11) & 0x1]);
    B_row[6] = make_half2(shC[(tmp0 >> 12) & 0x1], shC[(tmp0 >> 13) & 0x1]);
    B_row[7] = make_half2(shC[(tmp0 >> 14) & 0x1], shC[(tmp0 >> 15) & 0x1]);
    B_row[8] = make_half2(shC[(tmp0 >> 16) & 0x1], shC[(tmp0 >> 17) & 0x1]);
    B_row[9] = make_half2(shC[(tmp0 >> 18) & 0x1], shC[(tmp0 >> 19) & 0x1]);
    B_row[10] = make_half2(shC[(tmp0 >> 20) & 0x1], shC[(tmp0 >> 21) & 0x1]);
    B_row[11] = make_half2(shC[(tmp0 >> 22) & 0x1], shC[(tmp0 >> 23) & 0x1]);
    B_row[12] = make_half2(shC[(tmp0 >> 24) & 0x1], shC[(tmp0 >> 25) & 0x1]);
    B_row[13] = make_half2(shC[(tmp0 >> 26) & 0x1], shC[(tmp0 >> 27) & 0x1]);
    B_row[14] = make_half2(shC[(tmp0 >> 28) & 0x1], shC[(tmp0 >> 29) & 0x1]);
    B_row[15] = make_half2(shC[(tmp0 >> 30) & 0x1], shC[(tmp0 >> 31) & 0x1]);
}

template <>
__device__ __forceinline__ void pack_dequant<1, 2>(const uint32_t Bcode_row[1], half2 B_row[16], half shC[8]) {
    uint32_t tmp0 = Bcode_row[0];
    B_row[0] = ((half2 *)shC)[(tmp0 >> 0) & 0x3];
    B_row[1] = ((half2 *)shC)[(tmp0 >> 2) & 0x3];
    B_row[2] = ((half2 *)shC)[(tmp0 >> 4) & 0x3];
    B_row[3] = ((half2 *)shC)[(tmp0 >> 6) & 0x3];
    B_row[4] = ((half2 *)shC)[(tmp0 >> 8) & 0x3];
    B_row[5] = ((half2 *)shC)[(tmp0 >> 10) & 0x3];
    B_row[6] = ((half2 *)shC)[(tmp0 >> 12) & 0x3];
    B_row[7] = ((half2 *)shC)[(tmp0 >> 14) & 0x3];
    B_row[8] = ((half2 *)shC)[(tmp0 >> 16) & 0x3];
    B_row[9] = ((half2 *)shC)[(tmp0 >> 18) & 0x3];
    B_row[10] = ((half2 *)shC)[(tmp0 >> 20) & 0x3];
    B_row[11] = ((half2 *)shC)[(tmp0 >> 22) & 0x3];
    B_row[12] = ((half2 *)shC)[(tmp0 >> 24) & 0x3];
    B_row[13] = ((half2 *)shC)[(tmp0 >> 26) & 0x3];
    B_row[14] = ((half2 *)shC)[(tmp0 >> 28) & 0x3];
    B_row[15] = ((half2 *)shC)[(tmp0 >> 30) & 0x3];
}


template <>
__device__ __forceinline__ void pack_dequant<1, 4>(const uint32_t Bcode_row[1], half2 B_row[16], half shC[64]) {
    uint32_t tmp0 = Bcode_row[0];
    ((float2 *)B_row)[0] = ((float2 *)shC)[(tmp0 >> 0) & 0xf];
    ((float2 *)B_row)[1] = ((float2 *)shC)[(tmp0 >> 4) & 0xf];
    ((float2 *)B_row)[2] = ((float2 *)shC)[(tmp0 >> 8) & 0xf];
    ((float2 *)B_row)[3] = ((float2 *)shC)[(tmp0 >> 12) & 0xf];
    ((float2 *)B_row)[4] = ((float2 *)shC)[(tmp0 >> 16) & 0xf];
    ((float2 *)B_row)[5] = ((float2 *)shC)[(tmp0 >> 20) & 0xf];
    ((float2 *)B_row)[6] = ((float2 *)shC)[(tmp0 >> 24) & 0xf];
    ((float2 *)B_row)[7] = ((float2 *)shC)[(tmp0 >> 28) & 0xf];
}

template <>
__device__ __forceinline__ void pack_dequant<1, 8>(const uint32_t Bcode_row[1], half2 B_row[16], half shC[2048]) {
    uint32_t tmp0 = Bcode_row[0];
    ((float4 *)B_row)[0] = ((float4 *)shC)[(tmp0 >> 0) & 0xff];
    ((float4 *)B_row)[1] = ((float4 *)shC)[(tmp0 >> 8) & 0xff];
    ((float4 *)B_row)[2] = ((float4 *)shC)[(tmp0 >> 16) & 0xff];
    ((float4 *)B_row)[3] = ((float4 *)shC)[(tmp0 >> 24) & 0xff];
}

template <>
__device__ __forceinline__ void pack_dequant<2, 1>(const uint32_t Bcode_row[2], half2 B_row[16], half shC[4]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    B_row[0] = make_half2(shC[(tmp0 >> 0) & 0x3], shC[(tmp0 >> 2) & 0x3]);
    B_row[1] = make_half2(shC[(tmp0 >> 4) & 0x3], shC[(tmp0 >> 6) & 0x3]);
    B_row[2] = make_half2(shC[(tmp0 >> 8) & 0x3], shC[(tmp0 >> 10) & 0x3]);
    B_row[3] = make_half2(shC[(tmp0 >> 12) & 0x3], shC[(tmp0 >> 14) & 0x3]);
    B_row[4] = make_half2(shC[(tmp0 >> 16) & 0x3], shC[(tmp0 >> 18) & 0x3]);
    B_row[5] = make_half2(shC[(tmp0 >> 20) & 0x3], shC[(tmp0 >> 22) & 0x3]);
    B_row[6] = make_half2(shC[(tmp0 >> 24) & 0x3], shC[(tmp0 >> 26) & 0x3]);
    B_row[7] = make_half2(shC[(tmp0 >> 28) & 0x3], shC[(tmp0 >> 30) & 0x3]);
    B_row[8] = make_half2(shC[(tmp1 >> 0) & 0x3], shC[(tmp1 >> 2) & 0x3]);
    B_row[9] = make_half2(shC[(tmp1 >> 4) & 0x3], shC[(tmp1 >> 6) & 0x3]);
    B_row[10] = make_half2(shC[(tmp1 >> 8) & 0x3], shC[(tmp1 >> 10) & 0x3]);
    B_row[11] = make_half2(shC[(tmp1 >> 12) & 0x3], shC[(tmp1 >> 14) & 0x3]);
    B_row[12] = make_half2(shC[(tmp1 >> 16) & 0x3], shC[(tmp1 >> 18) & 0x3]);
    B_row[13] = make_half2(shC[(tmp1 >> 20) & 0x3], shC[(tmp1 >> 22) & 0x3]);
    B_row[14] = make_half2(shC[(tmp1 >> 24) & 0x3], shC[(tmp1 >> 26) & 0x3]);
    B_row[15] = make_half2(shC[(tmp1 >> 28) & 0x3], shC[(tmp1 >> 30) & 0x3]);
}

template <>
__device__ __forceinline__ void pack_dequant<2, 2>(const uint32_t Bcode_row[2], half2 B_row[16], half shC[32]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    B_row[0] = ((half2 *)shC)[(tmp0 >> 0) & 0xf];
    B_row[1] = ((half2 *)shC)[(tmp0 >> 4) & 0xf];
    B_row[2] = ((half2 *)shC)[(tmp0 >> 8) & 0xf];
    B_row[3] = ((half2 *)shC)[(tmp0 >> 12) & 0xf];
    B_row[4] = ((half2 *)shC)[(tmp0 >> 16) & 0xf];
    B_row[5] = ((half2 *)shC)[(tmp0 >> 20) & 0xf];
    B_row[6] = ((half2 *)shC)[(tmp0 >> 24) & 0xf];
    B_row[7] = ((half2 *)shC)[(tmp0 >> 28) & 0xf];
    B_row[8] = ((half2 *)shC)[(tmp1 >> 0) & 0xf];
    B_row[9] = ((half2 *)shC)[(tmp1 >> 4) & 0xf];
    B_row[10] = ((half2 *)shC)[(tmp1 >> 8) & 0xf];
    B_row[11] = ((half2 *)shC)[(tmp1 >> 12) & 0xf];
    B_row[12] = ((half2 *)shC)[(tmp1 >> 16) & 0xf];
    B_row[13] = ((half2 *)shC)[(tmp1 >> 20) & 0xf];
    B_row[14] = ((half2 *)shC)[(tmp1 >> 24) & 0xf];
    B_row[15] = ((half2 *)shC)[(tmp1 >> 28) & 0xf];
}

template <>
__device__ __forceinline__ void pack_dequant<2, 4>(const uint32_t Bcode_row[2], half2 B_row[16], half shC[1024]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    ((float2 *)B_row)[0] = ((float2 *)shC)[(tmp0 >> 0) & 0xff];
    ((float2 *)B_row)[1] = ((float2 *)shC)[(tmp0 >> 8) & 0xff];
    ((float2 *)B_row)[2] = ((float2 *)shC)[(tmp0 >> 16) & 0xff];
    ((float2 *)B_row)[3] = ((float2 *)shC)[(tmp0 >> 24) & 0xff];
    ((float2 *)B_row)[4] = ((float2 *)shC)[(tmp1 >> 0) & 0xff];
    ((float2 *)B_row)[5] = ((float2 *)shC)[(tmp1 >> 8) & 0xff];
    ((float2 *)B_row)[6] = ((float2 *)shC)[(tmp1 >> 16) & 0xff];
    ((float2 *)B_row)[7] = ((float2 *)shC)[(tmp1 >> 24) & 0xff];
}

template <>
__device__ __forceinline__ void pack_dequant<3, 1>(const uint32_t Bcode_row[3], half2 B_row[16], half shC[8]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    uint32_t tmp2 = Bcode_row[2];
    B_row[0] = make_half2(shC[(tmp0 >> 0) & 0x7], shC[(tmp0 >> 3) & 0x7]);
    B_row[1] = make_half2(shC[(tmp0 >> 6) & 0x7], shC[(tmp0 >> 9) & 0x7]);
    B_row[2] = make_half2(shC[(tmp0 >> 12) & 0x7], shC[(tmp0 >> 15) & 0x7]);
    B_row[3] = make_half2(shC[(tmp0 >> 18) & 0x7], shC[(tmp0 >> 21) & 0x7]);
    B_row[4] = make_half2(shC[(tmp0 >> 24) & 0x7], shC[(tmp0 >> 27) & 0x7]);
    uint32_t tmp = (((tmp0 >> 30) & 0x3) | ((tmp1 & 0x1) << 2));
    
    tmp1 >>= 1;
    B_row[5] = make_half2(shC[tmp], shC[(tmp1 >> 0) & 0x7]);
    B_row[6] = make_half2(shC[(tmp1 >> 3) & 0x7], shC[(tmp1 >> 6) & 0x7]);
    B_row[7] = make_half2(shC[(tmp1 >> 9) & 0x7], shC[(tmp1 >> 12) & 0x7]);
    B_row[8] = make_half2(shC[(tmp1 >> 15) & 0x7], shC[(tmp1 >> 18) & 0x7]);
    B_row[9] = make_half2(shC[(tmp1 >> 21) & 0x7], shC[(tmp1 >> 24) & 0x7]);
    tmp = (((tmp1 >> 30) & 0x1) | ((tmp2 & 0x3) << 1));
    tmp2 >>= 2;
    B_row[10] = make_half2(shC[(tmp1 >> 27) & 0x7], shC[tmp]);
    B_row[11] = make_half2(shC[(tmp2 >> 0) & 0x7], shC[(tmp2 >> 3) & 0x7]);
    B_row[12] = make_half2(shC[(tmp2 >> 6) & 0x7], shC[(tmp2 >> 9) & 0x7]);
    B_row[13] = make_half2(shC[(tmp2 >> 12) & 0x7], shC[(tmp2 >> 15) & 0x7]);
    B_row[14] = make_half2(shC[(tmp2 >> 18) & 0x7], shC[(tmp2 >> 21) & 0x7]);
    B_row[15] = make_half2(shC[(tmp2 >> 24) & 0x7], shC[(tmp2 >> 27) & 0x7]);
}

template <>
__device__ __forceinline__ void pack_dequant<3, 2>(const uint32_t Bcode_row[3], half2 B_row[16], half shC[128]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    uint32_t tmp2 = Bcode_row[2];
    B_row[0] = ((half2 *)shC)[(tmp0 >> 0) & 0x3f];
    B_row[1] = ((half2 *)shC)[(tmp0 >> 6) & 0x3f];
    B_row[2] = ((half2 *)shC)[(tmp0 >> 12) & 0x3f];
    B_row[3] = ((half2 *)shC)[(tmp0 >> 18) & 0x3f];
    B_row[4] = ((half2 *)shC)[(tmp0 >> 24) & 0x3f];
    uint32_t tmp = ((tmp0 >> 30) & 0x3) | ((tmp1 << 2) & 0x3c);
    tmp1 >>= 4;
    B_row[5] = ((half2 *)shC)[tmp];
    B_row[6] = ((half2 *)shC)[(tmp1 >> 0) & 0x3f];
    B_row[7] = ((half2 *)shC)[(tmp1 >> 6) & 0x3f];
    B_row[8] = ((half2 *)shC)[(tmp1 >> 12) & 0x3f];
    B_row[9] = ((half2 *)shC)[(tmp1 >> 18) & 0x3f];
    tmp = ((tmp1 >> 24) & 0xf) | ((tmp2 << 4) & 0x30);
    tmp2 >>= 2;
    B_row[10] = ((half2 *)shC)[tmp];
    B_row[11] = ((half2 *)shC)[(tmp2 >> 0) & 0x3f];
    B_row[12] = ((half2 *)shC)[(tmp2 >> 6) & 0x3f];
    B_row[13] = ((half2 *)shC)[(tmp2 >> 12) & 0x3f];
    B_row[14] = ((half2 *)shC)[(tmp2 >> 18) & 0x3f];
    B_row[15] = ((half2 *)shC)[(tmp2 >> 24) & 0x3f];
}

template <>
__device__ __forceinline__ void pack_dequant<4, 1>(const uint32_t Bcode_row[4], half2 B_row[16], half shC[16]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    uint32_t tmp2 = Bcode_row[2];
    uint32_t tmp3 = Bcode_row[3];
    B_row[0] = make_half2(shC[(tmp0 >> 0) & 0xf], shC[(tmp0 >> 4) & 0xf]);
    B_row[1] = make_half2(shC[(tmp0 >> 8) & 0xf], shC[(tmp0 >> 12) & 0xf]);
    B_row[2] = make_half2(shC[(tmp0 >> 16) & 0xf], shC[(tmp0 >> 20) & 0xf]);
    B_row[3] = make_half2(shC[(tmp0 >> 24) & 0xf], shC[(tmp0 >> 28) & 0xf]);
    B_row[4] = make_half2(shC[(tmp1 >> 0) & 0xf], shC[(tmp1 >> 4) & 0xf]);
    B_row[5] = make_half2(shC[(tmp1 >> 8) & 0xf], shC[(tmp1 >> 12) & 0xf]);
    B_row[6] = make_half2(shC[(tmp1 >> 16) & 0xf], shC[(tmp1 >> 20) & 0xf]);
    B_row[7] = make_half2(shC[(tmp1 >> 24) & 0xf], shC[(tmp1 >> 28) & 0xf]);
    B_row[8] = make_half2(shC[(tmp2 >> 0) & 0xf], shC[(tmp2 >> 4) & 0xf]);
    B_row[9] = make_half2(shC[(tmp2 >> 8) & 0xf], shC[(tmp2 >> 12) & 0xf]);
    B_row[10] = make_half2(shC[(tmp2 >> 16) & 0xf], shC[(tmp2 >> 20) & 0xf]);
    B_row[11] = make_half2(shC[(tmp2 >> 24) & 0xf], shC[(tmp2 >> 28) & 0xf]);
    B_row[12] = make_half2(shC[(tmp3 >> 0) & 0xf], shC[(tmp3 >> 4) & 0xf]);
    B_row[13] = make_half2(shC[(tmp3 >> 8) & 0xf], shC[(tmp3 >> 12) & 0xf]);
    B_row[14] = make_half2(shC[(tmp3 >> 16) & 0xf], shC[(tmp3 >> 20) & 0xf]);
    B_row[15] = make_half2(shC[(tmp3 >> 24) & 0xf], shC[(tmp3 >> 28) & 0xf]);
}

template <>
__device__ __forceinline__ void pack_dequant<4, 2>(const uint32_t Bcode_row[4], half2 B_row[16], half shC[512]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    uint32_t tmp2 = Bcode_row[2];
    uint32_t tmp3 = Bcode_row[3];
    B_row[0] = ((half2 *)shC)[(tmp0 >> 0) & 0xff];
    B_row[1] = ((half2 *)shC)[(tmp0 >> 8) & 0xff];
    B_row[2] = ((half2 *)shC)[(tmp0 >> 16) & 0xff];
    B_row[3] = ((half2 *)shC)[(tmp0 >> 24) & 0xff];
    B_row[4] = ((half2 *)shC)[(tmp1 >> 0) & 0xff];
    B_row[5] = ((half2 *)shC)[(tmp1 >> 8) & 0xff];
    B_row[6] = ((half2 *)shC)[(tmp1 >> 16) & 0xff];
    B_row[7] = ((half2 *)shC)[(tmp1 >> 24) & 0xff];
    B_row[8] = ((half2 *)shC)[(tmp2 >> 0) & 0xff];
    B_row[9] = ((half2 *)shC)[(tmp2 >> 8) & 0xff];
    B_row[10] = ((half2 *)shC)[(tmp2 >> 16) & 0xff];
    B_row[11] = ((half2 *)shC)[(tmp2 >> 24) & 0xff];
    B_row[12] = ((half2 *)shC)[(tmp3 >> 0) & 0xff];
    B_row[13] = ((half2 *)shC)[(tmp3 >> 8) & 0xff];
    B_row[14] = ((half2 *)shC)[(tmp3 >> 16) & 0xff];
    B_row[15] = ((half2 *)shC)[(tmp3 >> 24) & 0xff];
}

template <>
__device__ __forceinline__ void pack_dequant<5, 1>(const uint32_t Bcode_row[5], half2 B_row[16], half shC[32]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    uint32_t tmp2 = Bcode_row[2];
    uint32_t tmp3 = Bcode_row[3];
    uint32_t tmp4 = Bcode_row[4];
    B_row[0] = make_half2(shC[(tmp0 >> 0) & 0x1f], shC[(tmp0 >> 5) & 0x1f]);
    B_row[1] = make_half2(shC[(tmp0 >> 10) & 0x1f], shC[(tmp0 >> 15) & 0x1f]);
    B_row[2] = make_half2(shC[(tmp0 >> 20) & 0x1f], shC[(tmp0 >> 25) & 0x1f]);
    uint32_t tmp = (((tmp0 >> 30) & 0x3) | ((tmp1 & 0x7) << 2));
    
    tmp1 >>= 3;
    B_row[3] = make_half2(shC[tmp], shC[(tmp1 >> 0) & 0x1f]);
    B_row[4] = make_half2(shC[(tmp1 >> 5) & 0x1f], shC[(tmp1 >> 10) & 0x1f]);
    B_row[5] = make_half2(shC[(tmp1 >> 15) & 0x1f], shC[(tmp1 >> 20) & 0x1f]);
    tmp = (((tmp1 >> 25) & 0xf) | ((tmp2 & 0x1) << 4));
    tmp2 >>= 1;

    B_row[6] = make_half2(shC[tmp], shC[(tmp2 >> 0) & 0x1f]);
    B_row[7] = make_half2(shC[(tmp2 >> 5) & 0x1f], shC[(tmp2 >> 10) & 0x1f]);
    B_row[8] = make_half2(shC[(tmp2 >> 15) & 0x1f], shC[(tmp2 >> 20) & 0x1f]);
    tmp = (((tmp2 >> 30) & 0x1) | ((tmp3 & 0xf) << 1));
    tmp3 >>= 4;

    B_row[9] = make_half2(shC[(tmp2 >> 25) & 0x1f], shC[(tmp) & 0x1f]);
    B_row[10] = make_half2(shC[(tmp3 >> 0) & 0x1f], shC[(tmp3 >> 5) & 0x1f]);
    B_row[11] = make_half2(shC[(tmp3 >> 10) & 0x1f], shC[(tmp3 >> 15) & 0x1f]);
    tmp = (((tmp3 >> 25) & 0x7) | ((tmp4 & 0x3) << 3));
    tmp4 >>= 2;

    B_row[12] = make_half2(shC[(tmp3 >> 20) & 0x1f], shC[tmp]);
    B_row[13] = make_half2(shC[(tmp4 >> 0) & 0x1f], shC[(tmp4 >> 5) & 0x1f]);
    B_row[14] = make_half2(shC[(tmp4 >> 10) & 0x1f], shC[(tmp4 >> 15) & 0x1f]);
    B_row[15] = make_half2(shC[(tmp4 >> 20) & 0x1f], shC[(tmp4 >> 25) & 0x1f]);
}


template <>
__device__ __forceinline__ void pack_dequant<6, 1>(const uint32_t Bcode_row[6], half2 B_row[16], half shC[64]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    uint32_t tmp2 = Bcode_row[2];
    uint32_t tmp3 = Bcode_row[3];
    uint32_t tmp4 = Bcode_row[4];
    uint32_t tmp5 = Bcode_row[5];

    B_row[0] = make_half2(shC[(tmp0 >> 0) & 0x3f], shC[(tmp0 >> 6) & 0x3f]);
    B_row[1] = make_half2(shC[(tmp0 >> 12) & 0x3f], shC[(tmp0 >> 18) & 0x3f]);
    uint32_t tmp = ((tmp0 >> 30) & 0x3) | ((tmp1 & 0xf) << 2);
    tmp1 >>= 4;
    B_row[2] = make_half2(shC[(tmp0 >> 24) & 0x3f], shC[tmp]);
    B_row[3] = make_half2(shC[(tmp1 >> 0) & 0x3f], shC[(tmp1 >> 6) & 0x3f]);
    B_row[4] = make_half2(shC[(tmp1 >> 12) & 0x3f], shC[(tmp1 >> 18) & 0x3f]);
    tmp = ((tmp1 >> 24) & 0xf) | ((tmp2 & 0x3) << 4);
    tmp2 >>= 2;
    B_row[5] = make_half2(shC[tmp], shC[(tmp2 >> 0) & 0x3f]);
    B_row[6] = make_half2(shC[(tmp2 >> 6) & 0x3f], shC[(tmp2 >> 12) & 0x3f]);
    B_row[7] = make_half2(shC[(tmp2 >> 18) & 0x3f], shC[(tmp2 >> 24) & 0x3f]);
    
    B_row[8] = make_half2(shC[(tmp3 >> 0) & 0x3f], shC[(tmp3 >> 6) & 0x3f]);
    B_row[9] = make_half2(shC[(tmp3 >> 12) & 0x3f], shC[(tmp3 >> 18) & 0x3f]);
    tmp = ((tmp3 >> 30) & 0x3) | ((tmp4 & 0xf) << 2);
    tmp4 >>= 4;
    B_row[10] = make_half2(shC[(tmp3 >> 24) & 0x3f], shC[tmp]);
    B_row[11] = make_half2(shC[(tmp4 >> 0) & 0x3f], shC[(tmp4 >> 6) & 0x3f]);
    B_row[12] = make_half2(shC[(tmp4 >> 12) & 0x3f], shC[(tmp4 >> 18) & 0x3f]);
    tmp = ((tmp4 >> 24) & 0xf) | ((tmp5 & 0x3) << 4);
    tmp5 >>= 2;

    B_row[13] = make_half2(shC[tmp], shC[(tmp5 >> 0) & 0x3f]);
    B_row[14] = make_half2(shC[(tmp5 >> 6) & 0x3f], shC[(tmp5 >> 12) & 0x3f]);
    B_row[15] = make_half2(shC[(tmp5 >> 18) & 0x3f], shC[(tmp5 >> 24) & 0x3f]);
}


template <>
__device__ __forceinline__ void pack_dequant<7, 1>(const uint32_t Bcode_row[7], half2 B_row[16], half shC[64]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    uint32_t tmp2 = Bcode_row[2];
    uint32_t tmp3 = Bcode_row[3];
    uint32_t tmp4 = Bcode_row[4];
    uint32_t tmp5 = Bcode_row[5];
    uint32_t tmp6 = Bcode_row[6];

    B_row[0] = make_half2(shC[(tmp0 >> 0) & 0x7f], shC[(tmp0 >> 7) & 0x7f]);
    B_row[1] = make_half2(shC[(tmp0 >> 14) & 0x7f], shC[(tmp0 >> 21) & 0x7f]);
    uint32_t tmp = ((tmp0 >> 28) & 0xf) | ((tmp1 & 0x7) << 4);
    tmp1 >>= 3;
    B_row[2] = make_half2(shC[tmp], shC[(tmp1 >> 0) & 0x7f]);
    B_row[3] = make_half2(shC[(tmp1 >> 7) & 0x7f], shC[(tmp1 >> 14) & 0x7f]);
    tmp = ((tmp1 >> 28) & 0x1) | ((tmp2 & 0x3f) << 1);
    tmp2 >>= 6;
    B_row[4] = make_half2(shC[(tmp1 >> 21) & 0x7f], shC[tmp]);
    B_row[5] = make_half2(shC[(tmp2 >> 0) & 0x7f], shC[(tmp2 >> 7) & 0x7f]);
    tmp = ((tmp2 >> 21) & 0x1f) | ((tmp3 & 0x3) << 5);
    tmp3 >>= 2;
    B_row[6] = make_half2(shC[(tmp2 >> 14) & 0x7f], shC[tmp]);
    B_row[7] = make_half2(shC[(tmp3 >> 0) & 0x7f], shC[(tmp3 >> 7) & 0x7f]);
    B_row[8] = make_half2(shC[(tmp3 >> 14) & 0x7f], shC[(tmp3 >> 21) & 0x7f]);
    tmp = ((tmp3 >> 28) & 0x3) | ((tmp4 & 0x1f) << 2);
    tmp4 >>= 5;
    B_row[9] = make_half2(shC[tmp], shC[(tmp4 >> 0) & 0x7f]);
    B_row[10] = make_half2(shC[(tmp4 >> 7) & 0x7f], shC[(tmp4 >> 14) & 0x7f]);
    tmp = ((tmp4 >> 21) & 0x3f) | ((tmp5 & 0x1) << 6);
    tmp5 >>= 1;
    B_row[11] = make_half2(shC[tmp], shC[(tmp5 >> 0) & 0x7f]);
    B_row[12] = make_half2(shC[(tmp5 >> 7) & 0x7f], shC[(tmp5 >> 14) & 0x7f]);
    tmp = ((tmp5 >> 28) & 0x7) | ((tmp6 & 0xf) << 3);
    tmp6 >>= 4;
    B_row[13] = make_half2(shC[(tmp5 >> 21) & 0x7f], shC[tmp]);
    B_row[14] = make_half2(shC[(tmp6 >> 0) & 0x7f], shC[(tmp6 >> 7) & 0x7f]);
    B_row[15] = make_half2(shC[(tmp6 >> 14) & 0x7f], shC[(tmp6 >> 21) & 0x7f]);
}


template <>
__device__ __forceinline__ void pack_dequant<8, 1>(const uint32_t Bcode_row[8], half2 B_row[16], half shC[64]) {
    uint32_t tmp0 = Bcode_row[0];
    uint32_t tmp1 = Bcode_row[1];
    uint32_t tmp2 = Bcode_row[2];
    uint32_t tmp3 = Bcode_row[3];
    uint32_t tmp4 = Bcode_row[4];
    uint32_t tmp5 = Bcode_row[5];
    uint32_t tmp6 = Bcode_row[6];
    uint32_t tmp7 = Bcode_row[7];

    B_row[0] = make_half2(shC[(tmp0 >> 0) & 0xff], shC[(tmp0 >> 8) & 0xff]);
    B_row[1] = make_half2(shC[(tmp0 >> 16) & 0xff], shC[(tmp0 >> 24) & 0xff]);
    B_row[2] = make_half2(shC[(tmp1 >> 0) & 0xff], shC[(tmp1 >> 8) & 0xff]);
    B_row[3] = make_half2(shC[(tmp1 >> 16) & 0xff], shC[(tmp1 >> 24) & 0xff]);
    B_row[4] = make_half2(shC[(tmp2 >> 0) & 0xff], shC[(tmp2 >> 8) & 0xff]);
    B_row[5] = make_half2(shC[(tmp2 >> 16) & 0xff], shC[(tmp2 >> 24) & 0xff]);
    B_row[6] = make_half2(shC[(tmp3 >> 0) & 0xff], shC[(tmp3 >> 8) & 0xff]);
    B_row[7] = make_half2(shC[(tmp3 >> 16) & 0xff], shC[(tmp3 >> 24) & 0xff]);
    B_row[8] = make_half2(shC[(tmp4 >> 0) & 0xff], shC[(tmp4 >> 8) & 0xff]);
    B_row[9] = make_half2(shC[(tmp4 >> 16) & 0xff], shC[(tmp4 >> 24) & 0xff]);
    B_row[10] = make_half2(shC[(tmp5 >> 0) & 0xff], shC[(tmp5 >> 8) & 0xff]);
    B_row[11] = make_half2(shC[(tmp5 >> 16) & 0xff], shC[(tmp5 >> 24) & 0xff]);
    B_row[12] = make_half2(shC[(tmp6 >> 0) & 0xff], shC[(tmp6 >> 8) & 0xff]);
    B_row[13] = make_half2(shC[(tmp6 >> 16) & 0xff], shC[(tmp6 >> 24) & 0xff]);
    B_row[14] = make_half2(shC[(tmp7 >> 0) & 0xff], shC[(tmp7 >> 8) & 0xff]);
    B_row[15] = make_half2(shC[(tmp7 >> 16) & 0xff], shC[(tmp7 >> 24) & 0xff]);
}



template <int nbits>
__global__ void pack_dequant_kbit_store(const uint32_t K, const uint32_t *Bcode, const half *lut, half *O) {
    static_assert(nbits <= 8, "nbits must be <= 8");
    const int multi_bits = (nbits >=2 && nbits <= 4) ? 2 : 1;
    constexpr int num_centroids = 1 << nbits;
    constexpr int shC_siz = multi_bits * (1 << (nbits * multi_bits));

    __shared__ half shC[shC_siz];

    if constexpr (multi_bits == 1) {
        if constexpr (nbits <= 5){
            if (threadIdx.x < num_centroids) {
                shC[threadIdx.x] = lut[threadIdx.x];
            }
        } else if constexpr (nbits == 6) {
            ((half2*) shC)[threadIdx.x] = ((half2*) lut)[threadIdx.x];
        } else if constexpr (nbits == 7) {
            ((float2*) shC)[threadIdx.x] = ((float2*) lut)[threadIdx.x];
        } else if constexpr (nbits == 8) {
            ((float4*) shC)[threadIdx.x] = ((float4*) lut)[threadIdx.x];
        }
    } else if constexpr (multi_bits == 2) {
        if (threadIdx.x < num_centroids * num_centroids) {
            const int xx = threadIdx.x % num_centroids, yy = threadIdx.x / num_centroids;
            const half fracX = lut[xx];
            const int iter = max(1, shC_siz / blockDim.x / 2);
            #pragma unroll
            for (int i = 0; i < iter; i++) {
                const int yidx = yy | (i * blockDim.x / num_centroids);
                const half fracY = lut[yidx];
                ((half2 *)shC)[yidx * num_centroids | xx] = make_half2(fracX, fracY);
            }
        }
    } 
    __syncthreads();

    int n_idx_base = blockIdx.x * ANYPREC_NUM_ROWS + threadIdx.y;
    int K_iter = DIV_ROUND_UP(K, 32 * blockDim.x);
    __half2 B_row[16];
    uint32_t Bcode_row[nbits];
    int eff_warp_size = blockDim.x;

    #pragma unroll
    for (int k = 0; k < K_iter; k++) {
        if (k == K / (32 * blockDim.x)){
            eff_warp_size = (K % (32 * blockDim.x)) / 32;
            if (threadIdx.x >= eff_warp_size) break;
        }
        const int k_code = k * nbits * blockDim.x + threadIdx.x;
        const int k_val = k * 4 * blockDim.x + threadIdx.x;
        const int n_idx = n_idx_base;

        // Load B_code_row
        #pragma unroll
        for (int j = 0; j < nbits; j++) {
            const int k_code_idx = k_code + j * eff_warp_size;
            Bcode_row[j] = Bcode[n_idx * (K * nbits / 32) + k_code_idx];
        }

        // Load B_row
        pack_dequant<nbits, multi_bits>(Bcode_row, B_row, shC);
        
        // Save B_row to O
        #pragma unroll
        for (int j = 0; j < 4; j++) {
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

template <int maxm, int multi_row, int nbits, AccumType mode>
__global__ void sq_gemm_fp16(const uint32_t M, const uint32_t N, const uint32_t K,
                                const half *A, const uint32_t *Bcode, const half *lut, half *C){
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
   static_assert(nbits <= 8, "nbits must be <= 8");
   const int multi_bits = (nbits >= 2 && nbits <= 4) ? 2 : 1;

    // load lut
    constexpr int num_centroids = 1 << nbits;
    constexpr int shC_siz = multi_bits * (1 << (nbits * multi_bits));
    __shared__ half shC[shC_siz];
    if constexpr (multi_bits == 1) {
        if constexpr (nbits <= 5){
            if (threadIdx.x < num_centroids) {
                shC[threadIdx.x] = lut[threadIdx.x];
            }
        } else if constexpr (nbits == 6) {
            ((half2*) shC)[threadIdx.x] = ((half2*) lut)[threadIdx.x];
        } else if constexpr (nbits == 7) {
            ((float2*) shC)[threadIdx.x] = ((float2*) lut)[threadIdx.x];
        } else if constexpr (nbits == 8) {
            ((float4*) shC)[threadIdx.x] = ((float4*) lut)[threadIdx.x];
        }
    } else if constexpr (multi_bits == 2) {
        if (threadIdx.x < num_centroids * num_centroids) {
            const int xx = threadIdx.x % num_centroids, yy = threadIdx.x / num_centroids;
            const half fracX = lut[xx];
            const int iter = max(1, shC_siz / blockDim.x / 2);
            #pragma unroll
            for (int i = 0; i < iter; i++) {
                const int yidx = yy | (i * blockDim.x / num_centroids);
                const half fracY = lut[yidx];
                ((half2 *)shC)[yidx * num_centroids | xx] = make_half2(fracX, fracY);
            }
        }
    } 
    __syncthreads();

    // const int multi_row = (maxm == 1 ? 1 : 4);
    int n_idx_base = blockIdx.x * ANYPREC_NUM_ROWS * multi_row + threadIdx.y;
    int K_iter = DIV_ROUND_UP(K, 32 * blockDim.x);

    __half2 B_row[16];
    uint32_t Bcode_row[nbits];
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
        if (k == K / (32 * blockDim.x)){
            eff_warp_size = (K % (32 * blockDim.x)) / 32;
            if (threadIdx.x >= eff_warp_size) break;
        }
        #pragma unroll
        for (int h=0; h<multi_row; h++){
            const int k_code = k * nbits * blockDim.x + threadIdx.x;
            const int k_val = k * 4 * blockDim.x + threadIdx.x;
            const int n_idx = n_idx_base + h * ANYPREC_NUM_ROWS;

            // load B_code_row.
            // Bcode : N x (K * nbits / 32)
            // Bcode_row : B_code[n_idx * (K*nbits/32) + k * nbits * blockDim.x + threadIdx.x + blockDim.x * j]
            #pragma unroll
            for (int j = 0; j < nbits; j++) {
                const int k_code_idx = k_code + j * eff_warp_size;
                Bcode_row[j] = Bcode[n_idx * (K*nbits/32) + k_code_idx];
            }

            // load B_row 1 x K (1 x 32 half per each thread)
            pack_dequant<nbits, multi_bits>(Bcode_row, B_row, shC);
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
                for (int j=0; j<4; j++){
                    const int k_val_idx = k_val + j * eff_warp_size;

                    float4 A_row = ((float4 *)A)[i * (K / 8) + k_val_idx];
                    half2 * A_row_half2 = (half2 *) &A_row;
                    for (int k=0; k<4; k++){
                        if constexpr (mode == AccumType::HalfAccum){
                            sum_ = __hfma2(A_row_half2[k], B_row[4*j+k], sum_);
                        } else if constexpr (mode == AccumType::FloatAccumHmul){
                            sum_.x = sum_.x + __half2float(__hmul(A_row_half2[k].x, B_row[4*j+k].x));
                            sum_.y = sum_.y + __half2float(__hmul(A_row_half2[k].y, B_row[4*j+k].y));
                        } else if constexpr (mode == AccumType::FloatAccumFmul){
                            sum_.x = sum_.x + __half2float(A_row_half2[k].x) * __half2float(B_row[4*j+k].x);
                            sum_.y = sum_.y + __half2float(A_row_half2[k].y) * __half2float(B_row[4*j+k].y);
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
            int n_idx = n_idx_base + h * ANYPREC_NUM_ROWS;
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

using matmul_func = void (*)(
        const uint32_t, const uint32_t, const uint32_t,
        const half *, const uint32_t *,
        const half *, half *
);

template<int s, int e>
struct get_matmul_func {
    void operator()(matmul_func func[][9][2]) const {
        if constexpr(s <= e)
        {
            func[s][1][0] = sq_gemm_fp16<1, 1, s, AccumType::HalfAccum>;
            func[s][2][0] = sq_gemm_fp16<2, 4, s, AccumType::HalfAccum>;
            func[s][3][0] = sq_gemm_fp16<3, 4, s, AccumType::HalfAccum>;
            func[s][4][0] = sq_gemm_fp16<4, 4, s, AccumType::HalfAccum>;
            func[s][5][0] = sq_gemm_fp16<5, 4, s, AccumType::HalfAccum>;
            func[s][6][0] = sq_gemm_fp16<6, 4, s, AccumType::HalfAccum>;
            func[s][7][0] = sq_gemm_fp16<7, 4, s, AccumType::HalfAccum>;
            func[s][8][0] = sq_gemm_fp16<8, 4, s, AccumType::HalfAccum>;
            get_matmul_func<s + 1, e>()(func);
        }
    }
};

using dequant_func = void (*)(
        const uint32_t, const uint32_t *,
        const half *, half *
);

template<int s, int e>
struct get_dequant_func {
    void operator()(dequant_func func[]) const {
        if constexpr(s <= e)
        {
            func[s] = pack_dequant_kbit_store<s>;
            get_dequant_func<s + 1, e>()(func);
        }
    }
};

bool matmul_initialized = false;

matmul_func matmul_functions[9][9][2] = {nullptr};

void pack_matmul(
    half *in,        // [M, K]
    half *out,       // [M, N]
    uint32_t *qweight,   // [N, wbits * K/32]
    half *lut,       // [out_size, num_centroids]
    uint32_t M,           // batch size
    uint32_t N,           // output size
    uint32_t K,           // input size
    int w_bits,            // bit width
    cudaStream_t stream
) {
    assert(M >= 1 && M <= 8 && w_bits >= 2 && w_bits <= 8);
    // Initialize the function pointers if they haven't been initialized for this type
    if (!matmul_initialized) {
    get_matmul_func<2, 8>()(matmul_functions);
    matmul_initialized = true;
    }

    // Compute grid and block dimensions
    const int multi_row = (M == 1 ? 1 : 4);
    const int use_ksplit = 0; // M == 1 && K > 4096 && w_bits >= 7;
    const int num_ksplit = (use_ksplit ? DIV_ROUND_UP(K, 4096) : 1);

    dim3 grid(N / (ANYPREC_NUM_ROWS * multi_row)), block(32, ANYPREC_NUM_ROWS, num_ksplit);

    // Use the initialized function pointers for the kernel launch
    matmul_functions[w_bits][M][use_ksplit]<<<grid, block, 0, stream>>>(
        M, N, K, in, qweight, lut, out
    );
}

bool dequant_initalized = false;

dequant_func dequant_functions[9] = {nullptr};


void pack_dequant_kbit(
    const uint32_t *qweight,
    const uint32_t N, const uint32_t K,
    const half *lut, half *weight,
    int w_bits,
    cudaStream_t stream
) {
    assert(w_bits >= 2 && w_bits <= 8);

    if (!dequant_initalized) {
        get_dequant_func<2, 8>()(dequant_functions);
        dequant_initalized = true;
    }

    dim3 grid(N / ANYPREC_NUM_ROWS), block(32, ANYPREC_NUM_ROWS);
    dequant_functions[w_bits]<<<grid, block, 0, stream>>>(
        K, qweight, lut, weight
    );
}