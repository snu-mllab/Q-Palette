#include <cstdio>
#include <cassert>
#include <climits>

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <mma.h>
// #include <c10/cuda/CUDAStream.h>

#include "inference.h"

using namespace nvcuda;


#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)          do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (false)

#define BLOCKS_PER_SM 1
#define MMA_M                   16
#define MMA_N                   8
#define MMA_K                   16

#define BLOCK_COUNT             128
//#define MAX_THREADS_PER_SM      2048
#define WARP_SIZE               32
#define BLOCK_SIZE              1024
#define WARPS_PER_BLOCK         (BLOCK_SIZE/WARP_SIZE)

#define PREFETCHW               4
#define PREFETCHX               4
#define BLOCKS_PER_SM           1

#define FULL_MASK               0xFFFFFFFFU
#define ROUND_UP(a, b) ((a + b - 1) / b)


template <uint32_t N>
__inline__ __device__ void reduce_gather_routine(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][N][16]);

template <>
__inline__ __device__ void reduce_gather_routine<1>(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][1][16])
{
    for (int pi = 0; pi < 2; pi++) {
        if (laneId % 4 == 0) {
            reduce_gather[warpId][pi][0][laneId / 4] = reg_p[pi].x;
            reduce_gather[warpId][pi][0][laneId / 4 + 8] = reg_p[pi].z;
        }
    }
}

template <>
__inline__ __device__ void reduce_gather_routine<2>(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][2][16])
{
    for (int pi = 0; pi < 2; pi++) {
        if (laneId % 4 == 0) {
            reduce_gather[warpId][pi][0][laneId / 4] = reg_p[pi].x;
            reduce_gather[warpId][pi][0][laneId / 4 + 8] = reg_p[pi].z;
            reduce_gather[warpId][pi][1][laneId / 4] = reg_p[pi].y;
            reduce_gather[warpId][pi][1][laneId / 4 + 8] = reg_p[pi].w;
        }
    }
}

template <>
__inline__ __device__ void reduce_gather_routine<3>(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][3][16])
{
    for (int pi = 0; pi < 2; pi++) {
        if (laneId % 4 <= 1) {
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4] = reg_p[pi].x;
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4 + 8] = reg_p[pi].z;
        }
        if (laneId % 4 == 0) {
            reduce_gather[warpId][pi][1][laneId / 4] = reg_p[pi].y;
            reduce_gather[warpId][pi][1][laneId / 4 + 8] = reg_p[pi].w;
        }
    }
}

template <>
__inline__ __device__ void reduce_gather_routine<4>(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][4][16])
{
    for (int pi = 0; pi < 2; pi++) {
        if (laneId % 4 <= 1) {
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4] = reg_p[pi].x;
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4 + 8] = reg_p[pi].z;

            reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4] = reg_p[pi].y;
            reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4 + 8] = reg_p[pi].w;
        }
    }
}

template <>
__inline__ __device__ void reduce_gather_routine<5>(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][5][16])
{
    for (int pi = 0; pi < 2; pi++) {
        if (laneId % 4 <= 2) {
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4] = reg_p[pi].x;
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4 + 8] = reg_p[pi].z;
        }
        if (laneId % 4 <= 1) {
            reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4] = reg_p[pi].y;
            reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4 + 8] = reg_p[pi].w;
        }
    }
}

template <>
__inline__ __device__ void reduce_gather_routine<6>(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][6][16])
{
    for (int pi = 0; pi < 2; pi++) {
        if (laneId % 4 <= 2) {
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4] = reg_p[pi].x;
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4 + 8] = reg_p[pi].z;
            reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4] = reg_p[pi].y;
            reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4 + 8] = reg_p[pi].w;
        }
    }
}

template <>
__inline__ __device__ void reduce_gather_routine<7>(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][7][16])
{
    for (int pi = 0; pi < 2; pi++) {
        if (laneId % 4 <= 3) {
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4] = reg_p[pi].x;
            reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4 + 8] = reg_p[pi].z;
        }
        if (laneId % 4 <= 2) {
            reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4] = reg_p[pi].y;
            reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4 + 8] = reg_p[pi].w;
        }
    }
}

template <>
__inline__ __device__ void reduce_gather_routine<8>(float4 reg_p[2], int warpId, int laneId, float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][8][16])
{
    for (int pi = 0; pi < 2; pi++) {
        reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4] = reg_p[pi].x;
        reduce_gather[warpId][pi][(laneId%4) * 2][laneId / 4 + 8] = reg_p[pi].z;
        reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4] = reg_p[pi].y;
        reduce_gather[warpId][pi][(laneId%4) * 2 + 1][laneId / 4 + 8] = reg_p[pi].w;
    }
}

__inline__ __device__ uint32_t ld_cs(const uint32_t* p)
{
    uint32_t out;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

__inline__ __device__ uint2 ld_cs(const uint2* p)
{
    uint2 out;
    asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(out.x), "=r"(out.y) : "l"(p));
    //asm("ld.weak.global.cs.L2::256B.v2.u32 {%0, %1}, [%2];" : "=r"(out.x), "=r"(out.y) : "l"(p));
    // the compiler doesn't know how to infer load(p) and load(p+4096) from loop unrolling with this :(
    return out;
}
__inline__ __device__ uint3 ld_cs(const uint3* p)
{
    uint3 out;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(out.x) : "l"(p));
    asm("ld.global.cs.u32 %0, [%1+4];" : "=r"(out.y) : "l"(p));
    asm("ld.global.cs.u32 %0, [%1+8];" : "=r"(out.z) : "l"(p));
    return out;
}
__inline__ __device__ uint4 ld_cs(const uint4* p)
{
    uint4 out;
    asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(p));
    return out;
}
__inline__ __device__ uint2 ld_x(const uint32_t* p, uint32_t x_idx, int subki)
{
    uint2 out;
    // the indexing is written as int32 math instead of lsu constant offset because
    // apparently using lsu offset adds lots of MIO pressure!
    if (subki == 0) {
        asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out.x) : "l"(p+x_idx));
        asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out.y) : "l"(p+(x_idx+4)));
    } else {
        asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out.x) : "l"(p+(x_idx+8)));
        asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out.y) : "l"(p+(x_idx+12)));
    }
    return out;
}
__inline__ __device__ uint32_t ld_x(const uint32_t* p)
{
    uint32_t out;
    asm("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}

__inline__ __device__ void prefetch(uint32_t *a){
    asm("prefetch.global.L1 [%0];"::"l"(a));
}

template <uint32_t R, LUT_TYPE L>
__device__ inline void load_reg_cs(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next); 

template <>
__device__ inline void load_reg_cs<1U, LUT_TYPE::SQ_LUT>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ld_cs((uint *) &compressed[weight_idx]);
        reg_cs_next.x = reg_load & 0xff;
        reg_cs_next.y = (reg_load >> 8) & 0xff;
        reg_cs_next.z = (reg_load >> 16) & 0xff;
        reg_cs_next.w = (reg_load >> 24) & 0xff;
}

template <>
__device__ inline void load_reg_cs<2U, LUT_TYPE::SQ_LUT>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        ditto2 reg_load = {.u32x2 = ld_cs((uint2 *) &compressed[weight_idx])};
        reg_cs_next.x = reg_load.u16x4.x;
        reg_cs_next.y = reg_load.u16x4.y;
        reg_cs_next.z = reg_load.u16x4.z;
        reg_cs_next.w = reg_load.u16x4.w;
}

template <>
__device__ inline void load_reg_cs<3U, LUT_TYPE::SQ_LUT>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint3 reg_load = ld_cs((uint3 *) &compressed[weight_idx]);
        reg_cs_next.x = reg_load.x & 0xffffff;
        reg_cs_next.y = ((reg_load.x >> 24) | (reg_load.y << 8)) & 0xffffff;
        reg_cs_next.z = ((reg_load.y >> 16) | (reg_load.z << 16)) & 0xffffff;
        reg_cs_next.w = (reg_load.z >> 8) & 0xffffff;
}

template <>
__device__ inline void load_reg_cs<4U, LUT_TYPE::SQ_LUT>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint4 reg_load = ld_cs((uint4 *) &compressed[weight_idx]);
        reg_cs_next.x = reg_load.x;
        reg_cs_next.y = reg_load.y;
        reg_cs_next.z = reg_load.z;
        reg_cs_next.w = reg_load.w;
}

template <>
__device__ inline void load_reg_cs<5U, LUT_TYPE::SQ_LUT>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint reg_load = ld_cs((uint *) &compressed[weight_idx]);
        uint reg_load2 = ld_cs((uint *) &compressed[weight_idx + 2]);
        uint reg_load3 = ld_cs((uint *) &compressed[weight_idx + 4]);
        uint reg_load4 = ld_cs((uint *) &compressed[weight_idx + 6]);
        uint reg_load5 = ld_cs((uint *) &compressed[weight_idx + 8]);

        reg_cs_next.x = reg_load;
        reg_cs2_next.x = reg_load2 & 0xff;
        reg_cs_next.y = ((reg_load2 >> 8) | (reg_load3 << 24));
        reg_cs2_next.y = ((reg_load3 >> 8) & 0xff);
        reg_cs_next.z = ((reg_load3 >> 16) | (reg_load4 << 16));
        reg_cs2_next.z = ((reg_load4 >> 16) & 0xff);
        reg_cs_next.w = ((reg_load4 >> 24) | (reg_load5 << 8));
        reg_cs2_next.w = ((reg_load5 >> 24) & 0xff);
}

template <>
__device__ inline void load_reg_cs<6U, LUT_TYPE::SQ_LUT>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint2 reg_load = ld_cs((uint2 *) &compressed[weight_idx]);
        uint2 reg_load2 = ld_cs((uint2 *) &compressed[weight_idx + 4]);
        uint2 reg_load3 = ld_cs((uint2 *) &compressed[weight_idx + 8]);

        reg_cs_next.x = reg_load.x;
        reg_cs2_next.x = reg_load.y & 0xffff;
        reg_cs_next.y = ((reg_load.y >> 16) | (reg_load2.x << 16));
        reg_cs2_next.y = (reg_load2.x >> 16);
        reg_cs_next.z = reg_load2.y;
        reg_cs2_next.z = (reg_load3.x & 0xffff);
        reg_cs_next.w = ((reg_load3.x >> 16) | (reg_load3.y << 16));
        reg_cs2_next.w = (reg_load3.y >> 16);
}

template <>
__device__ inline void load_reg_cs<7U, LUT_TYPE::SQ_LUT>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint reg_load = ld_cs((uint *) &compressed[weight_idx]);
        uint reg_load2 = ld_cs((uint *) &compressed[weight_idx + 2]);
        uint reg_load3 = ld_cs((uint *) &compressed[weight_idx + 4]);
        uint reg_load4 = ld_cs((uint *) &compressed[weight_idx + 6]);
        uint reg_load5 = ld_cs((uint *) &compressed[weight_idx + 8]);
        uint reg_load6 = ld_cs((uint *) &compressed[weight_idx + 10]);
        uint reg_load7 = ld_cs((uint *) &compressed[weight_idx + 12]);

        reg_cs_next.x = reg_load;
        reg_cs2_next.x = reg_load2 & 0xffffff;
        reg_cs_next.y = ((reg_load2 >> 24) | (reg_load3 << 8));
        reg_cs2_next.y = ((reg_load3 >> 24) | ((reg_load4 & 0xffff) << 8));
        reg_cs_next.z = ((reg_load4 >> 16) | (reg_load5 << 16));
        reg_cs2_next.z = ((reg_load5 >> 16) | ((reg_load6 & 0xff) << 16));
        reg_cs_next.w = ((reg_load6 >> 8) | (reg_load7 << 24));
        reg_cs2_next.w = (reg_load7 >> 8);
}


template <>
__device__ inline void load_reg_cs<8U, LUT_TYPE::SQ_LUT>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint4 reg_load = ld_cs((uint4 *) &compressed[weight_idx]);
        uint4 reg_load2 = ld_cs((uint4 *) &compressed[weight_idx + 8]);

        reg_cs_next.x = reg_load.x;
        reg_cs2_next.x = reg_load.y;
        reg_cs_next.y =  reg_load.z;
        reg_cs2_next.y = reg_load.w;
        reg_cs_next.z = reg_load2.x;
        reg_cs2_next.z = reg_load2.y;
        reg_cs_next.w = reg_load2.z;
        reg_cs2_next.w = reg_load2.w;
}

template <>
__device__ inline void load_reg_cs<1U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint16_t reg_load = compressed[weight_idx];
        reg_cs_next.x = reg_load & 0xf;
        reg_cs_next.y = (reg_load >> 4) & 0xf;
        reg_cs_next.z = (reg_load >> 8) & 0xf;
        reg_cs_next.w = (reg_load >> 12) & 0xf;
}

template <>
__device__ inline void load_reg_cs<2U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ld_cs((uint *) &compressed[weight_idx]);
        reg_cs_next.x = reg_load & 0xff;
        reg_cs_next.y = (reg_load >> 8) & 0xff;
        reg_cs_next.z = (reg_load >> 16) & 0xff;
        reg_cs_next.w = (reg_load >> 24) & 0xff;
}

template <>
__device__ inline void load_reg_cs<3U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint16_t reg_load2 = compressed[weight_idx + 2];
        reg_cs_next.x = reg_load & 0xfff;
        reg_cs_next.y = (reg_load >> 12) & 0xfff;
        reg_cs_next.z = (reg_load >> 24) | ((reg_load2 & 0xf) << 8);
        reg_cs_next.w = (reg_load2 >> 4);
}

template <>
__device__ inline void load_reg_cs<4U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        ditto2 reg_load = {.u32x2 = ld_cs((uint2 *) &compressed[weight_idx])};
        reg_cs_next.x = reg_load.u16x4.x;
        reg_cs_next.y = reg_load.u16x4.y;
        reg_cs_next.z = reg_load.u16x4.z;
        reg_cs_next.w = reg_load.u16x4.w;
}

template <>
__device__ inline void load_reg_cs<5U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint32_t reg_load2 = ((uint32_t) compressed[weight_idx + 2]) | (((uint32_t) compressed[weight_idx + 3]) << 16);
        uint16_t reg_load3 = compressed[weight_idx + 4];
        reg_cs_next.x = reg_load & 0xfffff;
        reg_cs_next.y = (reg_load >> 20) | ((reg_load2 & 0xff) << 12);
        reg_cs_next.z = (reg_load2 >> 8) & 0xfffff;
        reg_cs_next.w = (reg_load2 >> 28) | (((uint32_t) reg_load3) << 4);
}

template <>
__device__ inline void load_reg_cs<6U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint3 reg_load = ld_cs((uint3 *) &compressed[weight_idx]);
        reg_cs_next.x = reg_load.x & 0xffffff;
        reg_cs_next.y = ((reg_load.x >> 24) | (reg_load.y << 8)) & 0xffffff;
        reg_cs_next.z = ((reg_load.y >> 16) | (reg_load.z << 16)) & 0xffffff;
        reg_cs_next.w = (reg_load.z >> 8) & 0xffffff;
}

template <>
__device__ inline void load_reg_cs<7U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint32_t reg_load2 = ((uint32_t) compressed[weight_idx+2]) | (((uint32_t) compressed[weight_idx + 3]) << 16);
        uint32_t reg_load3 = ((uint32_t) compressed[weight_idx+4]) | (((uint32_t) compressed[weight_idx + 5]) << 16);
        uint16_t reg_load4 = compressed[weight_idx + 6];
        reg_cs_next.x = reg_load & 0xfffffff;
        reg_cs_next.y = ((reg_load >> 28) | ((reg_load2 & 0xffffff) << 4));
        reg_cs_next.z = ((reg_load2 >> 24) | ((reg_load3 & 0xfffff) << 8));
        reg_cs_next.w = (reg_load3 >> 20) | ((((uint32_t) reg_load4) << 12));
}

template <>
__device__ inline void load_reg_cs<8U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint4 reg_load = ld_cs((uint4 *) &compressed[weight_idx]);
        reg_cs_next.x = reg_load.x;
        reg_cs_next.y = reg_load.y;
        reg_cs_next.z = reg_load.z;
        reg_cs_next.w = reg_load.w;
}

template <>
__device__ inline void load_reg_cs<9U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint32_t reg_load2 = ((uint32_t) compressed[weight_idx+2]) | (((uint32_t) compressed[weight_idx + 3]) << 16);
        uint32_t reg_load3 = ((uint32_t) compressed[weight_idx+4]) | (((uint32_t) compressed[weight_idx + 5]) << 16);
        uint32_t reg_load4 = ((uint32_t) compressed[weight_idx+6]) | (((uint32_t) compressed[weight_idx + 7]) << 16);
        uint16_t reg_load5 = compressed[weight_idx + 8];
        reg_cs_next.x = reg_load;
        reg_cs2_next.x = reg_load2 & 0xf;
        reg_cs_next.y = ((reg_load2 >> 4) | (reg_load3 << 28));
        reg_cs2_next.y = ((reg_load3 >> 4) & 0xf);
        reg_cs_next.z = ((reg_load3 >> 8) | (reg_load4 << 24));
        reg_cs2_next.z = ((reg_load4 >> 8) & 0xf);
        reg_cs_next.w = ((reg_load4 >> 12) | (((uint32_t) reg_load5) << 20));
        reg_cs2_next.w = (((uint32_t) reg_load5) >> 12);
}

template <>
__device__ inline void load_reg_cs<10U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint reg_load = ld_cs((uint *) &compressed[weight_idx]);
        uint reg_load2 = ld_cs((uint *) &compressed[weight_idx + 2]);
        uint reg_load3 = ld_cs((uint *) &compressed[weight_idx + 4]);
        uint reg_load4 = ld_cs((uint *) &compressed[weight_idx + 6]);
        uint reg_load5 = ld_cs((uint *) &compressed[weight_idx + 8]);

        reg_cs_next.x = reg_load;
        reg_cs2_next.x = reg_load2 & 0xff;
        reg_cs_next.y = ((reg_load2 >> 8) | (reg_load3 << 24));
        reg_cs2_next.y = ((reg_load3 >> 8) & 0xff);
        reg_cs_next.z = ((reg_load3 >> 16) | (reg_load4 << 16));
        reg_cs2_next.z = ((reg_load4 >> 16) & 0xff);
        reg_cs_next.w = ((reg_load4 >> 24) | (reg_load5 << 8));
        reg_cs2_next.w = ((reg_load5 >> 24) & 0xff);
}

template <>
__device__ inline void load_reg_cs<11U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint32_t reg_load2 = ((uint32_t) compressed[weight_idx+2]) | (((uint32_t) compressed[weight_idx + 3]) << 16);
        uint32_t reg_load3 = ((uint32_t) compressed[weight_idx+4]) | (((uint32_t) compressed[weight_idx + 5]) << 16);
        uint32_t reg_load4 = ((uint32_t) compressed[weight_idx+6]) | (((uint32_t) compressed[weight_idx + 7]) << 16);
        uint32_t reg_load5 = ((uint32_t) compressed[weight_idx+8]) | (((uint32_t) compressed[weight_idx + 9]) << 16);
        uint16_t reg_load6 = compressed[weight_idx + 10];
        reg_cs_next.x = reg_load;
        reg_cs2_next.x = reg_load2 & 0xfff;
        reg_cs_next.y = ((reg_load2 >> 12) | (reg_load3 << 20));
        reg_cs2_next.y = ((reg_load3 >> 12) & 0xfff);
        reg_cs_next.z = ((reg_load3 >> 24) | (reg_load4 << 8));
        reg_cs2_next.z = ((reg_load4 >> 24) | ((reg_load5 & 0xf) << 8));
        reg_cs_next.w = ((reg_load5 >> 4) | (((uint32_t) reg_load6) << 28));
        reg_cs2_next.w = (((uint32_t) reg_load6) >> 4);
}

template <>
__device__ inline void load_reg_cs<12U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint2 reg_load = ld_cs((uint2 *) &compressed[weight_idx]);
        uint2 reg_load2 = ld_cs((uint2 *) &compressed[weight_idx + 4]);
        uint2 reg_load3 = ld_cs((uint2 *) &compressed[weight_idx + 8]);

        reg_cs_next.x = reg_load.x;
        reg_cs2_next.x = reg_load.y & 0xffff;
        reg_cs_next.y = ((reg_load.y >> 16) | (reg_load2.x << 16));
        reg_cs2_next.y = (reg_load2.x >> 16);
        reg_cs_next.z = reg_load2.y;
        reg_cs2_next.z = (reg_load3.x & 0xffff);
        reg_cs_next.w = ((reg_load3.x >> 16) | (reg_load3.y << 16));
        reg_cs2_next.w = (reg_load3.y >> 16);
}

template <>
__device__ inline void load_reg_cs<13U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint32_t reg_load2 = ((uint32_t) compressed[weight_idx+2]) | (((uint32_t) compressed[weight_idx + 3]) << 16);
        uint32_t reg_load3 = ((uint32_t) compressed[weight_idx+4]) | (((uint32_t) compressed[weight_idx + 5]) << 16);
        uint32_t reg_load4 = ((uint32_t) compressed[weight_idx+6]) | (((uint32_t) compressed[weight_idx + 7]) << 16);
        uint32_t reg_load5 = ((uint32_t) compressed[weight_idx+8]) | (((uint32_t) compressed[weight_idx + 9]) << 16);
        uint32_t reg_load6 = ((uint32_t) compressed[weight_idx+10]) | (((uint32_t) compressed[weight_idx + 11]) << 16);
        uint16_t reg_load7 = compressed[weight_idx + 12];
        reg_cs_next.x = reg_load;
        reg_cs2_next.x = reg_load2 & 0xfffff;
        reg_cs_next.y = ((reg_load2 >> 20) | (reg_load3 << 12));
        reg_cs2_next.y = ((reg_load3 >> 20) | ((reg_load4 & 0xff) << 12));
        reg_cs_next.z = ((reg_load4 >> 8) | (reg_load5 << 24));
        reg_cs2_next.z = ((reg_load5 >> 8) & 0xfffff);
        reg_cs_next.w = ((reg_load5 >> 28) | (reg_load6 << 4));
        reg_cs2_next.w = ((reg_load6 >> 28) | (((uint32_t) reg_load7) << 4));
}

template <>
__device__ inline void load_reg_cs<14U, LUT_TYPE::VQ_LUT_2>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint reg_load = ld_cs((uint *) &compressed[weight_idx]);
        uint reg_load2 = ld_cs((uint *) &compressed[weight_idx + 2]);
        uint reg_load3 = ld_cs((uint *) &compressed[weight_idx + 4]);
        uint reg_load4 = ld_cs((uint *) &compressed[weight_idx + 6]);
        uint reg_load5 = ld_cs((uint *) &compressed[weight_idx + 8]);
        uint reg_load6 = ld_cs((uint *) &compressed[weight_idx + 10]);
        uint reg_load7 = ld_cs((uint *) &compressed[weight_idx + 12]);

        reg_cs_next.x = reg_load;
        reg_cs2_next.x = reg_load2 & 0xffffff;
        reg_cs_next.y = ((reg_load2 >> 24) | (reg_load3 << 8));
        reg_cs2_next.y = ((reg_load3 >> 24) | ((reg_load4 & 0xffff) << 8));
        reg_cs_next.z = ((reg_load4 >> 16) | (reg_load5 << 16));
        reg_cs2_next.z = ((reg_load5 >> 16) | ((reg_load6 & 0xff) << 16));
        reg_cs_next.w = ((reg_load6 >> 8) | (reg_load7 << 24));
        reg_cs2_next.w = (reg_load7 >> 8);
}

template <>
__device__ inline void load_reg_cs<2U, LUT_TYPE::VQ_LUT_4>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint16_t reg_load = compressed[weight_idx];
        reg_cs_next.x = reg_load & 0xf;
        reg_cs_next.y = (reg_load >> 4) & 0xf;
        reg_cs_next.z = (reg_load >> 8) & 0xf;
        reg_cs_next.w = (reg_load >> 12) & 0xf;
}

template <>
__device__ inline void load_reg_cs<4U, LUT_TYPE::VQ_LUT_4>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ld_cs((uint *) &compressed[weight_idx]);
        reg_cs_next.x = reg_load & 0xff;
        reg_cs_next.y = (reg_load >> 8) & 0xff;
        reg_cs_next.z = (reg_load >> 16) & 0xff;
        reg_cs_next.w = (reg_load >> 24) & 0xff;
}

template <>
__device__ inline void load_reg_cs<6U, LUT_TYPE::VQ_LUT_4>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint16_t reg_load2 = compressed[weight_idx + 2];
        reg_cs_next.x = reg_load & 0xfff;
        reg_cs_next.y = (reg_load >> 12) & 0xfff;
        reg_cs_next.z = (reg_load >> 24) | ((reg_load2 & 0xf) << 8);
        reg_cs_next.w = (reg_load2 >> 4);
}

template <>
__device__ inline void load_reg_cs<8U, LUT_TYPE::VQ_LUT_4>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        ditto2 reg_load = {.u32x2 = ld_cs((uint2 *) &compressed[weight_idx])};
        reg_cs_next.x = reg_load.u16x4.x;
        reg_cs_next.y = reg_load.u16x4.y;
        reg_cs_next.z = reg_load.u16x4.z;
        reg_cs_next.w = reg_load.u16x4.w;
}

template <>
__device__ inline void load_reg_cs<10U, LUT_TYPE::VQ_LUT_4>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint32_t reg_load2 = ((uint32_t) compressed[weight_idx + 2]) | (((uint32_t) compressed[weight_idx + 3]) << 16);
        uint16_t reg_load3 = compressed[weight_idx + 4];
        reg_cs_next.x = reg_load & 0xfffff;
        reg_cs_next.y = (reg_load >> 20) | ((reg_load2 & 0xff) << 12);
        reg_cs_next.z = (reg_load2 >> 8) & 0xfffff;
        reg_cs_next.w = (reg_load2 >> 28) | (((uint32_t) reg_load3) << 4);
}

template <>
__device__ inline void load_reg_cs<12U, LUT_TYPE::VQ_LUT_4>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint3 reg_load = ld_cs((uint3 *) &compressed[weight_idx]);
        reg_cs_next.x = reg_load.x & 0xffffff;
        reg_cs_next.y = ((reg_load.x >> 24) | (reg_load.y << 8)) & 0xffffff;
        reg_cs_next.z = ((reg_load.y >> 16) | (reg_load.z << 16)) & 0xffffff;
        reg_cs_next.w = (reg_load.z >> 8) & 0xffffff;
}

template <>
__device__ inline void load_reg_cs<14U, LUT_TYPE::VQ_LUT_4>(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
        uint32_t reg_load = ((uint32_t) compressed[weight_idx]) | (((uint32_t) compressed[weight_idx + 1]) << 16);
        uint32_t reg_load2 = ((uint32_t) compressed[weight_idx+2]) | (((uint32_t) compressed[weight_idx + 3]) << 16);
        uint32_t reg_load3 = ((uint32_t) compressed[weight_idx+4]) | (((uint32_t) compressed[weight_idx + 5]) << 16);
        uint16_t reg_load4 = compressed[weight_idx + 6];
        reg_cs_next.x = reg_load & 0xfffffff;
        reg_cs_next.y = ((reg_load >> 28) | ((reg_load2 & 0xffffff) << 4));
        reg_cs_next.z = ((reg_load2 >> 24) | ((reg_load3 & 0xfffff) << 8));
        reg_cs_next.w = (reg_load3 >> 20) | ((((uint32_t) reg_load4) << 12));
}


template <uint32_t R, uint32_t M, uint32_t N, uint32_t K, Q_MODE Q, LUT_TYPE L, LOAD_TYPE LD>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress_gemm(
    float *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const half2 *__restrict__ x,
    const half *__restrict__ codebook
) {
    
    // ** cursed indexing math **

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    uint32_t r_mask = (1 << R) - 1;
    uint32_t r_mask2 = (1 << (2 * R)) - 1;
    const int vec_sz = (1 << ((int)L));

    constexpr uint32_t tileCountM = M / MMA_M;
    constexpr uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;


    // ** load codebook **
    extern __shared__ __align__(128) half2 smem_codebook[];
    
    constexpr int num_centroids = 1 << R;
    constexpr int num_centroids_sq = num_centroids * num_centroids;

    int offset_codebook;
    if constexpr(LD == LOAD_TYPE::DUP) {
        if constexpr(Q == Q_MODE::PER_COLUMN) {
            offset_codebook = (MMA_M * 2 * num_centroids_sq * vec_sz);
        } else if constexpr(Q == Q_MODE::PER_TENSOR) {
            offset_codebook = num_centroids_sq * vec_sz;
        } 
    } else {
        if constexpr(Q == Q_MODE::PER_COLUMN) {
            offset_codebook = (MMA_M * 2 * num_centroids * vec_sz) / 2;
        } else if constexpr(Q == Q_MODE::PER_TENSOR) {
            offset_codebook = (num_centroids * vec_sz) / 2;
        } 
    }

    ditto2 (*x_buf)[BLOCK_SIZE / WARP_SIZE][4][4] =
        reinterpret_cast<decltype(x_buf)>(smem_codebook + offset_codebook);

    float (*reduce_gather)[2][N][16] =
        reinterpret_cast<decltype(reduce_gather)>(smem_codebook + offset_codebook); // reuse

    static_assert (tileCountM % 2 == 0);
    constexpr uint32_t m_per_block = ROUND_UP(tileCountM, (2 * BLOCK_COUNT));
    // tiles are iterated along k in groups of 2
    //static_assert (tileCountK >= warps_per_block * 2);
    constexpr uint32_t k_per_block = tileCountK / (warps_per_block * 4) * 2;
    // we sync at ki%4==0, make sure this is safe
    //constexpr bool enable_kim4_sync = !(M == 4096 && K==4096) && (tileCountK % (warps_per_block * 2) == 0 || k_per_block % 4 != 0);
    // some warps have more k tiles
    static_assert((tileCountK % (warps_per_block * 4)) % 4 == 0);
    uint32_t this_warp_k = (warpId < (tileCountK % (warps_per_block * 4)) / 4) ? k_per_block + 2 : k_per_block;

    constexpr uint32_t u16_per_compressed_tile = MMA_M * MMA_K * R / (16 * vec_sz);
    static_assert((MMA_M * MMA_K * R) % (16 * vec_sz) == 0);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block = u16_per_compressed_tile * weight_block; // one tile block per warp at a time
    constexpr uint32_t weight_step = warps_per_block * u16_per_tile_block;
    constexpr uint32_t weight_row_step = tileCountK * u16_per_compressed_tile * 2;  // 2 rows of tiles
    
    
    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        // ** load weight, start loop **
        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2 + laneId * (u16_per_tile_block / WARP_SIZE);
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        load_reg_cs<R, L>((const uint16_t * __restrict__) compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs;
        uint4 reg_cs2;

        // define acc
        float4 reg_p[2] = {};

        uint32_t x_idx = warpId * f16x2_per_x_tile * 4 + laneId;  // every warp does 4 k tiles per iteration
        uint32_t x_idx_step = warps_per_block * f16x2_per_x_tile * 4;

        // ** load codebook **
        if constexpr( (Q == Q_MODE::PER_TENSOR)) {
            if (mi == 0) {
                if constexpr(LD == LOAD_TYPE::DUP) {
                    int loop_count = max(num_centroids_sq / BLOCK_SIZE, 1);
                    if (threadIdx.x < num_centroids_sq){
                        const int xx = threadIdx.x % num_centroids;
                        const half fracX = codebook[xx];
                        for (int li = 0; li < loop_count; li++){
                            int cur_idx = threadIdx.x + li * BLOCK_SIZE;
                            const int yy = cur_idx / num_centroids;
                            const half fracY = codebook[yy];
                            smem_codebook[yy * num_centroids | xx] = make_half2(fracX, fracY);
                        }
                    }
                } else {
                    int loop_count = max(num_centroids / BLOCK_SIZE, 1);
                    if (threadIdx.x < num_centroids){       
                        for (int li = 0; li < loop_count; li++){
                            int cur_idx = threadIdx.x + li * BLOCK_SIZE;
                            if constexpr(L == LUT_TYPE::SQ_LUT) {
                                ((half*) smem_codebook)[cur_idx] = ((half*) codebook)[cur_idx];
                            } else if constexpr(L == LUT_TYPE::VQ_LUT_2) {
                                smem_codebook[cur_idx] = ((half2*) codebook)[cur_idx];
                            } else if constexpr(L == LUT_TYPE::VQ_LUT_4) {
                                ((float2*)smem_codebook)[cur_idx] = ((float2*) codebook)[cur_idx];
                            } else if constexpr(L == LUT_TYPE::VQ_LUT_8) {
                                ((float4*)smem_codebook)[cur_idx] = ((float4*) codebook)[cur_idx];
                            }
                        }
                    }
                }
            } 
        } else if constexpr(Q == Q_MODE::PER_COLUMN) {
            // implement only SQ_LUT for PER_COLUMN
            if constexpr(LD == LOAD_TYPE::DUP) {
                int m_start = tileIdM * 2 * MMA_M;
                int loop_count = max((MMA_M * 2 * num_centroids_sq) / BLOCK_SIZE, 1);
                if (threadIdx.x < MMA_M * 2 * num_centroids_sq){
                    const int xx = threadIdx.x % num_centroids;
                    const int yy = (threadIdx.x / num_centroids) % num_centroids;
                    for (int li = 0; li < loop_count; li++) {
                        int cur_idx = threadIdx.x + li * BLOCK_SIZE;
                        int m_res = cur_idx / num_centroids_sq;
                        int m_idx = m_start + m_res;
                        const half fracX = codebook[m_idx * num_centroids | xx];
                        const half fracY = codebook[m_idx * num_centroids | yy];
                        smem_codebook[((m_res * num_centroids_sq) | (yy * num_centroids)) | xx] = make_half2(fracX, fracY);
                    }
                }
            } else {
                int m_start = tileIdM * 2 * MMA_M;
                int loop_count = max((MMA_M * 2 * num_centroids) / BLOCK_SIZE, 1);
                if (threadIdx.x < MMA_M * 2 * num_centroids){
                    for (int li = 0; li < loop_count; li++) {
                        int cur_idx = threadIdx.x + li * BLOCK_SIZE;
                        int m_res = cur_idx / num_centroids;
                        int m_idx = m_start + m_res;
                        int xx = cur_idx % num_centroids;
                        ((half*)smem_codebook)[((m_res * num_centroids) | xx)] = codebook[m_idx * num_centroids | xx];
                    }
                }
            }
        }
        __syncthreads();

        uint32_t x_line;
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            if (ki + 1 != this_warp_k && ki % 2 == 1) {
                weight_idx += weight_step * 2; // fixme: this costs 10GB/s
            }
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            load_reg_cs<R, L>((const uint16_t * __restrict__) compressed, weight_idx + (1 - ki % 2) * u16_per_tile_block, laneId, reg_cs_next, reg_cs2_next);

            if (ki % 2 == 0) {
                __syncwarp();
#pragma unroll
                for (uint32_t i = 0; i < N; i++) {
                    x_buf[i][warpId][laneId / 8][laneId % 4].u32[(laneId % 8) / 4] = ld_x(reinterpret_cast<const uint32_t *>(x) + x_idx + i * K / 2);
                }
                __syncwarp();
                x_idx += x_idx_step;
            }

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {
                // load activations
                // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
                ditto2 reg_a;
                ditto2 reg_a2;        
                if constexpr(N <= 8) {
                    if (laneId < 4 * N) {
                        reg_a.u32x2 = x_buf[laneId / 4][warpId][ki % 2 * 2 + subki][laneId % 4].u32x2;
                    }
                } else if constexpr(N <= 16) {
                    reg_a.u32x2 = x_buf[laneId / 4][warpId][ki % 2 * 2 + subki][laneId % 4].u32x2;
                    if (laneId < 2 * N) {
                        reg_a2.u32x2 = x_buf[8 + laneId / 4][warpId][ki % 2 * 2 + subki][laneId % 4].u32x2;
                    }
                }

#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi++) {
                    uint32_t reg_c, reg_c2;
                    uint64_t reg_c_64;
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    int cur_m_res = 0;
                    // ** decode weights **
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    if constexpr(L == LUT_TYPE::SQ_LUT || L == LUT_TYPE::VQ_LUT_2) {
                        if constexpr(((R >= 5) && (L == LUT_TYPE::SQ_LUT)) || ((R >= 9) && (L == LUT_TYPE::VQ_LUT_2))) {
                            reg_c_64 = ((uint64_t) reg_c) | (((uint64_t) reg_c2) << 32);
                        }

                        if constexpr(((L == LUT_TYPE::SQ_LUT) && (LD == LOAD_TYPE::DUP)) || (L == LUT_TYPE::VQ_LUT_2)) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; j += 1) {
                                uint32_t idx;
                                if constexpr(L == LUT_TYPE::SQ_LUT) {
                                    if constexpr(R >= 5) {
                                        idx = (reg_c_64 >> (2 * R * j)) & r_mask2;
                                    } else {
                                        idx = (reg_c >> (2 * R * j)) & r_mask2;
                                    }
                                } else if constexpr(L == LUT_TYPE::VQ_LUT_2) {
                                    if constexpr(R >= 9) {
                                        idx = (reg_c_64 >> (R * j)) & r_mask;
                                    } else {
                                        idx = (reg_c >> (R * j)) & r_mask;
                                    }
                                }

                                if constexpr(Q == Q_MODE::PER_TENSOR) {
                                    reg_w.f16x2[j] = smem_codebook[idx];
                                } else if constexpr(Q == Q_MODE::PER_COLUMN) {
                                    cur_m_res = submi * MMA_M + (laneId / 4) + (j % 2) * 8;
                                    if constexpr(L == LUT_TYPE::SQ_LUT) {
                                        reg_w.f16x2[j] = smem_codebook[(cur_m_res * num_centroids_sq) | idx];
                                    } 
                                } 
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t j = 0; j < 8; j += 1) {
                                uint32_t idx;
                                if constexpr(L == LUT_TYPE::SQ_LUT) {
                                    if constexpr(R >= 5) {
                                        idx = (reg_c_64 >> (R * j)) & r_mask;
                                    } else {
                                        idx = (reg_c >> (R * j)) & r_mask;
                                    }
                                }
                                if constexpr(Q == Q_MODE::PER_TENSOR) {
                                    reg_w.f16[j] = ((half*)smem_codebook)[idx];
                                } else if constexpr(Q == Q_MODE::PER_COLUMN) {
                                    cur_m_res = submi * MMA_M + (laneId / 4) + ((j/2) % 2) * 8;
                                    if constexpr(L == LUT_TYPE::SQ_LUT) {
                                        reg_w.f16[j] = ((half*)smem_codebook)[(cur_m_res * num_centroids) | idx];
                                    } 
                                } 
                            }
                        }
                    } 
                    asm volatile (
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                            " {%0, %1, %2, %3},"
                            " {%4, %5, %6, %7},"
                            " {%8, %9},"
                            " {%0, %1, %2, %3};"
                            : "+f"(reg_p[submi].x), "+f"(reg_p[submi].y), "+f"(reg_p[submi].z), "+f"(reg_p[submi].w)
                            :  "r"(reg_w.u32[0]), "r"(reg_w.u32[1]), "r"(reg_w.u32[2]), "r"(reg_w.u32[3]),
                            "r"(reg_a.u32[0]), "r"(reg_a.u32[1])
                    );
                }

            }
            if (ki % 2 == 0) {
#pragma unroll
                for (uint32_t i = 0; i < N; i++) {
                    prefetch((uint32_t *) (x + x_idx + x_idx_step*4 + i * K / 2));
                }
            }
        }

        // __shared__ __align__(16 * 8*32) float reduce_gather[BLOCK_SIZE / WARP_SIZE][2][N][16];
        __syncthreads();
        reduce_gather_routine<N>(reg_p, warpId, laneId, reduce_gather);
        __syncthreads();
        float reduced = 0.0;
        if (warpId < N) {
            int pi = laneId / 16;
            for (int warpi = 0; warpi < BLOCK_SIZE / WARP_SIZE; warpi++) {
                reduced += reduce_gather[warpi][pi][warpId][laneId % 16];
            }

            // TODO: https://forums.developer.nvidia.com/t/can-float4-be-used-for-atomicadd-efficiently/215692
            // two rows at a time
            float *out_tile = out + (tileIdM * 2) * f32_per_out_tile + warpId * M;
            out_tile[laneId] = reduced;
        }
        if constexpr(m_per_block > 1) __syncthreads();
        tileIdM += 1;
    }
}


template <uint32_t R, Q_MODE Q, LUT_TYPE L, LOAD_TYPE LD>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress(
    half2 *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const half *__restrict__ codebook,
    uint32_t M,
    uint32_t K
) {
    
    // ** cursed indexing math **

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    uint32_t r_mask = (1 << R) - 1;
    uint32_t r_mask2 = (1 << (2 * R)) - 1;
    const int vec_sz = (1 << ((int)L));

    uint32_t tileCountM = M / MMA_M;
    uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;
    constexpr int num_centroids = 1 << R;
    constexpr int num_centroids_sq = num_centroids * num_centroids;

    // ** load codebook **
    extern __shared__ __align__(128) half2 smem_codebook[];

    uint32_t m_per_block = ROUND_UP(tileCountM, (2 * BLOCK_COUNT));
    uint32_t k_per_block = tileCountK / (warps_per_block * 4) * 2;
    uint32_t this_warp_k = (warpId < (tileCountK % (warps_per_block * 4)) / 4) ? k_per_block + 2 : k_per_block;

    constexpr uint32_t u16_per_compressed_tile = MMA_M * MMA_K * R / (16 * vec_sz);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block = u16_per_compressed_tile * weight_block; // one tile block per warp at a time
    constexpr uint32_t weight_step = warps_per_block * u16_per_tile_block;
    uint32_t weight_row_step = tileCountK * u16_per_compressed_tile * 2;  // 2 rows of tiles
    
    
    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        // ** load weight, start loop **
        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2 + laneId * (u16_per_tile_block / WARP_SIZE);
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        load_reg_cs<R, L>((const uint16_t * __restrict__) compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs;
        uint4 reg_cs2;

        // ** load codebook **
        if constexpr( (Q == Q_MODE::PER_TENSOR)) {
            if (mi == 0) {
                if constexpr(LD == LOAD_TYPE::DUP) {
                    int loop_count = max(num_centroids_sq / BLOCK_SIZE, 1);
                    if (threadIdx.x < num_centroids_sq){
                        const int xx = threadIdx.x % num_centroids;
                        const half fracX = codebook[xx];
                        for (int li = 0; li < loop_count; li++){
                            int cur_idx = threadIdx.x + li * BLOCK_SIZE;
                            const int yy = cur_idx / num_centroids;
                            const half fracY = codebook[yy];
                            smem_codebook[yy * num_centroids | xx] = make_half2(fracX, fracY);
                        }
                    }
                } else {
                    int loop_count = max(num_centroids / BLOCK_SIZE, 1);
                    if (threadIdx.x < num_centroids){       
                        for (int li = 0; li < loop_count; li++){
                            int cur_idx = threadIdx.x + li * BLOCK_SIZE;
                            if constexpr(L == LUT_TYPE::SQ_LUT) {
                                ((half*) smem_codebook)[cur_idx] = ((half*) codebook)[cur_idx];
                            } else if constexpr(L == LUT_TYPE::VQ_LUT_2) {
                                smem_codebook[cur_idx] = ((half2*) codebook)[cur_idx];
                            } else if constexpr(L == LUT_TYPE::VQ_LUT_4) {
                                ((float2*)smem_codebook)[cur_idx] = ((float2*) codebook)[cur_idx];
                            } else if constexpr(L == LUT_TYPE::VQ_LUT_8) {
                                ((float4*)smem_codebook)[cur_idx] = ((float4*) codebook)[cur_idx];
                            }
                        }
                    }
                }
            } 
        } else if constexpr(Q == Q_MODE::PER_COLUMN) {
            // implement only SQ_LUT for PER_COLUMN
            if constexpr(LD == LOAD_TYPE::DUP) {
                int m_start = tileIdM * 2 * MMA_M;
                int loop_count = max((MMA_M * 2 * num_centroids_sq) / BLOCK_SIZE, 1);
                if (threadIdx.x < MMA_M * 2 * num_centroids_sq){
                    const int xx = threadIdx.x % num_centroids;
                    const int yy = (threadIdx.x / num_centroids) % num_centroids;
                    for (int li = 0; li < loop_count; li++) {
                        int cur_idx = threadIdx.x + li * BLOCK_SIZE;
                        int m_res = cur_idx / num_centroids_sq;
                        int m_idx = m_start + m_res;
                        const half fracX = codebook[m_idx * num_centroids | xx];
                        const half fracY = codebook[m_idx * num_centroids | yy];
                        smem_codebook[((m_res * num_centroids_sq) | (yy * num_centroids)) | xx] = make_half2(fracX, fracY);
                    }
                }
            } else {
                int m_start = tileIdM * 2 * MMA_M;
                int loop_count = max((MMA_M * 2 * num_centroids) / BLOCK_SIZE, 1);
                if (threadIdx.x < MMA_M * 2 * num_centroids){
                    for (int li = 0; li < loop_count; li++) {
                        int cur_idx = threadIdx.x + li * BLOCK_SIZE;
                        int m_res = cur_idx / num_centroids;
                        int m_idx = m_start + m_res;
                        int xx = cur_idx % num_centroids;
                        ((half*)smem_codebook)[((m_res * num_centroids) | xx)] = codebook[m_idx * num_centroids | xx];
                    }
                }
            }
        }
        __syncthreads();

        uint32_t x_line;
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            if (ki + 1 != this_warp_k && ki % 2 == 1) {
                weight_idx += weight_step * 2; // fixme: this costs 10GB/s
            }
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            load_reg_cs<R, L>((const uint16_t * __restrict__) compressed, weight_idx + (1 - ki % 2) * u16_per_tile_block, laneId, reg_cs_next, reg_cs2_next);

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {

#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi++) {
                    uint32_t reg_c, reg_c2;
                    uint64_t reg_c_64;
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    int cur_m_res = 0;
                    // ** decode weights **
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    if constexpr(L == LUT_TYPE::SQ_LUT || L == LUT_TYPE::VQ_LUT_2) {
                        if constexpr(((R >= 5) && (L == LUT_TYPE::SQ_LUT)) || ((R >= 9) && (L == LUT_TYPE::VQ_LUT_2))) {
                            reg_c_64 = ((uint64_t) reg_c) | (((uint64_t) reg_c2) << 32);
                        }

                        if constexpr(((L == LUT_TYPE::SQ_LUT) && (LD == LOAD_TYPE::DUP)) || (L == LUT_TYPE::VQ_LUT_2)) {
                            #pragma unroll
                            for (uint32_t j = 0; j < 4; j += 1) {
                                uint32_t idx;
                                if constexpr(L == LUT_TYPE::SQ_LUT) {
                                    if constexpr(R >= 5) {
                                        idx = (reg_c_64 >> (2 * R * j)) & r_mask2;
                                    } else {
                                        idx = (reg_c >> (2 * R * j)) & r_mask2;
                                    }
                                } else if constexpr(L == LUT_TYPE::VQ_LUT_2) {
                                    if constexpr(R >= 9) {
                                        idx = (reg_c_64 >> (R * j)) & r_mask;
                                    } else {
                                        idx = (reg_c >> (R * j)) & r_mask;
                                    }
                                }

                                if constexpr(Q == Q_MODE::PER_TENSOR) {
                                    reg_w.f16x2[j] = smem_codebook[idx];
                                } else if constexpr(Q == Q_MODE::PER_COLUMN) {
                                    cur_m_res = submi * MMA_M + (laneId / 4) + (j % 2) * 8;
                                    if constexpr(L == LUT_TYPE::SQ_LUT) {
                                        reg_w.f16x2[j] = smem_codebook[(cur_m_res * num_centroids_sq) | idx];
                                    } 
                                }
                            }
                        } else {
                            #pragma unroll
                            for (uint32_t j = 0; j < 8; j += 1) {
                                uint32_t idx;
                                if constexpr(L == LUT_TYPE::SQ_LUT) {
                                    if constexpr(R >= 5) {
                                        idx = (reg_c_64 >> (R * j)) & r_mask;
                                    } else {
                                        idx = (reg_c >> (R * j)) & r_mask;
                                    }
                                }
                                if constexpr(Q == Q_MODE::PER_TENSOR) {
                                    reg_w.f16[j] = ((half*)smem_codebook)[idx];
                                } else if constexpr(Q == Q_MODE::PER_COLUMN) {
                                    cur_m_res = submi * MMA_M + (laneId / 4) + ((j/2) % 2) * 8;
                                    if constexpr(L == LUT_TYPE::SQ_LUT) {
                                        reg_w.f16[j] = ((half*)smem_codebook)[(cur_m_res * num_centroids) | idx];
                                    } 
                                } 
                            }
                        }
                    } 
                    for (uint32_t j = 0; j < 4; j += 1) {
                        // write to out
                        // m tile idx : tileIdM * 2 + submi
                        // k tile idx : ((ki / 2) * 4) * warps_per_block + warpId * 4 + (ki % 2) * 2 + subki
                        // m idx : m_tile_id * 16 + laneId / 8 + (j % 2) * 8
                        // k idx : k_tile_id * 8 + laneId % 4 + (j / 2) * 4
                        int m_tile_id = tileIdM * 2 + submi;
                        int k_tile_id = ((ki / 2) * 4) * warps_per_block + warpId * 4 + (ki % 2) * 2 + subki;
                        int m_idx = m_tile_id * 16 + laneId / 4 + (j % 2) * 8;
                        int k_idx = k_tile_id * 8 + laneId % 4 + (j / 2) * 4;
                        out[m_idx * K / 2 + k_idx] = reg_w.f16x2[j];
                    }
                }
            }
        }
        tileIdM += 1;
    }
}

// R: bits per weight
// V: log2(VQ dimension)
template <uint32_t R, uint32_t M, uint32_t N, uint32_t K, Q_MODE Q, LUT_TYPE L, LOAD_TYPE LD>
__host__ static void decompress_gemm_ptr(
    float *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed,    // m-by-k
    const half2 * __restrict__ x,               // k-by-n
    const half * __restrict__ codebook,
    CUstream_st *stream
) {
    static_assert(Q == Q_MODE::PER_TENSOR || Q == Q_MODE::PER_COLUMN, "Quantization mode = PER_TENSOR, PER_COLUMN for now");
    static_assert(L == LUT_TYPE::SQ_LUT || L == LUT_TYPE::VQ_LUT_2 || L == LUT_TYPE::VQ_LUT_4 || L == LUT_TYPE::VQ_LUT_8, "LUT type = SQ_LUT, VQ_LUT_2, VQ_LUT_4, VQ_LUT_8 for now");

    static_assert(M % MMA_M == 0);
    static_assert(N <= 8);
    static_assert(K % MMA_K == 0);

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;

    constexpr int num_centroids_sq = (1 << R) * (1 << R);
    constexpr int num_centroids = (1 << R);
    constexpr int vec_sz = (1 << ((int)L));

    constexpr uint32_t smemReduceGatherSize = N * BLOCK_SIZE * sizeof(float);
    if constexpr(LD == LOAD_TYPE::DUP) {
        if constexpr(Q == Q_MODE::PER_COLUMN) {
            constexpr int smemTotalSize = (MMA_M * 2 * num_centroids_sq * 2 * vec_sz) * sizeof(half) + smemReduceGatherSize;
            cudaFuncSetAttribute(kernel_decompress_gemm<R, M, N, K, Q, L, LD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smemTotalSize);
            kernel_decompress_gemm<R, M, N, K, Q, L, LD><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, x, codebook);
        } else if constexpr(Q == Q_MODE::PER_TENSOR) {
            constexpr int smemTotalSize = num_centroids_sq * 2 * vec_sz * sizeof(half) + smemReduceGatherSize;
            cudaFuncSetAttribute(kernel_decompress_gemm<R, M, N, K, Q, L, LD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smemTotalSize);
            kernel_decompress_gemm<R, M, N, K, Q, L, LD><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, x, codebook);
        } 
    } else {
        if constexpr(Q == Q_MODE::PER_COLUMN) {
            constexpr int smemTotalSize = (MMA_M * 2 * num_centroids * vec_sz) * sizeof(half) + smemReduceGatherSize;
            cudaFuncSetAttribute(kernel_decompress_gemm<R, M, N, K, Q, L, LD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smemTotalSize);
            kernel_decompress_gemm<R, M, N, K, Q, L, LD><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, x, codebook);
        } else if constexpr(Q == Q_MODE::PER_TENSOR) {
            constexpr int smemTotalSize = num_centroids * vec_sz * sizeof(half) + smemReduceGatherSize;
            cudaFuncSetAttribute(kernel_decompress_gemm<R, M, N, K, Q, L, LD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smemTotalSize);
            kernel_decompress_gemm<R, M, N, K, Q, L, LD><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, x, codebook);
        } 
    }
    gpuErrchk(cudaPeekAtLastError());
}


// R: bits per weight
// V: log2(VQ dimension)
template <uint32_t R, Q_MODE Q, LUT_TYPE L, LOAD_TYPE LD>
__host__ static void decompress_ptr(
    half2 *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed,    // m-by-k
    const half * __restrict__ codebook,
    uint32_t M,
    uint32_t K,
    CUstream_st *stream
) {
    static_assert(Q == Q_MODE::PER_TENSOR || Q == Q_MODE::PER_COLUMN, "Quantization mode = PER_TENSOR, PER_COLUMN for now");
    static_assert(L == LUT_TYPE::SQ_LUT || L == LUT_TYPE::VQ_LUT_2 || L == LUT_TYPE::VQ_LUT_4 || L == LUT_TYPE::VQ_LUT_8, "LUT type = SQ_LUT, VQ_LUT_2, VQ_LUT_4, VQ_LUT_8 for now");

    // static_assert(M % MMA_M == 0);
    // static_assert(K % MMA_K == 0);

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;

    constexpr int num_centroids_sq = (1 << R) * (1 << R);
    constexpr int num_centroids = (1 << R);
    constexpr int vec_sz = (1 << ((int)L));

    if constexpr(LD == LOAD_TYPE::DUP) {
        if constexpr(Q == Q_MODE::PER_COLUMN) {
            constexpr int smemTotalSize = (MMA_M * 2 * num_centroids_sq * 2 * vec_sz) * sizeof(half);
            cudaFuncSetAttribute(kernel_decompress<R, Q, L, LD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smemTotalSize);
            kernel_decompress<R, Q, L, LD><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, codebook, M, K);
        } else if constexpr(Q == Q_MODE::PER_TENSOR) {
            constexpr int smemTotalSize = num_centroids_sq * 2 * vec_sz * sizeof(half);
            cudaFuncSetAttribute(kernel_decompress<R, Q, L, LD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smemTotalSize);
            kernel_decompress<R, Q, L, LD><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, codebook, M, K);
        } 
    } else {
        if constexpr(Q == Q_MODE::PER_COLUMN) {
            constexpr int smemTotalSize = (MMA_M * 2 * num_centroids * vec_sz) * sizeof(half);
            cudaFuncSetAttribute(kernel_decompress<R, Q, L, LD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smemTotalSize);
            kernel_decompress<R, Q, L, LD><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, codebook, M, K);
        } else if constexpr(Q == Q_MODE::PER_TENSOR) {
            constexpr int smemTotalSize = num_centroids * vec_sz * sizeof(half);
            cudaFuncSetAttribute(kernel_decompress<R, Q, L, LD>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smemTotalSize);
            kernel_decompress<R, Q, L, LD><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, codebook, M, K);
        } 
    }
    gpuErrchk(cudaPeekAtLastError());
}

