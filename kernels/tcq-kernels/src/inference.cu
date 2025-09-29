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

template <uint32_t R>
__device__ inline void load_reg_cs(const uint16_t *__restrict__ compressed, int weight_idx, uint32_t laneId, uint4 &reg_cs_next, uint4 &reg_cs2_next) {
    if constexpr(R == 2) {
        ditto reg_load = {.u32 = ld_cs((uint32_t *) &compressed[weight_idx])};
        uint32_t next = __shfl_sync(FULL_MASK, reg_load.u32, laneId + 1);
        uint32_t next2 = __shfl_sync(FULL_MASK, reg_load.u32, laneId + 2);
        uint32_t packed = __byte_perm(next2, next, 0x5410);
        uint32_t packed2 = __byte_perm(next2, next, 0x7632);
        reg_cs_next.x = __byte_perm(packed, reg_load.u32, 0x0420);
        reg_cs_next.y = __byte_perm(packed, reg_load.u32, 0x0531);
        reg_cs_next.z = __byte_perm(packed2, reg_load.u32, 0x0620);
        reg_cs_next.w = __byte_perm(packed2, reg_load.u32, 0x0731);
    } else if constexpr(R == 3) {
        ditto reg_load;
        reg_load.u16x2.x = compressed[weight_idx];
        reg_load.u16x2.y = compressed[weight_idx + 1];
        uint16_t reg_load2 = compressed[weight_idx + 2];

        ditto2 reg_12;
        reg_12.u16x4.x = reg_load.u32 & 0xfff;
        reg_12.u16x4.y = (reg_load.u32 >> 12) & 0xfff;
        reg_12.u16x4.z = ((reg_load.u32 >> 24) & 0xff) | ((reg_load2 & 0xf) << 8);
        reg_12.u16x4.w = (reg_load2 >> 4) & 0xfff;

        // upper 8bits
        uint32_t upper = __byte_perm(reg_12.u32x2.x, reg_12.u32x2.y, 0x7531);
        uint32_t next_upper = __shfl_sync(FULL_MASK, upper, laneId + 1);

        // merge upper bits to send
        ditto2 merged;
        merged.u16x4.x = (reg_12.u16x4.x << 4) | ((next_upper) & 0xf);
        merged.u16x4.y = (reg_12.u16x4.y << 4) | ((next_upper >> 8) & 0xf);
        merged.u16x4.z = (reg_12.u16x4.z << 4) | ((next_upper >> 16) & 0xf);
        merged.u16x4.w = (reg_12.u16x4.w << 4) | ((next_upper >> 24) & 0xf);

        uint32_t next = __shfl_sync(FULL_MASK, merged.u32x2.x, laneId + 1);
        uint32_t next2 = __shfl_sync(FULL_MASK, merged.u32x2.y, laneId + 1);
        reg_cs_next.x = __byte_perm(next, reg_12.u32x2.x, 0x5410);
        reg_cs_next.y = __byte_perm(next, reg_12.u32x2.x, 0x7632);
        reg_cs_next.z = __byte_perm(next2, reg_12.u32x2.y, 0x5410);
        reg_cs_next.w = __byte_perm(next2, reg_12.u32x2.y, 0x7632);
    } else if constexpr(R == 4) {
        ditto2 reg_load = {.u32x2 = ld_cs((uint2 *) &compressed[weight_idx])};
        uint32_t next1 = __shfl_sync(FULL_MASK, reg_load.u32x2.x, laneId + 1);
        uint32_t next2 = __shfl_sync(FULL_MASK, reg_load.u32x2.y, laneId + 1);
        reg_cs_next.x = __byte_perm(next1, reg_load.u32x2.x, 0x5410);
        reg_cs_next.y = __byte_perm(next1, reg_load.u32x2.x, 0x7632);
        reg_cs_next.z = __byte_perm(next2, reg_load.u32x2.y, 0x5410);
        reg_cs_next.w = __byte_perm(next2, reg_load.u32x2.y, 0x7632);
    } else if constexpr(R == 5) {
        ditto reg_load, reg_load2;
        reg_load.u16x2.x = compressed[weight_idx];
        reg_load.u16x2.y = compressed[weight_idx + 1];
        reg_load2.u16x2.x = compressed[weight_idx + 2];
        reg_load2.u16x2.y = compressed[weight_idx + 3];
        uint32_t reg_load3 = (uint32_t) compressed[weight_idx + 4];

        uint32_t reg_20_1 = reg_load.u32 & 0xfffff;
        uint32_t reg_20_2 = ((reg_load.u32 >> 20) | (reg_load2.u32 << 12)) & 0xfffff;
        uint32_t reg_20_3 = (reg_load2.u32 >> 8) & 0xfffff;
        uint32_t reg_20_4 = (reg_load2.u32 >> 28) | (reg_load3 << 4);

        // send high 16 bits to prev thread
        uint32_t pack1 = (reg_20_1 >> 8) | ((reg_20_2 & 0xfff00) << 8);
        uint32_t pack3 = (reg_20_3 >> 8) | ((reg_20_4 & 0xfff00) << 8);

        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);

        reg_cs_next.x = ((reg_20_1) << 12) | (next1 & 0xfff);        
        reg_cs_next.y = ((reg_20_2) << 12) | ((next1 >> 16) & 0xfff);        
        reg_cs_next.z = ((reg_20_3) << 12) | (next3 & 0xfff);        
        reg_cs_next.w = ((reg_20_4) << 12) | ((next3 >> 16) & 0xfff);        

    } else if constexpr(R == 6) {
        uint3 reg_load = ld_cs((uint3 *) &compressed[weight_idx]);
        uint32_t reg_load1 = reg_load.x, reg_load2 = reg_load.y, reg_load3 = reg_load.z;

        uint32_t reg_24_1 = reg_load1 & 0xffffff;
        uint32_t reg_24_2 = ((reg_load1 >> 24) | (reg_load2 << 8)) & 0xffffff;
        uint32_t reg_24_3 = ((reg_load2 >> 16) | (reg_load3 << 16)) & 0xffffff;
        uint32_t reg_24_4 = (reg_load3 >> 8) & 0xffffff;

        // send high 16 bits to prev thread
        uint32_t pack1 = (reg_24_1 >> 8) | ((reg_24_2 << 8) & 0xffff0000);
        uint32_t pack3 = (reg_24_3 >> 8) | ((reg_24_4 << 8) & 0xffff0000);

        // receive high 16 bits from next thread
        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);

        reg_cs_next.x = __byte_perm(next1, reg_24_1, 0x6541);
        reg_cs_next.y = __byte_perm(next1, reg_24_2, 0x6543);
        reg_cs_next.z = __byte_perm(next3, reg_24_3, 0x6541);
        reg_cs_next.w = __byte_perm(next3, reg_24_4, 0x6543);

        reg_cs2_next.x = ((next1 >> 6) & 0b11'1111'1111) | (reg_24_1 << 10);
        reg_cs2_next.y = ((next1 >> (6 + 16) & 0b11'1111'1111)) | (reg_24_2 << 10);
        reg_cs2_next.z = ((next3 >> 6) & 0b11'1111'1111) | (reg_24_3 << 10);
        reg_cs2_next.w = ((next3 >> (6 + 16) & 0b11'1111'1111)) | (reg_24_4 << 10);
    } else if constexpr(R == 7) {
        ditto reg_load, reg_load2, reg_load3;
        reg_load.u16x2.x = compressed[weight_idx];
        reg_load.u16x2.y = compressed[weight_idx + 1];
        reg_load2.u16x2.x = compressed[weight_idx + 2];
        reg_load2.u16x2.y = compressed[weight_idx + 3];
        reg_load3.u16x2.x = compressed[weight_idx + 4];
        reg_load3.u16x2.y = compressed[weight_idx + 5];
        uint32_t reg_load4 = (uint32_t) compressed[weight_idx + 6];
        uint32_t reg_28_1 = reg_load.u32 & 0xfffffff;
        uint32_t reg_28_2 = ((reg_load.u32 >> 28) | (reg_load2.u32 << 4)) & 0xfffffff;
        uint32_t reg_28_3 = ((reg_load2.u32 >> 24) | (reg_load3.u32 << 8)) & 0xfffffff;
        uint32_t reg_28_4 = (reg_load3.u32 >> 20) | (reg_load4 << 12);

        // send high 16 bits to prev thread
        uint32_t pack1 = (reg_28_1 >> 12) | ((reg_28_2 << 4) & 0xffff0000);
        uint32_t pack3 = (reg_28_3 >> 12) | ((reg_28_4 << 4) & 0xffff0000);

        // receive high 16 bits from next thread
        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);

        reg_cs_next.x = (reg_28_1 << 2) | ((next1 >> 14) & 0b11);
        reg_cs_next.y = (reg_28_2 << 2) | ((next1 >> 30) & 0b11);
        reg_cs_next.z = (reg_28_3 << 2) | ((next3 >> 14) & 0b11);
        reg_cs_next.w = (reg_28_4 << 2) | ((next3 >> 30) & 0b11);

        reg_cs2_next.x = ((next1 >> 7) & 0b1'1111'1111) | (reg_28_1 << 9);
        reg_cs2_next.y = (((next1 >> (7 + 16)) & 0b1'1111'1111)) | (reg_28_2 << 9);
        reg_cs2_next.z = ((next3 >> 7) & 0b1'1111'1111) | (reg_28_3 << 9);
        reg_cs2_next.w = (((next3 >> (7 + 16)) & 0b1'1111'1111)) | (reg_28_4 << 9);

    } else if constexpr(R == 8) {
        uint4 reg_load = ld_cs((uint4 *) &compressed[weight_idx]);

        uint32_t reg_load1 = reg_load.x, reg_load2 = reg_load.y, reg_load3 = reg_load.z, reg_load4 = reg_load.w;

        // send high 16 bits to prev thread
        uint32_t pack1 = (reg_load1 >> 16) | (reg_load2 & 0xffff0000);
        uint32_t pack3 = (reg_load3 >> 16) | (reg_load4 & 0xffff0000);

        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);

        reg_cs_next.x = reg_load1;
        reg_cs_next.y = reg_load2;
        reg_cs_next.z = reg_load3;
        reg_cs_next.w = reg_load4;

        reg_cs2_next.x = __byte_perm(next1, reg_load1, 0x0041);
        reg_cs2_next.y = __byte_perm(next1, reg_load2, 0x0043);
        reg_cs2_next.z = __byte_perm(next3, reg_load3, 0x0041);
        reg_cs2_next.w = __byte_perm(next3, reg_load4, 0x0043);
    } else if constexpr(R == 9){
        ditto reg_load, reg_load2, reg_load3, reg_load4;
        reg_load.u16x2.x = compressed[weight_idx];
        reg_load.u16x2.y = compressed[weight_idx + 1];
        reg_load2.u16x2.x = compressed[weight_idx + 2];
        reg_load2.u16x2.y = compressed[weight_idx + 3];
        reg_load3.u16x2.x = compressed[weight_idx + 4];
        reg_load3.u16x2.y = compressed[weight_idx + 5];
        reg_load4.u16x2.x = compressed[weight_idx + 6];
        reg_load4.u16x2.y = compressed[weight_idx + 7];
        uint32_t reg_load5 = (uint32_t) compressed[weight_idx + 8];    

        reg_cs_next.x = (reg_load.u32 >> 4) | ((reg_load2.u32 & 0xf) << 28);
        reg_cs_next.y = (reg_load2.u32 >> 8) | ((reg_load3.u32 & 0xff) << 24);
        reg_cs_next.z = (reg_load3.u32 >> 12) | ((reg_load4.u32 & 0xfff) << 20);
        reg_cs_next.w = (reg_load4.u32 >> 16) | (reg_load5 << 16);
        // send high 16 bits to prev thread
        uint32_t pack1 = (reg_cs_next.x >> 16) | (reg_cs_next.y& 0xffff0000) ;
        uint32_t pack3 = (reg_cs_next.z >> 16) | (reg_cs_next.w& 0xffff0000) ;
        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);
        
        reg_cs2_next.x = (((next1 >> 9) & 0b111'1111) | ((reg_cs_next.x & 0x3fff) << 11)) | ((reg_load.u32 & 0xf) << 7);
        reg_cs2_next.y = (((next1 >> 25) & 0b111'1111) | ((reg_cs_next.y & 0x3fff) << 11)) | ((reg_load2.u32 & 0xf0) << 3);
        reg_cs2_next.z = (((next3 >> 9) & 0b111'1111) | ((reg_cs_next.z & 0x3fff) << 11)) | ((reg_load3.u32 & 0xf00) >> 1);
        reg_cs2_next.w = (((next3 >> 25) & 0b111'1111)) | ((reg_cs_next.w & 0x3fff) << 11) | ((reg_load4.u32 & 0xf000) >> 5);
    } else if constexpr(R == 10) {
        uint reg_load, reg_load2, reg_load3, reg_load4, reg_load5;
        reg_load = ld_cs((uint *) &compressed[weight_idx]);
        reg_load2 = ld_cs((uint *) &compressed[weight_idx + 2]);
        reg_load3 = ld_cs((uint *) &compressed[weight_idx + 4]);
        reg_load4 = ld_cs((uint *) &compressed[weight_idx + 6]);
        reg_load5 = ld_cs((uint *) &compressed[weight_idx + 8]);
        
        reg_cs_next.x = (reg_load >> 8) | ((reg_load2 & 0xff) << 24);
        reg_cs_next.y = (reg_load2 >> 16) | ((reg_load3 & 0xffff) << 16);
        reg_cs_next.z = (reg_load3 >> 24) | ((reg_load4 & 0xffffff) << 8);
        reg_cs_next.w = reg_load5;
        // send high 16 bits to prev thread
        uint32_t pack1 = (reg_cs_next.x >> 16) | (reg_cs_next.y& 0xffff0000) ;
        uint32_t pack3 = (reg_cs_next.z >> 16) | (reg_cs_next.w& 0xffff0000) ;
        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);

        reg_cs2_next.x = (((next1 >> 10) & 0b11'1111) | ((reg_cs_next.x & 0xfff) << 14)) | ((reg_load & 0xff) << 6);
        reg_cs2_next.y = (((next1 >> 26) & 0b11'1111) | ((reg_cs_next.y & 0xfff) << 14)) | ((reg_load2 & 0xff00) >> 2);
        reg_cs2_next.z = (((next3 >> 10) & 0b11'1111) | ((reg_cs_next.z & 0xfff) << 14)) | ((reg_load3 & 0xff0000) >> 10);
        reg_cs2_next.w = (((next3 >> 26) & 0b11'1111) | ((reg_cs_next.w & 0xfff) << 14)) | ((reg_load4 & 0xff000000) >> 18);
    } 
}

template <uint32_t L, uint32_t S, uint32_t R, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress_gemm(
    float *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const half2 *__restrict__ x,
    const half2 *__restrict__ codebook
) {
        // ** load codebook **
    extern __shared__ __align__(1<<(5+V+1)) half2 smem_codebook[];

    int offset_codebook = 1<<14;
    ditto2 (*x_buf)[BLOCK_SIZE / WARP_SIZE][4][4] =
        reinterpret_cast<decltype(x_buf)>(smem_codebook + offset_codebook);

    float (*reduce_gather)[2][N][16] =
        reinterpret_cast<decltype(reduce_gather)>(smem_codebook + offset_codebook); // reuse

    // ** cursed indexing math **

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    constexpr uint32_t tileCountM = M / MMA_M;
    constexpr uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;

#define ROUND_UP(a, b) ((a + b - 1) / b)

    static_assert (tileCountM % 2 == 0);
    constexpr uint32_t m_per_block = ROUND_UP(tileCountM, (2 * BLOCK_COUNT));
    constexpr uint32_t k_per_block = tileCountK / (warps_per_block * 4) * 2;
    static_assert((tileCountK % (warps_per_block * 4)) % 4 == 0);
    uint32_t this_warp_k = (warpId < (tileCountK % (warps_per_block * 4)) / 4) ? k_per_block + 2 : k_per_block;

    constexpr uint32_t u16_per_compressed_tile = MMA_M * MMA_K * R / 32;
    static_assert((MMA_M * MMA_K * R) % 32 == 0);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block = u16_per_compressed_tile * weight_block; // one tile block per warp at a time
    constexpr uint32_t weight_step = warps_per_block * u16_per_tile_block;
    constexpr uint32_t weight_row_step = tileCountK * u16_per_compressed_tile * 2;  // 2 rows of tiles

    constexpr uint32_t dup_bits = 14 - S;
    constexpr uint32_t s_mask = (1<<S) - 1;
    constexpr uint32_t dup_mask = (1<<dup_bits) - 1;
    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        // ** load weight, start loop **
        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2 + laneId * (u16_per_tile_block / WARP_SIZE);
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        load_reg_cs<R>((const uint16_t * __restrict__) compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs;
        uint4 reg_cs2;

        // define acc
        float4 reg_p[2] = {};

        uint32_t x_idx = warpId * f16x2_per_x_tile * 4 + laneId;  // every warp does 4 k tiles per iteration
        uint32_t x_idx_step = warps_per_block * f16x2_per_x_tile * 4;
        if (mi == 0) {
            if constexpr (S == 9) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 2) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ (threadIdx.x & dup_mask) ^ (threadIdx.x >> S))] = my_codebook_element;
                }
            } else if constexpr (S == 10) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 1) & dup_mask))] = my_codebook_element;
                }
            } else if constexpr (S == 11) {
                uint32_t my_cb_idx = threadIdx.x;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
                my_cb_idx += 1024;
                my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
            }
            __syncthreads();
        }

        uint32_t x_line;
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2; // fixme: this costs 10GB/s
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            load_reg_cs<R>((const uint16_t * __restrict__) compressed, weight_idx + (1 - ki % 2) * u16_per_tile_block, laneId, reg_cs_next, reg_cs2_next);

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
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    // ** decode weights **
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j += 1) {
                        uint32_t idx;
                        if constexpr(R == 2) {
                            idx = reg_c >> (2 * (4-j));
                        } else if constexpr(R == 3) {
                            idx = reg_c >> (3 * (4-j));
                        } else if constexpr(R == 4) {
                            idx = reg_c >> (4 * (4-j));
                        } else if constexpr(R == 5) {
                            idx = reg_c >> (5 * (3-j)+1);
                        } else if constexpr(R == 6) {
                            idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                        } else if constexpr(R == 7) {
                            idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                        } else if constexpr(R == 8) {
                            idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                        } else if constexpr(R == 9) {
                            idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                        } else if constexpr(R == 10) {
                            idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                        } else if constexpr(R == 11) {
                            idx = (j < 2) ? (reg_c >> (11 * (1-j) + 5)) : (reg_c2 >> (11 * (3-j)));
                        } else if constexpr(R == 12) {
                            idx = (j < 2) ? (reg_c >> (12 * (1-j) + 4)) : (reg_c2 >> (12 * (3-j)));
                        } 

                        static_assert(L==16);
                        idx = idx * (idx+1);
                        uint32_t masked_idx;
                        if constexpr (S==9){
                            masked_idx = ((idx & 0b0111111111000000) | (laneId << 1)); // this /2 will not be elided automatically
                        } else if constexpr (S==10){
                            masked_idx = ((idx & 0b0111111111100000) | ((laneId << 1) & 0b11110)); // this /2 will not be elided automatically
                        } else if constexpr (S==11){
                            masked_idx = ((idx & 0b0111111111110000) | ((laneId << 2) & 0b1110)); // this /2 will not be elided automatically
                        }
                        __builtin_assume(masked_idx % 2 == 0);
                        reg_w.f16x2[j] = smem_codebook[masked_idx/2];
                        uint32_t selector = 0b00000000'00000000'10000000'00000000;
                        reg_w.u32[j] = reg_w.u32[j] ^ (selector & idx);
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

        __syncthreads();
        reduce_gather_routine<N>(reg_p, warpId, laneId, reduce_gather);
        __syncthreads();
        float reduced = 0.0;
        if (warpId < N) {
            int pi = laneId / 16;
            for (int warpi = 0; warpi < BLOCK_SIZE / WARP_SIZE; warpi++) {
                reduced += reduce_gather[warpi][pi][warpId][laneId % 16];
            }

            float *out_tile = out + (tileIdM * 2) * f32_per_out_tile + warpId * M;
            out_tile[laneId] = reduced;
        }
        if constexpr(m_per_block > 1) __syncthreads();
        tileIdM += 1;
    }
}


template <uint32_t L, uint32_t S, uint32_t R1, uint32_t R2, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress_gemm_combt(
    float *__restrict__ out,
    const uint32_t *__restrict__ compressed1,
    const uint32_t *__restrict__ compressed2,
    const half2 *__restrict__ x,
    const half2 *__restrict__ codebook
) {
        // ** load codebook **
    extern __shared__ __align__(1<<(5+V+1)) half2 smem_codebook[];

    int offset_codebook = 1<<14;
    // int offset_codebook = 1<<(S+5+V+1-2);
    ditto2 (*x_buf)[BLOCK_SIZE / WARP_SIZE][4][4] =
        reinterpret_cast<decltype(x_buf)>(smem_codebook + offset_codebook);

    // int offset_x_buf = offset_codebook + sizeof(ditto2) * N * (BLOCK_SIZE / WARP_SIZE) * 4 * 4 / sizeof(half2);
    float (*reduce_gather)[2][N][16] =
        reinterpret_cast<decltype(reduce_gather)>(smem_codebook + offset_codebook); // reuse

    // ** cursed indexing math **

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    constexpr uint32_t tileCountM = M / MMA_M;
    constexpr uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;

#define ROUND_UP(a, b) ((a + b - 1) / b)

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

    constexpr uint32_t u16_per_compressed_tile1 = MMA_M * MMA_K * R1 / 32;
    constexpr uint32_t u16_per_compressed_tile2 = MMA_M * MMA_K * R2 / 32;
    static_assert((MMA_M * MMA_K * R1) % 32 == 0);
    static_assert((MMA_M * MMA_K * R2) % 32 == 0);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block1 = u16_per_compressed_tile1 * weight_block; // one tile block per warp at a time
    constexpr uint32_t u16_per_tile_block2 = u16_per_compressed_tile2 * weight_block; // one tile block per warp at a time
    constexpr uint32_t weight_row_step1 = tileCountK * u16_per_compressed_tile1;  // 2 rows of tiles for half tileCountK
    constexpr uint32_t weight_row_step2 = tileCountK * u16_per_compressed_tile2;  // 2 rows of tiles for half tileCountK

    constexpr uint32_t dup_bits = 14 - S;
    constexpr uint32_t s_mask = (1<<S) - 1;
    constexpr uint32_t dup_mask = (1<<dup_bits) - 1;
    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        // ** load weight, start loop **
        int cur_tileIdK = warpId * weight_block * 2; 
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        int weight_idx;
        int use_R1 = 1;
        if (cur_tileIdK < tileCountK) { 
            weight_idx = tileIdM * weight_row_step1 + cur_tileIdK * u16_per_compressed_tile1 + laneId * (u16_per_tile_block1 / WARP_SIZE);
            load_reg_cs<R1>((const uint16_t * __restrict__) compressed1, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        } else {
            use_R1 = 0;
            weight_idx = tileIdM * weight_row_step2 + (cur_tileIdK - tileCountK) * u16_per_compressed_tile2 + laneId * (u16_per_tile_block2 / WARP_SIZE);
            load_reg_cs<R2>((const uint16_t * __restrict__) compressed2, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        }

        uint4 reg_cs;
        uint4 reg_cs2;

        // define acc
        float4 reg_p[2] = {};

        uint32_t x_idx = warpId * f16x2_per_x_tile * 4 + laneId;  // every warp does 4 k tiles per iteration
        uint32_t x_idx_step = warps_per_block * f16x2_per_x_tile * 4;
        if (mi == 0) {
            if constexpr (S == 9) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 2) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ (threadIdx.x & dup_mask) ^ (threadIdx.x >> S))] = my_codebook_element;
                }
            } else if constexpr (S == 10) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 1) & dup_mask))] = my_codebook_element;
                }
            } else if constexpr (S == 11) {
                uint32_t my_cb_idx = threadIdx.x;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
                my_cb_idx += 1024;
                my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
            }
            // for (uint32_t i = 0; i < 32; i+= 1) { assert(smem_codebook[(my_cb_idx << 5) + i] == my_codebook_element); }
            __syncthreads();
        }

        uint32_t x_line;
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            if (cur_tileIdK >= tileCountK) use_R1=0;
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            if (ki + 1 != this_warp_k && ki % 2 == 1) {
                cur_tileIdK += warps_per_block * weight_block * 2;
                if (cur_tileIdK < tileCountK) {
                    weight_idx = tileIdM * weight_row_step1 + cur_tileIdK * u16_per_compressed_tile1 + laneId * (u16_per_tile_block1 / WARP_SIZE);
                } else {
                    weight_idx = tileIdM * weight_row_step2 + (cur_tileIdK - tileCountK) * u16_per_compressed_tile2 + laneId * (u16_per_tile_block2 / WARP_SIZE);
                }
            }
            if (cur_tileIdK < tileCountK) {
                load_reg_cs<R1>((const uint16_t * __restrict__) compressed1, weight_idx + (1 - ki % 2) * u16_per_tile_block1, laneId, reg_cs_next, reg_cs2_next);
            } else {
                load_reg_cs<R2>((const uint16_t * __restrict__) compressed2, weight_idx + (1 - ki % 2) * u16_per_tile_block2, laneId, reg_cs_next, reg_cs2_next);
            }


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
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    // ** decode weights **
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j += 1) {
                        uint32_t idx;
                        if (use_R1) {
                            if constexpr(R1 == 2) {
                                idx = reg_c >> (2 * (4-j));
                            } else if constexpr(R1 == 3) {
                                idx = reg_c >> (3 * (4-j));
                            } else if constexpr(R1 == 4) {
                                idx = reg_c >> (4 * (4-j));
                            } else if constexpr(R1 == 5) {
                                idx = reg_c >> (5 * (3-j)+1);
                            } else if constexpr(R1 == 6) {
                                idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                            } else if constexpr(R1 == 7) {
                                idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                            } else if constexpr(R1 == 8) {
                                idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                            } else if constexpr(R1 == 9) {
                                idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                            } else if constexpr(R1 == 10) {
                                idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                            } else if constexpr(R1 == 11) {
                                idx = (j < 2) ? (reg_c >> (11 * (1-j) + 5)) : (reg_c2 >> (11 * (3-j)));
                            } else if constexpr(R1 == 12) {
                                idx = (j < 2) ? (reg_c >> (12 * (1-j) + 4)) : (reg_c2 >> (12 * (3-j)));
                            } 
                        } else {
                            if constexpr(R2 == 2) {
                                idx = reg_c >> (2 * (4-j));
                            } else if constexpr(R2 == 3) {
                                idx = reg_c >> (3 * (4-j));
                            } else if constexpr(R2 == 4) {
                                idx = reg_c >> (4 * (4-j));
                            } else if constexpr(R2 == 5) {
                                idx = reg_c >> (5 * (3-j)+1);
                            } else if constexpr(R2 == 6) {
                                idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                            } else if constexpr(R2 == 7) {
                                idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                            } else if constexpr(R2 == 8) {
                                idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                            } else if constexpr(R2 == 9) {
                                idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                            } else if constexpr(R2 == 10) {
                                idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                            } else if constexpr(R2 == 11) {
                                idx = (j < 2) ? (reg_c >> (11 * (1-j) + 5)) : (reg_c2 >> (11 * (3-j)));
                            } else if constexpr(R2 == 12) {
                                idx = (j < 2) ? (reg_c >> (12 * (1-j) + 4)) : (reg_c2 >> (12 * (3-j)));
                            } 
                        }

                        static_assert(L==16);
                        idx = idx * (idx+1);
                        uint32_t masked_idx;
                        if constexpr (S==9){
                            masked_idx = ((idx & 0b0111111111000000) | (laneId << 1)); // this /2 will not be elided automatically
                        } else if constexpr (S==10){
                            masked_idx = ((idx & 0b0111111111100000) | ((laneId << 1) & 0b11110)); // this /2 will not be elided automatically
                        } else if constexpr (S==11){
                            masked_idx = ((idx & 0b0111111111110000) | ((laneId << 2) & 0b1110)); // this /2 will not be elided automatically
                        }
                        __builtin_assume(masked_idx % 2 == 0);
                        reg_w.f16x2[j] = smem_codebook[masked_idx/2];
                        //asm("ld.shared.u32 %0, [%1];" : "=r"(reg_w.u32[j]) : "r"((masked_idx * 2 + (uint16_t) smem_codebook)));
                        // sign flip
                        uint32_t selector = 0b00000000'00000000'10000000'00000000;
                        reg_w.u32[j] = reg_w.u32[j] ^ (selector & idx);
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
            //if constexpr(enable_kim4_sync) {if (ki % 4 == 0) __syncthreads();} // slower with 7b even with this if constexpr thing fsr
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


template <uint32_t L, uint32_t S, uint32_t R1, uint32_t R2, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress_gemm_comb(
    float *__restrict__ out,
    const uint32_t *__restrict__ compressed1,
    const uint32_t *__restrict__ compressed2,
    const half2 *__restrict__ x,
    const half2 *__restrict__ codebook
) {
        // ** load codebook **
    extern __shared__ __align__(1<<(5+V+1)) half2 smem_codebook[];

    int offset_codebook = 1<<14;
    // int offset_codebook = 1<<(S+5+V+1-2);
    ditto2 (*x_buf)[BLOCK_SIZE / WARP_SIZE][4][4] =
        reinterpret_cast<decltype(x_buf)>(smem_codebook + offset_codebook);

    // int offset_x_buf = offset_codebook + sizeof(ditto2) * N * (BLOCK_SIZE / WARP_SIZE) * 4 * 4 / sizeof(half2);
    float (*reduce_gather)[2][N][16] =
        reinterpret_cast<decltype(reduce_gather)>(smem_codebook + offset_codebook); // reuse

    // ** cursed indexing math **

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    constexpr uint32_t tileCountM = M / MMA_M;
    constexpr uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;

#define ROUND_UP(a, b) ((a + b - 1) / b)

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

    constexpr uint32_t u16_per_compressed_tile1 = MMA_M * MMA_K * R1 / 32;
    static_assert((MMA_M * MMA_K * R1) % 32 == 0);
    constexpr uint32_t u16_per_compressed_tile2 = MMA_M * MMA_K * R2 / 32;
    static_assert((MMA_M * MMA_K * R2) % 32 == 0);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block1 = u16_per_compressed_tile1 * weight_block; // one tile block per warp at a time
    constexpr uint32_t u16_per_tile_block2 = u16_per_compressed_tile2 * weight_block; // one tile block per warp at a time
    constexpr uint32_t weight_step1 = warps_per_block * u16_per_tile_block1;
    constexpr uint32_t weight_step2 = warps_per_block * u16_per_tile_block2;
    constexpr uint32_t weight_row_step1 = tileCountK * u16_per_compressed_tile1 * 2;  // 2 rows of tiles
    constexpr uint32_t weight_row_step2 = tileCountK * u16_per_compressed_tile2 * 2;  // 2 rows of tiles

    constexpr uint32_t half_tileIdM_max = tileCountM / 4;
    constexpr uint32_t dup_bits = 14 - S;
    constexpr uint32_t s_mask = (1<<S) - 1;
    constexpr uint32_t dup_mask = (1<<dup_bits) - 1;

    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return; 
        // ** load weight, start loop **
        int weight_idx;
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        if (tileIdM < half_tileIdM_max) {
            weight_idx = tileIdM * weight_row_step1 + warpId * u16_per_tile_block1 * 2 + laneId * (u16_per_tile_block1 / WARP_SIZE);
            load_reg_cs<R1>((const uint16_t * __restrict__) compressed1, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        } else {
            weight_idx = (tileIdM - half_tileIdM_max) * weight_row_step2 + warpId * u16_per_tile_block2 * 2 + laneId * (u16_per_tile_block2 / WARP_SIZE);
            load_reg_cs<R2>((const uint16_t * __restrict__) compressed2, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        }
        uint4 reg_cs;
        uint4 reg_cs2;

        // define acc
        float4 reg_p[2] = {};

        uint32_t x_idx = warpId * f16x2_per_x_tile * 4 + laneId;  // every warp does 4 k tiles per iteration
        uint32_t x_idx_step = warps_per_block * f16x2_per_x_tile * 4;
        if (mi == 0) {
            if constexpr (S == 9) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 2) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ (threadIdx.x & dup_mask) ^ (threadIdx.x >> S))] = my_codebook_element;
                }
            } else if constexpr (S == 10) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 1) & dup_mask))] = my_codebook_element;
                }
            } else if constexpr (S == 11) {
                uint32_t my_cb_idx = threadIdx.x;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
                my_cb_idx += 1024;
                my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
            }
            // for (uint32_t i = 0; i < 32; i+= 1) { assert(smem_codebook[(my_cb_idx << 5) + i] == my_codebook_element); }
            __syncthreads();
        }

        uint32_t x_line;
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            if (tileIdM < half_tileIdM_max) {
                if (ki + 1 != this_warp_k && ki % 2 == 1) {
                    weight_idx += weight_step1 * 2; // fixme: this costs 10GB/s
                }
                load_reg_cs<R1>((const uint16_t * __restrict__) compressed1, weight_idx + (1 - ki % 2) * u16_per_tile_block1, laneId, reg_cs_next, reg_cs2_next);
            } else {
                if (ki + 1 != this_warp_k && ki % 2 == 1) {
                    weight_idx += weight_step2 * 2; // fixme: this costs 10GB/s
                }
                load_reg_cs<R2>((const uint16_t * __restrict__) compressed2, weight_idx + (1 - ki % 2) * u16_per_tile_block2, laneId, reg_cs_next, reg_cs2_next);
            }


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
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    // ** decode weights **
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j += 1) {
                        uint32_t idx;
                        if (tileIdM < half_tileIdM_max) {
                            if constexpr(R1 == 2) {
                                idx = reg_c >> (2 * (4-j));
                            } else if constexpr(R1 == 3) {
                                idx = reg_c >> (3 * (4-j));
                            } else if constexpr(R1 == 4) {
                                idx = reg_c >> (4 * (4-j));
                            } else if constexpr(R1 == 5) {
                                idx = reg_c >> (5 * (3-j)+1);
                            } else if constexpr(R1 == 6) {
                                idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                            } else if constexpr(R1 == 7) {
                                idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                            } else if constexpr(R1 == 8) {
                                idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                            } else if constexpr(R1 == 9) {
                                idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                            } else if constexpr(R1 == 10) {
                                idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                            }
                        } else {
                            if constexpr(R2 == 2) {
                                idx = reg_c >> (2 * (4-j));
                            } else if constexpr(R2 == 3) {
                                idx = reg_c >> (3 * (4-j));
                            } else if constexpr(R2 == 4) {
                                idx = reg_c >> (4 * (4-j));
                            } else if constexpr(R2 == 5) {
                                idx = reg_c >> (5 * (3-j)+1);
                            } else if constexpr(R2 == 6) {
                                idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                            } else if constexpr(R2 == 7) {
                                idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                            } else if constexpr(R2 == 8) {
                                idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                            } else if constexpr(R2 == 9) {
                                idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                            } else if constexpr(R2 == 10) {
                                idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                            }
                        }

                        static_assert(L==16);
                        idx = idx * (idx+1);
                        uint32_t masked_idx;
                        if constexpr (S==9){
                            masked_idx = ((idx & 0b0111111111000000) | (laneId << 1)); // this /2 will not be elided automatically
                        } else if constexpr (S==10){
                            masked_idx = ((idx & 0b0111111111100000) | ((laneId << 1) & 0b11110)); // this /2 will not be elided automatically
                        } else if constexpr (S==11){
                            masked_idx = ((idx & 0b0111111111110000) | ((laneId << 2) & 0b1110)); // this /2 will not be elided automatically
                        }
                        __builtin_assume(masked_idx % 2 == 0);
                        reg_w.f16x2[j] = smem_codebook[masked_idx/2];
                        //asm("ld.shared.u32 %0, [%1];" : "=r"(reg_w.u32[j]) : "r"((masked_idx * 2 + (uint16_t) smem_codebook)));
                        // sign flip
                        uint32_t selector = 0b00000000'00000000'10000000'00000000;
                        reg_w.u32[j] = reg_w.u32[j] ^ (selector & idx);
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
            //if constexpr(enable_kim4_sync) {if (ki % 4 == 0) __syncthreads();} // slower with 7b even with this if constexpr thing fsr
            if ((ki % 2 == 0)) {
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



template <uint32_t L, uint32_t S, uint32_t R, uint32_t V>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress(
    half2 *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const half2 *__restrict__ codebook,
    uint32_t M,
    uint32_t K
) {
        // ** load codebook **
    extern __shared__ __align__(1<<(5+V+1)) half2 smem_codebook[];

    int offset_codebook = 1<<14;
    // int offset_codebook = 1<<(S+5+V+1-2);
    // ** cursed indexing math **

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    uint32_t tileCountM = M / MMA_M;
    uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;

    uint32_t m_per_block = ROUND_UP(tileCountM, (2 * BLOCK_COUNT));
    uint32_t k_per_block = tileCountK / (warps_per_block * 4) * 2;
    uint32_t this_warp_k = (warpId < (tileCountK % (warps_per_block * 4)) / 4) ? k_per_block + 2 : k_per_block;

    constexpr uint32_t u16_per_compressed_tile = MMA_M * MMA_K * R / 32;
    static_assert((MMA_M * MMA_K * R) % 32 == 0);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block = u16_per_compressed_tile * weight_block; // one tile block per warp at a time
    constexpr uint32_t weight_step = warps_per_block * u16_per_tile_block;
    uint32_t weight_row_step = tileCountK * u16_per_compressed_tile * 2;  // 2 rows of tiles

    constexpr uint32_t dup_bits = 14 - S;
    constexpr uint32_t s_mask = (1<<S) - 1;
    constexpr uint32_t dup_mask = (1<<dup_bits) - 1;
    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        // ** load weight, start loop **
        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2 + laneId * (u16_per_tile_block / WARP_SIZE);
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        load_reg_cs<R>((const uint16_t * __restrict__) compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs;
        uint4 reg_cs2;
        if (mi == 0) {
            if constexpr (S == 9) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 2) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ (threadIdx.x & dup_mask) ^ (threadIdx.x >> S))] = my_codebook_element;
                }
            } else if constexpr (S == 10) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 1) & dup_mask))] = my_codebook_element;
                }
            } else if constexpr (S == 11) {
                uint32_t my_cb_idx = threadIdx.x;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
                my_cb_idx += 1024;
                my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
            }
            // for (uint32_t i = 0; i < 32; i+= 1) { assert(smem_codebook[(my_cb_idx << 5) + i] == my_codebook_element); }
            __syncthreads();
        }
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2; // fixme: this costs 10GB/s
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            load_reg_cs<R>((const uint16_t * __restrict__) compressed, weight_idx + (1 - ki % 2) * u16_per_tile_block, laneId, reg_cs_next, reg_cs2_next);

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {
#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi++) {
                    uint32_t reg_c, reg_c2;
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    // ** decode weights **
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j += 1) {
                        uint32_t idx;
                        if constexpr(R == 2) {
                            idx = reg_c >> (2 * (4-j));
                        } else if constexpr(R == 3) {
                            idx = reg_c >> (3 * (4-j));
                        } else if constexpr(R == 4) {
                            idx = reg_c >> (4 * (4-j));
                        } else if constexpr(R == 5) {
                            idx = reg_c >> (5 * (3-j)+1);
                        } else if constexpr(R == 6) {
                            idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                        } else if constexpr(R == 7) {
                            idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                        } else if constexpr(R == 8) {
                            idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                        } else if constexpr(R == 9) {
                            idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                        } else if constexpr(R == 10) {
                            idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                        } else if constexpr(R == 11) {
                            idx = (j < 2) ? (reg_c >> (11 * (1-j) + 5)) : (reg_c2 >> (11 * (3-j)));
                        } else if constexpr(R == 12) {
                            idx = (j < 2) ? (reg_c >> (12 * (1-j) + 4)) : (reg_c2 >> (12 * (3-j)));
                        }

                        static_assert(L==16);
                        idx = idx * (idx+1);
                        uint32_t masked_idx;
                        if constexpr (S==9){
                            masked_idx = ((idx & 0b0111111111000000) | (laneId << 1)); // this /2 will not be elided automatically
                        } else if constexpr (S==10){
                            masked_idx = ((idx & 0b0111111111100000) | ((laneId << 1) & 0b11110)); // this /2 will not be elided automatically
                        } else if constexpr (S==11){
                            masked_idx = ((idx & 0b0111111111110000) | ((laneId << 2) & 0b1110)); // this /2 will not be elided automatically
                        }
                        __builtin_assume(masked_idx % 2 == 0);
                        reg_w.f16x2[j] = smem_codebook[masked_idx/2];
                        uint32_t selector = 0b00000000'00000000'10000000'00000000;
                        reg_w.u32[j] = reg_w.u32[j] ^ (selector & idx);

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


template <uint32_t L, uint32_t S, uint32_t R1, uint32_t R2, uint32_t V>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress_comb(
    half2 *__restrict__ out,
    const uint32_t *__restrict__ compressed1,
    const uint32_t *__restrict__ compressed2,
    const half2 *__restrict__ codebook,
    uint32_t M,
    uint32_t K
) {
        // ** load codebook **
    extern __shared__ __align__(1<<(5+V+1)) half2 smem_codebook[];

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    uint32_t tileCountM = M / MMA_M;
    uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;

    uint32_t m_per_block = ROUND_UP(tileCountM, (2 * BLOCK_COUNT));
    uint32_t k_per_block = tileCountK / (warps_per_block * 4) * 2;
    uint32_t this_warp_k = (warpId < (tileCountK % (warps_per_block * 4)) / 4) ? k_per_block + 2 : k_per_block;

    constexpr uint32_t u16_per_compressed_tile1 = MMA_M * MMA_K * R1 / 32;
    constexpr uint32_t u16_per_compressed_tile2 = MMA_M * MMA_K * R2 / 32;
    static_assert((MMA_M * MMA_K * R1) % 32 == 0);
    static_assert((MMA_M * MMA_K * R2) % 32 == 0);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block1 = u16_per_compressed_tile1 * weight_block; // one tile block per warp at a time
    constexpr uint32_t u16_per_tile_block2 = u16_per_compressed_tile2 * weight_block; // one tile block per warp at a time
    constexpr uint32_t weight_step1 = warps_per_block * u16_per_tile_block1;
    constexpr uint32_t weight_step2 = warps_per_block * u16_per_tile_block2;
    uint32_t weight_row_step1 = tileCountK * u16_per_compressed_tile1 * 2;  // 2 rows of tiles
    uint32_t weight_row_step2 = tileCountK * u16_per_compressed_tile2 * 2;  // 2 rows of tiles
    uint32_t half_tileIdM_max = tileCountM / 4;

    constexpr uint32_t dup_bits = 14 - S;
    constexpr uint32_t s_mask = (1<<S) - 1;
    constexpr uint32_t dup_mask = (1<<dup_bits) - 1;
    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        // ** load weight, start loop **
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        int weight_idx;
        if (tileIdM < half_tileIdM_max) {
            weight_idx = tileIdM * weight_row_step1 + warpId * u16_per_tile_block1 * 2 + laneId * (u16_per_tile_block1 / WARP_SIZE);
            load_reg_cs<R1>((const uint16_t * __restrict__) compressed1, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        } else {
            weight_idx = (tileIdM - half_tileIdM_max) * weight_row_step2 + warpId * u16_per_tile_block2 * 2 + laneId * (u16_per_tile_block2 / WARP_SIZE);
            load_reg_cs<R2>((const uint16_t * __restrict__) compressed2, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        }
        uint4 reg_cs;
        uint4 reg_cs2;
        if (mi == 0) {
            if constexpr (S == 9) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 2) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ (threadIdx.x & dup_mask) ^ (threadIdx.x >> S))] = my_codebook_element;
                }
            } else if constexpr (S == 10) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 1) & dup_mask))] = my_codebook_element;
                }
            } else if constexpr (S == 11) {
                uint32_t my_cb_idx = threadIdx.x;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
                my_cb_idx += 1024;
                my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
            }
            // for (uint32_t i = 0; i < 32; i+= 1) { assert(smem_codebook[(my_cb_idx << 5) + i] == my_codebook_element); }
            __syncthreads();
        }
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            if (tileIdM < half_tileIdM_max) {
                if (ki + 1 != this_warp_k && ki % 2 == 1) {
                    weight_idx += weight_step1 * 2; // fixme: this costs 10GB/s
                }
                load_reg_cs<R1>((const uint16_t * __restrict__) compressed1, weight_idx + (1 - ki % 2) * u16_per_tile_block1, laneId, reg_cs_next, reg_cs2_next);
            } else {
                if (ki + 1 != this_warp_k && ki % 2 == 1) {
                    weight_idx += weight_step2 * 2; // fixme: this costs 10GB/s
                }
                load_reg_cs<R2>((const uint16_t * __restrict__) compressed2, weight_idx + (1 - ki % 2) * u16_per_tile_block2, laneId, reg_cs_next, reg_cs2_next);
            }

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {
#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi++) {
                    uint32_t reg_c, reg_c2;
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    // ** decode weights **
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j += 1) {
                        uint32_t idx;
                        if (tileIdM < half_tileIdM_max) {
                            if constexpr(R1 == 2) {
                                idx = reg_c >> (2 * (4-j));
                            } else if constexpr(R1 == 3) {
                                idx = reg_c >> (3 * (4-j));
                            } else if constexpr(R1 == 4) {
                                idx = reg_c >> (4 * (4-j));
                            } else if constexpr(R1 == 5) {
                                idx = reg_c >> (5 * (3-j)+1);
                            } else if constexpr(R1 == 6) {
                                idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                            } else if constexpr(R1 == 7) {
                                idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                            } else if constexpr(R1 == 8) {
                                idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                            } else if constexpr(R1 == 9) {
                                idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                            } else if constexpr(R1 == 10) {
                                idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                            } else if constexpr(R1 == 11) {
                                idx = (j < 2) ? (reg_c >> (11 * (1-j) + 5)) : (reg_c2 >> (11 * (3-j)));
                            } else if constexpr(R1 == 12) {
                                idx = (j < 2) ? (reg_c >> (12 * (1-j) + 4)) : (reg_c2 >> (12 * (3-j)));
                            }
                        } else {
                            if constexpr(R2 == 2) {
                                idx = reg_c >> (2 * (4-j));
                            } else if constexpr(R2 == 3) {
                                idx = reg_c >> (3 * (4-j));
                            } else if constexpr(R2 == 4) {
                                idx = reg_c >> (4 * (4-j));
                            } else if constexpr(R2 == 5) {
                                idx = reg_c >> (5 * (3-j)+1);
                            } else if constexpr(R2 == 6) {
                                idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                            } else if constexpr(R2 == 7) {
                                idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                            } else if constexpr(R2 == 8) {
                                idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                            } else if constexpr(R2 == 9) {
                                idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                            } else if constexpr(R2 == 10) {
                                idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                            } else if constexpr(R2 == 11) {
                                idx = (j < 2) ? (reg_c >> (11 * (1-j) + 5)) : (reg_c2 >> (11 * (3-j)));
                            } else if constexpr(R2 == 12) {
                                idx = (j < 2) ? (reg_c >> (12 * (1-j) + 4)) : (reg_c2 >> (12 * (3-j)));
                            }
                        }

                        static_assert(L==16);
                        idx = idx * (idx+1);
                        uint32_t masked_idx;
                        if constexpr (S==9){
                            masked_idx = ((idx & 0b0111111111000000) | (laneId << 1)); // this /2 will not be elided automatically
                        } else if constexpr (S==10){
                            masked_idx = ((idx & 0b0111111111100000) | ((laneId << 1) & 0b11110)); // this /2 will not be elided automatically
                        } else if constexpr (S==11){
                            masked_idx = ((idx & 0b0111111111110000) | ((laneId << 2) & 0b1110)); // this /2 will not be elided automatically
                        }
                        __builtin_assume(masked_idx % 2 == 0);
                        reg_w.f16x2[j] = smem_codebook[masked_idx/2];
                        uint32_t selector = 0b00000000'00000000'10000000'00000000;
                        reg_w.u32[j] = reg_w.u32[j] ^ (selector & idx);

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



template <uint32_t L, uint32_t S, uint32_t R1, uint32_t R2, uint32_t V>
__global__ static void
__launch_bounds__(BLOCK_SIZE, 1)
kernel_decompress_combt(
    half2 *__restrict__ out,
    const uint32_t *__restrict__ compressed1,
    const uint32_t *__restrict__ compressed2,
    const half2 *__restrict__ codebook,
    uint32_t M,
    uint32_t K
) {
        // ** load codebook **
    extern __shared__ __align__(1<<(5+V+1)) half2 smem_codebook[];

    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    uint32_t tileCountM = M / MMA_M;
    uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;

    uint32_t m_per_block = ROUND_UP(tileCountM, (2 * BLOCK_COUNT));
    uint32_t k_per_block = tileCountK / (warps_per_block * 4) * 2;
    uint32_t this_warp_k = (warpId < (tileCountK % (warps_per_block * 4)) / 4) ? k_per_block + 2 : k_per_block;

    constexpr uint32_t u16_per_compressed_tile1 = MMA_M * MMA_K * R1 / 32;
    constexpr uint32_t u16_per_compressed_tile2 = MMA_M * MMA_K * R2 / 32;
    static_assert((MMA_M * MMA_K * R1) % 32 == 0);
    static_assert((MMA_M * MMA_K * R2) % 32 == 0);
    constexpr uint32_t f16x2_per_x_tile = MMA_K / 2;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block1 = u16_per_compressed_tile1 * weight_block; // one tile block per warp at a time
    constexpr uint32_t u16_per_tile_block2 = u16_per_compressed_tile2 * weight_block; // one tile block per warp at a time
    uint32_t weight_row_step1 = tileCountK * u16_per_compressed_tile1;  // 2 rows of tiles for half tileCountK
    uint32_t weight_row_step2 = tileCountK * u16_per_compressed_tile2;  // 2 rows of tiles for half tileCountK


    constexpr uint32_t dup_bits = 14 - S;
    constexpr uint32_t s_mask = (1<<S) - 1;
    constexpr uint32_t dup_mask = (1<<dup_bits) - 1;
    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        // ** load weight, start loop **
        int cur_tileIdK = warpId * weight_block * 2; 
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        int weight_idx;
        int use_R1 = 1;
        if (cur_tileIdK < tileCountK) { 
            weight_idx = tileIdM * weight_row_step1 + cur_tileIdK * u16_per_compressed_tile1 + laneId * (u16_per_tile_block1 / WARP_SIZE);
            load_reg_cs<R1>((const uint16_t * __restrict__) compressed1, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        } else {
            use_R1 = 0;
            weight_idx = tileIdM * weight_row_step2 + (cur_tileIdK - tileCountK) * u16_per_compressed_tile2 + laneId * (u16_per_tile_block2 / WARP_SIZE);
            load_reg_cs<R2>((const uint16_t * __restrict__) compressed2, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        }
        uint4 reg_cs;
        uint4 reg_cs2;
        if (mi == 0) {
            if constexpr (S == 9) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 2) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ (threadIdx.x & dup_mask) ^ (threadIdx.x >> S))] = my_codebook_element;
                }
            } else if constexpr (S == 10) {
                uint32_t my_cb_idx = threadIdx.x & s_mask;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 1) & dup_mask))] = my_codebook_element;
                }
            } else if constexpr (S == 11) {
                uint32_t my_cb_idx = threadIdx.x;
                half2 my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
                my_cb_idx += 1024;
                my_codebook_element = codebook[my_cb_idx];
                for (uint32_t i = 0; i < (1<<dup_bits); i+= 1) {
                    smem_codebook[(my_cb_idx << dup_bits)|(i ^ ((threadIdx.x >> 2) & dup_mask))] = my_codebook_element;
                }
            }
            // for (uint32_t i = 0; i < 32; i+= 1) { assert(smem_codebook[(my_cb_idx << 5) + i] == my_codebook_element); }
            __syncthreads();
        }
#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load this 2x2 block of weight tiles
            if (cur_tileIdK >= tileCountK) use_R1=0;
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            if (ki + 1 != this_warp_k && ki % 2 == 1) {
                cur_tileIdK += warps_per_block * weight_block * 2;
                if (cur_tileIdK < tileCountK) {
                    weight_idx = tileIdM * weight_row_step1 + cur_tileIdK * u16_per_compressed_tile1 + laneId * (u16_per_tile_block1 / WARP_SIZE);
                } else {
                    weight_idx = tileIdM * weight_row_step2 + (cur_tileIdK - tileCountK) * u16_per_compressed_tile2 + laneId * (u16_per_tile_block2 / WARP_SIZE);
                }
            }
            if (cur_tileIdK < tileCountK) {
                load_reg_cs<R1>((const uint16_t * __restrict__) compressed1, weight_idx + (1 - ki % 2) * u16_per_tile_block1, laneId, reg_cs_next, reg_cs2_next);
            } else {
                load_reg_cs<R2>((const uint16_t * __restrict__) compressed2, weight_idx + (1 - ki % 2) * u16_per_tile_block2, laneId, reg_cs_next, reg_cs2_next);
            }

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {
#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi++) {
                    uint32_t reg_c, reg_c2;
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;

                    // ** decode weights **
                    // at R = 2, 16 bit -> 8 weights -> 4 half2
                    ditto4 reg_w;
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; j += 1) {
                        uint32_t idx;
                        if (use_R1) {
                            if constexpr(R1 == 2) {
                                idx = reg_c >> (2 * (4-j));
                            } else if constexpr(R1 == 3) {
                                idx = reg_c >> (3 * (4-j));
                            } else if constexpr(R1 == 4) {
                                idx = reg_c >> (4 * (4-j));
                            } else if constexpr(R1 == 5) {
                                idx = reg_c >> (5 * (3-j)+1);
                            } else if constexpr(R1 == 6) {
                                idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                            } else if constexpr(R1 == 7) {
                                idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                            } else if constexpr(R1 == 8) {
                                idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                            } else if constexpr(R1 == 9) {
                                idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                            } else if constexpr(R1 == 10) {
                                idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                            } else if constexpr(R1 == 11) {
                                idx = (j < 2) ? (reg_c >> (11 * (1-j) + 5)) : (reg_c2 >> (11 * (3-j)));
                            } else if constexpr(R1 == 12) {
                                idx = (j < 2) ? (reg_c >> (12 * (1-j) + 4)) : (reg_c2 >> (12 * (3-j)));
                            }
                        } else {
                            if constexpr(R2 == 2) {
                                idx = reg_c >> (2 * (4-j));
                            } else if constexpr(R2 == 3) {
                                idx = reg_c >> (3 * (4-j));
                            } else if constexpr(R2 == 4) {
                                idx = reg_c >> (4 * (4-j));
                            } else if constexpr(R2 == 5) {
                                idx = reg_c >> (5 * (3-j)+1);
                            } else if constexpr(R2 == 6) {
                                idx = (j < 3) ? (reg_c >> (6 * (2-j) + 4)) : reg_c2;
                            } else if constexpr(R2 == 7) {
                                idx = (j < 3) ? (reg_c >> (7 * (2-j))) : reg_c2;
                            } else if constexpr(R2 == 8) {
                                idx = (j < 3) ? (reg_c >> (8 * (2-j))) : reg_c2;
                            } else if constexpr(R2 == 9) {
                                idx = (j < 2) ? (reg_c >> (9 * (1-j) + 7)) : (reg_c2 >> (9 * (3-j)));
                            } else if constexpr(R2 == 10) {
                                idx = (j < 2) ? (reg_c >> (10 * (1-j) + 6)) : (reg_c2 >> (10 * (3-j)));
                            } else if constexpr(R2 == 11) {
                                idx = (j < 2) ? (reg_c >> (11 * (1-j) + 5)) : (reg_c2 >> (11 * (3-j)));
                            } else if constexpr(R2 == 12) {
                                idx = (j < 2) ? (reg_c >> (12 * (1-j) + 4)) : (reg_c2 >> (12 * (3-j)));
                            }
                        }

                        static_assert(L==16);
                        idx = idx * (idx+1);
                        uint32_t masked_idx;
                        if constexpr (S==9){
                            masked_idx = ((idx & 0b0111111111000000) | (laneId << 1)); // this /2 will not be elided automatically
                        } else if constexpr (S==10){
                            masked_idx = ((idx & 0b0111111111100000) | ((laneId << 1) & 0b11110)); // this /2 will not be elided automatically
                        } else if constexpr (S==11){
                            masked_idx = ((idx & 0b0111111111110000) | ((laneId << 2) & 0b1110)); // this /2 will not be elided automatically
                        }
                        __builtin_assume(masked_idx % 2 == 0);
                        reg_w.f16x2[j] = smem_codebook[masked_idx/2];
                        uint32_t selector = 0b00000000'00000000'10000000'00000000;
                        reg_w.u32[j] = reg_w.u32[j] ^ (selector & idx);

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


// L: shift register bit-width
// S: codebook index bit-width
// R: bits per weight
// V: log2(VQ dimension)
template <uint32_t L, uint32_t S, uint32_t R, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__host__ static void decompress_gemm_ptr(
    float *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed,    // m-by-k
    const half2 * __restrict__ x,               // k-by-n
    const half2 * __restrict__ codebook,
    CUstream_st *stream
) {
    static_assert(L <= 16, "Shift register should fit in uint16_t");
    static_assert(L >= S, "Shift register state space must not be smaller than codebook size");
    static_assert(S + V >= 3, "Codebook must have at least eight float16 elements as smem copy operates on uint4");
    static_assert(S==9 || S==10 || S==11, "S=9 or 10 or 11 for now"); 
    // static_assert(S + 5 + V + 1 <= 16, "We can only use 64 KiB shared memory"); // warpSize is 1<<5, sizeof(half) is 1<<1
    static_assert(R==2 || R==3 || R == 4 || R == 5 || R == 6 || R == 7 || R == 8 || R == 9 || R == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(V == 1, "Always quantize two weights at a time");

    static_assert(M % MMA_M == 0);
    static_assert(N <= 8);
    static_assert(K % MMA_K == 0);

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;
    constexpr uint32_t smemCodebookSize = 1<<16;
    constexpr uint32_t smemReduceGatherSize = N * BLOCK_SIZE * sizeof(float);
    constexpr uint32_t smemTotalSize = smemCodebookSize + smemReduceGatherSize;
    cudaFuncSetAttribute(kernel_decompress_gemm<L, S, R, V, M, N, K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemTotalSize);

    kernel_decompress_gemm<L, S, R, V, M, N, K><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, x, codebook);
    
    gpuErrchk(cudaPeekAtLastError());
}

template <uint32_t L, uint32_t S, uint32_t R, uint32_t V>
__host__ static void decompress_ptr(
    half2 *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed,    // m-by-k
    const half2 * __restrict__ codebook,
    uint32_t M,
    uint32_t K,
    CUstream_st *stream
) {
    static_assert(L <= 16, "Shift register should fit in uint16_t");
    static_assert(L >= S, "Shift register state space must not be smaller than codebook size");
    static_assert(S + V >= 3, "Codebook must have at least eight float16 elements as smem copy operates on uint4");
    // static_assert(S + 5 + V + 1 <= 16, "We can only use 64 KiB shared memory"); // warpSize is 1<<5, sizeof(half) is 1<<1
    static_assert(S==9 || S==10 || S==11,  "S=9 or 10 or 11 for now"); 
    static_assert(R==2 || R==3 || R == 4 || R == 5 || R == 6 || R == 7 || R == 8 || R == 9 || R == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(V == 1, "Always quantize two weights at a time");

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;
    constexpr uint32_t smemTotalSize = 1<<16;
    cudaFuncSetAttribute(kernel_decompress<L, S, R, V>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemTotalSize);

    kernel_decompress<L, S, R, V><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed, codebook, M, K);
    
    gpuErrchk(cudaPeekAtLastError());
}


template <uint32_t L, uint32_t S, uint32_t R1, uint32_t R2, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__host__ static void decompress_gemm_comb_ptr(
    float *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed1,    // m-by-k
    const uint32_t *__restrict__ compressed2,    // m-by-k
    const half2 * __restrict__ x,               // k-by-n
    const half2 * __restrict__ codebook,
    CUstream_st *stream
) {
    static_assert(L <= 16, "Shift register should fit in uint16_t");
    static_assert(L >= S, "Shift register state space must not be smaller than codebook size");
    static_assert(S + V >= 3, "Codebook must have at least eight float16 elements as smem copy operates on uint4");
    static_assert(S==9 || S==10 || S==11, "S=9 or 10 or 11 for now"); 
    // static_assert(S + 5 + V + 1 <= 16, "We can only use 64 KiB shared memory"); // warpSize is 1<<5, sizeof(half) is 1<<1
    static_assert(R1==2 || R1==3 || R1 == 4 || R1 == 5 || R1 == 6 || R1 == 7 || R1 == 8 || R1 == 9 || R1 == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(R2==2 || R2==3 || R2 == 4 || R2 == 5 || R2 == 6 || R2 == 7 || R2 == 8 || R2 == 9 || R2 == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(V == 1, "Always quantize two weights at a time");

    static_assert(M % MMA_M == 0);
    static_assert(N <= 8);
    static_assert(K % MMA_K == 0);

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;
    constexpr uint32_t smemCodebookSize = 1<<16;
    constexpr uint32_t smemReduceGatherSize = N * BLOCK_SIZE * sizeof(float);
    constexpr uint32_t smemTotalSize = smemCodebookSize + smemReduceGatherSize;
    cudaFuncSetAttribute(kernel_decompress_gemm_comb<L, S, R1, R2, V, M, N, K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemTotalSize);

    kernel_decompress_gemm_comb<L, S, R1, R2, V, M, N, K><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed1, compressed2, x, codebook);
    
    gpuErrchk(cudaPeekAtLastError());
}

template <uint32_t L, uint32_t S, uint32_t R1, uint32_t R2, uint32_t V, uint32_t M, uint32_t N, uint32_t K>
__host__ static void decompress_gemm_combt_ptr(
    float *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed1,    // m-by-k
    const uint32_t *__restrict__ compressed2,    // m-by-k
    const half2 * __restrict__ x,               // k-by-n
    const half2 * __restrict__ codebook,
    CUstream_st *stream
) {
    static_assert(L <= 16, "Shift register should fit in uint16_t");
    static_assert(L >= S, "Shift register state space must not be smaller than codebook size");
    static_assert(S + V >= 3, "Codebook must have at least eight float16 elements as smem copy operates on uint4");
    static_assert(S==9 || S==10 || S==11, "S=9 or 10 or 11 for now"); 
    // static_assert(S + 5 + V + 1 <= 16, "We can only use 64 KiB shared memory"); // warpSize is 1<<5, sizeof(half) is 1<<1
    static_assert(R1==2 || R1==3 || R1 == 4 || R1 == 5 || R1 == 6 || R1 == 7 || R1 == 8 || R1 == 9 || R1 == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(R2==2 || R2==3 || R2 == 4 || R2 == 5 || R2 == 6 || R2 == 7 || R2 == 8 || R2 == 9 || R2 == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(V == 1, "Always quantize two weights at a time");

    static_assert(M % MMA_M == 0);
    static_assert(N <= 8);
    static_assert(K % MMA_K == 0);

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;
    constexpr uint32_t smemCodebookSize = 1<<16;
    constexpr uint32_t smemReduceGatherSize = N * BLOCK_SIZE * sizeof(float);
    constexpr uint32_t smemTotalSize = smemCodebookSize + smemReduceGatherSize;
    cudaFuncSetAttribute(kernel_decompress_gemm_combt<L, S, R1, R2, V, M, N, K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemTotalSize);

    kernel_decompress_gemm_combt<L, S, R1, R2, V, M, N, K><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed1, compressed2, x, codebook);
    
    gpuErrchk(cudaPeekAtLastError());
}

template <uint32_t L, uint32_t S, uint32_t R1, uint32_t R2, uint32_t V>
__host__ static void decompress_comb_ptr(
    half2 *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed1,    // m-by-k
    const uint32_t *__restrict__ compressed2,    // m-by-k
    const half2 * __restrict__ codebook,
    uint32_t M,
    uint32_t K,
    CUstream_st *stream
) {
    static_assert(L <= 16, "Shift register should fit in uint16_t");
    static_assert(L >= S, "Shift register state space must not be smaller than codebook size");
    static_assert(S + V >= 3, "Codebook must have at least eight float16 elements as smem copy operates on uint4");
    static_assert(S==9 || S==10 || S==11, "S=9 or 10 or 11 for now"); 
    // static_assert(S + 5 + V + 1 <= 16, "We can only use 64 KiB shared memory"); // warpSize is 1<<5, sizeof(half) is 1<<1
    static_assert(R1==2 || R1==3 || R1 == 4 || R1 == 5 || R1 == 6 || R1 == 7 || R1 == 8 || R1 == 9 || R1 == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(R2==2 || R2==3 || R2 == 4 || R2 == 5 || R2 == 6 || R2 == 7 || R2 == 8 || R2 == 9 || R2 == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(V == 1, "Always quantize two weights at a time");

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;
    constexpr uint32_t smemTotalSize = 1<<16;
    cudaFuncSetAttribute(kernel_decompress_comb<L, S, R1, R2, V>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemTotalSize);

    kernel_decompress_comb<L, S, R1, R2, V><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed1, compressed2, codebook, M, K);
    
    gpuErrchk(cudaPeekAtLastError());
}


template <uint32_t L, uint32_t S, uint32_t R1, uint32_t R2, uint32_t V>
__host__ static void decompress_combt_ptr(
    half2 *__restrict__ out,                    // m-by-n
    const uint32_t *__restrict__ compressed1,    // m-by-k
    const uint32_t *__restrict__ compressed2,    // m-by-k
    const half2 * __restrict__ codebook,
    uint32_t M,
    uint32_t K,
    CUstream_st *stream
) {
    static_assert(L <= 16, "Shift register should fit in uint16_t");
    static_assert(L >= S, "Shift register state space must not be smaller than codebook size");
    static_assert(S + V >= 3, "Codebook must have at least eight float16 elements as smem copy operates on uint4");
    static_assert(S==9 || S==10 || S==11, "S=9 or 10 or 11 for now"); 
    // static_assert(S + 5 + V + 1 <= 16, "We can only use 64 KiB shared memory"); // warpSize is 1<<5, sizeof(half) is 1<<1
    static_assert(R1==2 || R1==3 || R1 == 4 || R1 == 5 || R1 == 6 || R1 == 7 || R1 == 8 || R1 == 9 || R1 == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(R2==2 || R2==3 || R2 == 4 || R2 == 5 || R2 == 6 || R2 == 7 || R2 == 8 || R2 == 9 || R2 == 10, "Quantization rate = 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for now");
    static_assert(V == 1, "Always quantize two weights at a time");

    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    constexpr uint32_t gridSize = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;
    constexpr uint32_t smemTotalSize = 1<<16;
    cudaFuncSetAttribute(kernel_decompress_combt<L, S, R1, R2, V>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemTotalSize);

    kernel_decompress_combt<L, S, R1, R2, V><<<gridSize, blockSize, smemTotalSize, stream>>>(out, compressed1, compressed2, codebook, M, K);
    
    gpuErrchk(cudaPeekAtLastError());
}

