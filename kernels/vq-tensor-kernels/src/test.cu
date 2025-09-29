#include <cstdio>
#include <cuda.h>

#include "inference.h"
#include "inference.cu"

#define M 4096
#define N 1
#define K 4096

int main() {
    constexpr uint32_t S = 9;
    constexpr uint32_t R = 2;

    size_t m = M;
    size_t n = N;
    size_t k = K;

    float *out;
    uint32_t *compressed;
    half2 *x;
    half2 *codebook;
    uint32_t codebook_ones[(1 << R) * m];
    for (int i = 0; i < (1 << R) * m / 2; i++) {
        codebook_ones[i] = 0x3c003c00;
    }
    uint16_t x_ones[N * K];
    for (int i = 0; i < N * K; i++) {
        x_ones[i] = 0x3c00;
    }

    gpuErrchk(cudaMalloc(&out, m * n * sizeof *out));
    gpuErrchk(cudaMalloc(&compressed, m * k * R / CHAR_BIT));
    gpuErrchk(cudaMalloc(&x, k * n * sizeof *x / 2));
    gpuErrchk(cudaMalloc(&codebook, (1<<R) * m * sizeof(half)));

    gpuErrchk(cudaMemset(out, 0, m * n * sizeof *out));
    gpuErrchk(cudaMemset(compressed, 0, m * k * R / CHAR_BIT));
    gpuErrchk(cudaMemcpy(x, x_ones, k * n * sizeof *x / 2, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(codebook, codebook_ones, (1<<R) * m * sizeof(half), cudaMemcpyHostToDevice));

    decompress_matvec_ptr_sq<16, 9, R, 1, M, N, K, -1>((float*) out, (uint32_t*)compressed, (half2*)x, (half*)codebook, NULL);
    gpuErrchk(cudaDeviceSynchronize());

    float *out_h;
    gpuErrchk(cudaMallocHost(&out_h, m * n * sizeof *out));
    gpuErrchk(cudaMemcpy(out_h, out, m * n * sizeof *out, cudaMemcpyDeviceToHost));
    int incorrect = 0;
    float sum = 0.f;
    for (uint32_t i = 0; i < m * n; i += 1) {
        incorrect += out_h[i] != K;
        sum += out_h[i];
        if (out_h[i] != K) {
            printf("incorrect: ref: %f actual: %f\n", (float)K, out_h[i]);
        }
    }
    printf("incorrect = %d\n", incorrect);
    printf("sum = %f\n", sum);
    gpuErrchk(cudaFreeHost(out_h));

    gpuErrchk(cudaFree(out));
    gpuErrchk(cudaFree(compressed));
    gpuErrchk(cudaFree(x));
    gpuErrchk(cudaFree(codebook));

    return incorrect;
}
