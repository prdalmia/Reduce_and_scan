// Author: Nic Olsen

#include <cuda.h>
#include <iostream>

#include "scan.cuh"

// Scans each block of g_idata separately and writes the result to g_odata.
// g_idata and g_odata are arrays available on device of length n
// Writes the sum of each block to lasts[blockIdx.x]
__global__ void hillis_steele(float* g_odata, float* lasts, const float* g_idata, unsigned int n, bool write_lasts) {
    extern volatile __shared__ float s[];

    int tid = threadIdx.x;
    unsigned int index = blockDim.x * blockIdx.x + tid;
    int pout = 0;
    int pin = 1;

    if (index >= n) {
        s[tid] = 0.f;
    } else if (tid == 0) {
        s[tid] = 0.f;
    } else {
        s[tid] = g_idata[index - 1];
    }
    
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (tid >= offset) {
            s[pout * blockDim.x + tid] = s[pin * blockDim.x + tid] + s[pin * blockDim.x + tid - offset];
        } else {
            s[pout * blockDim.x + tid] = s[pin * blockDim.x + tid];
        }
        __syncthreads();
    }
    if (index < n) {
        g_odata[index] = s[pout * blockDim.x + tid];
        if(gridDim.x == 1)
        printf("g_odata is %f at index %d\n", g_odata[index], index);
    }

    if (write_lasts && threadIdx.x == 0) {
        unsigned int block_end = blockIdx.x * blockDim.x + blockDim.x - 1;
        lasts[blockIdx.x] = s[pout * blockDim.x + blockDim.x - 1] + g_idata[block_end];
    }
}

// Increment each element corresponding to block b_i of arr by lasts[b_i]
__global__ void inc_blocks(float* arr, const float* lasts, unsigned int n) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        arr[index] = arr[index] + lasts[blockIdx.x];
    }
}

__host__ void scan(float* in, float* out, unsigned int n, unsigned int threads_per_block) {
    // Sort each block indiviually
    unsigned int nBlocks = (n + threads_per_block - 1) / threads_per_block;
    float* lasts;
    cudaMallocManaged(&lasts, nBlocks * sizeof(float));
    unsigned int shmem = 2 * threads_per_block * sizeof(float);
    hillis_steele<<<nBlocks, threads_per_block, shmem>>>(out, lasts, in, n, true);
    cudaDeviceSynchronize();

    // Scan lasts
    
    hillis_steele<<<1, threads_per_block, shmem>>>(lasts, nullptr, lasts, nBlocks, false);
    cudaDeviceSynchronize();

    // Add starting value to each block
    inc_blocks<<<nBlocks, threads_per_block>>>(out, lasts, n);
    cudaDeviceSynchronize();

    cudaFree(lasts);
}
