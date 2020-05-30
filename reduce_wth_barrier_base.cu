// Author: Nic Olsen

#include <iostream>
#include <stdio.h>
#include "reduce.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

 __global__ void reduce_kernel(int* g_idata, int* g_odata, unsigned int N) {
    extern __shared__ int sdata[];
  
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int n = N; n > 1; n = (n + blockDim.x - 1) / blockDim.x){
    if (i < n) {
        sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }

    __syncthreads();

    // Sequential addressing alteration
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write out reduced portion of the output
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
        printf("Sum for blockIdx.x is %d and %d\n", blockIdx.x, g_odata[blockIdx.x]);
    }

    cg::grid_group grid = cg::this_grid(); 
    cg::sync(grid);

    int* tmp = g_idata;
    g_idata = g_odata;
    g_odata = tmp;
}

}

__host__ int reduce(const int* arr, unsigned int N, unsigned int threads_per_block) {
    // Workspace NOTE: Could be smaller
    int* a;
    int* b;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMemcpy(a, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    //for (unsigned int n = N; n > 1; n = (n + threads_per_block - 1) / threads_per_block) {
        reduce_kernel<<<(N + threads_per_block - 1) / threads_per_block, threads_per_block,
                        threads_per_block * sizeof(int)>>>(a, b, N);

        // Swap input and output arrays
        //int* tmp = a;
        //a = b;
        //b = tmp;
   // }
    cudaDeviceSynchronize();

    int sum = a[0];

    cudaFree(a);
    cudaFree(b);

    return sum;
}
