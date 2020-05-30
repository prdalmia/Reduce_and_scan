// Author: Nic Olsen

#include <iostream>
#include <stdio.h>
#include "reduce.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

 __global__ void reduce_kernel(int* g_idata, int* g_odata, unsigned int N, int* output) {
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
    }
    cg::grid_group grid = cg::this_grid(); 
    cg::sync(grid);
    int* tmp = g_idata;
    g_idata = g_odata;
    g_odata = tmp;
}

if(tid ==0 and blockIdx.x ==0){
    *output = g_idata[0];
} 
 
}

__host__ int reduce(const int* arr, unsigned int N, unsigned int threads_per_block) {
    // Workspace NOTE: Could be smaller
    int* a;
    int* b;
    int* output;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&output, sizeof(int));
    cudaMemcpy(a, arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //for (unsigned int n = N; n > 1; n = (n + threads_per_block - 1) / threads_per_block) {
        reduce_kernel<<<(N + threads_per_block - 1) / threads_per_block, threads_per_block,
                        threads_per_block * sizeof(int)>>>(a, b, N, output);

        // Swap input and output arrays
        //int* tmp = a;
        //a = b;
        //b = tmp;
   // }cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "time cuda only(ms) " << ms << std::endl;

    int sum1 = *output;
    printf("sum1 is %d", *output);
    int sum = a[0];

    cudaFree(a);
    cudaFree(b);

    return sum;
}
