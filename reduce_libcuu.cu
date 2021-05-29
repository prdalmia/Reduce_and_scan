// Author: Nic Olsen

#include <iostream>
#include <stdio.h>
#include "reduce.cuh"
#include <cuda/std/barrier>
using barrier = cuda::barrier<cuda::thread_scope_device>;

 __global__ void reduce_kernel(int* g_idata, int* g_odata, unsigned int N, int* output , barrier* sync_point ) {
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
    
  // __syncthreads();
   
    sync_point->arrive_and_wait();
   
    //__threadfence();
    
    int* tmp = g_idata;
    g_idata = g_odata;
    g_odata = tmp;
}

*output = g_idata[0];
/*
long long int stop = clock64();
if(i == 0){
    *time = (stop - start);
    }	
*/
}

__host__ int reduce(const int* arr, unsigned int N, unsigned int threads_per_block) {
    // Workspace NOTE: Could be smaller
    int* a;
    int* b;
    int* output;
    //long long int* time;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    barrier* sync_point;
    cudaMallocManaged(&sync_point, sizeof(barrier));
    new (sync_point) barrier(((N + threads_per_block - 1) / threads_per_block));
    
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&output, sizeof(int));
    //cudaMallocManaged(&time, sizeof(long long int ));
    cudaMemcpy(a, arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_kernel<<<(N + threads_per_block - 1) / threads_per_block, threads_per_block,
    threads_per_block * sizeof(int)>>>(a, b, N, output, sync_point);
      cudaEventRecord(stop);
    cudaDeviceSynchronize();
    sync_point->~barrier();
    cudaFree(sync_point);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    //std::cout << "time cuda only(ms) " << ms <<" barrier time is " << *time <<   std::endl;
    printf("time cuda only(ms) is %f", ms) ;
    int sum = *output;

    cudaFree(a);
    cudaFree(b);

    return sum;
}
