// Author: Nic Olsen

#include <cuda.h>
#include <iostream>
#include <stdio.h>

#include "scan.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// Scans each block of g_idata separately and writes the result to g_odata.
// g_idata and g_odata are arrays available on device of length n
// Writes the sum of each block to lasts[blockIdx.x]


__global__ void hillis_steele(float* g_odata, float* lasts,  float* g_idata, unsigned int n, bool write_lasts) {
    extern volatile __shared__ float s[];
    
    float *tmp1;
    float * tmp2;
    float* tmp3;
    bool write_p = write_lasts;
    cg::grid_group grid = cg::this_grid(); 
    int a = n;
    int tid = threadIdx.x;
    unsigned int index = blockDim.x * blockIdx.x + tid;
    int pout = 0;
    int pin = 1;
   for( int i = 0 ; i < 2 ; i++){
       pout = 0;
       pin = 1;
    if (index >= a) {
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
    if (index < a ) {
        g_odata[index] = s[pout * blockDim.x + tid];
    }

    if (write_p && threadIdx.x == 0) {
        unsigned int block_end = blockIdx.x * blockDim.x + blockDim.x - 1;
        lasts[blockIdx.x] = s[pout * blockDim.x + blockDim.x - 1] + g_idata[block_end];
    }
    __syncthreads();
    cg::sync(grid); 
    if(a == n){
      tmp1 = g_idata;
      tmp2 = g_odata;
      tmp3 = lasts;
      g_idata = lasts;
      g_odata = lasts;
      lasts = nullptr;
      write_p = false;
      a = (n + blockDim.x - 1) / blockDim.x;
    }
   }
   cg::sync(grid); 
    lasts = tmp3;
    g_odata = tmp2;
    __syncthreads();
    if (index < n) {
        g_odata[index] = g_odata[index] + lasts[blockIdx.x];
      //  printf("g_odata is %f at index %d\n", g_odata[index], index);
    }
}

// Increment each element corresponding to block b_i of arr by lasts[b_i]
__global__ void inc_blocks(float* arr, float* lasts, unsigned int n) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        arr[index] = arr[index] + lasts[blockIdx.x];
    }
}


__host__ void scan( float* in, float* out, unsigned int n, unsigned int threads_per_block) {
    // Sort each block indiviually
    unsigned int nBlocks = (n + threads_per_block - 1) / threads_per_block;
    float* lasts;
    cudaMallocManaged(&lasts, nBlocks * sizeof(float));
    bool write_lasts = true;
    unsigned int shmem = 4 * threads_per_block * sizeof(float);
   // hillis_steele<<<nBlocks, threads_per_block, shmem>>>(out, lasts, in, n, true);
    //cudaDeviceSynchronize();
   //for (unsigned int a = n; a > 1; a = (a + threads_per_block - 1) / threads_per_block) {
    void *kernelArgs[] = {
        (void *)&out,  (void *)&lasts, (void *)&in, (void *)&n, (void *)&write_lasts  
    };
    cudaLaunchCooperativeKernel((void*)hillis_steele, nBlocks, threads_per_block,  kernelArgs, shmem, 0);
    //hillis_steele<<<nBlocks, threads_per_block, shmem>>>(out, lasts, in, n, true);
    // Swap input and output arrays
 //   float* tmp = in;
 //   in = lasts;
 //   lasts = tmp;
 //   std::cout << in[a-1] << std::endl;
 //  }
    // Scan lasts
    //hillis_steele<<<1, threads_per_block, shmem>>>(lasts, nullptr, lasts, nBlocks, false);
    cudaDeviceSynchronize();

    cudaFree(lasts);
}
