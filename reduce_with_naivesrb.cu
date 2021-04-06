// Author: Nic Olsen

#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include "reduce.cuh"

inline __device__ void cudaBarrierAtomicSubSRB(unsigned int * globalBarr,
    // numBarr represents the number of
    // TBs going to the barrier
    const unsigned int numBarr,
    int backoff,
    const bool isMasterThread,
    bool * volatile sense,
    bool * volatile global_sense)
  {
  __syncthreads();
  if (isMasterThread)
  {
  //printf("Inside global Barrier for blockID %d and sense is %d and global sense is %d\n", blockIdx.x, *sense, *global_sense);
  // atomicInc acts as a store release, need TF to enforce ordering
  __threadfence();
  // atomicInc effectively adds 1 to atomic for each TB that's part of the
  // global barrier.
  atomicInc(globalBarr, 0x7FFFFFFF);
  printf("Global barr is %d\n", *globalBarr);
  }
  __syncthreads();
  
  while (*global_sense != *sense)
  {
  if (isMasterThread)
  {
  //printf("Global sense hili\n");
  /*
  For the tree barrier we expect only 1 TB from each SM to enter the
  global barrier.  Since we are assuming an equal amount of work for all
  SMs, we can use the # of TBs reaching the barrier for the compare value
  here.  Once the atomic's value == numBarr, then reset the value to 0 and
  proceed because all of the TBs have reached the global barrier.
  */
  if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
  // atomicCAS acts as a load acquire, need TF to enforce ordering
  __threadfence();
  *global_sense = *sense;
  }
  else { // increase backoff to avoid repeatedly hammering global barrier
  // (capped) exponential backoff
  backoff = (((backoff << 1) + 1) & (64-1));
  }
  }
  __syncthreads();
  
  // do exponential backoff to reduce the number of times we pound the global
  // barrier
    if (*global_sense != *sense) {
    for (int i = 0; i < backoff; ++i) { ; }
    }
    __syncthreads();
    //}
    }
}
  
  inline __device__ void cudaBarrierAtomicSRB(unsigned int * barrierBuffers,
  // numBarr represents the number of
  // TBs going to the barrier
  const unsigned int numBarr,
  const bool isMasterThread,
  bool * volatile sense,
  bool * volatile global_sense)
  {
  __shared__ int backoff;
  
  if (isMasterThread) {
  backoff = 1;
  }
  __syncthreads();
  
  cudaBarrierAtomicSubSRB(barrierBuffers, numBarr, backoff, isMasterThread, sense, global_sense);
  }
  
  
  
  /*
  Helper function for joining the barrier with the atomic tree barrier.
  */
  __device__ void joinBarrier_helperNaiveSRB(bool * global_sense,
  bool * perSMsense,
  unsigned int* global_count,
  const unsigned int numBlocksAtBarr,
  const int smID,
  const int perSM_blockID,
  const int numTBs_perSM,
  const bool isMasterThread) {         
    if(isMasterThread) {
        perSMsense[blockIdx.x] = !(*global_sense);
      }
      __syncthreads();
                            
  cudaBarrierAtomicSRB(global_count, numBlocksAtBarr, isMasterThread,  &perSMsense[blockIdx.x], global_sense);
  }
  
  
  __device__ void kernelAtomicTreeBarrierUniqNaiveSRB( bool * global_sense,
  bool * perSMsense,
  unsigned int* global_count,
  const int NUM_SM)
  {
  
  // local variables
  // thread 0 is master thread
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
  (threadIdx.z == 0));
  // represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
  // fewer TBs than SMs).
  const unsigned int numBlocksAtBarr = gridDim.x;
  const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID
  
  // all thread blocks on the same SM access unique locations because the
  // barrier can't ensure DRF between TBs
  const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  
  int numTBs_perSM = (int)ceil((float)gridDim.x / numBlocksAtBarr);
  
  
  joinBarrier_helperNaiveSRB(global_sense, perSMsense, global_count, numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
  isMasterThread);
  }
__global__ void reduce_kernel(int* g_idata, int* g_odata, unsigned int N, int* output, bool * global_sense,
    bool * perSMsense,
    unsigned int* global_count,
    const int NUM_SM) {
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
    __syncthreads();
 kernelAtomicTreeBarrierUniqNaiveSRB(global_sense, perSMsense, global_count, NUM_SM);     
    int* tmp = g_idata;
    g_idata = g_odata;
    g_odata = tmp;
}
 *output = g_idata[0];
}


__host__ int reduce(const int* arr, unsigned int N, unsigned int threads_per_block ) {
    // Workspace NOTE: Could be smaller
    int* a;
    int* b;
    int * output;
    unsigned int* global_count;
    bool * global_sense;
    bool* perSMsense;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&output, sizeof(int));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int NUM_SM = deviceProp.multiProcessorCount;
    cudaMallocManaged((void **)&global_sense,sizeof(bool));
    cudaMallocManaged((void **)&perSMsense,((N + threads_per_block - 1) / threads_per_block) *sizeof(bool));
    cudaMallocManaged((void **)&global_count,sizeof(unsigned int));
    
    cudaMemset(global_sense, false, sizeof(bool));
    cudaMemset(global_count, 0, sizeof(unsigned int));

    for (int i = 0; i < ((N + threads_per_block - 1) / threads_per_block); ++i) {
       cudaMemset(&perSMsense[i], false, sizeof(bool));
     }
    cudaMemcpy(a, arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //for (unsigned int n = N; n > 1; n = (n + threads_per_block - 1) / threads_per_block) {
        reduce_kernel<<<(N + threads_per_block - 1) / threads_per_block, threads_per_block,
                        threads_per_block * sizeof(int)>>>(a, b, N, output, global_sense, perSMsense, global_count, NUM_SM);

        // Swap input and output arrays
        //int* tmp = a;
        //a = b;
        //b = tmp;
   // }
   cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "time cuda only(ms) " << ms << std::endl;

    int sum = *output;

    cudaFree(a);
    cudaFree(b);

    return sum;
}
