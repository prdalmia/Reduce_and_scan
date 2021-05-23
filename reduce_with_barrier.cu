// Author: Nic Olsen

#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include "reduce.cuh"
__device__ __forceinline__ bool ld_gbl_cg (const bool *addr)
{
    short t;
#if defined(__LP64__) || defined(_WIN64)
    asm ("ld.global.cg.u8 %0, [%1];" : "=h"(t) : "l"(addr));
#else
    asm ("ld.global.cg.u8 %0, [%1];" : "=h"(t) : "r"(addr));
#endif
    return (bool)t;
}

inline __device__ void cudaBarrierAtomicNaiveSRB(unsigned int *globalBarr,
  // numBarr represents the number
  // of TBs going to the barrier
  const unsigned int numBarr,
  int backoff,
  const bool isMasterThread,
  bool * volatile global_sense) {
__syncthreads();
__shared__ bool s;
if (isMasterThread) {
s = !(ld_gbl_cg(global_sense));
__threadfence();
// atomicInc effectively adds 1 to atomic for each TB that's part of the
// global barrier.
atomicInc(globalBarr, 0x7FFFFFFF);
 //printf("Global barr is %d and numbarr is %d\n", *globalBarr, numBarr);
}
__syncthreads();

while (ld_gbl_cg(global_sense) != s) {
if (isMasterThread) {
/*
Once the atomic's value == numBarr, then reset the value to 0 and
proceed because all of the TBs have reached the global barrier.
*/
if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
// atomicCAS acts as a load acquire, need TF to enforce ordering
__threadfence();
*global_sense = s;
} else { // increase backoff to avoid repeatedly hammering global barrier
// (capped) exponential backoff
backoff = (((backoff << 1) + 1) & (1024 - 1));
}
}
__syncthreads();

// do exponential backoff to reduce the number of times we pound the global
// barrier
if (ld_gbl_cg(global_sense) != s) {
for (int i = 0; i < backoff; ++i) {
;
}
__syncthreads();
}
}
}

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
  //printf("Global barr is %d and numBarr is %d\n", *globalBarr, numBarr);
  }
  __syncthreads();
  
  while (*global_sense != ld_gbl_cg(sense))
  {
  if (isMasterThread)
  {
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
  *global_sense = ld_gbl_cg(sense);
  }
  else { // increase backoff to avoid repeatedly hammering global barrier
  // (capped) exponential backoff
  backoff = (((backoff << 1) + 1) & (1024-1));
  }
  }
  __syncthreads();
  
  // do exponential backoff to reduce the number of times we pound the global
  // barrier
  if(isMasterThread){
  //if (*global_sense != *sense) {
  //*for (int i = 0; i < backoff; ++i) { ; }
  }
  __syncthreads();
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
  
  inline __device__ void cudaBarrierAtomicSubLocalSRB(unsigned int * perSMBarr,
         const unsigned int numTBs_thisSM,
         const bool isMasterThread,
         bool * sense,
         const int smID,
         unsigned int* last_block)
  
  {
  __syncthreads();
  __shared__ bool s;
  if (isMasterThread)
  {
  s = !(ld_gbl_cg(sense));
  // atomicInc acts as a store release, need TF to enforce ordering locally
  __threadfence_block();
  /*
  atomicInc effectively adds 1 to atomic for each TB that's part of the
  barrier.  For the local barrier, this requires using the per-CU
  locations.
  */
  atomicInc(perSMBarr, 0x7FFFFFFF);
  }
  __syncthreads();
  
  while (ld_gbl_cg(sense) != s)
  {
  if (isMasterThread)
  {
  /*
  Once all of the TBs on this SM have incremented the value at atomic,
  then the value (for the local barrier) should be equal to the # of TBs
  on this SM.  Once that is true, then we want to reset the atomic to 0
  and proceed because all of the TBs on this SM have reached the local
  barrier.
  */
  if (atomicCAS(perSMBarr, numTBs_thisSM, 0) == numTBs_thisSM) {
  // atomicCAS acts as a load acquire, need TF to enforce ordering
  // locally
  __threadfence_block();
  *sense = s;
  *last_block = blockIdx.x;
  }
  }
  __syncthreads();
  }
  }
  
  //Implements PerSM sense reversing barrier
  inline __device__ void cudaBarrierAtomicLocalSRB(unsigned int * perSMBarrierBuffers,
               unsigned int * last_block,
               const unsigned int smID,
               const unsigned int numTBs_thisSM,
               const bool isMasterThread,
               bool*  sense)
  {
  // each SM has MAX_BLOCKS locations in barrierBuffers, so my SM's locations
  // start at barrierBuffers[smID*MAX_BLOCKS]
  cudaBarrierAtomicSubLocalSRB(perSMBarrierBuffers, numTBs_thisSM, isMasterThread, sense, smID, last_block);
  }
  
  /*
  Helper function for joining the barrier with the atomic tree barrier.
  */
  __device__ void joinBarrier_helperSRB(bool * volatile global_sense,
  bool * volatile perSMsense,
  bool *  done,
  unsigned int* global_count,
  unsigned int* local_count,
  unsigned int* last_block,
  const unsigned int numBlocksAtBarr,
  const int smID,
  const int perSM_blockID,
  const int numTBs_perSM,
  const bool isMasterThread,
  bool naive) {
     //*done = 0;                                 
  __syncthreads();
  if (numTBs_perSM > 1 && naive == false) {
  cudaBarrierAtomicLocalSRB(&local_count[smID], &last_block[smID], smID, numTBs_perSM, isMasterThread, &perSMsense[smID]);
  
  // only 1 TB per SM needs to do the global barrier since we synchronized
  // the TBs locally first
  if (blockIdx.x == last_block[smID]) {
  cudaBarrierAtomicSRB(global_count, numBlocksAtBarr, isMasterThread , &perSMsense[smID], global_sense);
 
  }
  else {
  if(isMasterThread){
    while (*global_sense != ld_gbl_cg(&perSMsense[smID])){  
        __threadfence();
  }
  } 
  __syncthreads();
  }    
  } else { // if only 1 TB on the SM, no need for the local barriers
    __shared__ int backoff;
    if (isMasterThread) {
      backoff = 1;
    }
    __syncthreads();
  cudaBarrierAtomicNaiveSRB(global_count, (numBlocksAtBarr*numTBs_perSM), backoff,  isMasterThread,  global_sense);
  }
  }
  
  
  __device__ void kernelAtomicTreeBarrierUniqSRB( bool * volatile global_sense,
  bool * volatile perSMsense,
  bool * done,
  unsigned int* global_count,
  unsigned int* local_count,
  unsigned int* last_block,
  const int NUM_SM,
  bool naive)
  {
  
  // local variables
  // thread 0 is master thread
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
  (threadIdx.z == 0));
  // represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
  // fewer TBs than SMs).
  const unsigned int numBlocksAtBarr = ((gridDim.x < NUM_SM) ? gridDim.x :
  NUM_SM);
  const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID
  
  // all thread blocks on the same SM access unique locations because the
  // barrier can't ensure DRF between TBs
  const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  
  int numTBs_perSM = (int)ceil((float)gridDim.x / numBlocksAtBarr);
  
  
  joinBarrier_helperSRB(global_sense, perSMsense, done, global_count, local_count, last_block,
  numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
  isMasterThread, naive);
  }
/*
__device__ void __gpu_sync(int blocks_to_synch)
{
    __syncthreads();
    //thread ID in a block
    int tid_in_block= threadIdx.x;


    // only thread 0 is used for synchronization
    if (tid_in_block == 0)
    {
        atomicAdd((int *)&g_mutex, 1);
        //only when all blocks add 1 to g_mutex will
        //g_mutex equal to blocks_to_synch
        while(g_mutex < blocks_to_synch);
    }
    __syncthreads();
}
*/
__global__ void reduce_kernel(int* g_idata, int* g_odata, unsigned int N, int* output, bool * volatile  global_sense,
    bool * volatile perSMsense,
    bool * volatile done,
    unsigned int* global_count,
    unsigned int* local_count,
    unsigned int* last_block,
    const int NUM_SM,
    bool naive,
    long long int* time) {
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
 long long int start = clock64(); 
 kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive);     
 long long int stop = clock64();
  if(i == 0){
  *time = (stop - start);
  }	 

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
    unsigned int* local_count; 
    unsigned int *last_block;
    long long int* time;
    bool * volatile global_sense;
    bool* volatile perSMsense;
    bool * volatile done;
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&output, sizeof(int));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int NUM_SM = deviceProp.multiProcessorCount;
    bool naive = false;
    cudaMallocManaged((void **)&global_sense,sizeof(bool));
    cudaMallocManaged((void **)&done,sizeof(bool));
    cudaMallocManaged((void **)&perSMsense,NUM_SM*sizeof(bool));
    cudaMallocManaged((void **)&last_block,sizeof(unsigned int)*(NUM_SM));
    cudaMallocManaged((void **)&local_count,  NUM_SM*sizeof(unsigned int));
    cudaMallocManaged((void **)&global_count,sizeof(unsigned int));
    cudaMallocManaged(&time, sizeof(long long int ));
    int clockrate = deviceProp.clockRate;
    cudaMemset(global_sense, false, sizeof(bool));
    cudaMemset(done, false, sizeof(bool));
    cudaMemset(global_count, 0, sizeof(unsigned int));

    for (int i = 0; i < NUM_SM; ++i) {
       cudaMemset(&perSMsense[i], false, sizeof(bool));
       cudaMemset(&local_count[i], 0, sizeof(unsigned int));
       cudaMemset(&last_block[i], 0, sizeof(unsigned int));
     }
    cudaMemcpy(a, arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //for (unsigned int n = N; n > 1; n = (n + threads_per_block - 1) / threads_per_block) {
        reduce_kernel<<<(N + threads_per_block - 1) / threads_per_block, threads_per_block,
                        threads_per_block * sizeof(int)>>>(a, b, N, output, global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM, naive, time);

        // Swap input and output arrays
        //int* tmp = a;
        //a = b;
        //b = tmp;
   // }
   cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("time cuda only(ms) is %f and barries time is %llu and clock rate is %d\n", ms, ((*time)/clockrate), clockrate) ;

    int sum = *output;

    cudaFree(a);
    cudaFree(b);

    return sum;
}
