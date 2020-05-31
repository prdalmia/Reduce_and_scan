// Author: Nic Olsen

#include <cuda.h>
#include <iostream>
#include <stdio.h>

#include "scan.cuh"

// Scans each block of g_idata separately and writes the result to g_odata.
// g_idata and g_odata are arrays available on device of length n
// Writes the sum of each block to lasts[blockIdx.x]
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
printf("globalBarr is %d\n", *globalBarr);
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
 printf("Setting global sense = sense \n");
}
else { // increase backoff to avoid repeatedly hammering global barrier
// (capped) exponential backoff
backoff = (((backoff << 1) + 1) & (1024-1));
}
}
__syncthreads();

// do exponential backoff to reduce the number of times we pound the global
// barrier
if (*global_sense != *sense) {
for (int i = 0; i < backoff; ++i) { ; }
__syncthreads();
}
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
s = !(*sense);
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

while (*sense != s)
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
               bool* sense)
{
// each SM has MAX_BLOCKS locations in barrierBuffers, so my SM's locations
// start at barrierBuffers[smID*MAX_BLOCKS]
cudaBarrierAtomicSubLocalSRB(perSMBarrierBuffers, numTBs_thisSM, isMasterThread, sense, smID, last_block);
}

/*
Helper function for joining the barrier with the atomic tree barrier.
*/
__device__ void joinBarrier_helperSRB(bool * global_sense,
bool * perSMsense,
bool * done,
unsigned int* global_count,
unsigned int* local_count,
unsigned int* last_block,
const unsigned int numBlocksAtBarr,
const int smID,
const int perSM_blockID,
const int numTBs_perSM,
const bool isMasterThread) {                                 
__syncthreads();
if (numTBs_perSM > 1) {
cudaBarrierAtomicLocalSRB(&local_count[smID], &last_block[smID], smID, numTBs_perSM, isMasterThread, &perSMsense[smID]);

// only 1 TB per SM needs to do the global barrier since we synchronized
// the TBs locally first
if (blockIdx.x == last_block[smID]) {
    if(isMasterThread && perSM_blockID == 0){    
    }
    __syncthreads();
cudaBarrierAtomicSRB(global_count, numBlocksAtBarr, isMasterThread , &perSMsense[smID], global_sense);  
//*done = 1;
}
else {
if(isMasterThread){
while (*global_sense != perSMsense[smID]   ){  
__threadfence();
}
}
__syncthreads();
}    
} else { // if only 1 TB on the SM, no need for the local barriers
    if(isMasterThread){
    perSMsense[smID] = ~perSMsense[smID];
    }
    __syncthreads();
cudaBarrierAtomicSRB(global_count, numBlocksAtBarr, isMasterThread,  &perSMsense[smID], global_sense);
}
}


__device__ void kernelAtomicTreeBarrierUniqSRB( bool * global_sense,
bool * perSMsense,
bool * done,
unsigned int* global_count,
unsigned int* local_count,
unsigned int* last_block,
const int NUM_SM)
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
isMasterThread);

}


__global__ void hillis_steele(float* g_odata, float* lasts,  float* g_idata, unsigned int n, bool write_lasts, bool * global_sense,
    bool * perSMsense,
    bool * done,
    unsigned int* global_count,
    unsigned int* local_count,
    unsigned int* last_block,
    const int NUM_SM) {
    extern volatile __shared__ float s[];
    float *tmp1;
    float * tmp2;
    bool write_p = write_lasts;
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
    __syncthreads();
    if (write_p && threadIdx.x == 0) {
        unsigned int block_end = blockIdx.x * blockDim.x + blockDim.x - 1;
        lasts[blockIdx.x] = s[pout * blockDim.x + blockDim.x - 1] + g_idata[block_end];
    }
    if(a==n){
    __syncthreads();
    }
    kernelAtomicTreeBarrierUniqSRB(global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM);      
    if(a == n ){
      tmp1 = g_idata;
      tmp2 = g_odata;
      g_idata = lasts;
      g_odata = lasts;
      write_p = false;
      a = (n + blockDim.x - 1) / blockDim.x;
    }
   }

    g_odata = tmp2;
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
    unsigned int shmem = 2 * threads_per_block * sizeof(float);
   // hillis_steele<<<nBlocks, threads_per_block, shmem>>>(out, lasts, in, n, true);
    //cudaDeviceSynchronize();
   //for (unsigned int a = n; a > 1; a = (a + threads_per_block - 1) / threads_per_block) {
    unsigned int* global_count;
    unsigned int* local_count; 
    unsigned int *last_block;
    bool * global_sense;
    bool* perSMsense;
    bool * done;
    int NUM_SM = 80;
    cudaMallocManaged((void **)&global_sense,sizeof(bool));
    cudaMallocManaged((void **)&done,sizeof(bool));
    cudaMallocManaged((void **)&perSMsense,NUM_SM*sizeof(bool));
    cudaMallocManaged((void **)&last_block,sizeof(unsigned int)*(NUM_SM));
    cudaMallocManaged((void **)&local_count,  NUM_SM*sizeof(unsigned int));
    cudaMallocManaged((void **)&global_count,sizeof(unsigned int));
    
    cudaMemset(global_sense, false, sizeof(bool));
    cudaMemset(done, false, sizeof(bool));
    cudaMemset(global_count, 0, sizeof(unsigned int));

    for (int i = 0; i < NUM_SM; ++i) {
       cudaMemset(&perSMsense[i], false, sizeof(bool));
       cudaMemset(&local_count[i], 0, sizeof(unsigned int));
       cudaMemset(&last_block[i], 0, sizeof(unsigned int));
     }
    //cudaLaunchCooperativeKernel((void*)hillis_steele, nBlocks, threads_per_block,  kernelArgs, shmem, 0);
    hillis_steele<<<nBlocks, threads_per_block, shmem>>>(out, lasts, in, n, write_lasts, global_sense, perSMsense, done, global_count, local_count, last_block, NUM_SM);
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
