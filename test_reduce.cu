// Author: Nic Olsen

#include <cuda.h>
#include <iostream>

#include "reduce.cuh"

int main(int argc, char* argv[]) {
    int N = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    int* arr = new int[N];

    for (int i = 0; i < N; i++) {
        arr[i] = 1;
    }
    int correct_sum = N;

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int sum = reduce(arr, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Correct sum: " << correct_sum << std::endl;
    std::cout << "time (ms) " << ms << std::endl;

    delete[] arr;
    return 0;
}
