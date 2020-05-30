// Author: Nic Olsen

#include <cuda.h>

#include <iostream>

#include "scan.cuh"

int main(int argc, char* argv[]) {
    unsigned int n = std::atoi(argv[1]);
    unsigned int threads_per_block = 1024;

    float* arr;
    cudaMallocManaged(&arr, n * sizeof(float));
    for (unsigned int i = 0; i < n; i++) {
        arr[i] = 1.f;
    }

    float* output;
    cudaMallocManaged(&output, n * sizeof(float));

    scan(arr, output, n, threads_per_block);

    bool correct = true;
    for (unsigned int i = 0; i < n; i++) {
        correct = correct && (output[i] == i);
        std::cout << output[i] << std::endl;
    }
    if (correct) {
        std::cout << "CORRECT" << std::endl;
    } else {
        std::cout << "INCORRECT" << std::endl;
    }

    cudaFree(arr);
    cudaFree(output);
    return 0;
}
