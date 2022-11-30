#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <chrono>

#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_fp16.h>
#include <cufftXt.h>

// FP16 if specified, otherwise FP32
#ifdef HALF_PRECISION
typedef half2 Complex;
const cudaDataType_t DTYPE = CUDA_C_16F;
#else
typedef float2 Complex;
const cudaDataType_t DTYPE = CUDA_C_32F;
#endif

/**
 * Performs the experiment with the given parameters
 * @param vectorSize the vector size (n) used for the experiment
 * @param batchSize the batch size (m) used for the experiment
 * @param numIterations the number of iterations in the experiment (used for average iteration time)
 */
void runTest(long long vectorSize, long long batchSize, long long numIterations){

    // Events used for timing
    float time;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Allocate host memory
    Complex* h_data = new Complex[vectorSize * batchSize];
    for (long long i = 0; i < vectorSize * batchSize; i++)
        h_data[i].x = rand() / (float)RAND_MAX, h_data[i].y = rand() / (float)RAND_MAX;

    // Allocate device memory
    Complex* d_data;
    checkCudaErrors(cudaMalloc(&d_data, vectorSize * batchSize * sizeof(Complex)));

    // Create FFT plan
    cufftHandle plan;
    checkCudaErrors(cufftCreate(&plan));
    size_t ws = 0;
    checkCudaErrors(cufftXtMakePlanMany(plan, 1, &vectorSize, NULL, 1, 1, DTYPE, NULL, 1, 1, DTYPE, batchSize, &ws, DTYPE));

    // Copy to device
    checkCudaErrors(cudaMemcpy(d_data, h_data, vectorSize * batchSize * sizeof(Complex), cudaMemcpyHostToDevice));

    // Execute FFT
    checkCudaErrors(cudaEventRecord(start, 0));

    std::cout << "START" << std::endl;

    for (int i = 0; i < numIterations; i++) {
        checkCudaErrors(cufftXtExec(plan, (void *) d_data, (void *) d_data, CUFFT_FORWARD));
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    std::cout << "STOP" << std::endl;

    // Print results
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    std::cout << time / numIterations << std::endl;

    // Cleanup
    checkCudaErrors(cufftDestroy(plan));
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaDeviceReset());

}

/**
 * Parses the given parameters and executes the experiment.
 * Execution format: ./fft VECTOR_SIZE BATCH_SIZE NUM_ITERATIONS
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv){

    assert(argc == 4);

    // Parse VECTOR_SIZE
    long long vectorSize = atol(argv[1]);

    // Parse BATCH_SIZE
    long long batchSize = atol(argv[2]);

    // Parse NUM_ITERATIONS
    long long numIterations = atol(argv[3]);

    // Run the experiment
    runTest(vectorSize, batchSize, numIterations);

    return 0;

}