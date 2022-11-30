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
typedef half Real;
typedef half2 Complex;
const cudaDataType_t CDTYPE = CUDA_C_16F;
const cudaDataType_t RDTYPE = CUDA_R_16F;
#else
typedef float Real;
typedef float2 Complex;
const cudaDataType_t CDTYPE = CUDA_C_32F;
const cudaDataType_t RDTYPE = CUDA_R_32F;
#endif

static __device__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMul(Complex *, const Complex *, int);

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
    Real* hx_data = new Real[vectorSize * batchSize];
    for (long long i = 0; i < vectorSize * batchSize; i++)
        hx_data[i] = rand() / (float)RAND_MAX;
    Real* hy_data = new Real[vectorSize * batchSize];
    for (long long i = 0; i < vectorSize * batchSize; i++)
        hy_data[i] = rand() / (float)RAND_MAX;

    // Allocate device memory
    Complex* dx_data;
    checkCudaErrors(cudaMalloc(&dx_data, vectorSize * batchSize * sizeof(Complex)));
    Complex* dy_data;
    checkCudaErrors(cudaMalloc(&dy_data, vectorSize * batchSize * sizeof(Complex)));

    // Create FFT plan
    cufftHandle R2CPlan;
    checkCudaErrors(cufftCreate(&R2CPlan));
    size_t R2Cws = 0;
    checkCudaErrors(cufftXtMakePlanMany(R2CPlan, 1, &vectorSize, NULL, 1, 1, RDTYPE, NULL, 1, 1, CDTYPE, batchSize, &R2Cws, CDTYPE));
    cufftHandle C2CPlan;
    size_t C2Cws = 0;
    checkCudaErrors(cufftCreate(&C2CPlan));
    checkCudaErrors(cufftXtMakePlanMany(C2CPlan, 1, &vectorSize, NULL, 1, 1, CDTYPE, NULL, 1, 1, CDTYPE, batchSize, &C2Cws, CDTYPE));

    // Copy to device
    checkCudaErrors(cudaMemcpy(dx_data, hx_data, vectorSize * batchSize * sizeof(Real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dy_data, hy_data, vectorSize * batchSize * sizeof(Real), cudaMemcpyHostToDevice));

    // Execute polynomial multiplication
    checkCudaErrors(cudaEventRecord(start, 0));

    std::cout << "START" << std::endl;

    for (int i = 0; i < numIterations; i++) {

        // Perform forward FFTs
        checkCudaErrors(cufftXtExec(R2CPlan, (void *) dx_data, (void *) dx_data, CUFFT_FORWARD));
        checkCudaErrors(cufftXtExec(R2CPlan, (void *) dy_data, (void *) dy_data, CUFFT_FORWARD));

        // Perform element-wise multiplication
        ComplexPointwiseMul<<<(vectorSize * batchSize) / 256, 256>>>(dx_data, dy_data, vectorSize * batchSize);
        getLastCudaError("Kernel execution failed [ ComplexPointwiseMul ]");

        // Perform inverse FFT
        checkCudaErrors(cufftXtExec(C2CPlan, (void *) dx_data, (void *) dx_data, CUFFT_INVERSE));

    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    std::cout << "STOP" << std::endl;

    // Print results
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    std::cout << time / numIterations << std::endl;

    // Cleanup
    checkCudaErrors(cufftDestroy(R2CPlan));
    checkCudaErrors(cufftDestroy(C2CPlan));
    checkCudaErrors(cudaFree(dx_data));
    checkCudaErrors(cudaFree(dy_data));
    checkCudaErrors(cudaDeviceReset());

}

static __device__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

/**
 * Performs a pointwise multiplication
 */
static __global__ void ComplexPointwiseMul(Complex *a, const Complex *b, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = ComplexMul(a[i], b[i]);
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
