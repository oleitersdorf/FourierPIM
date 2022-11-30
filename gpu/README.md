# cuFFT Baseline: NVIDIA RTX 3070 & NVIDIA A100
## Overview
We detail here the GPU baseline utilized towards the evaluation of FourierPIM. The baseline is programmed directly in the 
CUDA language and utilizes the cuFFT library to perform the FFT operations efficiently. Latency is measured through 
the `cudaEvents` API (measuring only the cuFFT execution time) and energy/memory are measured 
via an independent synchronized process that automatically queries the `nvidia-smi` tool (and only measures during the cuFFT execution time
due to the synchronization). The batch dimension in each experiment is chosen as the maximum value that can fit without CUDA errors.

Each of the three benchmarks is represented via a CUDA file (i.e., `fft.cu`, `real_poly.cu`, `complex_poly.cu`) that 
receives experiment parameters as command-line arguments and executes the experiment while measuring latency. Each benchmark
begins by initializing random data in the GPU memory, then performs the desired operation `numIterations` times, and lastly records the 
latency as the average latency of the `numIterations` trials. The benchmarks are compiled using the provided Makefile, with the option
to compile either a full-precision or a half-precision version. The python script `run.py` provides an interface for executing experiments easily,
and also incorporates the synchronized power/memory measurements through `nvidia-smi`. We also include the raw outputs from
the `run.py` script for the NVIDIA RTX 3070 and the NVIDIA A100 in `rtx3070.out` and `a100.out`, respectively.

## User Information
### Dependencies
This project requires the following dependencies:
1. CUDA Toolkit 11.7, including `nvcc` and `cuFFT`.
2. python3

### Usage
To manually execute an experiment, begin by compiling the source files using `make` (using `make DTYPE=half` for half-precision).
Then execute one of the executable files (i.e., `fft`, `complex_poly`, `real_poly`) with the vector dimension, batch size, and 
number of iterations as arguments. For example, 
```
./fft 2048 250000 1024
```
executes 250000 independent 2048-dimensional FFTs 1024 times (averaging the latency across the 1024 trials). Alternatively,
the same experiment can be invoked using the python script via
```
runTest('fft', 2048, 250000, 1024)
```
to receive the more-detailed output
```
fft(2048, 250000): 20.29ms, 164.78W, 7974.02MiB -> 12320126.16 Tput, 74767.43 Tput/W
```
which includes the power/memory measurements, and extends these towards Throughput and Throughput/Watt. Executing the 
python file directly will perform a pre-determined sequence of experiments, see `runAllTests`; the results of 
those experiments on the NVIDIA RTX 3070 and the NVIDIA A100 are provided in `rtx3070.out` and `a100.out`, respectively.