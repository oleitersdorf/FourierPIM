# FourierPIM: High-Throughput In-Memory Fast Fourier Transform and Polynomial Multiplication
## Overview
This is the simulation environment for the following paper, 

`Anonymous Authors, “FourierPIM: High-Throughput In-Memory Fast Fourier Transform and Polynomial Multiplication,” 2022.` 

The goal of the simulator is to verify the correctness of the algorithms proposed in the paper,
to measure the performance of the algorithms (latency, area, energy), and to serve as an open-source library that 
may be utilized in other works. The simulator consists of three separate parts: a logical simulator for the functionality
of a memristor crossbar array, a set of implementations for the algorithms proposed in FourierPIM, and a set of testers that  
verify the correctness of the proposed algorithms and also measure performance. Below we include details on the execution of the
simulator, as well as more detailed explanations for the three parts of the simulator.

See the `gpu` folder for the derivation of the baseline GPU results.

## User Information
### Dependencies
The simulation environment is implemented via `numpy` to enable fast bitwise operations. Therefore, 
the project requires the following libraries:
1. python3
2. numpy

### Organization

1. `simulator`: Includes the logical simulation for a memristive crossbar array.
2. `algorithms`: Includes all the algorithms developed for FourierPIM, alongside an adaption of the AritPIM [1] algorithms
for floating-point arithmetic.
3.  `test`: Includes the testers for the proposed algorithms.
4. `util`: Miscellaneous helper functions (e.g., converting between different number representations).

### Logical Simulation

The fundamental idea of the logical simulator is to represent a memristor crossbar array as a binary matrix that supports
bitwise operations on rows and columns of the array. Specifically, we assume the following:

1. The memory supports the NOT/MIN3 set of logic gates, proposed by FELIX [2]. 
2. Only a single initialization is allowed per cycle. When initialization cycles are not performed, then the result is ANDed with the previous value of the cell (see FELIX [2]).
3. The memory supports a write operation that writes a single number to a single location in each crossbar in a single cycle.

### Proposed Algorithms

The proposed algorithms are divided into three parts:

#### Proposed Algorithms: Arithmetic

We adapt the arithmetic functions from AritPIM [1] within the simulation code. Further, we extend the 
functions to support arithmetic with complex numbers. Overall, we assume only half-precision and full-precision numbers
for simplicity, corresponding to the IEEE 16-bit and IEEE 32-bit standards.

#### Proposed Algorithms: FFT

We begin by extending the complex arithmetic towards an element-parallel butterfly operation. Then, we provide
implementations for each of the configurations (r, 2r, 2rbeta) by utilizing both the butterfly operations and a sequence of swap operations. The swap operations are performed without requiring additional intermediate rows by utilizing
the intermediate columns that are typically used for arithmetic.

#### Proposed Algorithms: Polynomial Multiplication

We extend the FFT towards polynomial multiplication through the convolution theorem. While complex polynomial multiplication
follows immediately from the proposed FFT, we find that real FFT requires several techniques to perform the FFT packing efficiently
(i.e., computing the reverse conjugate and applying it to the correct locations requires complex swap operations).


## References
[1] O. Leitersdorf, D. Leitersdorf, J. Gal, M. Dahan, R. Ronen, and S. Kvatinsky, “AritPIM: High-Throughput In-Memory Arithmetic,” 2022.

[2] S. Gupta, M. Imani and T. Rosing, "FELIX: Fast and Energy-Efficient Logic in Memory," IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2018, pp. 1-7.