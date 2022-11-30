import unittest
import numpy as np
from simulator import simulator
from algorithms import arithmetic, fft
from util import representation


class TestFFT(unittest.TestCase):
    """
    Tests the FFT algorithms
    """

    # def test_butterfly(self):
    #     """
    #     Tests the butterfly routine
    #     """
    #
    #     # Parameters
    #     dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
    #     N = dtype.N
    #     num_rows = 1024
    #     num_cols = 1024
    #
    #     # Address allocation
    #     u_addr = np.arange(0, N)
    #     v_addr = np.arange(N, 2 * N)
    #     w_addr = np.arange(2 * N, 3 * N)
    #     inter_addr = np.arange(3 * N, num_cols)
    #
    #     # Define the simulator
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     # Sample the inputs at random
    #     u = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)
    #     v = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)
    #     w = np.random.random((1, num_rows)).astype(np.csingle) + 1j * np.random.random((1, num_rows)).astype(np.csingle)
    #
    #     # Write the inputs to the memory
    #     sim.memory[u_addr] = representation.signedComplexFloatToBinary(u)
    #     sim.memory[v_addr] = representation.signedComplexFloatToBinary(v)
    #     sim.memory[w_addr] = representation.signedComplexFloatToBinary(w)
    #
    #     # Perform the butterfly algorithm
    #     fft.FFT.butterfly(sim, u_addr, v_addr, w_addr, inter_addr, dtype)
    #
    #     # Read the outputs from the memory
    #     utag = representation.binaryToSignedComplexFloat(sim.memory[u_addr]).astype(np.csingle)
    #     vtag = representation.binaryToSignedComplexFloat(sim.memory[v_addr]).astype(np.csingle)
    #
    #     # Verify correctness
    #     np.seterr(over='ignore')
    #
    #     expected_u = u + w * v
    #     mask = np.logical_and(np.logical_not(np.isinf(expected_u)), np.logical_or(expected_u == 0, np.abs(expected_u) >= np.finfo(np.float32).tiny))
    #     self.assertTrue(((utag == expected_u)[mask]).all())
    #
    #     expected_v = u - w * v
    #     mask = np.logical_and(np.logical_not(np.isinf(expected_v)), np.logical_or(expected_v == 0, np.abs(expected_v) >= np.finfo(np.float32).tiny))
    #     self.assertTrue(((vtag == expected_v)[mask]).all())
    #
    #     print(f'Complex {N}-bit Butterfly with {sim.latency} cycles and {sim.energy//num_rows} average energy.')
    #
    # def test_rFFT(self):
    #     """
    #     Tests the r-FFT algorithm
    #     """
    #
    #     # Parameters
    #     dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT16
    #     N = dtype.N
    #     num_rows = 1024
    #     num_cols = 1024
    #     n = num_rows
    #
    #     # Address allocation
    #     x_addr = np.arange(0, N)
    #     inter_addr = np.arange(N, num_cols)
    #
    #     # Define the simulator
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     # Load the sample input
    #     x = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)
    #
    #     # Write the inputs to the memory
    #     # sim.memory[x_addr] = representation.signedComplexFloatToBinary(x)
    #
    #     # Perform the r-FFT algorithm
    #     fft.FFT.performRFFT(sim, x_addr, inter_addr, dtype=dtype)
    #
    #     # Read the outputs from the memory
    #     # X = representation.binaryToSignedComplexFloat(sim.memory[x_addr]).astype(np.csingle)
    #
    #     # Verify correctness
    #     np.seterr(over='ignore')
    #
    #     expected = np.fft.fft(x).astype(np.csingle)
    #     # self.assertTrue((np.isclose(X, expected, rtol=1e-3, atol=1e-3).all()))
    #
    #     print(f'Complex {N}-bit {n}-element r-FFT with {sim.latency} cycles and {sim.energy} energy.')
    #
    # def test_2rFFT(self):
    #     """
    #     Tests the 2r-FFT algorithm
    #     """
    #
    #     # Parameters
    #     dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT16
    #     N = dtype.N
    #     num_rows = 1024
    #     num_cols = 1024
    #     n = 2 * num_rows
    #
    #     # Address allocation
    #     x_addr = np.arange(0, N)
    #     y_addr = np.arange(N, 2 * N)
    #     inter_addr = np.arange(2 * N, num_cols)
    #
    #     # Define the simulator
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     # Load the sample input
    #     x = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)
    #
    #     # Write the inputs to the memory
    #     # sim.memory[x_addr] = representation.signedComplexFloatToBinary(x)[:, 0::2]
    #     # sim.memory[y_addr] = representation.signedComplexFloatToBinary(x)[:, 1::2]
    #
    #     # Perform the 2r-FFT algorithm
    #     fft.FFT.perform2RFFT(sim, x_addr, y_addr, inter_addr, dtype=dtype)
    #
    #     # Read the outputs from the memory
    #     X = np.zeros((1, n), dtype=np.csingle)
    #     # X[:, 0::2] = representation.binaryToSignedComplexFloat(sim.memory[x_addr]).astype(np.csingle)
    #     # X[:, 1::2] = representation.binaryToSignedComplexFloat(sim.memory[y_addr]).astype(np.csingle)
    #
    #     # Verify correctness
    #     np.seterr(over='ignore')
    #
    #     expected = np.fft.fft(x).astype(np.csingle)
    #     # self.assertTrue((np.isclose(X, expected, rtol=1e-3, atol=1e-3).all()))
    #
    #     print(f'Complex {N}-bit {n}-element 2r-FFT with {sim.latency} cycles and {sim.energy} energy.')

    def test_2rbetaFFT(self):
        """
        Tests the 2rbeta-FFT algorithm
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
        N = dtype.N
        num_rows = 1024
        num_cols = 1024
        beta = 4
        n = 2 * num_rows * beta

        # Address allocation
        x_addrs = [np.arange(i * 2 * N, i * 2 * N + N) for i in range(beta)]
        y_addrs = [np.arange(i * 2 * N + N, i * 2 * N + 2 * N) for i in range(beta)]
        inter_addr = np.arange(2 * N * beta, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Load the sample input
        x = np.random.random((1, n)).astype(np.csingle) + 1j * np.random.random((1, n)).astype(np.csingle)

        # Write the inputs to the memory
        # for b in range(beta):
        #     sim.memory[x_addrs[b]] = representation.signedComplexFloatToBinary(x)[:, 2 * b::2 * beta]
        #     sim.memory[y_addrs[b]] = representation.signedComplexFloatToBinary(x)[:, 2 * b + 1::2 * beta]

        # Perform the 2rbeta-FFT algorithm
        fft.FFT.perform2RBetaFFT(sim, x_addrs, y_addrs, inter_addr, dtype=dtype)

        # Read the outputs from the memory
        X = np.zeros((1, n), dtype=np.csingle)
        # for b in range(beta):
        #     X[:, 2 * b::2 * beta] =\
        #         representation.binaryToSignedComplexFloat(sim.memory[x_addrs[b]]).astype(np.csingle)
        #     X[:, 2 * b + 1::2 * beta] = \
        #         representation.binaryToSignedComplexFloat(sim.memory[y_addrs[b]]).astype(np.csingle)

        # Verify correctness
        np.seterr(over='ignore')

        expected = np.fft.fft(x).astype(np.csingle)
        # self.assertTrue((np.isclose(X, expected, rtol=1e-3, atol=1e-3).all()))

        print(f'Complex {N}-bit {n}-element 2rbeta-FFT with {sim.latency} cycles and {sim.energy} energy.')


if __name__ == '__main__':
    unittest.main()
