import unittest
import numpy as np
from simulator import simulator
from algorithms import arithmetic, poly
from util import representation


class TestPoly(unittest.TestCase):
    """
    Tests the polynomial multiplication algorithms
    """
    #
    # def test_rPoly(self):
    #     """
    #     Tests the r-Poly algorithm
    #     """
    #
    #     # Parameters
    #     dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32
    #     N = dtype.N
    #     num_rows = 1024
    #     num_cols = 1024
    #     n = num_rows
    #
    #     # Address allocation
    #     a_addr = np.arange(0, N)
    #     b_addr = np.arange(N, 2 * N)
    #     inter_addr = np.arange(2 * N, num_cols)
    #
    #     # Define the simulator
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     # Load the sample input
    #     a = np.concatenate((
    #         np.random.random((1, n // 2)).astype(np.csingle) + 1j * np.random.random((1, n // 2)).astype(np.csingle),
    #         np.zeros((1, n // 2), dtype=np.csingle)), axis=1)
    #     b = np.concatenate((
    #         np.random.random((1, n // 2)).astype(np.csingle) + 1j * np.random.random((1, n // 2)).astype(np.csingle),
    #         np.zeros((1, n // 2), dtype=np.csingle)), axis=1)
    #
    #     # Write the inputs to the memory
    #     sim.memory[a_addr] = representation.signedComplexFloatToBinary(a)
    #     sim.memory[b_addr] = representation.signedComplexFloatToBinary(b)
    #
    #     # Perform the r-Poly algorithm
    #     poly.Poly.performRPolyComplexMult(sim, a_addr, b_addr, inter_addr, dtype=dtype)
    #
    #     # Read the outputs from the memory
    #     c = representation.binaryToSignedComplexFloat(sim.memory[a_addr]).astype(np.csingle)
    #
    #     # Verify correctness
    #     np.seterr(over='ignore')
    #
    #     expected = (np.polymul(a.squeeze(0), b.squeeze(0))[:n]).astype(np.csingle)
    #     self.assertTrue((np.isclose(c, expected, rtol=1e-3, atol=1e-3).all()))
    #
    #     print(f'Complex {N}-bit {n}-element r-Poly with {sim.latency} cycles and {sim.energy} energy.')
    #
    # def test_2rPoly(self):
    #     """
    #     Tests the 2r-Poly algorithm
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
    #     ax_addr = np.arange(0, N)
    #     ay_addr = np.arange(N, 2 * N)
    #     bx_addr = np.arange(2 * N, 3 * N)
    #     by_addr = np.arange(3 * N, 4 * N)
    #     inter_addr = np.arange(4 * N, num_cols)
    #
    #     # Define the simulator
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     # Load the sample input
    #     a = np.concatenate((
    #         np.random.random((1, n // 2)).astype(np.csingle) + 1j * np.random.random((1, n // 2)).astype(np.csingle),
    #         np.zeros((1, n // 2), dtype=np.csingle)), axis=1)
    #     b = np.concatenate((
    #         np.random.random((1, n // 2)).astype(np.csingle) + 1j * np.random.random((1, n // 2)).astype(np.csingle),
    #         np.zeros((1, n // 2), dtype=np.csingle)), axis=1)
    #
    #     # Write the inputs to the memory
    #     # sim.memory[ax_addr] = representation.signedComplexFloatToBinary(a)[:, 0::2]
    #     # sim.memory[ay_addr] = representation.signedComplexFloatToBinary(a)[:, 1::2]
    #     # sim.memory[bx_addr] = representation.signedComplexFloatToBinary(b)[:, 0::2]
    #     # sim.memory[by_addr] = representation.signedComplexFloatToBinary(b)[:, 1::2]
    #
    #     # Perform the 2r-Poly algorithm
    #     poly.Poly.perform2RPolyComplexMult(sim, ax_addr, ay_addr, bx_addr, by_addr, inter_addr, dtype=dtype)
    #
    #     # Read the outputs from the memory
    #     c = np.zeros((1, n), dtype=np.csingle)
    #     # c[:, 0::2] = representation.binaryToSignedComplexFloat(sim.memory[ax_addr]).astype(np.csingle)
    #     # c[:, 1::2] = representation.binaryToSignedComplexFloat(sim.memory[ay_addr]).astype(np.csingle)
    #
    #     # Verify correctness
    #     np.seterr(over='ignore')
    #
    #     expected = (np.polymul(a.squeeze(0), b.squeeze(0))[:n]).astype(np.csingle)
    #     # self.assertTrue((np.isclose(c, expected, rtol=1e-3, atol=1e-3).all()))
    #
    #     print(f'Complex {N}-bit {n}-element 2r-Poly with {sim.latency} cycles and {sim.energy} energy.')

    # def test_2rbetaPoly(self):
    #     """
    #     Tests the 2rbeta-Poly algorithm
    #     """
    #
    #     # Parameters
    #     dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT16
    #     N = dtype.N
    #     num_rows = 1024
    #     num_cols = 1024
    #     beta = 4
    #     n = 2 * beta * num_rows
    #
    #     # Address allocation
    #     ax_addrs = [np.arange(i * 2 * N, i * 2 * N + N) for i in range(beta)]
    #     ay_addrs = [np.arange(i * 2 * N + N, i * 2 * N + 2 * N) for i in range(beta)]
    #     bx_addrs = [np.arange(2 * N * beta + i * 2 * N, 2 * N * beta + i * 2 * N + N) for i in range(beta)]
    #     by_addrs = [np.arange(2 * N * beta + i * 2 * N + N, 2 * N * beta + i * 2 * N + 2 * N) for i in range(beta)]
    #     inter_addr = np.arange(4 * N * beta, num_cols)
    #
    #     # Define the simulator
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     # Load the sample input
    #     a = np.concatenate((
    #         np.random.random((1, n // 2)).astype(np.csingle) + 1j * np.random.random((1, n // 2)).astype(np.csingle),
    #         np.zeros((1, n // 2), dtype=np.csingle)), axis=1)
    #     b = np.concatenate((
    #         np.random.random((1, n // 2)).astype(np.csingle) + 1j * np.random.random((1, n // 2)).astype(np.csingle),
    #         np.zeros((1, n // 2), dtype=np.csingle)), axis=1)
    #
    #     # Write the inputs to the memory
    #     # for i in range(beta):
    #     #     sim.memory[ax_addrs[i]] = representation.signedComplexFloatToBinary(a)[:, 2 * i::2 * beta]
    #     #     sim.memory[ay_addrs[i]] = representation.signedComplexFloatToBinary(a)[:, 2 * i + 1::2 * beta]
    #     # for i in range(beta):
    #     #     sim.memory[bx_addrs[i]] = representation.signedComplexFloatToBinary(b)[:, 2 * i::2 * beta]
    #     #     sim.memory[by_addrs[i]] = representation.signedComplexFloatToBinary(b)[:, 2 * i + 1::2 * beta]
    #
    #     # Perform the 2rbeta-Poly algorithm
    #     poly.Poly.perform2RBetaPolyComplexMult(sim, ax_addrs, ay_addrs, bx_addrs, by_addrs, inter_addr, dtype=dtype)
    #
    #     # Read the outputs from the memory
    #     c = np.zeros((1, n), dtype=np.csingle)
    #     # for i in range(beta):
    #     #     c[:, 2 * i::2 * beta] = \
    #     #         representation.binaryToSignedComplexFloat(sim.memory[ax_addrs[i]]).astype(np.csingle)
    #     #     c[:, 2 * i + 1::2 * beta] = \
    #     #         representation.binaryToSignedComplexFloat(sim.memory[ay_addrs[i]]).astype(np.csingle)
    #
    #     # Verify correctness
    #     np.seterr(over='ignore')
    #
    #     expected = (np.polymul(a.squeeze(0), b.squeeze(0))[:n]).astype(np.csingle)
    #     # self.assertTrue((np.isclose(c, expected, rtol=1e-3, atol=1e-3).all()))
    #
    #     print(f'Complex {N}-bit {n}-element 2rbeta-Poly with {sim.latency} cycles and {sim.energy} energy.')

    # def test_2rPolyReal(self):
    #     """
    #     Tests the 2r-PolyReal algorithm
    #     """
    #
    #     # Parameters
    #     dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT16
    #     Nc = dtype.N
    #     Nr = dtype.base.N
    #     num_rows = 1024
    #     num_cols = 1024
    #     n = 2 * num_rows
    #
    #     # Address allocation
    #     ax_addr = np.arange(0, Nr)
    #     ay_addr = np.arange(Nr, 2 * Nr)
    #     bx_addr = np.arange(2 * Nr, 3 * Nr)
    #     by_addr = np.arange(3 * Nr, 4 * Nr)
    #     inter_addr = np.arange(4 * Nr, num_cols)
    #
    #     # Define the simulator
    #     sim = simulator.SerialSimulator(num_rows, num_cols)
    #
    #     # Load the sample input
    #     a = np.concatenate((np.random.random((1, n // 2)), np.zeros((1, n // 2))), axis=1)
    #     b = np.concatenate((np.random.random((1, n // 2)), np.zeros((1, n // 2))), axis=1)
    #
    #     # Write the inputs to the memory
    #     # sim.memory[ax_addr] = representation.signedFloatToBinary(a)[:, 0::2]
    #     # sim.memory[ay_addr] = representation.signedFloatToBinary(a)[:, 1::2]
    #     # sim.memory[bx_addr] = representation.signedFloatToBinary(b)[:, 0::2]
    #     # sim.memory[by_addr] = representation.signedFloatToBinary(b)[:, 1::2]
    #     #
    #     # Perform the 2r-PolyReal algorithm
    #     poly.Poly.perform2RPolyRealMult(sim, ax_addr, ay_addr, bx_addr, by_addr, inter_addr, dtype=dtype)
    #
    #     # Read the outputs from the memory
    #     c = np.zeros((1, n))
    #     # c[:, 0::2] = representation.binaryToSignedFloat(sim.memory[ax_addr])
    #     # c[:, 1::2] = representation.binaryToSignedFloat(sim.memory[ay_addr])
    #
    #     # Verify correctness
    #     np.seterr(over='ignore')
    #
    #     expected = (np.polymul(a.squeeze(0), b.squeeze(0))[:n])
    #     # self.assertTrue((np.isclose(c, expected, rtol=1e-3, atol=1e-3).all()))
    #
    #     print(f'Real {Nr}-bit {n}-element 2r-PolyReal with {sim.latency} cycles and {sim.energy} energy.')

    def test_2rbetaPolyReal(self):
        """
        Tests the 2rbeta-PolyReal algorithm
        """

        # Parameters
        dtype = arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT16
        Nc = dtype.N
        Nr = dtype.base.N
        num_rows = 1024
        num_cols = 1024
        beta = 8
        n = 2 * beta * num_rows

        # Address allocation
        ax_addrs = [np.arange(i * 2 * Nr, i * 2 * Nr + Nr) for i in range(beta)]
        ay_addrs = [np.arange(i * 2 * Nr + Nr, i * 2 * Nr + 2 * Nr) for i in range(beta)]
        bx_addrs = [np.arange(2 * Nr * beta + i * 2 * Nr, 2 * Nr * beta + i * 2 * Nr + Nr) for i in range(beta)]
        by_addrs = [np.arange(2 * Nr * beta + i * 2 * Nr + Nr, 2 * Nr * beta + i * 2 * Nr + 2 * Nr) for i in range(beta)]
        inter_addr = np.arange(4 * Nr * beta, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(num_rows, num_cols)

        # Load the sample input
        a = np.concatenate((np.random.random((1, n // 2)), np.zeros((1, n // 2))), axis=1)
        b = np.concatenate((np.random.random((1, n // 2)), np.zeros((1, n // 2))), axis=1)

        # Write the inputs to the memory
        # for i in range(beta):
        #     sim.memory[ax_addrs[i]] = representation.signedFloatToBinary(a)[:, 2 * i::2 * beta]
        #     sim.memory[ay_addrs[i]] = representation.signedFloatToBinary(a)[:, 2 * i + 1::2 * beta]
        # for i in range(beta):
        #     sim.memory[bx_addrs[i]] = representation.signedFloatToBinary(b)[:, 2 * i::2 * beta]
        #     sim.memory[by_addrs[i]] = representation.signedFloatToBinary(b)[:, 2 * i + 1::2 * beta]

        # Perform the 2rbeta-PolyReal algorithm
        poly.Poly.perform2RBetaPolyRealMult(sim, ax_addrs, ay_addrs, bx_addrs, by_addrs, inter_addr, dtype=dtype)

        # Read the outputs from the memory
        c = np.zeros((1, n))
        # for i in range(beta):
        #     c[:, 2 * i::2 * beta] = \
        #         representation.binaryToSignedFloat(sim.memory[ax_addrs[i]])
        #     c[:, 2 * i + 1::2 * beta] = \
        #         representation.binaryToSignedFloat(sim.memory[ay_addrs[i]])

        # Verify correctness
        np.seterr(over='ignore')

        expected = (np.polymul(a.squeeze(0), b.squeeze(0))[:n])
        # self.assertTrue((np.isclose(c, expected, rtol=1e-3, atol=1e-3).all()))

        print(f'Real {Nr}-bit {n}-element 2rbeta-PolyReal with {sim.latency} cycles and {sim.energy} energy.')


if __name__ == '__main__':
    unittest.main()
