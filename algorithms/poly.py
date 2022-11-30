import numpy as np
from simulator import simulator
from algorithms import arithmetic, fft


class Poly:
    """
    The proposed polynomial multiplication algorithms for r, 2r, and 2rbeta configurations.
    """

    @staticmethod
    def performRPolyComplexMult(sim: simulator.SerialSimulator, a_addr: np.ndarray, b_addr: np.ndarray,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs an in-place r-Poly algorithm on the given columns.
        :param sim: the simulation environment
        :param a_addr: the column addresses of the first input polynomial (and the output polynomial)
        :param b_addr: the column addresses of the first second polynomial
        :param inter: addresses for inter
        :param dtype: the type of numbers
        """

        fft.FFT.performRFFT(sim, a_addr, inter, dtype)
        fft.FFT.performRFFT(sim, b_addr, inter, dtype)

        arithmetic.ComplexArithmetic.mult(sim, a_addr, b_addr, inter[:dtype.N], inter[dtype.N:], dtype)
        arithmetic.ComplexArithmetic.copy(sim, inter[:dtype.N], a_addr, inter[dtype.N:], dtype)

        fft.FFT.performRFFT(sim, a_addr, inter, dtype, inv=True)

    @staticmethod
    def perform2RPolyComplexMult(sim: simulator.SerialSimulator, ax_addr: np.ndarray, ay_addr: np.ndarray,
            bx_addr: np.ndarray, by_addr: np.ndarray,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs an in-place 2r-Poly algorithm on the given columns.
        :param sim: the simulation environment
        :param ax_addr: the column addresses of the even indices of the first input polynomial (and the output polynomial)
        :param ay_addr: the column addresses of the odd indices of the first input polynomial (and the output polynomial)
        :param bx_addr: the column addresses of the even indices of the second input polynomial
        :param by_addr: the column addresses of the odd indices of the second input polynomial
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        fft.FFT.perform2RFFT(sim, ax_addr, ay_addr, inter, dtype)
        fft.FFT.perform2RFFT(sim, bx_addr, by_addr, inter, dtype)

        arithmetic.ComplexArithmetic.mult(sim, ax_addr, bx_addr, inter[:dtype.N], inter[dtype.N:], dtype)
        arithmetic.ComplexArithmetic.copy(sim, inter[:dtype.N], ax_addr, inter[dtype.N:], dtype)

        arithmetic.ComplexArithmetic.mult(sim, ay_addr, by_addr, inter[:dtype.N], inter[dtype.N:], dtype)
        arithmetic.ComplexArithmetic.copy(sim, inter[:dtype.N], ay_addr, inter[dtype.N:], dtype)

        fft.FFT.perform2RFFT(sim, ax_addr, ay_addr, inter, dtype, inv=True)

    @staticmethod
    def perform2RBetaPolyComplexMult(sim: simulator.SerialSimulator, ax_addrs, ay_addrs, bx_addrs, by_addrs,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs an in-place 2rbeta-Poly algorithm on the given columns.
        :param sim: the simulation environment
        :param ax_addrs: the column addresses of the even indices of the first input polynomial (and the output polynomial)
        :param ay_addrs: the column addresses of the odd indices of the first input polynomial (and the output polynomial)
        :param bx_addrs: the column addresses of the even indices of the second input polynomial
        :param by_addrs: the column addresses of the odd indices of the second input polynomial
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        fft.FFT.perform2RBetaFFT(sim, ax_addrs, ay_addrs, inter, dtype)
        fft.FFT.perform2RBetaFFT(sim, bx_addrs, by_addrs, inter, dtype)

        for ax_addr, bx_addr in zip(ax_addrs, bx_addrs):
            arithmetic.ComplexArithmetic.mult(sim, ax_addr, bx_addr, inter[:dtype.N], inter[dtype.N:], dtype)
            arithmetic.ComplexArithmetic.copy(sim, inter[:dtype.N], ax_addr, inter[dtype.N:], dtype)

        for ay_addr, by_addr in zip(ay_addrs, by_addrs):
            arithmetic.ComplexArithmetic.mult(sim, ay_addr, by_addr, inter[:dtype.N], inter[dtype.N:], dtype)
            arithmetic.ComplexArithmetic.copy(sim, inter[:dtype.N], ay_addr, inter[dtype.N:], dtype)

        fft.FFT.perform2RBetaFFT(sim, ax_addrs, ay_addrs, inter, dtype, inv=True)

    @staticmethod
    def perform2RPolyRealMult(sim: simulator.SerialSimulator, ax_addr: np.ndarray, ay_addr: np.ndarray,
            bx_addr: np.ndarray, by_addr: np.ndarray,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs an in-place 2r-Poly algorithm on the given columns.
        :param sim: the simulation environment
        :param ax_addr: the column addresses of the even indices of the first input polynomial (and the output polynomial)
        :param ay_addr: the column addresses of the odd indices of the first input polynomial (and the output polynomial)
        :param bx_addr: the column addresses of the even indices of the second input polynomial
        :param by_addr: the column addresses of the odd indices of the second input polynomial
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers (complex)
        """

        # Perform a single complex FFT
        abx_addr = np.concatenate((ax_addr, bx_addr))
        aby_addr = np.concatenate((ay_addr, by_addr))
        fft.FFT.perform2RFFT(sim, abx_addr, aby_addr, inter, dtype)

        # Perform the multiplication

        # Reverse abx_addr except for the 0 element and the r/2 element
        temp_addr = inter[:dtype.N]
        arithmetic.ComplexArithmetic.copy(sim, abx_addr, temp_addr, inter[dtype.N:], dtype)
        rng = [x for x in range(sim.num_rows) if x != 0 and x != sim.num_rows // 2]
        fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[dtype.N:])
        # Set abx = 0.25j * (temp* ^ 2 - abx ^ 2)
        arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[dtype.N:], dtype)
        temp2_addr = inter[dtype.N:dtype.N * 2]
        arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.mult(sim, abx_addr, abx_addr, temp_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr, np.concatenate((abx_addr[dtype.base.N:], abx_addr[:dtype.base.N])), inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, abx_addr, 2, inter, dtype)
        arithmetic.RealArithmetic.inv(sim, abx_addr[:dtype.base.N], inter, dtype.base)

        # Reverse aby_addr
        temp_addr = inter[:dtype.N]
        arithmetic.ComplexArithmetic.copy(sim, aby_addr, temp_addr, inter[dtype.N:], dtype)
        rng = list(range(sim.num_rows))
        fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[dtype.N:])
        # Set aby = 0.25j * (temp* ^ 2 - aby ^ 2)
        arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[dtype.N:], dtype)
        temp2_addr = inter[dtype.N:dtype.N * 2]
        arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.mult(sim, aby_addr, aby_addr, temp_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr,
                                         np.concatenate((aby_addr[dtype.base.N:], aby_addr[:dtype.base.N])),
                                         inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, aby_addr, 2, inter, dtype)
        arithmetic.RealArithmetic.inv(sim, aby_addr[:dtype.base.N], inter, dtype.base)

        fft.FFT.perform2RFFT(sim, abx_addr, aby_addr, inter, dtype, inv=True)

    @staticmethod
    def perform2RBetaPolyRealMult(sim: simulator.SerialSimulator, ax_addrs, ay_addrs, bx_addrs, by_addrs,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs an in-place 2rbeta-Poly algorithm on the given columns.
        :param sim: the simulation environment
        :param ax_addrs: the column addresses of the even indices of the first input polynomial (and the output polynomial)
        :param ay_addrs: the column addresses of the odd indices of the first input polynomial (and the output polynomial)
        :param bx_addrs: the column addresses of the even indices of the second input polynomial
        :param by_addrs: the column addresses of the odd indices of the second input polynomial
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers (complex)
        """

        beta = len(ax_addrs)

        # Perform a single complex FFT
        abx_addrs = [np.concatenate((ax_addr, bx_addr)) for (ax_addr, bx_addr) in zip(ax_addrs, bx_addrs)]
        aby_addrs = [np.concatenate((ay_addr, by_addr)) for (ay_addr, by_addr) in zip(ay_addrs, by_addrs)]
        fft.FFT.perform2RBetaFFT(sim, abx_addrs, aby_addrs, inter, dtype)

        # Perform the multiplication

        # Reverse abx_addrs[0] except for the 0 element and the r/2 element
        temp_addr = inter[:dtype.N]
        arithmetic.ComplexArithmetic.copy(sim, abx_addrs[0], temp_addr, inter[dtype.N:], dtype)
        rng = [x for x in range(sim.num_rows) if x != 0 and x != sim.num_rows // 2]
        fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[dtype.N:])
        # Set abx = 0.25j * (temp* ^ 2 - abx ^ 2)
        arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[dtype.N:], dtype)
        temp2_addr = inter[dtype.N:dtype.N * 2]
        arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.mult(sim, abx_addrs[0], abx_addrs[0], temp_addr, inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr,
                                         np.concatenate((abx_addrs[0][dtype.base.N:], abx_addrs[0][:dtype.base.N])),
                                         inter[dtype.N * 2:], dtype)
        arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, abx_addrs[0], 2, inter, dtype)
        arithmetic.RealArithmetic.inv(sim, abx_addrs[0][:dtype.base.N], inter, dtype.base)

        for i in range(1, beta):

            # Columns abx_addrs[i] receive from abx_addrs[beta-i] and abx_addrs[beta-i] from abx_addrs[i]

            if i < beta - i:

                # Reverse abx_addrs[beta-i]
                abx_copy = inter[:dtype.N]
                arithmetic.ComplexArithmetic.copy(sim, abx_addrs[i], abx_copy, inter[dtype.N:], dtype)
                temp_addr = inter[dtype.N:2 * dtype.N]
                arithmetic.ComplexArithmetic.copy(sim, abx_addrs[beta-i], temp_addr, inter[2 * dtype.N:], dtype)
                rng = list(range(sim.num_rows))
                fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[2 * dtype.N:])
                # Set abx_addrs[i] = 0.25j * (temp* ^ 2 - abx_addrs[i] ^ 2)
                arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[2 * dtype.N:], dtype)
                temp2_addr = inter[2 * dtype.N:3 * dtype.N]
                arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 3:], dtype)
                arithmetic.ComplexArithmetic.mult(sim, abx_addrs[i], abx_addrs[i], temp_addr, inter[dtype.N * 3:], dtype)
                arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr,
                                                 np.concatenate((abx_addrs[i][dtype.base.N:], abx_addrs[i][:dtype.base.N])),
                                                 inter[3 * dtype.N:], dtype)
                arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, abx_addrs[i], 2, inter[dtype.N:], dtype)
                arithmetic.RealArithmetic.inv(sim, abx_addrs[i][:dtype.base.N], inter[dtype.N:], dtype.base)

                temp_addr = inter[dtype.N:2 * dtype.N]
                arithmetic.ComplexArithmetic.copy(sim, abx_copy, temp_addr, inter[2 * dtype.N:], dtype)
                rng = list(range(sim.num_rows))
                fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[dtype.N * 2:])
                # Set abx_addrs[i] = 0.25j * (temp* ^ 2 - abx_addrs[i] ^ 2)
                arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[2 * dtype.N:], dtype)
                temp2_addr = inter[2 * dtype.N:3 * dtype.N]
                arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 3:], dtype)
                arithmetic.ComplexArithmetic.mult(sim, abx_addrs[beta-i], abx_addrs[beta-i], temp_addr, inter[dtype.N * 3:], dtype)
                arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr,
                                                 np.concatenate(
                                                     (abx_addrs[beta-i][dtype.base.N:], abx_addrs[beta-i][:dtype.base.N])),
                                                 inter[3 * dtype.N:], dtype)
                arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, abx_addrs[beta-i], 2, inter[dtype.N:], dtype)
                arithmetic.RealArithmetic.inv(sim, abx_addrs[beta-i][:dtype.base.N], inter[dtype.N:], dtype.base)

            elif i == beta - i:

                # Reverse x_addrs[beta-i]
                temp_addr = inter[:dtype.N]
                arithmetic.ComplexArithmetic.copy(sim, abx_addrs[beta-i], temp_addr, inter[dtype.N:], dtype)
                rng = list(range(sim.num_rows))
                fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[dtype.N:])
                # Set abx_addrs[i] = 0.25j * (temp* ^ 2 - abx_addrs[i] ^ 2)
                arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[dtype.N:], dtype)
                temp2_addr = inter[dtype.N:dtype.N * 2]
                arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 2:], dtype)
                arithmetic.ComplexArithmetic.mult(sim, abx_addrs[i], abx_addrs[i], temp_addr, inter[dtype.N * 2:], dtype)
                arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr,
                                                 np.concatenate((abx_addrs[i][dtype.base.N:], abx_addrs[i][:dtype.base.N])),
                                                 inter[dtype.N * 2:], dtype)
                arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, abx_addrs[i], 2, inter, dtype)
                arithmetic.RealArithmetic.inv(sim, abx_addrs[i][:dtype.base.N], inter, dtype.base)

        for i in range(beta):

            if i < beta-i-1:

                # Columns aby_addrs[i] receive from aby_addrs[beta-i-1]

                # Reverse aby_addrs[beta-i]
                aby_copy = inter[:dtype.N]
                arithmetic.ComplexArithmetic.copy(sim, aby_addrs[i], aby_copy, inter[dtype.N:], dtype)
                temp_addr = inter[dtype.N:2 * dtype.N]
                arithmetic.ComplexArithmetic.copy(sim, aby_addrs[beta-i-1], temp_addr, inter[2 * dtype.N:], dtype)
                rng = list(range(sim.num_rows))
                fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[2 * dtype.N:])
                # Set abx_addrs[i] = 0.25j * (temp* ^ 2 - abx_addrs[i] ^ 2)
                arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[2 * dtype.N:], dtype)
                temp2_addr = inter[2 * dtype.N:3 * dtype.N]
                arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 3:], dtype)
                arithmetic.ComplexArithmetic.mult(sim, aby_addrs[i], aby_addrs[i], temp_addr, inter[dtype.N * 3:], dtype)
                arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr,
                                                 np.concatenate((aby_addrs[i][dtype.base.N:], aby_addrs[i][:dtype.base.N])),
                                                 inter[3 * dtype.N:], dtype)
                arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, aby_addrs[i], 2, inter[dtype.N:], dtype)
                arithmetic.RealArithmetic.inv(sim, aby_addrs[i][:dtype.base.N], inter[dtype.N:], dtype.base)

                temp_addr = inter[dtype.N:2 * dtype.N]
                arithmetic.ComplexArithmetic.copy(sim, aby_copy, temp_addr, inter[2 * dtype.N:], dtype)
                rng = list(range(sim.num_rows))
                fft.FFT.swapRows(sim, temp_addr, rng[:len(rng) // 2], np.flip(rng[len(rng) // 2:]), inter[2 * dtype.N:])
                # Set abx_addrs[i] = 0.25j * (temp* ^ 2 - abx_addrs[i] ^ 2)
                arithmetic.ComplexArithmetic.conjugate(sim, temp_addr, inter[2 * dtype.N:], dtype)
                temp2_addr = inter[2 * dtype.N:3 * dtype.N]
                arithmetic.ComplexArithmetic.mult(sim, temp_addr, temp_addr, temp2_addr, inter[dtype.N * 3:], dtype)
                arithmetic.ComplexArithmetic.mult(sim, aby_addrs[beta-i-1], aby_addrs[beta-i-1], temp_addr, inter[dtype.N * 3:], dtype)
                arithmetic.ComplexArithmetic.sub(sim, temp2_addr, temp_addr,
                                                 np.concatenate(
                                                     (aby_addrs[beta-i-1][dtype.base.N:], aby_addrs[beta-i-1][:dtype.base.N])),
                                                 inter[3 * dtype.N:], dtype)
                arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, aby_addrs[beta-i-1], 2, inter[dtype.N:], dtype)
                arithmetic.RealArithmetic.inv(sim, aby_addrs[beta-i-1][:dtype.base.N], inter[dtype.N:], dtype.base)

        fft.FFT.perform2RBetaFFT(sim, abx_addrs, aby_addrs, inter, dtype, inv=True)
