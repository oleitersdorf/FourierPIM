import numpy as np
from math import log2, ceil
from typing import List
from simulator import simulator
from algorithms import arithmetic
from util import representation


class FFT:
    """
    The proposed FFT algorithms for r, 2r, and 2rbeta configurations.
    """

    @staticmethod
    def butterfly(sim: simulator.SerialSimulator, u_addr: np.ndarray, v_addr: np.ndarray, w_addr: np.ndarray,
            inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32):
        """
        Performs an in-place butterfly operation on the given columns.
        :param sim: the simulation environment
        :param u_addr: the addresses of input u
        :param v_addr: the addresses of input v
        :param w_addr: the addresses of output w
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        wv_addr = inter[:dtype.N]

        arithmetic.ComplexArithmetic.mult(sim, w_addr, v_addr, wv_addr, inter[dtype.N:], dtype)

        arithmetic.ComplexArithmetic.copy(sim, u_addr, inter[dtype.N:2*dtype.N], inter[2*dtype.N:], dtype)

        arithmetic.ComplexArithmetic.add(sim, inter[dtype.N:2*dtype.N], inter[:dtype.N], u_addr, inter[2*dtype.N:], dtype)

        arithmetic.ComplexArithmetic.sub(sim, inter[dtype.N:2 * dtype.N], inter[:dtype.N], v_addr, inter[2 * dtype.N:], dtype)

    @staticmethod
    def performRFFT(sim: simulator.SerialSimulator, x_addr: np.ndarray, inter, dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32, inv=False):
        """
        Performs the r-FFT algorithm on the given columns.
        :param sim: the simulation environment
        :param x_addr: the column addresses of the input vector
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        :param inv: whether to perform an inverse FFT
        """

        N = dtype.N
        n = sim.num_rows

        # Pre-compute the constants
        # w = np.exp((2j if inv else -2j) * np.pi / n * np.arange(n)).astype(dtype.np_dtype).reshape(1, n)
        # w_bin = representation.signedComplexFloatToBinary(w)
        w_bin = np.zeros((N, n), dtype=np.bool)

        # Allocate the address for the pair column
        y_addr = inter[:N]
        inter = inter[N:]

        # Allocate the address for the constants
        w_addr = inter[:N]
        inter = inter[N:]

        # Perform the bit-reversal permutation
        source_rows, dest_rows = [], []
        for k in range(n):
            j = FFT.__bitRev(k, ceil(log2(n)))
            if k < j:
                source_rows.append(k)
                dest_rows.append(j)
        FFT.swapRows(sim, x_addr, source_rows, dest_rows, inter)

        # Perform the r-iterations
        for k in range(ceil(log2(n))):

            # Perform the shift
            # Initialize y
            for i in range(N):
                sim.perform(simulator.GateType.INIT1, [], [y_addr[i]], simulator.GateDirection.IN_ROW)
            # Move to the right
            for i in range(N):
                sim.perform(simulator.GateType.NOT, [x_addr[i]], [y_addr[i]], simulator.GateDirection.IN_ROW,
                    np.array([i for i in range(n) if i & (1 << k)]))
            # Move upwards
            for i in range(n):
                if i & (1 << k):
                    sim.perform(simulator.GateType.NOT, [i], [i - (1 << k)], simulator.GateDirection.IN_COLUMN, y_addr)

            # Write the constants
            for i in range(n):
                if not i & (1 << k):
                    sim.write(i, w_addr, w_bin[:, (i % (1 << k)) * (n // (1 << (1 + k)))])

            # Perform the butterfly operation
            FFT.butterfly(sim, x_addr, y_addr, w_addr, inter, dtype)

            # Perform the shift
            # Initialize x
            for i in range(N):
                sim.perform(simulator.GateType.INIT1, [], [x_addr[i]], simulator.GateDirection.IN_ROW,
                    np.array([i for i in range(n) if i & (1 << k)]))
            # Initialize y
            for i in range(N):
                sim.perform(simulator.GateType.INIT1, [], [y_addr[i]], simulator.GateDirection.IN_ROW,
                    np.array([i for i in range(n) if i & (1 << k)]))
            # Move downwards
            for i in range(n):
                if i & (1 << k):
                    sim.perform(simulator.GateType.NOT, [i - (1 << k)], [i], simulator.GateDirection.IN_COLUMN, y_addr)
            # Move to the left
            for i in range(N):
                sim.perform(simulator.GateType.NOT, [y_addr[i]], [x_addr[i]], simulator.GateDirection.IN_ROW,
                            np.array([i for i in range(n) if i & (1 << k)]))

        # If inverse, divide all outputs by n
        if inv:
            arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, x_addr, ceil(log2(n)), inter, dtype=dtype)

    @staticmethod
    def perform2RFFT(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, inter,
                    dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32, inv=False):
        """
        Performs the 2r-FFT algorithm on the given columns.
        :param sim: the simulation environment
        :param x_addr: the column addresses of the even part of the input vector
        :param y_addr: the column addresses of the odd part of the input vector
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        :param inv: whether to perform an inverse FFT
        """

        N = dtype.N
        n = 2 * sim.num_rows

        # Pre-compute the constants
        # w = np.exp((2j if inv else -2j) * np.pi / n * np.arange(n)).astype(dtype.np_dtype).reshape(1, n)
        # w_bin = representation.signedComplexFloatToBinary(w)
        w_bin = np.zeros((N, n), dtype=np.bool)

        # Allocate the address for the constants
        w_addr = inter[:N]
        inter = inter[N:]

        # Perform the bit-reversal permutation
        # Swap within x_addr
        FFT.swapRows(sim, x_addr,
                       [i // 2 for i in range(0, n // 2, 2) if i < FFT.__bitRev(i, ceil(log2(n)))],
                       [FFT.__bitRev(i, ceil(log2(n))) // 2 for i in range(0, n // 2, 2) if i < FFT.__bitRev(i, ceil(log2(n)))], inter)
        # Swap within y_addr
        FFT.swapRows(sim, y_addr,
                       [i // 2 for i in range(n // 2 + 1, n, 2) if i < FFT.__bitRev(i, ceil(log2(n)))],
                       [FFT.__bitRev(i, ceil(log2(n))) // 2 for i in range(n // 2 + 1, n, 2) if i < FFT.__bitRev(i, ceil(log2(n)))], inter)
        # Swap between x_addr and y_addr
        FFT.swapRows(sim, x_addr,
                       [i // 2 for i in range(n // 2, n, 2)],
                       [FFT.__bitRev(i, ceil(log2(n))) // 2 for i in range(n // 2, n, 2)], inter, y_addr)

        # Perform the 2r-iterations
        for k in range(ceil(log2(n))):

            # Write the constants
            for i in range(n // 2):
                sim.write(i, w_addr, w_bin[:, (i % (1 << k)) * (n // (1 << (1 + k)))])

            # Perform the butterfly operation
            FFT.butterfly(sim, x_addr, y_addr, w_addr, inter, dtype)

            # Perform the swaps
            if k < ceil(log2(n)) - 1:
                x_rows = np.array([r for r in range(sim.num_rows) if r & (1 << k)])
                y_rows = np.array([r for r in range(sim.num_rows) if not r & (1 << k)])
                FFT.swapRows(sim, x_addr, x_rows, y_rows, inter, y_addr)

        # Reverse the swaps
        for k in reversed(range(ceil(log2(n)))):
            if k < ceil(log2(n)) - 1:
                x_rows = np.array([r for r in range(sim.num_rows) if r & (1 << k)])
                y_rows = np.array([r for r in range(sim.num_rows) if not r & (1 << k)])
                FFT.swapRows(sim, x_addr, x_rows, y_rows, inter, y_addr)

        # If inverse, divide all outputs by n
        if inv:
            arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, x_addr, ceil(log2(n)), inter, dtype=dtype)
            arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, y_addr, ceil(log2(n)), inter, dtype=dtype)

    @staticmethod
    def perform2RBetaFFT(sim: simulator.SerialSimulator, x_addrs: List[np.ndarray], y_addrs: List[np.ndarray], inter,
                    dtype=arithmetic.ComplexArithmetic.DataType.IEEE_CFLOAT32, inv=False):
        """
        Performs the 2rbeta-FFT algorithm on the given columns.
        :param sim: the simulation environment
        :param x_addrs: a list of the column addresses for the even parts of the input vector
        :param y_addrs: a list of the column addresses for the odd parts of the input vector
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        :param inv: whether to perform an inverse FFT
        """

        N = dtype.N
        beta = len(x_addrs)
        n = 2 * sim.num_rows * beta

        # Pre-compute the constants
        # w = np.exp((2j if inv else -2j) * np.pi / n * np.arange(n)).astype(dtype.np_dtype).reshape(1, n)
        # w_bin = representation.signedComplexFloatToBinary(w)
        w_bin = np.zeros((N, n), dtype=np.bool)

        # Allocate the address for the constants
        w_addr = inter[:N]
        inter = inter[N:]

        print(sim.latency)

        # Perform the bit-reversal permutation
        all_cols = [val for pair in zip(x_addrs, y_addrs) for val in pair]
        # Perform swaps within each column
        for b in range(len(all_cols)):
            FFT.swapRows(sim, all_cols[b],
                [i // (2 * beta) for i in range(b, n, 2 * beta)
                 if i < FFT.__bitRev(i, ceil(log2(n))) and (FFT.__bitRev(i, ceil(log2(n))) % (2 * beta) == b)],
                [FFT.__bitRev(i, ceil(log2(n))) // (2 * beta) for i in range(b, n, 2 * beta)
                 if i < FFT.__bitRev(i, ceil(log2(n))) and (FFT.__bitRev(i, ceil(log2(n))) % (2 * beta) == b)], inter)

        print(sim.latency)

        # Perform swaps between columns
        for b1 in range(len(all_cols)):
            for b2 in range(b1 + 1, len(all_cols)):
                FFT.swapRows(sim, all_cols[b1],
                    [i // (2 * beta) for i in range(b1, n, 2 * beta)
                     if (FFT.__bitRev(i, ceil(log2(n))) % (2 * beta) == b2)],
                    [FFT.__bitRev(i, ceil(log2(n))) // (2 * beta) for i in range(b1, n, 2 * beta)
                     if (FFT.__bitRev(i, ceil(log2(n))) % (2 * beta) == b2)], inter, all_cols[b2])

        print(sim.latency)

        # Perform the 2rbeta-iterations
        for k in range(ceil(log2(n))):

            print(sim.latency)

            # Perform the butterfly operations
            lat = sim.latency
            for b in range(beta):
                s0 = sim.latency
                for i in range(sim.num_rows):
                    sim.write(i, w_addr, w_bin[:, ((b + beta * i) % (1 << k)) * (n // (1 << (1 + k)))])
                FFT.butterfly(sim, x_addrs[b], y_addrs[b], w_addr, inter, dtype)
                s1 = sim.latency
            sim.latency = lat + (s1 - s0)

            print(sim.latency)

            # Perform the swaps
            if k < ceil(log2(beta)):
                for b in range(beta):
                    if b & (1 << k):
                        FFT.swapCols(sim, y_addrs[b - (1 << k)], x_addrs[b], inter)
            elif k < ceil(log2(n)) - 1:
                x_rows = np.array([r for r in range(sim.num_rows) if r & (1 << (k - ceil(log2(beta))))])
                y_rows = np.array([r for r in range(sim.num_rows) if not r & (1 << (k - ceil(log2(beta))))])
                for b in range(beta):
                    FFT.swapRows(sim, x_addrs[b], x_rows, y_rows, inter, y_addrs[b])

            print(sim.latency)

            print()

        # Reverse the swaps
        for k in reversed(range(ceil(log2(n)))):
            if k < ceil(log2(beta)):
                for b in range(beta):
                    if b & (1 << k):
                        FFT.swapCols(sim, y_addrs[b - (1 << k)], x_addrs[b], inter)
            elif k < ceil(log2(n)) - 1:
                x_rows = np.array([r for r in range(sim.num_rows) if r & (1 << (k - ceil(log2(beta))))])
                y_rows = np.array([r for r in range(sim.num_rows) if not r & (1 << (k - ceil(log2(beta))))])
                for b in range(beta):
                    FFT.swapRows(sim, x_addrs[b], x_rows, y_rows, inter, y_addrs[b])

        # If inverse, divide all outputs by n
        if inv:
            for x_addr in x_addrs:
                arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, x_addr, ceil(log2(n)), inter, dtype=dtype)
            for y_addr in y_addrs:
                arithmetic.ComplexArithmetic.divByPowerOfTwo(sim, y_addr, ceil(log2(n)), inter, dtype=dtype)

    @staticmethod
    def __bitRev(x, N):
        """
        Returns the bit-reverse of x (representation size N)
        :param x: the number to bit-reverse
        :param N: the representation size
        """
        bin_x = list(reversed([(x & (1 << i)) >> i for i in range(N)]))
        return sum([bin_x[i] << i for i in range(N)])

    @staticmethod
    def swapCols(sim, source_cols, dest_cols, inter: np.ndarray):
        """
        Swaps the given columns by using intermediate columns
        :param sim: the simulation environment
        :param source_cols: the column addresses of the source elements
        :param dest_cols: the column addresses of the destination elements
        :param inter: addresses for inter
        """

        N = len(source_cols)

        # Perform the swaps column-by-column
        for i in range(N):
            sim.perform(simulator.GateType.INIT1, [], [inter[0]])
            sim.perform(simulator.GateType.INIT1, [], [inter[1]])
            sim.perform(simulator.GateType.NOT, [source_cols[i]], [inter[0]])
            sim.perform(simulator.GateType.NOT, [dest_cols[i]], [inter[1]])
            sim.perform(simulator.GateType.INIT1, [], [source_cols[i]])
            sim.perform(simulator.GateType.INIT1, [], [dest_cols[i]])
            sim.perform(simulator.GateType.NOT, [inter[1]], [source_cols[i]])
            sim.perform(simulator.GateType.NOT, [inter[0]], [dest_cols[i]])

    @staticmethod
    def swapRows(sim, source_cols, source_rows, dest_rows, inter: np.ndarray, dest_cols=None):
        """
        Swaps the given elements at the given rows and columns by using intermediate columns
        :param sim: the simulation environment
        :param source_cols: the column addresses of the source elements
        :param dest_cols: the column addresses of the destination elements. If None, then assumed same as source_cols.
        :param source_rows: the row addresses of the source elements
        :param dest_rows: the row addresses of the destination elements
        :param inter: addresses for inter
        """

        N = len(source_cols)
        assert(len(source_rows) == len(dest_rows))
        n = len(source_rows)

        # Only supports cases when the source_rows and dest_rows are disjoint
        assert(len(np.intersect1d(source_rows, dest_rows)) == 0)

        all_rows = np.concatenate((source_rows, dest_rows))
        # If there will be no free rows, then split into two swap operations
        if len(all_rows) == sim.num_rows:
            FFT.swapRows(sim, source_cols, source_rows[:sim.num_rows // 4], dest_rows[:sim.num_rows // 4], inter, dest_cols)
            FFT.swapRows(sim, source_cols, source_rows[sim.num_rows // 4:], dest_rows[sim.num_rows // 4:], inter, dest_cols)
            return

        # Start by copying the data over to inter
        for i in range(N):
            sim.perform(simulator.GateType.INIT1, [], [inter[i]], simulator.GateDirection.IN_ROW, all_rows)
        if dest_cols is None:
            for i in range(N):
                sim.perform(simulator.GateType.NOT, [source_cols[i]], [inter[i]], simulator.GateDirection.IN_ROW, all_rows)
        else:
            for i in range(N):
                sim.perform(simulator.GateType.NOT, [source_cols[i]], [inter[i]], simulator.GateDirection.IN_ROW, source_rows)
                sim.perform(simulator.GateType.NOT, [dest_cols[i]], [inter[i]], simulator.GateDirection.IN_ROW, dest_rows)

        # Select two intermediate rows that are not used
        unused_rows = np.setdiff1d(np.arange(sim.num_rows), all_rows)
        assert(len(unused_rows) >= 2)
        row0 = unused_rows[0]
        row1 = unused_rows[1]

        # Perform the swaps
        for i in range(n):
            sim.perform(simulator.GateType.INIT1, [], [row0], simulator.GateDirection.IN_COLUMN, inter[:N])
            sim.perform(simulator.GateType.INIT1, [], [row1], simulator.GateDirection.IN_COLUMN, inter[:N])
            sim.perform(simulator.GateType.NOT, [source_rows[i]], [row0], simulator.GateDirection.IN_COLUMN, inter[:N])
            sim.perform(simulator.GateType.NOT, [dest_rows[i]], [row1], simulator.GateDirection.IN_COLUMN, inter[:N])
            sim.perform(simulator.GateType.INIT1, [], [source_rows[i]], simulator.GateDirection.IN_COLUMN, inter[:N])
            sim.perform(simulator.GateType.INIT1, [], [dest_rows[i]], simulator.GateDirection.IN_COLUMN, inter[:N])
            sim.perform(simulator.GateType.NOT, [row1], [source_rows[i]], simulator.GateDirection.IN_COLUMN, inter[:N])
            sim.perform(simulator.GateType.NOT, [row0], [dest_rows[i]], simulator.GateDirection.IN_COLUMN, inter[:N])

        # Copy the data back from inter
        if dest_cols is None:
            for i in range(N):
                sim.perform(simulator.GateType.INIT1, [], [source_cols[i]], simulator.GateDirection.IN_ROW, all_rows)
                sim.perform(simulator.GateType.NOT, [inter[i]], [source_cols[i]], simulator.GateDirection.IN_ROW, all_rows)
        else:
            for i in range(N):
                sim.perform(simulator.GateType.INIT1, [], [source_cols[i]], simulator.GateDirection.IN_ROW, source_rows)
                sim.perform(simulator.GateType.NOT, [inter[i]], [source_cols[i]], simulator.GateDirection.IN_ROW, source_rows)
                sim.perform(simulator.GateType.INIT1, [], [dest_cols[i]], simulator.GateDirection.IN_ROW, dest_rows)
                sim.perform(simulator.GateType.NOT, [inter[i]], [dest_cols[i]], simulator.GateDirection.IN_ROW, dest_rows)