import numpy as np
from math import log2, ceil
from enum import Enum
from simulator import simulator
from util import representation


class IntermediateAllocator:
    """
    Helper that assists in the allocation of intermediate cells
    """

    def __init__(self, cells: np.ndarray):
        """
        Initializes the allocator
        :param cells: a np list of the available cells
        """

        self.cells = cells
        self.cells_inverse = {cells[i]: i for i in range(len(cells))}
        self.allocated = np.zeros_like(cells, dtype=bool)  # vector containing 1 if allocated, 0 otherwise

    def malloc(self, num_cells: int):
        """
        Allocates num_cells cells
        :param num_cells: the number of cells to allocate
        :return: np array containing the allocated indices, or int if num_cells = 1
        """

        assert (num_cells >= 1)

        allocation = []

        # Search for available cells (first searching between previous allocations, then extending if necessary)
        for i in range(len(self.cells)):
            if not self.allocated[i]:
                allocation.append(i)
                # Mark the cell as allocated
                self.allocated[i] = True
            if len(allocation) == num_cells:
                break

        # Assert that there were enough cells
        assert (len(allocation) == num_cells)

        # Return the allocated cells
        if num_cells > 1:
            return np.array(self.cells[allocation], dtype=int)
        else:
            return self.cells[allocation[0]]

    def free(self, cells):
        """
        Frees the given cells
        :param cells: np array containing the cells to free, or int (if num_cells was 1)
        """

        if isinstance(cells, np.ndarray):
            self.allocated[np.array([self.cells_inverse[x] for x in cells])] = False
        else:
            self.allocated[self.cells_inverse[cells]] = False


class RealArithmetic:
    """
    Suite of arithmetic functions for real floating-point numbers. Adapted from AritPIM.
    """

    class DataType(Enum):
        """
        Represents a type of real data
        """
        IEEE_FLOAT16 = (1, 5, 10, np.float16)
        IEEE_FLOAT32 = (1, 8, 23, np.float32)

        def __init__(self, Ns, Ne, Nm, np_dtype):
            self.Ns = Ns
            self.Ne = Ne
            self.Nm = Nm
            self.N = Ns + Ne + Nm
            self.np_dtype = np_dtype

    @staticmethod
    def add(sim: simulator.SerialSimulator,
            x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter, dtype=DataType.IEEE_FLOAT32):
        """
        Performs an addition on the given columns.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param y_addr: the addresses of input y
        :param z_addr: the addresses of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        RealArithmetic.__floatingAddition(sim,
            x_addr[:dtype.Ns], x_addr[dtype.Ns:dtype.Ns+dtype.Ne], x_addr[dtype.Ns+dtype.Ne:],
            y_addr[:dtype.Ns], y_addr[dtype.Ns:dtype.Ns+dtype.Ne], y_addr[dtype.Ns+dtype.Ne:],
            z_addr[:dtype.Ns], z_addr[dtype.Ns:dtype.Ns+dtype.Ne], z_addr[dtype.Ns+dtype.Ne:], inter)

    @staticmethod
    def sub(sim: simulator.SerialSimulator,
            x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter, dtype=DataType.IEEE_FLOAT32):
        """
        Performs an addition on the given columns.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param y_addr: the addresses of input y
        :param z_addr: the addresses of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        # Invert the sign bit of y
        notys_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [notys_addr])
        sim.perform(simulator.GateType.NOT, y_addr[:dtype.Ns], [notys_addr])

        RealArithmetic.__floatingAddition(sim,
            x_addr[:dtype.Ns], x_addr[dtype.Ns:dtype.Ns+dtype.Ne], x_addr[dtype.Ns+dtype.Ne:],
            notys_addr, y_addr[dtype.Ns:dtype.Ns+dtype.Ne], y_addr[dtype.Ns+dtype.Ne:],
            z_addr[:dtype.Ns], z_addr[dtype.Ns:dtype.Ns+dtype.Ne], z_addr[dtype.Ns+dtype.Ne:], inter)

        inter.free(notys_addr)

    @staticmethod
    def inv(sim: simulator.SerialSimulator, x_addr: np.ndarray, inter, dtype=DataType.IEEE_FLOAT32):
        """
        Inverts (negates) the given number
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        copys_addr = inter.malloc(1)
        RealArithmetic.__id(sim, x_addr[0], copys_addr, inter)

        sim.perform(simulator.GateType.INIT1, [], [x_addr[0]])
        sim.perform(simulator.GateType.NOT, [copys_addr], [x_addr[0]])

        inter.free(copys_addr)

    @staticmethod
    def mult(sim: simulator.SerialSimulator,
            x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter, dtype=DataType.IEEE_FLOAT32):
        """
        Performs a multiplication on the given columns.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param y_addr: the addresses of input y
        :param z_addr: the addresses of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        RealArithmetic.__floatingMultiplication(sim,
            x_addr[:dtype.Ns], x_addr[dtype.Ns:dtype.Ns+dtype.Ne], x_addr[dtype.Ns+dtype.Ne:],
            y_addr[:dtype.Ns], y_addr[dtype.Ns:dtype.Ns+dtype.Ne], y_addr[dtype.Ns+dtype.Ne:],
            z_addr[:dtype.Ns], z_addr[dtype.Ns:dtype.Ns+dtype.Ne], z_addr[dtype.Ns+dtype.Ne:], inter)

    @staticmethod
    def copy(sim: simulator.SerialSimulator,
            x_addr: np.ndarray, z_addr: np.ndarray, inter, dtype=DataType.IEEE_FLOAT32):
        """
        Copies the data from x_addr to z_addr
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param z_addr: the addresses of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """
        for i in range(dtype.N):
            RealArithmetic.__id(sim, x_addr[i], z_addr[i], inter)

    @staticmethod
    def divByPowerOfTwo(sim: simulator, x_addr: np.ndarray, power, inter, dtype=DataType.IEEE_FLOAT32):
        """
        Divides the values in x_addr by 2^power
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param power: the power to divide by
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        # Write power to Ne + 1 intermediate columns
        power_addr = inter.malloc(dtype.Ne + 1)
        power_bin = representation.unsignedToBinaryFixed(np.array([[power]]), dtype.Ne + 1)
        for i in range(dtype.Ne + 1):
            if power_bin[i]:
                sim.perform(simulator.GateType.INIT1, [], [power_addr[i]])
            else:
                sim.perform(simulator.GateType.INIT0, [], [power_addr[i]])

        # Subtract power from the exponent of the numbers
        ze = inter.malloc(1)
        sim.perform(simulator.GateType.INIT0, [], [ze])

        RealArithmetic.__fixedSubtraction(sim, np.concatenate((x_addr[dtype.Ns:dtype.Ns + dtype.Ne], np.array([ze]))),
            power_addr, np.concatenate((x_addr[dtype.Ns:dtype.Ns + dtype.Ne], np.array([ze]))), inter)
        inter.free(power_addr)

        # If ze, then there was an underflow. Reset the output
        for x in x_addr:
            sim.perform(simulator.GateType.NOT, [ze], [x])

        inter.free(ze)

    @staticmethod
    def __fixedAddition(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray,
            inter, cin_addr=None, cout_addr=None):
        """
        Performs a fixed-point addition on the given columns. Supports both unsigned and signed numbers.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param y_addr: the addresses of input y (N-bit)
        :param z_addr: the addresses for the output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param cin_addr: the address for an optional input carry. "-1" designates constant 1 input carry.
        :param cout_addr: the address for an optional output carry
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        carry_addr1 = inter.malloc(1)
        carry_addr2 = inter.malloc(1)
        not_carry_addr1 = inter.malloc(1)
        not_carry_addr2 = inter.malloc(1)

        # Initialize the input carry
        if cin_addr is None:
            cin_addr = carry_addr1
            sim.perform(simulator.GateType.INIT0, [], [carry_addr1])
        elif cin_addr == -1:
            cin_addr = carry_addr1
            sim.perform(simulator.GateType.INIT1, [], [carry_addr1])
        sim.perform(simulator.GateType.INIT1, [], [not_carry_addr1])
        sim.perform(simulator.GateType.NOT, [cin_addr], [not_carry_addr1])

        # Perform the N iterations of full-adders
        for i in range(N):

            # The input and output carry locations
            in_carry_addr = (carry_addr1 if i % 2 == 0 else carry_addr2) if i > 0 else cin_addr
            out_carry_addr = (carry_addr1 if i % 2 == 1 else carry_addr2)
            in_not_carry_addr = (not_carry_addr1 if i % 2 == 0 else not_carry_addr2)
            out_not_carry_addr = (not_carry_addr1 if i % 2 == 1 else not_carry_addr2)
            if i == N - 1 and cout_addr is not None:
                out_carry_addr = cout_addr

            # Perform the full-adder
            RealArithmetic.__fullAdder(sim, x_addr[i], y_addr[i], in_carry_addr, z_addr[i], out_carry_addr, inter,
                                       notc_addr=in_not_carry_addr, notcout_addr=out_not_carry_addr)

        inter.free(carry_addr1)
        inter.free(carry_addr2)
        inter.free(not_carry_addr1)
        inter.free(not_carry_addr2)

    @staticmethod
    def __fixedSubtraction(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray,
            inter, cout_addr=None):
        """
        Performs a fixed-point subtraction on the given columns. Supports both unsigned and signed numbers.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param y_addr: the addresses of input y (N-bit)
        :param z_addr: the addresses for the output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param cout_addr: the address for an optional output carry
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        carry_addr1 = inter.malloc(1)
        carry_addr2 = inter.malloc(1)
        not_carry_addr1 = inter.malloc(1)
        not_carry_addr2 = inter.malloc(1)
        not_y_addr = inter.malloc(1)

        # Initialize the input carry
        sim.perform(simulator.GateType.INIT1, [], [carry_addr1])
        sim.perform(simulator.GateType.INIT0, [], [not_carry_addr1])

        # Perform the N iterations of full-subtractors
        for i in range(N):
            # The input and output carry locations
            in_carry_addr = (carry_addr1 if i % 2 == 0 else carry_addr2)
            out_carry_addr = (carry_addr1 if i % 2 == 1 else carry_addr2)
            in_not_carry_addr = (not_carry_addr1 if i % 2 == 0 else not_carry_addr2)
            out_not_carry_addr = (not_carry_addr1 if i % 2 == 1 else not_carry_addr2)
            if i == N - 1 and cout_addr is not None:
                out_carry_addr = cout_addr

            # Perform the full-subtractor
            sim.perform(simulator.GateType.INIT1, [], [not_y_addr])
            sim.perform(simulator.GateType.NOT, [y_addr[i]], [not_y_addr])
            RealArithmetic.__fullAdder(sim, x_addr[i], not_y_addr, in_carry_addr, z_addr[i], out_carry_addr, inter,
                                       notc_addr=in_not_carry_addr, notcout_addr=out_not_carry_addr)

        inter.free(carry_addr1)
        inter.free(carry_addr2)
        inter.free(not_carry_addr1)
        inter.free(not_carry_addr2)
        inter.free(not_y_addr)

    @staticmethod
    def __fixedMultiplication(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray,
            inter):
        """
        Performs a fixed-point multiplication on the given columns. Supports only unsigned numbers.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param y_addr: the addresses of input y (N-bit)
        :param z_addr: the addresses for the output z (2N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == 2 * N)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        notx_addr = inter.malloc(N)
        noty_bit_addr = inter.malloc(1)
        p_bit_addr = inter.malloc(1)
        one_bit_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        # Compute and store not(x) in advance
        for i in range(N):
            sim.perform(simulator.GateType.INIT1, [], [notx_addr[i]])
            sim.perform(simulator.GateType.NOT, [x_addr[i]], [notx_addr[i]])

        # Iterate over partial products
        for i in range(N):

            # Compute y_i'
            sim.perform(simulator.GateType.INIT1, [], [noty_bit_addr])
            sim.perform(simulator.GateType.NOT, [y_addr[i]], [noty_bit_addr])

            if i == 0:
                for j in range(N):
                    sim.perform(simulator.GateType.INIT1, [], [z_addr[j]])
                    sim.perform(simulator.GateType.MIN3, [notx_addr[j], noty_bit_addr, one_bit_addr], [z_addr[j]])
                for j in range(N, 2 * N):
                    sim.perform(simulator.GateType.INIT0, [], [z_addr[j]])
            else:
                # Perform partial product computation and addition
                carry_addr1 = inter.malloc(1)
                carry_addr2 = inter.malloc(1)
                not_carry_addr1 = inter.malloc(1)
                not_carry_addr2 = inter.malloc(1)
                sim.perform(simulator.GateType.INIT0, [], [carry_addr1])
                sim.perform(simulator.GateType.INIT1, [], [not_carry_addr1])

                for j in range(N):
                    # The input and output carry locations
                    in_carry_addr = (carry_addr1 if j % 2 == 0 else carry_addr2)
                    out_carry_addr = (carry_addr1 if j % 2 == 1 else carry_addr2)
                    in_not_carry_addr = (not_carry_addr1 if j % 2 == 0 else not_carry_addr2)
                    out_not_carry_addr = (not_carry_addr1 if j % 2 == 1 else not_carry_addr2)
                    if j == N - 1:
                        out_carry_addr = z_addr[i + N]

                    sim.perform(simulator.GateType.INIT1, [], [p_bit_addr])
                    sim.perform(simulator.GateType.MIN3, [notx_addr[j], noty_bit_addr, one_bit_addr], [p_bit_addr])

                    RealArithmetic.__fullAdder(sim, z_addr[i + j], p_bit_addr, in_carry_addr, z_addr[i + j], out_carry_addr,
                                               inter, notc_addr=in_not_carry_addr, notcout_addr=out_not_carry_addr)

                inter.free(carry_addr1)
                inter.free(carry_addr2)
                inter.free(not_carry_addr1)
                inter.free(not_carry_addr2)

        inter.free(notx_addr)
        inter.free(noty_bit_addr)
        inter.free(p_bit_addr)
        inter.free(one_bit_addr)

    @staticmethod
    def __floatingAddition(sim: simulator.SerialSimulator,
            xs_addr: np.ndarray, xe_addr: np.ndarray, xm_addr: np.ndarray, ys_addr: np.ndarray, ye_addr: np.ndarray, ym_addr: np.ndarray,
            zs_addr: np.ndarray, ze_addr: np.ndarray, zm_addr: np.ndarray, inter):
        """
        Performs a floating-point addition on the given columns. Supports only signed numbers.
        :param sim: the simulation environment
        :param xs_addr: the addresses of input xs (1-bit)
        :param xe_addr: the addresses of input xe (Ne-bit)
        :param xm_addr: the addresses of input xm (Nm-bit)
        :param ys_addr: the addresses of input ys (1-bit)
        :param ye_addr: the addresses of input ye (Ne-bit)
        :param ym_addr: the addresses of input yn (Nm-bit)
        :param zs_addr: the addresses of output zs (1-bit)
        :param ze_addr: the addresses of output ze (Ne-bit)
        :param zm_addr: the addresses of output zn (Nm-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        Ne = len(xe_addr)
        Nm = len(xm_addr)
        assert (len(ye_addr) == Ne)
        assert (len(ym_addr) == Nm)
        assert (len(ze_addr) == Ne)
        assert (len(zm_addr) == Nm)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        # Setup the hidden bits to be 1 if the exponent is non-zero
        xhidden_addr = inter.malloc(1)
        yhidden_addr = inter.malloc(1)
        zhidden_addr = inter.malloc(1)
        RealArithmetic.__reduceOR(sim, xe_addr, xhidden_addr, inter)
        RealArithmetic.__reduceOR(sim, ye_addr, yhidden_addr, inter)
        xm_addr = np.concatenate((xm_addr, np.array([xhidden_addr])))
        ym_addr = np.concatenate((ym_addr, np.array([yhidden_addr])))
        zm_addr = np.concatenate((zm_addr, np.array([zhidden_addr])))
        Nm = Nm + 1

        # Zero bit (constant zero used typically for padding)
        zero_bit_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])
        # One bit
        one_bit_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        # MSB of ze
        zemsb_addr = inter.malloc(1)
        ze_addr = np.concatenate((ze_addr, np.array([zemsb_addr])))

        # Compute deltaE and swap using fixed-point subtraction
        deltaE_addr = inter.malloc(Ne + 1)
        swap_addr = inter.malloc(1)
        notswap_addr = inter.malloc(1)
        RealArithmetic.__fixedSubtraction(sim, xe_addr, ye_addr, deltaE_addr[:Ne], inter, cout_addr=notswap_addr)
        sim.perform(simulator.GateType.INIT1, [], [deltaE_addr[Ne]])
        sim.perform(simulator.GateType.NOT, [notswap_addr], [deltaE_addr[Ne]])
        sim.perform(simulator.GateType.INIT1, [], [swap_addr])
        sim.perform(simulator.GateType.NOT, [notswap_addr], [swap_addr])

        # Perform conditional swap

        # ze = mux_swap(ye, xe)
        for i in range(Ne):
            RealArithmetic.__mux(sim, swap_addr, ye_addr[i], xe_addr[i], ze_addr[i], inter, nota_addr=notswap_addr, zero_bit_addr=zero_bit_addr)
        sim.perform(simulator.GateType.INIT0, [], [ze_addr[Ne]])

        # xmt = mux_swap(ym, xm)
        xmt_addr = zm_addr
        for i in range(Nm):
            RealArithmetic.__mux(sim, swap_addr, ym_addr[i], xm_addr[i], xmt_addr[i], inter, nota_addr=notswap_addr, zero_bit_addr=zero_bit_addr)

        # ymt = mux_swap(xm, ym)
        ymt_addr = inter.malloc(Nm)
        for i in range(Nm):
            RealArithmetic.__mux(sim, swap_addr, xm_addr[i], ym_addr[i], ymt_addr[i], inter, nota_addr=notswap_addr, zero_bit_addr=zero_bit_addr)

        # Compute absDeltaE = abs(deltaE)
        RealArithmetic.__abs(sim, deltaE_addr, deltaE_addr, inter, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)

        # Perform variable shift

        # We optimize the variable shift to use ceil(log2(Nm + 2)) bits instead of Nt by computing the OR of the top bits
        or_addr = inter.malloc(1)
        if Ne > ceil(log2(Nm + 2)):
            # Compute the OR of the top bits
            RealArithmetic.__reduceOR(sim, deltaE_addr[ceil(log2(Nm + 2)):], or_addr, inter)
            # If the OR is one, then zero the mantissa
            for i in range(Nm):
                sim.perform(simulator.GateType.NOT, [or_addr], [ymt_addr[i]])
        inter.free(or_addr)
        guard_addr = inter.malloc(1)
        round_addr = inter.malloc(1)
        sticky_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT0, [], [sticky_addr])
        sim.perform(simulator.GateType.INIT0, [], [guard_addr])
        sim.perform(simulator.GateType.INIT0, [], [round_addr])
        RealArithmetic.__variableShift(sim, np.concatenate((np.array([round_addr, guard_addr]), ymt_addr)),
            deltaE_addr[:ceil(log2(Nm + 2))], inter, sticky_addr=sticky_addr, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)

        inter.free(deltaE_addr)

        # Perform XOR on the signs of x and y
        sdiff_addr = inter.malloc(1)
        notsdiff_addr = inter.malloc(1)
        RealArithmetic.__xor(sim, xs_addr.item(), ys_addr.item(), sdiff_addr, inter, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        sim.perform(simulator.GateType.INIT1, [], [notsdiff_addr])
        sim.perform(simulator.GateType.NOT, [sdiff_addr], [notsdiff_addr])

        # Perform mantissa addition
        # Compute zm = (x_m' if (x_s == y_s) else -x_m') + y_m' = (x_m' XOR sdiff) + y_m + sdiff
        mantissa_carry_addr = inter.malloc(1)

        temp_carry_addr1 = inter.malloc(1)
        temp_carry_addr2 = inter.malloc(1)
        temp_not_carry_addr1 = inter.malloc(1)
        temp_not_carry_addr2 = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [temp_not_carry_addr1])
        sim.perform(simulator.GateType.NOT, [sdiff_addr], [temp_not_carry_addr1])

        for j in range(Nm):

            # The input and output carry locations
            in_carry_addr = (temp_carry_addr1 if j % 2 == 0 else temp_carry_addr2) if j > 0 else sdiff_addr
            out_carry_addr = (temp_carry_addr1 if j % 2 == 1 else temp_carry_addr2)
            in_not_carry_addr = (temp_not_carry_addr1 if j % 2 == 0 else temp_not_carry_addr2)
            out_not_carry_addr = (temp_not_carry_addr1 if j % 2 == 1 else temp_not_carry_addr2)
            if j == Nm - 1:
                out_carry_addr = mantissa_carry_addr

            xor_bit_addr = inter.malloc(1)

            RealArithmetic.__xor(sim, xmt_addr[j], sdiff_addr, xor_bit_addr, inter, notb_addr=notsdiff_addr,
                                 zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)

            RealArithmetic.__fullAdder(sim, ymt_addr[j], xor_bit_addr, in_carry_addr, zm_addr[j], out_carry_addr, inter,
                                       notc_addr=in_not_carry_addr, notcout_addr=out_not_carry_addr)

            inter.free(xor_bit_addr)

        inter.free(temp_carry_addr1)
        inter.free(temp_carry_addr2)
        inter.free(temp_not_carry_addr1)
        inter.free(temp_not_carry_addr2)

        inter.free(ymt_addr)

        # If sdiff and not mantissa_carry, then negative_m (if negative_m = 1, then zm is negative);
        # thus, negative_M = sdiff AND (NOT mantissa_carry)) = NOT(notsdiff OR mantissa_carry) = NOR(notsdiff, mantissa_carry)
        negativeM_addr = inter.malloc(1)
        notnegativeM_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [negativeM_addr])
        sim.perform(simulator.GateType.MIN3, [notsdiff_addr, mantissa_carry_addr, one_bit_addr], [negativeM_addr])
        sim.perform(simulator.GateType.INIT1, [], [notnegativeM_addr])
        sim.perform(simulator.GateType.NOT, [negativeM_addr], [notnegativeM_addr])

        # If negative, then set s = -s. Specifically, set s = (s XOR negative) and add with carry-in of negative
        # (implemented using the absolute value routine)
        negativeM_copy_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [negativeM_copy_addr])
        sim.perform(simulator.GateType.NOT, [notnegativeM_addr], [negativeM_copy_addr])
        RealArithmetic.__abs(sim,
                             np.concatenate((np.array([sticky_addr, round_addr, guard_addr]), zm_addr, np.array([negativeM_copy_addr]))),
                             np.concatenate((np.array([sticky_addr, round_addr, guard_addr]), zm_addr, np.array([negativeM_copy_addr]))),
                             inter, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        inter.free(negativeM_copy_addr)

        # if diff_signs, then mantissa_carry = False
        sim.perform(simulator.GateType.NOT, [sdiff_addr], [mantissa_carry_addr])  # X-MAGIC

        # Perform right-shift normalization
        # SerialArithmetic.__fixedAddBit(sim, ze_addr, ze_addr, inter, cin_addr=mantissa_carry_addr)  # performed as part of overflow addition
        mantissa_carry_copy_addr = inter.malloc(1)
        RealArithmetic.__id(sim, mantissa_carry_addr, mantissa_carry_copy_addr, inter)
        RealArithmetic.__variableShift(sim,
                                       np.concatenate((np.array([round_addr, guard_addr]), zm_addr, np.array([mantissa_carry_copy_addr]))),
                                       np.array([mantissa_carry_addr]), inter, sticky_addr=sticky_addr, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        inter.free(mantissa_carry_copy_addr)

        # Perform left-shift normalization
        left_shift_addr = inter.malloc(Ne + 1)
        RealArithmetic.__normalizeShift(sim,
                                        np.concatenate((np.array([round_addr, guard_addr]), zm_addr)),
            left_shift_addr[:ceil(log2(Nm + 2))], inter, direction=True, zero_bit_addr=zero_bit_addr)
        for col in left_shift_addr[ceil(log2(Nm + 2)):]:
            sim.perform(simulator.GateType.INIT0, [], [col])
        # Subtract from exponent
        RealArithmetic.__fixedSubtraction(sim, ze_addr, left_shift_addr, ze_addr, inter)
        inter.free(left_shift_addr)

        # Perform the round-to-nearest-tie-to-even
        # sticky_addr = OR(round_addr, sticky_addr)
        RealArithmetic.__or(sim, round_addr, sticky_addr, sticky_addr, inter, one_bit_addr=one_bit_addr)
        # should_round_addr = AND(guard, OR(sticky_addr, zm[0])) = NOR(NOT guard, NOR(sticky_addr, zm[0]))
        should_round_addr = inter.malloc(1)
        temps_addr = inter.malloc(2)
        sim.perform(simulator.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(simulator.GateType.MIN3, [sticky_addr, zm_addr[0], one_bit_addr], [temps_addr[0]])
        sim.perform(simulator.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(simulator.GateType.NOT, [guard_addr], [temps_addr[1]])
        sim.perform(simulator.GateType.INIT1, [], [should_round_addr])
        sim.perform(simulator.GateType.MIN3, [temps_addr[0], temps_addr[1], one_bit_addr], [should_round_addr])
        inter.free(temps_addr)

        inter.free(guard_addr)
        inter.free(round_addr)
        inter.free(sticky_addr)

        # Add the rounding correction bit to the mantissa
        # Store the carry-out (whether the rounding caused an overflow since the mantissa was all 1) in overflow_addr
        overflow_addr = inter.malloc(1)
        RealArithmetic.__fixedAddBit(sim, zm_addr, zm_addr, inter, should_round_addr, overflow_addr, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        inter.free(should_round_addr)
        # If such overflow occurred, increment the exponent
        # Perform the addition with the addition of mantissa_carry_addr
        RealArithmetic.__or(sim, zhidden_addr, overflow_addr, zhidden_addr, inter, one_bit_addr=one_bit_addr)
        temp_carry_addr = inter.malloc(1)
        RealArithmetic.__fullAdder(sim, mantissa_carry_addr, overflow_addr, ze_addr[0], ze_addr[0], temp_carry_addr, inter)
        RealArithmetic.__fixedAddBit(sim, ze_addr[1:], ze_addr[1:], inter, temp_carry_addr, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        inter.free(temp_carry_addr)

        inter.free(mantissa_carry_addr)

        inter.free(overflow_addr)

        # Computing the final sign

        # Idea: Control flow (before conversion to mux - for reference)
        # if xs == ys:
        #     zs = xs
        # else:
        #     if xs AND (NOT ys):
        #         zs = negativeM XOR swap
        #     else:
        #         zs = not negativeM XOR swap

        # Data flow. Observations:
        # 1. AND(xs, NOT ys) = NOR(NOT xs, ys)
        # 2. The top else evaluates to:
        # notNegativeM XOR swap XOR NOR(NOT xs, ys)
        # 3. Overall, we find:
        # zs = xs if XNOR(xs, ys) else (notNegativeM XOR swap XOR NOR(NOT xs, ys))  (implemented using mux)

        # Data flow. Implementation:
        xor_addr = inter.malloc(1)

        not_xs_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [not_xs_addr])
        sim.perform(simulator.GateType.NOT, [xs_addr], [not_xs_addr])

        sim.perform(simulator.GateType.INIT1, [], [xor_addr])
        sim.perform(simulator.GateType.MIN3, [ys_addr.item(), not_xs_addr, one_bit_addr], [xor_addr])

        inter.free(not_xs_addr)

        RealArithmetic.__xor(sim, xor_addr, swap_addr, xor_addr, inter, notb_addr=notswap_addr, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        RealArithmetic.__xor(sim, xor_addr, notnegativeM_addr, xor_addr, inter, notb_addr=negativeM_addr, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        RealArithmetic.__mux(sim, notsdiff_addr, xs_addr.item(), xor_addr, zs_addr.item(), inter, nota_addr=sdiff_addr, zero_bit_addr=zero_bit_addr)

        inter.free(xor_addr)

        inter.free(swap_addr)
        inter.free(notswap_addr)
        inter.free(sdiff_addr)
        inter.free(notsdiff_addr)
        inter.free(negativeM_addr)
        inter.free(notnegativeM_addr)

        # Set the output to zero if the zhidden is zero or exponent is negative
        # should_zero = OR(NOT zhidden, ze[-1])
        should_zero_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [should_zero_addr])
        sim.perform(simulator.GateType.NOT, [zhidden_addr], [should_zero_addr])
        RealArithmetic.__or(sim, should_zero_addr, ze_addr[-1], should_zero_addr, inter, one_bit_addr=one_bit_addr)
        for z in np.concatenate((zs_addr, ze_addr, zm_addr)):
            sim.perform(simulator.GateType.NOT, [should_zero_addr], [z])  # X-MAGIC
        inter.free(should_zero_addr)

        inter.free(zero_bit_addr)
        inter.free(one_bit_addr)
        inter.free(zemsb_addr)

        inter.free(xhidden_addr)
        inter.free(yhidden_addr)
        inter.free(zhidden_addr)

    @staticmethod
    def __floatingMultiplication(sim: simulator.SerialSimulator,
            xs_addr: np.ndarray, xe_addr: np.ndarray, xm_addr: np.ndarray, ys_addr: np.ndarray, ye_addr: np.ndarray, ym_addr: np.ndarray,
            zs_addr: np.ndarray, ze_addr: np.ndarray, zm_addr: np.ndarray, inter):
        """
        Performs a floating-point multiplication on the given columns. Supports only signed numbers.
        :param sim: the simulation environment
        :param xs_addr: the addresses of input xs (1-bit)
        :param xe_addr: the addresses of input xe (Ne-bit)
        :param xm_addr: the addresses of input xm (Nm-bit)
        :param ys_addr: the addresses of input ys (1-bit)
        :param ye_addr: the addresses of input ye (Ne-bit)
        :param ym_addr: the addresses of input yn (Nm-bit)
        :param zs_addr: the addresses of output zs (1-bit)
        :param ze_addr: the addresses of output ze (Ne-bit)
        :param zm_addr: the addresses of output zn (Nm-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        Ne = len(xe_addr)
        Nm = len(xm_addr)
        assert (len(ye_addr) == Ne)
        assert (len(ym_addr) == Nm)
        assert (len(ze_addr) == Ne)
        assert (len(zm_addr) == Nm)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        # Setup the hidden bits to be 1 if the exponent is non-zero
        xhidden_addr = inter.malloc(1)
        yhidden_addr = inter.malloc(1)
        zhidden_addr = inter.malloc(1)
        RealArithmetic.__reduceOR(sim, xe_addr, xhidden_addr, inter)
        RealArithmetic.__reduceOR(sim, ye_addr, yhidden_addr, inter)
        xm_addr = np.concatenate((xm_addr, np.array([xhidden_addr])))
        ym_addr = np.concatenate((ym_addr, np.array([yhidden_addr])))
        zm_addr = np.concatenate((zm_addr, np.array([zhidden_addr])))
        Nm = Nm + 1

        # Zero bit (constant zero used typically for padding)
        zero_bit_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])
        # One bit
        one_bit_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        # MSB of ze
        zemsb_addr = inter.malloc(1)
        ze_addr = np.concatenate((ze_addr, np.array([zemsb_addr])))

        # Compute the product of the mantissas
        mantissa_carry_addr = inter.malloc(1)
        lower_mult_addr = inter.malloc(Nm - 2)
        guard_addr = inter.malloc(1)
        RealArithmetic.__fixedMultiplication(sim, xm_addr, ym_addr,
                                           np.concatenate((lower_mult_addr, np.array([guard_addr]), zm_addr, np.array([mantissa_carry_addr]))), inter)
        sticky_addr = inter.malloc(1)
        RealArithmetic.__reduceOR(sim, lower_mult_addr, sticky_addr, inter)
        inter.free(lower_mult_addr)

        # Write -(1 << Ne - 1) to ze_addr
        sim.perform(simulator.GateType.INIT1, [], [ze_addr[0]])
        sim.perform(simulator.GateType.INIT1, [], [ze_addr[-2]])
        sim.perform(simulator.GateType.INIT1, [], [ze_addr[-1]])
        for z in ze_addr[1:-2]:
            sim.perform(simulator.GateType.INIT0, [], [z])

        # Increment exponent
        # performed as part of exponent addition below
        # SerialArithmetic.__fixedAddBit(sim, ze_addr, ze_addr, inter, cin_addr=mantissa_carry_addr)

        # Perform right-shift normalization
        mantissa_carry_copy_addr = inter.malloc(1)
        RealArithmetic.__id(sim, mantissa_carry_addr, mantissa_carry_copy_addr, inter)
        RealArithmetic.__variableShift(sim,
                                       np.concatenate((np.array([guard_addr]), zm_addr, np.array([mantissa_carry_copy_addr]))), np.array([mantissa_carry_addr]),
                                       inter, sticky_addr=sticky_addr, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        inter.free(mantissa_carry_copy_addr)

        # Perform the round-to-nearest-tie-to-even
        # should_round_addr = AND(guard, OR(sticky_addr, zm[0])) = NOR(NOT guard, NOR(sticky_addr, zm[0]))
        should_round_addr = inter.malloc(1)
        temps_addr = inter.malloc(2)
        sim.perform(simulator.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(simulator.GateType.MIN3, [sticky_addr, zm_addr[0], one_bit_addr], [temps_addr[0]])
        sim.perform(simulator.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(simulator.GateType.NOT, [guard_addr], [temps_addr[1]])
        sim.perform(simulator.GateType.INIT1, [], [should_round_addr])
        sim.perform(simulator.GateType.MIN3, [temps_addr[0], temps_addr[1], one_bit_addr], [should_round_addr])
        inter.free(temps_addr)

        inter.free(guard_addr)
        inter.free(sticky_addr)

        overflow_addr = inter.malloc(1)
        RealArithmetic.__fixedAddBit(sim, zm_addr, zm_addr, inter, should_round_addr, overflow_addr, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        inter.free(should_round_addr)
        # In case the rounding causes an overflow (the mantissa was all ones), increment the exponent
        RealArithmetic.__or(sim, zhidden_addr, overflow_addr, zhidden_addr, inter, one_bit_addr=one_bit_addr)
        # SerialArithmetic.__fixedAddBit(sim, ze_addr, ze_addr, inter, overflow_addr)  # performed as part of next addition

        # Compute the new exponent
        RealArithmetic.__fixedAddition(sim, np.concatenate((xe_addr, np.array([zero_bit_addr]))), ze_addr, ze_addr, inter, cin_addr=overflow_addr)
        RealArithmetic.__fixedAddition(sim, np.concatenate((ye_addr, np.array([zero_bit_addr]))), ze_addr, ze_addr, inter, cin_addr=mantissa_carry_addr)

        inter.free(mantissa_carry_addr)

        inter.free(overflow_addr)

        # Compute the XOR of the signs
        if xs_addr != ys_addr:
            RealArithmetic.__xor(sim, xs_addr.item(), ys_addr.item(), zs_addr.item(), inter, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)
        else:
            sim.perform(simulator.GateType.INIT0, [], [zs_addr.item()])

        # Set the output to zero if the zhidden is zero or exponent is negative
        # should_zero = OR(NOT zhidden, ze[-1])
        should_zero_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [should_zero_addr])
        sim.perform(simulator.GateType.NOT, [zhidden_addr], [should_zero_addr])
        RealArithmetic.__or(sim, should_zero_addr, ze_addr[-1], should_zero_addr, inter, one_bit_addr=one_bit_addr)
        for z in np.concatenate((zs_addr, ze_addr, zm_addr)):
            sim.perform(simulator.GateType.NOT, [should_zero_addr], [z])  # X-MAGIC
        inter.free(should_zero_addr)

        inter.free(zero_bit_addr)
        inter.free(one_bit_addr)
        inter.free(zemsb_addr)

        inter.free(xhidden_addr)
        inter.free(yhidden_addr)
        inter.free(zhidden_addr)

    @staticmethod
    def __id(sim: simulator.SerialSimulator, a_addr: int, z_addr: int, inter, notz_addr=None):
        """
        Performs z = ID(a) on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param z_addr: the index of the output
        :param notz_addr: the index of the optional output which stores the not of z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        custom_not_out_addr = notz_addr is None
        if custom_not_out_addr:
            notz_addr = inter.malloc(1)

        sim.perform(simulator.GateType.INIT1, [], [notz_addr])
        sim.perform(simulator.GateType.NOT, [a_addr], [notz_addr])

        sim.perform(simulator.GateType.INIT1, [], [z_addr])
        sim.perform(simulator.GateType.NOT, [notz_addr], [z_addr])

        if custom_not_out_addr:
            inter.free(notz_addr)

    @staticmethod
    def __or(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, z_addr: int, inter,
            nota_addr=None, notb_addr=None, notz_addr=None, one_bit_addr=None):
        """
        Performs z = OR(a,b) on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param z_addr: the index of the output
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param notz_addr: the index of the optional output which stores the not of z
        :param one_bit_addr: an index for an optional constant one column
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        custom_one_addr = one_bit_addr is None
        if custom_one_addr:
            one_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        custom_nor_out_addr = notz_addr is None
        if custom_nor_out_addr:
            notz_addr = inter.malloc(1)

        sim.perform(simulator.GateType.INIT1, [], [notz_addr])
        sim.perform(simulator.GateType.MIN3, [a_addr, b_addr, one_bit_addr], [notz_addr])

        sim.perform(simulator.GateType.INIT1, [], [z_addr])
        sim.perform(simulator.GateType.NOT, [notz_addr], [z_addr])

        if custom_nor_out_addr:
            inter.free(notz_addr)
        if custom_one_addr:
            inter.free(one_bit_addr)

    @staticmethod
    def __xor(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, z_addr: int, inter,
            nota_addr=None, notb_addr=None, zero_bit_addr=None, one_bit_addr=None):
        """
        Performs z = XOR(a,b) on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param z_addr: the index of the output
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param zero_bit_addr: an index for an optional constant zero column
        :param one_bit_addr: an index for an optional constant one column
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        custom_zero_addr = zero_bit_addr is None
        if custom_zero_addr:
            zero_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])
        custom_one_addr = one_bit_addr is None
        if custom_one_addr:
            one_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        # If in-place XOR
        if a_addr == z_addr or b_addr == z_addr:

            temps_addr = inter.malloc(3)

            sim.perform(simulator.GateType.INIT1, [], [temps_addr[0]])
            sim.perform(simulator.GateType.MIN3, [a_addr, b_addr, zero_bit_addr], [temps_addr[0]])

            sim.perform(simulator.GateType.INIT1, [], [temps_addr[1]])
            sim.perform(simulator.GateType.MIN3, [a_addr, temps_addr[0], zero_bit_addr], [temps_addr[1]])

            sim.perform(simulator.GateType.INIT1, [], [temps_addr[2]])
            sim.perform(simulator.GateType.MIN3, [b_addr, temps_addr[0], zero_bit_addr], [temps_addr[2]])

            sim.perform(simulator.GateType.INIT1, [], [z_addr])
            sim.perform(simulator.GateType.MIN3, [temps_addr[1], temps_addr[2], zero_bit_addr], [z_addr])

            inter.free(temps_addr)
        else:

            temp_addr = inter.malloc(1)

            sim.perform(simulator.GateType.INIT1, [], [temp_addr])
            sim.perform(simulator.GateType.MIN3, [a_addr, b_addr, one_bit_addr], [temp_addr])

            sim.perform(simulator.GateType.INIT1, [], [z_addr])
            sim.perform(simulator.GateType.MIN3, [a_addr, b_addr, zero_bit_addr], [z_addr])

            sim.perform(simulator.GateType.NOT, [temp_addr], [z_addr])

            inter.free(temp_addr)

        if custom_zero_addr:
            inter.free(zero_bit_addr)
        if custom_one_addr:
            inter.free(one_bit_addr)

    @staticmethod
    def __mux(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, c_addr: int, z_addr: int, inter,
            nota_addr=None, zero_bit_addr=None):
        """
        Performs a 1-bit mux_a(b,c) on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a (the condition)
        :param b_addr: the index of input b (if a if true)
        :param c_addr: the index of input c (if a is false)
        :param z_addr: the index of the output
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param nota_addr: the index of the optional input which stores the not of a
        :param zero_bit_addr: an index for an optional constant zero column
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        computed_not_a = nota_addr is None
        if computed_not_a:
            nota_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [nota_addr])
            sim.perform(simulator.GateType.NOT, [a_addr], [nota_addr])

        custom_zero_addr = zero_bit_addr is None
        if custom_zero_addr:
            zero_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])

        temp_addr = inter.malloc(1)

        sim.perform(simulator.GateType.INIT1, [], [temp_addr])
        sim.perform(simulator.GateType.MIN3, [b_addr, a_addr, zero_bit_addr], [temp_addr])
        sim.perform(simulator.GateType.MIN3, [c_addr, nota_addr, zero_bit_addr], [temp_addr])
        sim.perform(simulator.GateType.INIT1, [], [z_addr])
        sim.perform(simulator.GateType.NOT, [temp_addr], [z_addr])

        inter.free(temp_addr)

        if computed_not_a:
            inter.free(nota_addr)
        if custom_zero_addr:
            inter.free(zero_bit_addr)

    @staticmethod
    def __halfAdder(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, s_addr: int, cout_addr: int, inter,
            zero_bit_addr=None, one_bit_addr=None):
        """
        Performs a half-adder on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param s_addr: the index of the output sum
        :param cout_addr: the index of the output carry
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param zero_bit_addr: an index for an optional constant zero column
        :param one_bit_addr: an index for an optional constant one column
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        custom_zero_addr = zero_bit_addr is None
        if custom_zero_addr:
            zero_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])
        custom_one_addr = one_bit_addr is None
        if custom_one_addr:
            one_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        temps_addr = inter.malloc(2)

        sim.perform(simulator.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(simulator.GateType.MIN3, [a_addr, b_addr, zero_bit_addr], [temps_addr[0]])

        sim.perform(simulator.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(simulator.GateType.MIN3, [a_addr, b_addr, one_bit_addr], [temps_addr[1]])

        sim.perform(simulator.GateType.INIT1, [], [cout_addr])
        sim.perform(simulator.GateType.NOT, [temps_addr[0]], [cout_addr])

        sim.perform(simulator.GateType.INIT1, [], [s_addr])
        sim.perform(simulator.GateType.MIN3, [cout_addr, temps_addr[1], one_bit_addr], [s_addr])

        inter.free(temps_addr)

        if custom_zero_addr:
            inter.free(zero_bit_addr)
        if custom_one_addr:
            inter.free(one_bit_addr)

    @staticmethod
    def __fullAdder(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, c_addr: int, s_addr: int,
            cout_addr: int, inter, notc_addr=None, notcout_addr=None):
        """
        Performs a full-adder on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param c_addr: the index of input c
        :param s_addr: the index of the output sum
        :param cout_addr: the index of the output carry
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param notc_addr: the index of the optional input which stores the not of c
        :param notcout_addr: the index of the optional output which stores the not of cout
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        computed_not_c = notc_addr is None
        if computed_not_c:
            notc_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [notc_addr])
            sim.perform(simulator.GateType.NOT, [c_addr], [notc_addr])

        custom_not_cout_addr = notcout_addr is None
        if custom_not_cout_addr:
            notcout_addr = inter.malloc(1)

        temp_addr = inter.malloc(1)

        sim.perform(simulator.GateType.INIT1, [], [notcout_addr])
        sim.perform(simulator.GateType.MIN3, [a_addr, b_addr, c_addr], [notcout_addr])

        sim.perform(simulator.GateType.INIT1, [], [cout_addr])
        sim.perform(simulator.GateType.NOT, [notcout_addr], [cout_addr])

        sim.perform(simulator.GateType.INIT1, [], [temp_addr])
        sim.perform(simulator.GateType.MIN3, [a_addr, b_addr, notc_addr], [temp_addr])

        sim.perform(simulator.GateType.INIT1, [], [s_addr])
        sim.perform(simulator.GateType.MIN3, [cout_addr, notc_addr, temp_addr], [s_addr])

        inter.free(temp_addr)

        if computed_not_c:
            inter.free(notc_addr)
        if custom_not_cout_addr:
            inter.free(notcout_addr)

    @staticmethod
    def __fixedAddBit(sim: simulator.SerialSimulator, x_addr: np.ndarray, z_addr: np.ndarray, inter,
            cin_addr, cout_addr=None, zero_bit_addr=None, one_bit_addr=None):
        """
        Adds a single bit to the given number using half-adders
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param z_addr: the addresses for the output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param cin_addr: the address for the input carry. "-1" designates constant 1 input carry.
        :param cout_addr: the address for an optional output carry
        :param zero_bit_addr: an index for an optional constant one column
        :param one_bit_addr: an index for an optional constant one column
        """

        N = len(x_addr)
        assert (len(z_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        custom_zero_addr = zero_bit_addr is None
        if custom_zero_addr:
            zero_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])
        custom_one_addr = one_bit_addr is None
        if custom_one_addr:
            one_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        carry_addr = inter.malloc(1)

        if cin_addr == -1:
            cin_addr = carry_addr
            sim.perform(simulator.GateType.INIT1, [], [carry_addr])

        # Initialize the carry out
        if cout_addr is None:
            cout_addr = carry_addr

        for i in range(N):

            in_carry_addr = carry_addr if i > 0 else cin_addr
            out_carry_addr = carry_addr if i < N - 1 else cout_addr

            RealArithmetic.__halfAdder(sim, x_addr[i], in_carry_addr, z_addr[i], out_carry_addr, inter, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)

        inter.free(carry_addr)

        if custom_zero_addr:
            inter.free(zero_bit_addr)
        if custom_one_addr:
            inter.free(one_bit_addr)

    @staticmethod
    def __abs(sim: simulator.SerialSimulator, x_addr: np.ndarray, z_addr: np.ndarray, inter, zero_bit_addr=None, one_bit_addr=None):
        """
        Computes the absolute value of the given fixed-point number.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param z_addr: the addresses for the output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param zero_bit_addr: an index for an optional constant one column
        :param one_bit_addr: an index for an optional constant one column
        """

        N = len(x_addr)
        assert (len(z_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        custom_zero_addr = zero_bit_addr is None
        if custom_zero_addr:
            zero_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])
        custom_one_addr = one_bit_addr is None
        if custom_one_addr:
            one_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        carry_addr = inter.malloc(1)

        for i in range(N - 1):

            in_carry_addr = carry_addr if i > 0 else x_addr[-1]
            out_carry_addr = carry_addr

            xor_addr = inter.malloc(1)

            RealArithmetic.__xor(sim, x_addr[i], x_addr[-1], xor_addr, inter, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)

            RealArithmetic.__halfAdder(sim, xor_addr, in_carry_addr, z_addr[i], out_carry_addr, inter, zero_bit_addr=zero_bit_addr, one_bit_addr=one_bit_addr)

            inter.free(xor_addr)

        sim.perform(simulator.GateType.INIT0, [], [z_addr[-1]])

        inter.free(carry_addr)

        if custom_zero_addr:
            inter.free(zero_bit_addr)
        if custom_one_addr:
            inter.free(one_bit_addr)

    @staticmethod
    def __reduceOR(sim: simulator.SerialSimulator, x_addr: np.ndarray, z_addr: int, inter, notz_addr=None):
        """
        Performs an OR reduction on the x bits, storing the result in z.
        :param sim: the simulation environment
        :param x_addr: the addresses of the input columns
        :param z_addr: the address of the output column
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param notz_addr: the index of the optional output which stores the not of z
        """

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        # Performed using De Morgan

        custom_nor_out_addr = notz_addr is None
        if custom_nor_out_addr:
            notz_addr = inter.malloc(1)

        sim.perform(simulator.GateType.INIT1, [], [notz_addr])
        for x in x_addr:
            sim.perform(simulator.GateType.NOT, [x], [notz_addr])

        sim.perform(simulator.GateType.INIT1, [], [z_addr])
        sim.perform(simulator.GateType.NOT, [notz_addr], [z_addr])

        if custom_nor_out_addr:
            inter.free(notz_addr)

    @staticmethod
    def __variableShift(sim: simulator.SerialSimulator, x_addr: np.ndarray, t_addr: np.ndarray, inter,
            sticky_addr=None, direction=False, zero_bit_addr=None, one_bit_addr=None):
        """
        Performs the in-place variable shift operation on the given columns.
        :param sim: the simulation environment
        :param x_addr: the addresses of input & output x (Nx-bit)
        :param t_addr: the addresses of input t (Nt-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param sticky_addr: an optional column for a sticky bit (OR of all of the bits that were truncated).
        :param direction: the direction of the shift. False is right-shift, and True is left-shift.
        :param zero_bit_addr: an index for an optional constant zero column
        :param one_bit_addr: an index for an optional constant one column
        """

        Nx = len(x_addr)
        log2_Nx = ceil(log2(Nx))
        Nt = len(t_addr)
        assert (Nt <= log2_Nx)

        if direction:
            x_addr = np.flip(x_addr)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        custom_zero_addr = zero_bit_addr is None
        if custom_zero_addr:
            zero_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])
        custom_one_addr = one_bit_addr is None
        if custom_one_addr:
            one_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [one_bit_addr])

        for j in range(Nt):

            not_tj = inter.malloc(1)
            sim.perform(simulator.GateType.INIT1, [], [not_tj])
            sim.perform(simulator.GateType.NOT, [t_addr[j]], [not_tj])

            if sticky_addr is not None:

                # Compute the OR of the bits that are potentially lost in this step
                or_addr = inter.malloc(1)
                RealArithmetic.__reduceOR(sim, x_addr[:2 ** j], or_addr, inter)
                # Compute the AND with whether the shift actually occurs
                sim.perform(simulator.GateType.NOT, [not_tj], [or_addr])  # X-MAGIC
                # Compute the OR with the current sticky bit
                RealArithmetic.__or(sim, sticky_addr, or_addr, sticky_addr, inter, one_bit_addr=one_bit_addr)
                inter.free(or_addr)

            for i in range(Nx - 2 ** j):
                RealArithmetic.__mux(sim, t_addr[j], x_addr[i + (2 ** j)], x_addr[i], x_addr[i], inter, nota_addr=not_tj, zero_bit_addr=zero_bit_addr)

            for i in range(max(Nx - 2 ** j, 0), Nx):
                sim.perform(simulator.GateType.NOT, [t_addr[j]], [x_addr[i]])  # X-MAGIC

            inter.free(not_tj)

        if custom_zero_addr:
            inter.free(zero_bit_addr)
        if custom_one_addr:
            inter.free(one_bit_addr)

    @staticmethod
    def __normalizeShift(sim: simulator.SerialSimulator, x_addr: np.ndarray, t_addr: np.ndarray, inter, direction=False, zero_bit_addr=None):
        """
        Performs the in-place normalize shift operation on the given columns
        :param sim: the simulation environment
        :param x_addr: the addresses of input & output x (Nx-bit)
        :param t_addr: the addresses of output t (Nt-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param direction: the direction of the shift. False is right-shift, and True is left-shift.
        :param zero_bit_addr: an index for an optional constant one column
        """

        Nx = len(x_addr)
        Nt = len(t_addr)

        if direction:
            x_addr = np.flip(x_addr)

        if isinstance(inter, np.ndarray):
            inter = IntermediateAllocator(inter)

        custom_zero_addr = zero_bit_addr is None
        if custom_zero_addr:
            zero_bit_addr = inter.malloc(1)
            sim.perform(simulator.GateType.INIT0, [], [zero_bit_addr])

        for j in reversed(range(Nt)):

            not_tj = inter.malloc(1)

            RealArithmetic.__reduceOR(sim, x_addr[:(2 ** j)], not_tj, inter, notz_addr=t_addr[j])

            for i in range(Nx - 2 ** j):
                RealArithmetic.__mux(sim, t_addr[j], x_addr[i + (2 ** j)], x_addr[i], x_addr[i], inter, nota_addr=not_tj, zero_bit_addr=zero_bit_addr)

            for i in range(max(Nx - 2 ** j, 0), Nx):
                sim.perform(simulator.GateType.NOT, [t_addr[j]], [x_addr[i]])  # X-MAGIC

            inter.free(not_tj)

        # If didn't contain any ones, then we define shift amount as zero
        not_lsb_addr = inter.malloc(1)
        sim.perform(simulator.GateType.INIT1, [], [not_lsb_addr])
        sim.perform(simulator.GateType.NOT, [x_addr[0]], [not_lsb_addr])
        for t in t_addr:
            sim.perform(simulator.GateType.NOT, [not_lsb_addr], [t])  # X-MAGIC
        inter.free(not_lsb_addr)

        if custom_zero_addr:
            inter.free(zero_bit_addr)


class ComplexArithmetic:
    """
    Suite of arithmetic functions for complex floating-point numbers.
    """

    class DataType(Enum):
        """
        Represents a type of real data
        """
        IEEE_CFLOAT16 = (RealArithmetic.DataType.IEEE_FLOAT16, None)
        IEEE_CFLOAT32 = (RealArithmetic.DataType.IEEE_FLOAT32, np.csingle)

        def __init__(self, base, np_dtype):
            self.base = base
            self.N = base.N * 2
            self.np_dtype = np_dtype

    @staticmethod
    def add(sim: simulator.SerialSimulator,
            x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter, dtype=DataType.IEEE_CFLOAT32):
        """
        Performs a complex addition on the given columns.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param y_addr: the addresses of input y
        :param z_addr: the addresses of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        xa_addr, xb_addr = ComplexArithmetic.__splitAddr(x_addr)
        ya_addr, yb_addr = ComplexArithmetic.__splitAddr(y_addr)
        za_addr, zb_addr = ComplexArithmetic.__splitAddr(z_addr)

        RealArithmetic.add(sim, xa_addr, ya_addr, za_addr, inter, dtype.base)
        RealArithmetic.add(sim, xb_addr, yb_addr, zb_addr, inter, dtype.base)

    @staticmethod
    def sub(sim: simulator.SerialSimulator,
            x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter, dtype=DataType.IEEE_CFLOAT32):
        """
        Performs a complex subtraction on the given columns.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param y_addr: the addresses of input y
        :param z_addr: the addresses of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        xa_addr, xb_addr = ComplexArithmetic.__splitAddr(x_addr)
        ya_addr, yb_addr = ComplexArithmetic.__splitAddr(y_addr)
        za_addr, zb_addr = ComplexArithmetic.__splitAddr(z_addr)

        RealArithmetic.sub(sim, xa_addr, ya_addr, za_addr, inter, dtype.base)
        RealArithmetic.sub(sim, xb_addr, yb_addr, zb_addr, inter, dtype.base)

    @staticmethod
    def mult(sim: simulator.SerialSimulator,
            x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter, dtype=DataType.IEEE_CFLOAT32):
        """
        Performs a complex multiplication on the given columns.
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param y_addr: the addresses of input y
        :param z_addr: the addresses of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        temp1_addr = inter[:dtype.base.N]
        temp2_addr = inter[dtype.base.N:2*dtype.base.N]
        inter = inter[2*dtype.base.N:]

        xa_addr, xb_addr = ComplexArithmetic.__splitAddr(x_addr)
        ya_addr, yb_addr = ComplexArithmetic.__splitAddr(y_addr)
        za_addr, zb_addr = ComplexArithmetic.__splitAddr(z_addr)

        RealArithmetic.mult(sim, xa_addr, ya_addr, temp1_addr, inter, dtype.base)
        RealArithmetic.mult(sim, xb_addr, yb_addr, temp2_addr, inter, dtype.base)
        RealArithmetic.sub(sim, temp1_addr, temp2_addr, za_addr, inter, dtype.base)

        RealArithmetic.mult(sim, xa_addr, yb_addr, temp1_addr, inter, dtype.base)
        RealArithmetic.mult(sim, xb_addr, ya_addr, temp2_addr, inter, dtype.base)
        RealArithmetic.add(sim, temp1_addr, temp2_addr, zb_addr, inter, dtype.base)

    @staticmethod
    def conjugate(sim: simulator.SerialSimulator, x_addr: np.ndarray, inter, dtype=DataType.IEEE_CFLOAT32):
        """
        Performs an in-place conjugate operation
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        xa_addr, xb_addr = ComplexArithmetic.__splitAddr(x_addr)
        RealArithmetic.inv(sim, xb_addr, inter, dtype)

    @staticmethod
    def divByPowerOfTwo(sim: simulator, x_addr: np.ndarray, power, inter, dtype=DataType.IEEE_CFLOAT32):
        """
        Divides the values in x_addr by 2^power
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param power: the power to divide by
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        xa_addr, xb_addr = ComplexArithmetic.__splitAddr(x_addr)
        RealArithmetic.divByPowerOfTwo(sim, xa_addr, power, inter, dtype.base)
        RealArithmetic.divByPowerOfTwo(sim, xb_addr, power, inter, dtype.base)

    @staticmethod
    def copy(sim: simulator.SerialSimulator,
            x_addr: np.ndarray, z_addr: np.ndarray, inter, dtype=DataType.IEEE_CFLOAT32):
        """
        Copies the data from x_addr to z_addr
        :param sim: the simulation environment
        :param x_addr: the addresses of input x
        :param z_addr: the addresses of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param dtype: the type of numbers
        """

        xa_addr, xb_addr = ComplexArithmetic.__splitAddr(x_addr)
        za_addr, zb_addr = ComplexArithmetic.__splitAddr(z_addr)

        RealArithmetic.copy(sim, xa_addr, za_addr, inter, dtype.base)
        RealArithmetic.copy(sim, xb_addr, zb_addr, inter, dtype.base)

    @staticmethod
    def __splitAddr(x_addr):
        """
        Splits the given complex address according to the real parts
        :param x_addr: the overall complex address
        """
        return x_addr[:len(x_addr)//2], x_addr[len(x_addr)//2:]