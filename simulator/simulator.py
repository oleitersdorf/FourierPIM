import numpy as np
from enum import Enum


class GateType(Enum):
    """
    Represents a type of logic gate.
    """

    NOT = 0
    MIN3 = 1
    INIT0 = 2
    INIT1 = 3


class GateDirection(Enum):
    """
    Represents the orientation of a logical gate
    """

    IN_ROW = 0
    IN_COLUMN = 1


class SerialSimulator:
    """
    Simulates a single array that adheres to the standard abstract PIM model
    """

    def __init__(self, num_rows: int, num_cols: int):
        """
        Initializes the array simulator with the given number of rows and columns
        :param num_rows: the number of rows
        :param num_cols: the number of columns
        """
        self.memory = np.zeros((num_cols, num_rows), dtype=bool)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.latency = 0
        self.energy = 0

    def perform(self, type: GateType, inputs, outputs, direction=GateDirection.IN_ROW, mask=None):
        """
        Performs the given logic gate on the simulated array. For logic operations, the result is ANDed with the
            previous output value (to simulate the effect of logic initialization steps).
        :param type: the type of gate to perform
        :param inputs: the list (python list or numpy array) of input indices
        :param outputs: the list (python list or numpy array) of output indices
        :param direction: the direction of gate to perform
        :param mask: the mask of what rows (columns) to select for an IN_ROW (IN_COLUMN) operation
        """

        # Check no intersection in inputs and outputs
        assert(len(np.intersect1d(inputs, outputs)) == 0)
        # Check inputs are unique
        assert(len(np.unique(inputs)) == len(inputs))
        # Check outputs are unique
        assert(len(np.unique(outputs)) == len(outputs))

        if mask is None:
            mask = np.arange(self.num_rows if direction == GateDirection.IN_ROW else self.num_cols)

        memory_before = (self.memory[outputs[0], mask].copy() if direction == GateDirection.IN_ROW else self.memory[mask, outputs[0]].copy())

        if direction == GateDirection.IN_ROW:

            if type == GateType.NOT:
                self.memory[outputs[0], mask] = np.bitwise_and(np.bitwise_not(self.memory[inputs[0], mask]), self.memory[outputs[0], mask])
            elif type == GateType.MIN3:
                self.memory[outputs[0], mask] = np.bitwise_and(np.bitwise_not(
                    np.bitwise_or(np.bitwise_or(
                        np.bitwise_and(self.memory[inputs[0], mask], self.memory[inputs[1], mask]),
                        np.bitwise_and(self.memory[inputs[0], mask], self.memory[inputs[2], mask])),
                        np.bitwise_and(self.memory[inputs[1], mask], self.memory[inputs[2], mask]))), self.memory[outputs[0], mask])
            elif type == GateType.INIT0:
                self.memory[outputs[0], mask] = False
            elif type == GateType.INIT1:
                self.memory[outputs[0], mask] = True

        elif direction == GateDirection.IN_COLUMN:

            if type == GateType.NOT:
                self.memory[mask, outputs[0]] = np.bitwise_and(np.bitwise_not(self.memory[mask, inputs[0]]), self.memory[mask, outputs[0]])
            elif type == GateType.MIN3:
                self.memory[mask, outputs[0]] = np.bitwise_and(np.bitwise_not(
                    np.bitwise_or(np.bitwise_or(
                        np.bitwise_and(self.memory[mask, inputs[0]], self.memory[mask, inputs[1]]),
                        np.bitwise_and(self.memory[mask, inputs[0]], self.memory[mask, inputs[2]])),
                        np.bitwise_and(self.memory[mask, inputs[1]], self.memory[mask, inputs[2]]))), self.memory[mask, outputs[0]])
            elif type == GateType.INIT0:
                self.memory[mask, outputs[0]] = False
            elif type == GateType.INIT1:
                self.memory[mask, outputs[0]] = True

        self.latency += 1
        memory_after = (self.memory[outputs[0], mask].copy() if direction == GateDirection.IN_ROW else self.memory[mask, outputs[0]].copy())
        self.energy += (memory_before != memory_after).sum()

    def write(self, row, cols, data):
        """
        Performs a write operation at the given row and columns with the given data. Writes binary vector data to cells
            (row, cols[0]), (row, cols[1]), ..., (row, cols[len(data) - 1])
        :param row: the row address
        :param cols: the columns to write to
        :param data: binary vector of data
        """
        self.memory[cols, row] = data
        self.latency += 1
        self.energy += len(cols)
