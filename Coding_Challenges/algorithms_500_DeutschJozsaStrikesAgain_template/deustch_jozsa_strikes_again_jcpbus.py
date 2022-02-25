#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #
    wires = range(7)
    dev = qml.device("default.qubit", wires=wires)

    def deutsch_jozsa_basic(f):
        qml.PauliX(wires=2)

        for w in [0, 1, 2]:
            qml.Hadamard(w)

        f(wires=[0, 1, 2])

        for w in [0, 1]:
            qml.Hadamard(w)
            qml.PauliX(w)

        qml.Toffoli(wires=[0, 1, 3])

        for w in [0, 1]:
            qml.PauliX(w)

    U = qml.transforms.get_unitary_matrix(deutsch_jozsa_basic)

    def binary_string(x):
        # converts a number to a binary string
        x_bin = bin(x)[2:]

        # pad with zeros to length 2
        x_bin = x_bin.zfill(2)

    # global Deutsch Jozsa
    @qml.qnode(dev)
    def deutsch_jozsa_new(fs):
        qml.PauliX(wires=5)

        for w in [3, 4, 5]:
            qml.Hadamard(w)

        for i, f in enumerate(fs):
            qml.ControlledQubitUnitary(
                U(f),
                wires=[0, 1, 2, 5],
                control_wires=[3, 4],
                control_values=binary_string(i),
            )

        for w in [3, 4]:
            qml.Hadamard(w)

        return qml.probs(wires=[3, 4])
        # return qml.sample(wires=[3, 4])

    sample = deutsch_jozsa_new(fs)

    print(sample)

    # if sum(sample) == 0:
    #     return "4 same"
    # else:
    #     return "2 and 2"
    # # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
