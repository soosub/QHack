#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


def qfunc_adder(m, wires):
    """Quantum function capable of adding m units to a basic state given as input.

    Args:
        - m (int): units to add.
        - wires (list(int)): list of wires in which the function will be executed on.
    """

    qml.QFT(wires=wires)

    # QHACK #
    N = len(wires)
    m_binary = [int(x) for x in np.binary_repr(m)]  # binary representation of m
    m_binary = [0] * (N - len(m_binary)) + m_binary  # pad with zeros

    print("m_binary: ", m_binary)

    for i, x in enumerate(m_binary):
        phase = (np.pi) * (m / 2 ** (i))
        print(i, x, phase / (2 * np.pi))
        qml.RZ(phase, wires=0)

    # QHACK #
    qml.QFT(wires=wires).inv()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    m = int(inputs[0])
    n_wires = int(inputs[1])
    wires = range(n_wires)

    dev = qml.device("default.qubit", wires=wires, shots=1)

    @qml.qnode(dev)
    def test_circuit():
        # Input:  |2^{N-1}>
        qml.PauliX(wires=0)

        qfunc_adder(m, wires)
        return qml.sample()

    output = test_circuit()
    print(*output, sep=",")
