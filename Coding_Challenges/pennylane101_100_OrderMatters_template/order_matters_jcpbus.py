#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


def compare_circuits(angles):
    """Given two angles, compare two circuit outputs that have their order of operations flipped: RX then RY VERSUS RY then RX.

    Args:
        - angles (np.ndarray): Two angles

    Returns:
        - (float): | < \sigma^x >_1 - < \sigma^x >_2 |
    """

    # QHACK #

    n = 1
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(i, theta1, theta2):
        """Implements the Deutsch Jozsa algorithm."""

        # QHACK #
        if i == 1:
            qml.RX(theta1, wires=0)
            qml.RY(theta2, wires=0)
        else:
            qml.RY(theta2, wires=0)
            qml.RX(theta1, wires=0)

        return qml.expval(qml.PauliX(0))  # wires=range(n)

    theta1, theta2 = angles
    sample1 = circuit(1, theta1, theta2)
    sample2 = circuit(2, theta1, theta2)

    return np.abs(sample1 - sample2)
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    angles = np.array(sys.stdin.read().split(","), dtype=float)
    output = compare_circuits(angles)
    print(f"{output:.6f}")
