#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml
import networkx as nx

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}

def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #

    print(cnot, type(cnot))
    
    dev = qml.device('forest.qvm', device=nx.Graph(graph))

    @qml.qnode(dev)
    def circuit(i,j):
        """Implements the Deutsch Jozsa algorithm."""
        qml.CNOT(wires=[i,j])   

        return 'yes'

    drawer = qml.draw(circuit)
    print(drawer(cnot[0],cnot[1]))
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
