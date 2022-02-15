import sys
import pennylane as qml
from pennylane import numpy as np

def deutsch_jozsa(oracle):
    """This function will determine whether an oracle defined by a function f is constant or balanced.

    Args:
        - oracle (function): Encoding of the f function as a quantum gate. The first two qubits refer to the input and the third to the output.

    Returns:
        - (str): "constant" or "balanced"
    """
    n = 3
    dev = qml.device("default.qubit", wires=n, shots=1)

    @qml.qnode(dev)
    def circuit():
        """Implements the Deutsch Jozsa algorithm."""

        # QHACK #
        qml.PauliX(n-1)

        for i in range(n):
            qml.Hadamard(i)

        oracle()  # DO NOT MODIFY this line

        # QHACK #
        for i in range(n-1):
            qml.Hadamard(i)      

        return qml.sample(wires=range(n-1))

    sample = circuit()

    return 'constant' if np.sum(sample) == 0 else 'balanced'


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        for i in numbers:
            qml.CNOT(wires=[i, 2])

    output = deutsch_jozsa(oracle)
    print(output)