import sys
import pennylane as qml
from pennylane import numpy as np

NUM_WIRES = 6


def triple_excitation_matrix(gamma, n_wires=6):
    """The matrix representation of a triple-excitation Givens rotation.

    Args:
        - gamma (float): The angle of rotation

    Returns:
        - (np.ndarray): The matrix representation of a triple-excitation
    """

    assert n_wires % 2 == 0, "Number of wires must be even"

    def single_unit_matrix(i, j):
        """
        Returns matrix with zeros except at index [i,j] where it is 1.
        """
        m = np.zeros((2 ** n_wires, 2 ** n_wires))
        m[i, j] = 1
        return m

    # binary
    x = [0] * int(n_wires / 2) + [1] * int(n_wires / 2)
    y = x[::-1]

    # convert x to base 10 number
    x_dec = int("".join(map(str, x)), 2)
    y_dec = int("".join(map(str, y)), 2)

    m = (
        np.identity(2 ** n_wires)
        - single_unit_matrix(x_dec, x_dec)
        - single_unit_matrix(y_dec, y_dec)
    )

    m += np.cos(gamma / 2) * single_unit_matrix(2 ** n_wires - x_dec, x_dec)
    m += np.sin(gamma / 2) * single_unit_matrix(y_dec, x_dec)

    m += np.cos(gamma / 2) * single_unit_matrix(y_dec, y_dec)
    m += -np.sin(gamma / 2) * single_unit_matrix(x_dec, y_dec)

    return m


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit(angles):
    """Prepares the quantum state in the problem statement and returns qml.probs

    Args:
        - angles (list(float)): The relevant angles in the problem statement in this order:
        [alpha, beta, gamma]

    Returns:
        - (np.tensor): The probability of each computational basis state
    """
    alpha, beta, gamma = angles

    # QHACK #
    qml.PauliX(wires=[2])
    qml.PauliX(wires=[4])
    qml.PauliX(wires=[5])

    qml.SingleExcitation(-alpha, wires=[1, 4])
    qml.DoubleExcitation(beta, wires=[0, 1, 4, 5])
    qml.QubitUnitary(
        triple_excitation_matrix(gamma / 8), wires=[0, 1, 2, 3, 4, 5]
    )
    # QHACK #

    return qml.probs(wires=range(NUM_WIRES))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    probs = circuit(inputs).round(6)
    print(*probs, sep=",")
