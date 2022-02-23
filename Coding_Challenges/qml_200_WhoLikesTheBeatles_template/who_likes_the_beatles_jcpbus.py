#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def distance(A, B):
    """Function that returns the distance between two vectors.

    Args:
        - A (list[int]): person's information: [age, minutes spent watching TV].
        - B (list[int]): person's information: [age, minutes spent watching TV].

    Returns:
        - (float): distance between the two feature vectors.
    """

    # QHACK #

    # The Swap test is a method that allows you to calculate |<A|B>|^2 , you could use it to help you.
    # The qml.AmplitudeEmbedding operator could help you too.

    dev = qml.device("default.qubit", wires=3)

    def embed(P, wire):
        qml.RX(2 * np.pi * P[0] / 100, wires=wire)
        qml.RY(2 * np.pi * P[1] / 500, wires=wire)

    @qml.qnode(dev)
    def swap_test(A, B):
        """
        This function calculates P(Qubit 2 = 0), where qubit 2 is the ancillary
        swap test qubit.


        P(Qubit 2 = 0) = 1/2 + 1/2 |<psi_A|psi_B>|^2
        """

        embed(A, wire=0)
        embed(B, wire=1)

        print(A, B)

        qml.Hadamard(wires=2)
        qml.ControlledQubitUnitary(
            qml.SWAP.matrix, control_wires=2, wires=[0, 1]
        )
        qml.Hadamard(wires=2)
        return qml.probs(wires=2)

    def overlap(A, B):
        """
        Function that returns the overlap between two "people states".

        P(Qubit 2 = 0) = 1/2 + 1/2 |<psi_A|psi_B>|^2

        Meaning

        |<psi_A|psi_B>| = sqrt(2 * P(Qubit 2 = 0) - 1)
        """
        p_0 = swap_test(A, B)[0]
        return np.sqrt(2 * p_0 - 1)

    return np.sqrt(2 * (1 - overlap(A, B)))  # distance

    # QHACK #

    return distance(A, B)

    # QHACK #


def predict(dataset, new, k):
    """Function that given a dataset, determines if a new person do like Beatles or not.

    Args:
        - dataset (list): List with the age, minutes that different people watch TV, and if they like Beatles.
        - new (list(int)): Age and TV minutes of the person we want to classify.
        - k (int): number of nearby neighbors to be taken into account.

    Returns:
        - (str): "YES" if they like Beatles, "NO" otherwise.
    """

    # DO NOT MODIFY anything in this code block

    def k_nearest_classes():
        """Function that returns a list of k near neighbors."""
        distances = []
        for data in dataset:
            distances.append(distance(data[0], new))
        nearest = []
        for _ in range(k):
            indx = np.argmin(distances)
            nearest.append(indx)
            distances[indx] += 2

        return [dataset[i][1] for i in nearest]

    output = k_nearest_classes()

    return (
        "YES"
        if len([i for i in output if i == "YES"]) > len(output) / 2
        else "NO",
        float(distance(dataset[0][0], new)),
    )


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    dataset = []
    new = [int(inputs[0]), int(inputs[1])]
    k = int(inputs[2])
    for i in range(3, len(inputs), 3):
        dataset.append(
            [[int(inputs[i + 0]), int(inputs[i + 1])], str(inputs[i + 2])]
        )

    output = predict(dataset, new, k)
    sol = 0 if output[0] == "YES" else 1
    print(f"{sol},{output[1]}")
