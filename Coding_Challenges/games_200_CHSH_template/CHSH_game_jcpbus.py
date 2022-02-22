#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    a = alpha / norm

    phi = np.arccos(a)

    qml.RY(2 * phi, wires=0)
    qml.CNOT(wires=[0, 1])
    # QHACK #


@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #

    # Alice's measurement
    if x == 0:
        qml.RY(theta_A0, wires=0).inv()
    else:
        qml.RY(theta_A1, wires=0).inv()

    # Bob's measurement
    if y == 0:
        qml.RY(theta_B0, wires=1).inv()
    else:
        qml.RY(theta_B1, wires=1).inv()

    # QHACK #

    return qml.probs(wires=[0, 1])


def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    p = 0

    def probs(x, y):
        return chsh_circuit(
            params[0], params[1], params[2], params[3], x, y, alpha, beta
        )

    # P(succes) = P(X=0)P(Y=0|X=0) * (P(A=0,B=0|X=0,Y=0) + P(A=1,B=1|X=0,Y=0))
    #           + P(X=0)P(Y=1|X=0) * (P(A=0,B=0|X=1,Y=1) + P(A=1,B=1|X=1,Y=1))
    #           + P(X=1)P(Y=0|X=1) * (P(A=0,B=0|X=1,Y=0) + P(A=1,B=1|X=1,Y=0))
    #           + P(X=1)P(Y=1|X=1) * (P(A=1,B=0|X=1,Y=1) + P(A=0,B=1|X=1,Y=1))

    p = 0
    p += 0.5 ** 2 * (probs(0, 0)[0] + probs(0, 0)[3])
    p += 0.5 ** 2 * (probs(0, 1)[0] + probs(0, 1)[3])
    p += 0.5 ** 2 * (probs(1, 0)[0] + probs(1, 0)[3])
    p += 0.5 ** 2 * (probs(1, 1)[1] + probs(1, 1)[2])

    return p
    # QHACK #


def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return -winning_prob(params, alpha, beta)

    # QHACK #

    # Initialize parameters, choose an optimization method and number of steps
    init_params = np.random.random(4) * 2 * np.pi
    opt = qml.NesterovMomentumOptimizer(stepsize=1)
    steps = 100

    # QHACK #

    # set the initial parameter values
    params = init_params
    for i in range(steps):
        # update the circuit parameters
        # QHACK #

        params = opt.step(cost, params)
        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == "__main__":
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")
