'''
Test file checking for NE computation

Auther: Hanchen Wang
Date: 2025-06
'''

import nashpy
import numpy as np
import torch

import numpy as np
from scipy.optimize import minimize


def get_nbs_strategy_prob(disagreement_point, feasible_utilities):
    """
    Compute the Nash Bargaining Solution.

    Args:
        disagreement_point: Tuple (d1, d2) of fallback utilities.
        feasible_utilities: List of (u1, u2) achievable through cooperation.

    Returns:
        NBS utilities (u1, u2).
    """
    feasible_utils = np.array(feasible_utilities)

    # Maximize Nash product: (u1 - d1)(u2 - d2)
    def objective(x):
        return -((x[0] - d1) * (x[1] - d2))  # Negative for minimization

    d1, d2 = disagreement_point
    x0 = np.mean(feasible_utils, axis=0)  # Initial guess

    # Constraints: Utilities must be in feasible set
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - feasible_utils[:, 0]},
        {'type': 'ineq', 'fun': lambda x: x[1] - feasible_utils[:, 1]}
    ]

    result = minimize(objective, x0, constraints=constraints)
    return result.x


def get_nash(matrices):
    # Initialize the game
    game = nashpy.Game(matrices[0], matrices[1])

    # Compute Nash equilibria using support enumeration
    ne = []
    for eq in game.support_enumeration():
        ne.append(eq)

    return ne


def get_utilities(nash_equilibrium, payoff_A, payoff_B):
    # Compute expected payoffs for each Nash equilibrium
    result = []
    for eq in nash_equilibrium:
        row_stg, col_stg = eq
        utility_A = row_stg @ payoff_A @ col_stg.T
        utility_B = row_stg @ payoff_B @ col_stg.T
        result.append((utility_A, utility_B))
    return result


# Function to compute joint probabilities
def get_joint_probs(nash_equilibrium):
    result = []
    for eq in nash_equilibrium:
        row_stg, col_stg = eq
        CC = row_stg[0] * col_stg[0]
        CD = row_stg[0] * col_stg[1]
        DC = row_stg[1] * col_stg[0]
        DD = row_stg[1] * col_stg[1]
        result.append([CC, CD, DC, DD])
    return result


def compute_expected_row(p, q, payoffs):
    return (
            p * q * payoffs[0, 0].item() +  # CC
            p * (1 - q) * payoffs[0, 1].item() +  # CD
            (1 - p) * q * payoffs[1, 0].item() +  # DC
            (1 - p) * (1 - q) * payoffs[1, 1].item()  # DD
    )


if __name__ == '__main__':

    ''' PD '''
    A_PD = np.array([[3, 1],
                     [4, 2]])
    B_PD = np.array([[3, 4],
                     [1, 2]])

    ''' Harmony '''
    A_Ha = np.array([[4, 2],
                     [3, 1]])
    B_Ha = np.array([[4, 3],
                     [2, 1]])

    ''' Chicken '''
    A_Ch = np.array([[0, 7],
                     [2, 6]])
    B_Ch = np.array([[0, 2],
                     [7, 6]])

    ''' Battle '''
    A_Ba = np.array([[3, 1],
                     [0, 2]])
    B_Ba = np.array([[2, 1],
                     [0, 3]])

    A_Ba = np.array([[2, 3],
                     [4, 1]])
    B_Ba = np.array([[2, 4],
                     [3, 1]])

    matrices = [A_Ba, B_Ba]

    ne = get_nash(matrices)

    utilities = get_utilities(ne, matrices[0], matrices[1])

    probs = get_joint_probs(ne)

    for i, (eq, utility, prob) in enumerate(zip(ne, utilities, probs), 1):
        row_stg, col_stg = eq
        print(f"Nash Equilibrium {i}: [CC: {prob[0]:.2f}, CD: {prob[1]:.2f}, DC: {prob[2]:.2f}, DD: {prob[3]:.2f}]")
        print(f"\tPlayer A Strategy & Utility: {row_stg} -> {utility[0]}")
        print(f"\tPlayer B Strategy & Utility: {col_stg} -> {utility[1]}")
