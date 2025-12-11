'''
Build tit-for-tat

Auther: Hanchen Wang
Date: 2025-10
'''


import torch
import torch.nn.functional as F

# Set numerical stability constant
EPSILON = torch.finfo().eps


def calculate_efe(preference_matrix, q_opponent_prev, gamma=1.0):
    """
    Calculate Expected Free Energy (EFE) for tit-for-tat strategy
    with probabilistic beliefs about opponent's previous action.

    Args:
        preference_matrix: 2x2 tensor of preference values
        q_opponent_prev: Belief about opponent's previous action [P(coop), P(defect)]
        gamma: precision parameter
    """
    # Calculate expected log preferences
    # E[log p*(o_i, o_j)] = ∑_{o_j} q(o_j) log p*(o_i, o_j)
    expected_log_pref = torch.einsum('j,ij->i',
                                     q_opponent_prev,
                                     torch.log(preference_matrix + EPSILON)
                                     )

    # G[u] = -E[log p*(o_i, o_j)]
    EFE = -expected_log_pref

    # Convert EFE to action probabilities using softmax
    # q(u)=σ(−γG)
    q_u = F.softmax(-gamma * EFE, dim=0)

    return EFE, q_u


# Define preference matrices with different values but same ordinal relationships
matrices = {
    torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]]),

    torch.tensor([[0.99, 0.01],
                  [0.01, 0.99]]),

    torch.tensor([[0.51, 0.49],
                  [0.49, 0.51]])
}

# Test cases: opponent's last action
cases = [
    ("Certain cooperation", torch.tensor([1.0, 0.0])),
    ("Certain defection", torch.tensor([0.0, 1.0])),
    ("Half", torch.tensor([0.5, 0.5]))
]

# Calculate and print results
for matrix in matrices:
    print(f"\nPreference Matrix:")
    print(matrix.numpy())

    for case_name, opponent_action in cases:
        EFE, q_u = calculate_efe(matrix, opponent_action)
        print(f"\n  {case_name}:")
        print(f"  EFE: {EFE.detach().numpy().round(2)}")
        print(f"  Action probabilities: {q_u.detach().numpy().round(4)}")
        print(f"  Preferred action: {'Cooperate' if q_u[0] > q_u[1] else 'Defect'}")
