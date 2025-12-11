'''
Test file for manually checking constructing of reduced prior in BMR

Auther: Hanchen Wang
Date: 2025-09
'''

import numpy as np
import torch
from scipy.special import gammaln
import torch.nn.functional as F

EPSILON = torch.finfo().eps


def make_reduced_prior(
        a_prior: torch.Tensor,
        method: str = "epsilon",
        threshold: float = 0.1,
        strength: float = 8.0,
        preserve_total: bool = False,
) -> torch.Tensor:
    """
    Construct a reduced Dirichlet prior for a single conditional (1D Dirichlet vector).

    Args:
        a_prior (torch.Tensor): 1D tensor of posterior/prior concentration parameters (shape: K).
        method (str): Reduction method: "epsilon" (hard prune) or "softmax" (smooth sharpening).
        threshold (float): For "epsilon" method: quantile cutoff in [0,1] to prune small probs.
        strength (float): For "softmax" method: sharpening factor (STRENGTH > 1 -> sparser).
        preserve_total (bool): If True, rescale retained entries to preserve the original total
                               concentration (NOT Friston default). Default: False.

    Returns:
        a_red (torch.Tensor): 1D tensor of reduced Dirichlet concentrations (clamped >= EPSILON).
    Notes:
        - This function assumes `a_post_full` is 1D (one conditional row).
        - By default both methods produce a reduced total concentration (Friston-style).
          Set preserve_total=True to keep the original sum (Heuristic behaviour).
    """
    # Ensure 1D
    if a_prior.ndim != 1:
        raise ValueError("a_post_full must be a 1D tensor representing one Dirichlet vector.")

    # Numerical safety
    a = a_prior.clone().detach().float().clamp_min(EPSILON)
    total = a.sum()

    # Normalize to probabilities (posterior -> p_post)
    p_prior = a / total.clamp_min(EPSILON)

    if method == "epsilon":
        # Hard-prune entries below quantile cutoff -> set to EPSILON
        print("p_prior", p_prior)
        cutoff = torch.quantile(p_prior, threshold)
        print("cutoff", cutoff)
        mask = p_prior < cutoff
        print("mask", mask)

        a_red = a.clone()
        a_red[mask] = EPSILON

        if preserve_total:
            # Rescale retained entries to preserve the original total concentration
            retained_mask = ~mask
            retained_sum = a_red[retained_mask].sum()
            if retained_sum > 0:
                a_red[retained_mask] = a_red[retained_mask] * (
                        total - EPSILON * mask.sum()
                ) / retained_sum

    elif method == "softmax":
        # Smooth sharpening in log-space
        p_red = F.softmax(strength * torch.log(p_prior + EPSILON), dim=-1)

        # Friston-style: reduce concentration by multiplying with original concentrations,
        # so total concentration typically decreases when sharpening.
        a_red = (p_red * a).clone()

        if preserve_total:
            # If user requests preserve_total, rescale a_red to sum to original total
            sum_a_red = a_red.sum()
            if sum_a_red > 0:
                a_red = a_red * (total / sum_a_red)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'epsilon' or 'softmax'.")

    return a_red.clamp_min(EPSILON)


a_prior = torch.tensor([0.1, 0.2, 0.5, 1])
print("a_prior", a_prior)
a_red = make_reduced_prior(
    a_prior,
    method="epsilon",  # "epsilon" or "softmax"
    threshold=0.2,  # only used for "epsilon"
    strength=None,  # only used for "softmax"
    preserve_total=False  # Friston-style: do NOT preserve total by default
)
print("a_red", a_red)
