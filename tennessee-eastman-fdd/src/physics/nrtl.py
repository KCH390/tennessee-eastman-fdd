"""
General (N-component) NRTL activity-coefficient model.

Implements the original Renon & Prausnitz (1968) NRTL equation, the same
form used by the sources cited in vle_params.py:

    tau_ij = dg_ij / (R*T)          (dg_ij in J/mol, T in K)
    G_ij   = exp(-alpha_ij * tau_ij)
    tau_ii = 0, G_ii = 1

    ln(gamma_i) = [sum_j tau_ji * x_j * G_ji] / [sum_k x_k * G_ki]
                + sum_j { x_j * G_ij / [sum_k x_k * G_kj] *
                          ( tau_ij - [sum_m x_m * tau_mj * G_mj] / [sum_k x_k * G_kj] ) }

This module only does the math; it doesn't know anything about methyl
acetate specifically. See vle_params.py for this project's actual
component list, indices, and (partially) sourced dg/alpha matrices, and
test_nrtl.py for validation against known reference behavior (methanol +
water and methyl acetate + methanol, both from a DECHEMA-recommended
textbook table -- see vle_params.py for the exact citation).
"""

import numpy as np

R_GAS = 8.314462618  # J/(mol K)


def tau_matrix(dg: np.ndarray, T: float) -> np.ndarray:
    """dg: (n,n) array of Delta g_ij in J/mol (dg[i,i] ignored). Returns tau_ij(T)."""
    n = dg.shape[0]
    tau = dg / (R_GAS * T)
    np.fill_diagonal(tau, 0.0)
    return tau


def g_matrix(tau: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """G_ij = exp(-alpha_ij * tau_ij), G_ii = 1."""
    n = tau.shape[0]
    G = np.exp(-alpha * tau)
    np.fill_diagonal(G, 1.0)
    return G


def activity_coefficients(x: np.ndarray, dg: np.ndarray, alpha: np.ndarray, T: float) -> np.ndarray:
    """
    x: (n,) mole fractions, must sum to 1.
    dg, alpha: (n,n) NRTL parameter matrices (symmetric alpha, dg[i,j] != dg[j,i] in general).
    Returns gamma: (n,) activity coefficients.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    tau = tau_matrix(dg, T)
    G = g_matrix(tau, alpha)

    ln_gamma = np.zeros(n)
    denom_k = x @ G  # denom_k[j] = sum_k x_k * G_kj, shape (n,)

    for i in range(n):
        term1 = (tau[:, i] * x * G[:, i]).sum() / (x @ G[:, i])

        term2 = 0.0
        for j in range(n):
            denom_j = denom_k[j]
            weighted_tau = (x * tau[:, j] * G[:, j]).sum() / denom_j
            term2 += (x[j] * G[i, j] / denom_j) * (tau[i, j] - weighted_tau)

        ln_gamma[i] = term1 + term2

    return np.exp(ln_gamma)
