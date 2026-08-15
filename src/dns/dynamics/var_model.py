"""
src/dns/dynamics/var_model.py

Phase 5 : modélisation dynamique des facteurs latents par un VAR(1)
(Diebold-Li) :

    beta_t = (I - A) mu + A beta_{t-1} + eta_t,    eta_t ~ (0, Q)

Estimation par OLS (équation par équation, régresseurs identiques pour les
3 équations -> équivalent à une régression multivariée classique).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VARResult:
    A: np.ndarray            # (3,3) matrice de transition
    mu: np.ndarray           # (3,)  moyenne de long terme
    c: np.ndarray            # (3,)  constante = (I-A) @ mu
    Q: np.ndarray            # (3,3) covariance des résidus
    residuals: np.ndarray    # (T-1, 3)
    valeurs_propres_A: np.ndarray  # (3,) module des valeurs propres de A
    stationnaire: bool       # True si toutes les valeurs propres de A ont un module < 1


def estimer_var1(beta_df: pd.DataFrame, cols: tuple = ("beta0", "beta1", "beta2")) -> VARResult:
    """
    Estime un VAR(1) sur la série temporelle des facteurs, supposée déjà
    triée chronologiquement (une ligne par date, sans trou).
    """
    B = beta_df[list(cols)].values  # (T, 3)
    if B.shape[0] < 5:
        raise ValueError("au moins 5 observations sont nécessaires pour un VAR(1) exploitable")

    Y = B[1:]        # beta_t,  t = 2..T
    X = B[:-1]        # beta_{t-1}, t = 1..T-1
    n_obs = Y.shape[0]
    k = X.shape[1]

    X_aug = np.column_stack([np.ones(n_obs), X])  # (n_obs, 1+k)
    coeffs, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)  # (1+k, k)

    c = coeffs[0, :]
    A = coeffs[1:, :].T  # (k,k) : Y_t^T = c^T + beta_{t-1}^T @ A^T

    residuals = Y - X_aug @ coeffs  # (n_obs, k)
    dof = max(n_obs - X_aug.shape[1], 1)
    Q = (residuals.T @ residuals) / dof

    mu = np.linalg.solve(np.eye(k) - A, c)

    valeurs_propres = np.abs(np.linalg.eigvals(A))
    stationnaire = bool(np.all(valeurs_propres < 1.0))

    return VARResult(
        A=A,
        mu=mu,
        c=c,
        Q=Q,
        residuals=residuals,
        valeurs_propres_A=valeurs_propres,
        stationnaire=stationnaire,
    )