"""
src/dns/kalman/state_space.py

Construction des matrices du modèle espace-état (SSM) Diebold-Li.

État (centré) : z_t = beta_t - mu,  transition z_t = A z_{t-1} + eta_t, eta_t ~ (0,Q)
Observation    : Y_t = C beta_t + eps_t = C mu + C z_t + eps_t, eps_t ~ (0,H)
Déflation      : Y'_t = Y_t - C mu = C z_t + eps_t

C = matrice de loadings NS [1, loading_pente(tau,lambda), loading_courbure(tau,lambda)],
IDENTIQUE à la matrice de design de la calibration OLS (Phase 2) -- réutilisée
telle quelle, la grille de maturités tau étant fixe (33 points) sur tout
l'historique.
"""

import numpy as np

from ..ns.calibration import build_design_matrix


def matrice_observation(tau: np.ndarray, lambda_: float) -> np.ndarray:
    """C : (n_tau, 3), identique à la matrice de design OLS de la Phase 2."""
    return build_design_matrix(tau, lambda_)


def deflater_observations(Y: np.ndarray, C: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Y' = Y - C @ mu, appliqué ligne par ligne. Y : (T, n_tau)."""
    return Y - (C @ mu)[None, :]