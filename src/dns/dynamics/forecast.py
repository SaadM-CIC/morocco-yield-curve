"""
src/dns/dynamics/forecast.py

Phase 6 : prévision des facteurs latents à horizon h, et reconstruction de
la courbe des taux future à partir des betas prévus.

Formule du rapport :
    E_t(beta_{t+h}) = (sum_{i=0}^{h-1} A^i) (I-A) mu + A^h beta_t

Implémentée ici sous sa forme fermée équivalente (plus stable
numériquement, évite de sommer explicitement une série de puissances de
A) :
    E_t(beta_{t+h}) = mu + A^h (beta_t - mu)

Équivalence : sum_{i=0}^{h-1} A^i (I-A) = I - A^h (série géométrique
matricielle), donc (sum A^i)(I-A)mu + A^h beta_t
                  = (I - A^h) mu + A^h beta_t
                  = mu + A^h (beta_t - mu).
Vérifié numériquement dans les tests.
"""

import numpy as np
import pandas as pd

from ..ns.model import taux_ns


def prevoir_beta(beta_t: np.ndarray, A: np.ndarray, mu: np.ndarray, h: int) -> np.ndarray:
    """
    Prévision ponctuelle des facteurs à l'horizon h (h >= 1), forme fermée
    E_t(beta_{t+h}) = mu + A^h (beta_t - mu).
    """
    if h < 1:
        raise ValueError("h doit être >= 1")
    A_h = np.linalg.matrix_power(A, h)
    return mu + A_h @ (beta_t - mu)


def prevoir_trajectoire(beta_t: np.ndarray, A: np.ndarray, mu: np.ndarray, h_max: int) -> pd.DataFrame:
    """
    Prévision pour tous les horizons 1..h_max. Retourne un DataFrame avec
    une ligne par horizon (colonnes : horizon, beta0, beta1, beta2).
    """
    lignes = []
    for h in range(1, h_max + 1):
        beta_h = prevoir_beta(beta_t, A, mu, h)
        lignes.append({"horizon": h, "beta0": beta_h[0], "beta1": beta_h[1], "beta2": beta_h[2]})
    return pd.DataFrame(lignes)


def reconstruire_courbe(beta_forecast: np.ndarray, tau: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Reconstruit la courbe des taux prévue à partir des betas prévus et
    d'une grille de maturités tau (années, > 0).
    """
    beta0, beta1, beta2 = beta_forecast
    return taux_ns(tau, beta0, beta1, beta2, lambda_)