"""
src/dns/ns/calibration.py

Calibration du modèle Nelson-Siegel statique à lambda FIXÉ, par régression
linéaire OLS (étape 1 de la procédure en deux étapes de Diebold-Li).

Le modèle est linéaire en (beta0, beta1, beta2) une fois lambda connu :

    R(tau) = beta0 * 1 + beta1 * loading_pente(tau, lambda)
                       + beta2 * loading_courbure(tau, lambda)

On résout donc un simple problème de moindres carrés X @ beta = R.
"""

from dataclasses import dataclass

import numpy as np

from .model import loading_courbure, loading_pente


@dataclass
class CalibrationResult:
    beta0: float
    beta1: float
    beta2: float
    lambda_: float
    residuals: np.ndarray  # taux observés - taux ajustés, par point de la courbe
    rmse: float


def build_design_matrix(tau: np.ndarray, lambda_: float) -> np.ndarray:
    """Matrice X à 3 colonnes [1, loading_pente(tau), loading_courbure(tau)]."""
    tau = np.asarray(tau, dtype=float)
    ones = np.ones_like(tau)
    return np.column_stack([ones, loading_pente(tau, lambda_), loading_courbure(tau, lambda_)])


def calibrate_ns_ols(tau: np.ndarray, taux: np.ndarray, lambda_: float) -> CalibrationResult:
    """
    Calibre (beta0, beta1, beta2) par OLS pour un lambda fixé, sur une seule
    courbe (un jour donné).

    tau  : maturités résiduelles en années, shape (N,), toutes > 0.
    taux : taux observés (zéro-coupon) correspondants, shape (N,).
    """
    tau = np.asarray(tau, dtype=float)
    taux = np.asarray(taux, dtype=float)
    if tau.shape != taux.shape:
        raise ValueError("tau et taux doivent avoir la même forme")
    if tau.shape[0] < 3:
        raise ValueError("au moins 3 points sont nécessaires pour calibrer 3 paramètres")

    X = build_design_matrix(tau, lambda_)
    beta, _, _, _ = np.linalg.lstsq(X, taux, rcond=None)
    fitted = X @ beta
    residuals = taux - fitted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return CalibrationResult(
        beta0=float(beta[0]),
        beta1=float(beta[1]),
        beta2=float(beta[2]),
        lambda_=lambda_,
        residuals=residuals,
        rmse=rmse,
    )