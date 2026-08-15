"""
src/dns/kalman/mle.py

Phase 7 : estimation MLE conjointe (A, mu, Q, H) du modèle espace-état
Diebold-Li, par maximisation de la log-vraisemblance du filtre de Kalman,
initialisée à partir des résultats VAR (Phase 5).

Paramétrisation (pour respecter les contraintes de positivité) :
  - A, mu : libres (pas de contrainte).
  - Q = L L^T (décomposition de Cholesky, L triangulaire inférieure 3x3,
    6 paramètres libres) -> garantit Q semi-définie positive.
  - H = sigma^2 * I : UNE SEULE variance de mesure, partagée par toutes
    les maturités (hypothèse de modélisation par parcimonie -- le rapport
    ne précise pas la structure de H ; 33 variances séparées seraient
    peu identifiables avec 194 observations). Paramétrée via log(sigma^2)
    pour garantir la positivité.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .filter import filtrer
from .state_space import deflater_observations

K = 3  # dimension de l'etat (beta0, beta1, beta2)


@dataclass
class MLEResult:
    A: np.ndarray
    mu: np.ndarray
    Q: np.ndarray
    sigma2_H: float
    loglik: float
    success: bool
    n_iter: int


def _empaqueter(A: np.ndarray, mu: np.ndarray, Q: np.ndarray, sigma2_H: float) -> np.ndarray:
    L = np.linalg.cholesky(Q)
    L_vals = [L[0, 0], L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[2, 2]]
    return np.concatenate([A.flatten(), mu, L_vals, [np.log(sigma2_H)]])


def _depaqueter(theta: np.ndarray):
    idx = 0
    A = theta[idx:idx + K * K].reshape(K, K)
    idx += K * K
    mu = theta[idx:idx + K]
    idx += K
    L_vals = theta[idx:idx + 6]
    idx += 6
    L = np.array([
        [L_vals[0], 0.0, 0.0],
        [L_vals[1], L_vals[2], 0.0],
        [L_vals[3], L_vals[4], L_vals[5]],
    ])
    Q = L @ L.T
    sigma2_H = float(np.exp(theta[idx]))
    return A, mu, Q, sigma2_H


def _neg_loglik(theta: np.ndarray, Y: np.ndarray, C: np.ndarray) -> float:
    A, mu, Q, sigma2_H = _depaqueter(theta)
    H = np.eye(C.shape[0]) * sigma2_H
    Yd = deflater_observations(Y, C, mu)
    res = filtrer(Yd, C, A, Q, H)
    if not np.isfinite(res.loglik):
        return 1e10
    return -res.loglik


def estimer_mle(
    Y: np.ndarray,
    C: np.ndarray,
    A_init: np.ndarray,
    mu_init: np.ndarray,
    Q_init: np.ndarray,
    sigma2_H_init: float,
    maxiter: int = 500,
) -> MLEResult:
    """
    Y : (T, n_tau) courbes observées (non déflatées -- la déflation dépend
        de mu, ré-appliquée à chaque évaluation puisque mu est estimé).
    C : (n_tau, 3) matrice d'observation (fixe, lambda déjà incorporé).
    *_init : point de départ, typiquement issu du VAR (Phase 5) et du
        RMSE moyen de la calibration OLS (Phase 4) pour sigma2_H_init.
    """
    theta0 = _empaqueter(A_init, mu_init, Q_init, sigma2_H_init)

    resultat = minimize(
        _neg_loglik,
        theta0,
        args=(Y, C),
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    A_hat, mu_hat, Q_hat, sigma2_hat = _depaqueter(resultat.x)

    return MLEResult(
        A=A_hat,
        mu=mu_hat,
        Q=Q_hat,
        sigma2_H=sigma2_hat,
        loglik=float(-resultat.fun),
        success=bool(resultat.success),
        n_iter=int(resultat.nit),
    )