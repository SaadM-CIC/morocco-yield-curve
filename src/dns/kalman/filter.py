"""
src/dns/kalman/filter.py

Filtre de Kalman linéaire gaussien sur l'état centré z_t = beta_t - mu.

État initial : z_0 = 0 (moyenne inconditionnelle du VAR démeané),
P_0 = variance inconditionnelle du VAR, solution de l'équation de
Lyapunov discrète P0 = A P0 A' + Q (VAR stationnaire supposé).
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


@dataclass
class KalmanResult:
    loglik: float
    z_filtres: np.ndarray   # (T, k) états filtrés (centrés, = beta_t - mu)
    P_filtres: np.ndarray   # (T, k, k) covariances filtrées


def filtrer(
    Y_deflate: np.ndarray,
    C: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
) -> KalmanResult:
    """
    Y_deflate : (T, n_tau), observations déjà déflatées (Y - C@mu).
    C : (n_tau, k) matrice d'observation (fixe dans le temps).
    A : (k, k) matrice de transition.
    Q : (k, k) covariance du bruit d'état.
    H : (n_tau, n_tau) covariance du bruit de mesure.
    """
    T, n_tau = Y_deflate.shape
    k = A.shape[0]

    try:
        P0 = solve_discrete_lyapunov(A, Q)
    except Exception:
        P0 = Q.copy()  # repli si la résolution échoue (A quasi non-stationnaire)

    z_pred = np.zeros(k)
    P_pred = P0.copy()

    z_filtres = np.zeros((T, k))
    P_filtres = np.zeros((T, k, k))
    loglik = 0.0

    for t in range(T):
        v_t = Y_deflate[t] - C @ z_pred
        S_t = C @ P_pred @ C.T + H
        try:
            S_inv = np.linalg.inv(S_t)
        except np.linalg.LinAlgError:
            return KalmanResult(loglik=-np.inf, z_filtres=z_filtres, P_filtres=P_filtres)

        sign, logdet = np.linalg.slogdet(S_t)
        if sign <= 0:
            return KalmanResult(loglik=-np.inf, z_filtres=z_filtres, P_filtres=P_filtres)

        loglik += -0.5 * (n_tau * np.log(2 * np.pi) + logdet + v_t @ S_inv @ v_t)

        K_t = P_pred @ C.T @ S_inv
        z_filt = z_pred + K_t @ v_t
        P_filt = (np.eye(k) - K_t @ C) @ P_pred

        z_filtres[t] = z_filt
        P_filtres[t] = P_filt

        z_pred = A @ z_filt
        P_pred = A @ P_filt @ A.T + Q

    return KalmanResult(loglik=loglik, z_filtres=z_filtres, P_filtres=P_filtres)


def lisser(z_filtres: np.ndarray, P_filtres: np.ndarray, A: np.ndarray, Q: np.ndarray):
    """
    Lissage de Kalman (Rauch-Tung-Striebel), à partir des sorties du filtre
    (z_filtres, P_filtres). Nécessaire pour comparer les facteurs de
    l'approche SSM à ceux de l'approche deux étapes (le rapport lisse les
    états SSM avant comparaison, il ne compare pas les états filtrés bruts).
    """
    T, k = z_filtres.shape
    z_lisse = np.zeros_like(z_filtres)
    P_lisse = np.zeros_like(P_filtres)
    z_lisse[-1] = z_filtres[-1]
    P_lisse[-1] = P_filtres[-1]

    for t in range(T - 2, -1, -1):
        P_pred_suivant = A @ P_filtres[t] @ A.T + Q
        J_t = P_filtres[t] @ A.T @ np.linalg.inv(P_pred_suivant)
        z_lisse[t] = z_filtres[t] + J_t @ (z_lisse[t + 1] - A @ z_filtres[t])
        P_lisse[t] = P_filtres[t] + J_t @ (P_lisse[t + 1] - P_pred_suivant) @ J_t.T

    return z_lisse, P_lisse