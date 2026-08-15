"""
src/dns/ns/lambda_selection.py

Détermination de lambda par régression non linéaire (Phase 3, étape 1).

Approche "concentrée" (variable projection) : le modèle NS est linéaire en
(beta0, beta1, beta2) une fois lambda fixé (cf. calibration.py). On profite
de cette structure : pour un lambda candidat, les betas optimaux sont
obtenus par OLS, et on ne cherche numériquement que le lambda qui minimise
le RMSE résultant (recherche 1D bornée), au lieu d'un Levenberg-Marquardt
classique à 4 paramètres couplés. Optimum mathématiquement équivalent,
plus stable numériquement.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .calibration import CalibrationResult, build_design_matrix, calibrate_ns_ols

SEUIL_CONDITIONNEMENT = 1e8  # au-delà : matrice quasi-singulière, lambda écarté


@dataclass
class LambdaEstimationResult:
    lambda_opt: float
    beta0: float
    beta1: float
    beta2: float
    rmse: float
    success: bool


def _rmse_pour_lambda(lambda_: float, tau: np.ndarray, taux: np.ndarray) -> float:
    if lambda_ <= 0:
        return np.inf
    X = build_design_matrix(tau, lambda_)
    # Garde-fou : quand lambda -> 0, loading_pente -> 1 pour tout tau, la
    # colonne devient quasi-identique à la colonne du beta0 (intercept) ->
    # matrice quasi-singulière -> OLS numériquement instable (coefficients
    # qui explosent pour "faussement" bien fitter). On écarte ces lambda.
    if np.linalg.cond(X) > SEUIL_CONDITIONNEMENT:
        return np.inf
    beta, _, rank, _ = np.linalg.lstsq(X, taux, rcond=None)
    if rank < 3:
        return np.inf
    fitted = X @ beta
    return float(np.sqrt(np.mean((taux - fitted) ** 2)))


def estimate_lambda_nls(
    tau: np.ndarray,
    taux: np.ndarray,
    lambda_bounds: tuple[float, float] = (0.02, 3.0),
    n_grille: int = 500,
) -> LambdaEstimationResult:
    """
    Trouve le lambda qui minimise le RMSE de calibration OLS sur UNE courbe
    (un jour donné).

    Borne basse par défaut = 0.02 : en dessous, la "bosse" du modèle NS
    (maximale autour de tau=1/lambda) se situe au-delà de 50 ans, bien
    au-delà de notre horizon de données (30 ans). beta1/beta2 cessent
    d'être identifiables à partir de la courbure réellement observée, et
    la quasi-colinéarité des loadings permet une sur-paramétrisation
    (coefficients qui explosent pour capter du bruit). Ce n'est pas qu'un
    problème numérique : c'est une zone hors du domaine économiquement
    pertinent pour ce jeu de données.

    La surface RMSE(lambda) du modèle NS n'est pas convexe (minima locaux
    possibles) : on fait donc d'abord une recherche en grille grossière sur
    tout l'intervalle pour localiser la région du minimum global, puis on
    raffine localement par recherche scalaire bornée autour de ce point.
    """
    grille = np.linspace(lambda_bounds[0], lambda_bounds[1], n_grille)
    rmse_grille = np.array([_rmse_pour_lambda(l, tau, taux) for l in grille])
    idx_min = int(np.argmin(rmse_grille))
    lambda_grille = grille[idx_min]

    # Raffinement local autour du meilleur point de grille
    pas = grille[1] - grille[0]
    bornes_locales = (
        max(lambda_bounds[0], lambda_grille - 2 * pas),
        min(lambda_bounds[1], lambda_grille + 2 * pas),
    )
    opt = minimize_scalar(
        _rmse_pour_lambda,
        args=(tau, taux),
        bounds=bornes_locales,
        method="bounded",
    )

    # Garde-fou : si le raffinement local a dérivé vers un pire optimum
    # (ne devrait pas arriver avec un intervalle aussi resserré), on retombe
    # sur le meilleur point de grille.
    lambda_final = opt.x if opt.fun <= rmse_grille[idx_min] else lambda_grille

    calib: CalibrationResult = calibrate_ns_ols(tau, taux, lambda_final)

    return LambdaEstimationResult(
        lambda_opt=float(lambda_final),
        beta0=calib.beta0,
        beta1=calib.beta1,
        beta2=calib.beta2,
        rmse=calib.rmse,
        success=bool(opt.success),
    )