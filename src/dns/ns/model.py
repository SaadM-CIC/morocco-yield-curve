"""
src/dns/ns/model.py

Modèle de Nelson-Siegel statique (Diebold-Li) :

    R(tau) = beta0 + beta1 * loading_pente(tau, lambda)
                    + beta2 * loading_courbure(tau, lambda)

Convention :
    - tau : maturité résiduelle en ANNÉES (décimal), toujours > 0 dans ce
      pipeline (grille ZC construite en Phase 1 : maturité minimale = 91
      jours ≈ 0.249 an). Aucune évaluation à tau=0 n'est nécessaire ; la
      forme limite (0/0) n'est donc pas gérée ici, par choix explicite.
    - lambda_ : paramètre de décroissance exponentielle, en 1/année, > 0.
    - beta0, beta1, beta2 : facteurs latents (niveau, pente, courbure).
"""

import numpy as np


def loading_pente(tau, lambda_):
    """
    Facteur de charge associé à beta1 (pente) :

        (1 - exp(-lambda*tau)) / (lambda*tau)

    Valide uniquement pour tau > 0 (voir docstring du module).
    """
    tau = np.asarray(tau, dtype=float)
    if np.any(tau <= 0):
        raise ValueError("loading_pente: tau doit être strictement positif")
    x = lambda_ * tau
    return (1.0 - np.exp(-x)) / x


def loading_courbure(tau, lambda_):
    """
    Facteur de charge associé à beta2 (courbure) :

        (1 - exp(-lambda*tau)) / (lambda*tau) - exp(-lambda*tau)

    Valide uniquement pour tau > 0 (voir docstring du module).
    """
    tau = np.asarray(tau, dtype=float)
    if np.any(tau <= 0):
        raise ValueError("loading_courbure: tau doit être strictement positif")
    x = lambda_ * tau
    return (1.0 - np.exp(-x)) / x - np.exp(-x)


def taux_ns(tau, beta0, beta1, beta2, lambda_):
    """
    Taux Nelson-Siegel statique R(tau), combinaison des trois facteurs.
    """
    return (
        beta0
        + beta1 * loading_pente(tau, lambda_)
        + beta2 * loading_courbure(tau, lambda_)
    )