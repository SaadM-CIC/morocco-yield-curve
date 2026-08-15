import numpy as np
import pytest

from src.dns.ns.lambda_selection import estimate_lambda_nls
from src.dns.ns.model import taux_ns


def test_recupere_lambda_exact_sans_bruit():
    lambda_vrai = 0.0614
    beta0_vrai, beta1_vrai, beta2_vrai = 4.5, -1.8, 0.9
    tau_court = np.array([91, 182, 364]) / 365.25
    tau_long = np.arange(1, 31)
    tau = np.concatenate([tau_court, tau_long])
    taux = taux_ns(tau, beta0_vrai, beta1_vrai, beta2_vrai, lambda_vrai)

    res = estimate_lambda_nls(tau, taux)

    assert res.success
    assert res.lambda_opt == pytest.approx(lambda_vrai, abs=1e-3)
    assert res.beta0 == pytest.approx(beta0_vrai, abs=1e-3)
    assert res.rmse == pytest.approx(0.0, abs=1e-4)


def test_recupere_lambda_diebold_li_original():
    # Deuxieme valeur de lambda pour verifier qu'on ne retombe pas toujours
    # sur la meme solution par coincidence
    lambda_vrai = 0.7308
    tau_court = np.array([91, 182, 364]) / 365.25
    tau_long = np.arange(1, 31)
    tau = np.concatenate([tau_court, tau_long])
    taux = taux_ns(tau, 5.0, -2.0, 1.0, lambda_vrai)

    res = estimate_lambda_nls(tau, taux)

    assert res.lambda_opt == pytest.approx(lambda_vrai, abs=1e-3)


def test_rmse_faible_avec_leger_bruit():
    lambda_vrai = 0.0614
    tau_court = np.array([91, 182, 364]) / 365.25
    tau_long = np.arange(1, 31)
    tau = np.concatenate([tau_court, tau_long])
    taux_propre = taux_ns(tau, 4.5, -1.8, 0.9, lambda_vrai)
    rng = np.random.default_rng(0)
    taux_bruite = taux_propre + rng.normal(scale=0.01, size=tau.shape)

    res = estimate_lambda_nls(tau, taux_bruite)

    assert res.success
    assert res.rmse < 0.05
    assert res.lambda_opt > 0