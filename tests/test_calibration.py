import numpy as np
import pytest

from src.dns.ns.calibration import build_design_matrix, calibrate_ns_ols
from src.dns.ns.model import taux_ns


def test_recupere_exactement_les_betas_sans_bruit():
    lambda_ = 0.0614
    beta0_vrai, beta1_vrai, beta2_vrai = 5.0, -2.0, 1.5
    tau = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    taux = taux_ns(tau, beta0_vrai, beta1_vrai, beta2_vrai, lambda_)

    res = calibrate_ns_ols(tau, taux, lambda_)

    assert res.beta0 == pytest.approx(beta0_vrai, abs=1e-8)
    assert res.beta1 == pytest.approx(beta1_vrai, abs=1e-8)
    assert res.beta2 == pytest.approx(beta2_vrai, abs=1e-8)
    assert res.rmse == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(res.residuals, 0.0, atol=1e-8)


def test_grille_33_points_realiste_sans_bruit():
    # Grille identique à celle produite par le bootstrap de la Phase 1
    # (3 courtes + 30 années entières)
    lambda_ = 0.0614
    tau_court = np.array([91, 182, 364]) / 365.25
    tau_long = np.arange(1, 31)
    tau = np.concatenate([tau_court, tau_long])
    beta0_vrai, beta1_vrai, beta2_vrai = 4.5, -1.8, 0.9
    taux = taux_ns(tau, beta0_vrai, beta1_vrai, beta2_vrai, lambda_)

    res = calibrate_ns_ols(tau, taux, lambda_)

    assert res.beta0 == pytest.approx(beta0_vrai, abs=1e-6)
    assert res.beta1 == pytest.approx(beta1_vrai, abs=1e-6)
    assert res.beta2 == pytest.approx(beta2_vrai, abs=1e-6)


def test_rmse_positif_avec_bruit():
    lambda_ = 0.0614
    tau = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    taux_propre = taux_ns(tau, 5.0, -2.0, 1.5, lambda_)
    rng = np.random.default_rng(42)
    bruit = rng.normal(scale=0.05, size=tau.shape)
    taux_bruite = taux_propre + bruit

    res = calibrate_ns_ols(tau, taux_bruite, lambda_)

    assert res.rmse > 0.0
    # Le fit doit rester proche des vraies valeurs malgré le bruit
    assert res.beta0 == pytest.approx(5.0, abs=0.6)


def test_design_matrix_colonnes():
    tau = np.array([1.0, 2.0, 5.0])
    X = build_design_matrix(tau, lambda_=0.06)
    assert X.shape == (3, 3)
    assert np.allclose(X[:, 0], 1.0)  # colonne du beta0


def test_rejette_formes_incompatibles():
    with pytest.raises(ValueError):
        calibrate_ns_ols(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), lambda_=0.06)


def test_rejette_trop_peu_de_points():
    with pytest.raises(ValueError):
        calibrate_ns_ols(np.array([1.0, 2.0]), np.array([3.0, 4.0]), lambda_=0.06)