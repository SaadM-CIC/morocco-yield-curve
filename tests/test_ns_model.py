import numpy as np
import pytest

from src.dns.ns.model import loading_courbure, loading_pente, taux_ns


def test_loading_pente_tend_vers_zero_grand_tau():
    # x = lambda*tau grand -> (1-e^-x)/x -> 1/x (terme e^-x negligeable)
    val = loading_pente(tau=1e6, lambda_=1.0)
    assert val == pytest.approx(1e-6, abs=1e-7)


def test_loading_pente_tend_vers_un_petit_tau():
    # limite theorique en tau->0 : 1
    val = loading_pente(tau=1e-6, lambda_=0.06)
    assert val == pytest.approx(1.0, abs=1e-4)


def test_loading_courbure_tend_vers_zero_petit_tau():
    # limite theorique en tau->0 : 0
    val = loading_courbure(tau=1e-6, lambda_=0.06)
    assert val == pytest.approx(0.0, abs=1e-4)


def test_loading_courbure_tend_vers_zero_grand_tau():
    # x grand -> (1-e^-x)/x -> 1/x et e^-x -> 0 -> loading -> 1/x
    val = loading_courbure(tau=1e6, lambda_=1.0)
    assert val == pytest.approx(1e-6, abs=1e-7)


def test_taux_ns_proche_beta0_plus_beta1_petit_tau():
    beta0, beta1, beta2, lambda_ = 5.0, -2.0, 1.0, 0.06
    val = taux_ns(1e-6, beta0, beta1, beta2, lambda_)
    assert val == pytest.approx(beta0 + beta1, abs=1e-3)


def test_taux_ns_proche_beta0_grand_tau():
    beta0, beta1, beta2, lambda_ = 5.0, -2.0, 1.0, 0.06
    val = taux_ns(1e6, beta0, beta1, beta2, lambda_)
    assert val == pytest.approx(beta0, abs=1e-3)


def test_taux_ns_vectorise():
    tau = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
    val = taux_ns(tau, beta0=5.0, beta1=-2.0, beta2=1.0, lambda_=0.0614)
    assert val.shape == tau.shape
    assert np.all(np.isfinite(val))


def test_loading_pente_rejette_tau_zero():
    with pytest.raises(ValueError):
        loading_pente(0.0, 0.06)


def test_loading_courbure_rejette_tau_negatif():
    with pytest.raises(ValueError):
        loading_courbure(-1.0, 0.06)