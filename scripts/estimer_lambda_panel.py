"""
scripts/estimer_lambda_panel.py

Phase 3, étape 2 : applique estimate_lambda_nls() sur toutes les courbes
mensuelles, calcule lambda2 = moyenne des lambda estimés, et compare aux
deux valeurs candidates du rapport (lambda1 = 0.7308, lambda2 = moyenne
empirique) via le RMSE global de calibration OLS.

Usage :
    python scripts/estimer_lambda_panel.py --input data/zc_timeseries.csv
"""

import argparse

import numpy as np
import pandas as pd

from src.dns.ns.calibration import calibrate_ns_ols
from src.dns.ns.lambda_selection import estimate_lambda_nls

LAMBDA1_DIEBOLD_LI = 0.7308


def estimer_lambda_toutes_dates(df: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for d, day_df in df.groupby("date_courbe", sort=True):
        day_df = day_df.sort_values("maturite_annees")
        tau = day_df["maturite_annees"].values
        taux = day_df["taux_zc"].values
        res = estimate_lambda_nls(tau, taux)
        lignes.append({
            "date_courbe": d,
            "lambda_opt": res.lambda_opt,
            "rmse": res.rmse,
            "success": res.success,
            "sur_borne_basse": np.isclose(res.lambda_opt, 0.02, atol=1e-6),
            "sur_borne_haute": np.isclose(res.lambda_opt, 3.0, atol=1e-6),
        })
    return pd.DataFrame(lignes)


def rmse_global_pour_lambda_fixe(df: pd.DataFrame, lambda_: float) -> float:
    """RMSE moyen (sur toutes les dates) de la calibration OLS à lambda fixé."""
    rmses = []
    for _, day_df in df.groupby("date_courbe", sort=True):
        day_df = day_df.sort_values("maturite_annees")
        res = calibrate_ns_ols(day_df["maturite_annees"].values, day_df["taux_zc"].values, lambda_)
        rmses.append(res.rmse)
    return float(np.mean(rmses))


def main():
    parser = argparse.ArgumentParser(description="Estimation NLS de lambda sur tout le panel")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")

    print(f"Dates à traiter : {df['date_courbe'].nunique()}")
    res_df = estimer_lambda_toutes_dates(df)

    n_borne_basse = res_df["sur_borne_basse"].sum()
    n_borne_haute = res_df["sur_borne_haute"].sum()
    n_echec = (~res_df["success"]).sum()

    print(f"\nDiagnostics estimation par date :")
    print(f"  Dates sur la borne basse (0.02)  : {n_borne_basse} / {len(res_df)}")
    print(f"  Dates sur la borne haute (3.0)   : {n_borne_haute} / {len(res_df)}")
    print(f"  Échecs de convergence            : {n_echec} / {len(res_df)}")
    print(f"\nDistribution de lambda_opt :")
    print(res_df["lambda_opt"].describe())

    lambda2 = float(res_df["lambda_opt"].mean())
    lambda2_median = float(res_df["lambda_opt"].median())
    print(f"\nlambda2 (moyenne des lambda estimés)   : {lambda2:.4f}")
    print(f"lambda2 (médiane, moins sensible aux bornes) : {lambda2_median:.4f}")

    print("\nComparaison RMSE global (calibration OLS à lambda fixé, moyenné sur toutes les dates) :")
    rmse_lambda1 = rmse_global_pour_lambda_fixe(df, LAMBDA1_DIEBOLD_LI)
    rmse_lambda2 = rmse_global_pour_lambda_fixe(df, lambda2)
    rmse_lambda2_median = rmse_global_pour_lambda_fixe(df, lambda2_median)
    print(f"  lambda1 = {LAMBDA1_DIEBOLD_LI:.4f} (Diebold-Li original) -> RMSE moyen = {rmse_lambda1:.5f}")
    print(f"  lambda2 = {lambda2:.4f} (moyenne empirique)             -> RMSE moyen = {rmse_lambda2:.5f}")
    print(f"  lambda2 = {lambda2_median:.4f} (médiane empirique)      -> RMSE moyen = {rmse_lambda2_median:.5f}")

    res_df.to_csv("lambda_par_date.csv", index=False, sep=";")
    print("\nDétail par date sauvegardé : lambda_par_date.csv")


if __name__ == "__main__":
    main()