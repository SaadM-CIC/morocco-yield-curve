"""
scripts/comparer_lambda.py

Phase 3, étape 3 : compare lambda1 (Diebold-Li original, 0.7308) et lambda2
(moyenne empirique obtenue par estimate_lambda_nls sur le panel complet)
via RMSE global, RMSE par maturité, et distribution des résidus, pour
trancher le lambda final à retenir en Phase 4.

Usage :
    python scripts/comparer_lambda.py --input data/zc_timeseries.csv --lambda2 0.0605
"""

import argparse

import numpy as np
import pandas as pd

from src.dns.ns.calibration import calibrate_ns_ols

LAMBDA1 = 0.7308

SEGMENTS = {
    "court (<1 an)": lambda tau: tau < 1,
    "moyen (1-10 ans)": lambda tau: (tau >= 1) & (tau <= 10),
    "long (>10 ans)": lambda tau: tau > 10,
}


def calibrer_toutes_dates(df: pd.DataFrame, lambda_: float) -> tuple[np.ndarray, np.ndarray, list]:
    """Retourne (matrice de résidus [n_dates x n_maturites], maturités, dates)."""
    dates = sorted(df["date_courbe"].unique())
    residus_par_date = []
    maturites_ref = None
    for d in dates:
        day_df = df[df["date_courbe"] == d].sort_values("maturite_annees")
        tau = day_df["maturite_annees"].values
        taux = day_df["taux_zc"].values
        if maturites_ref is None:
            maturites_ref = tau
        res = calibrate_ns_ols(tau, taux, lambda_)
        residus_par_date.append(res.residuals)
    return np.array(residus_par_date), maturites_ref, dates


def rmse_par_segment(residus: np.ndarray, maturites: np.ndarray) -> dict:
    out = {}
    for nom, mask_fn in SEGMENTS.items():
        mask = mask_fn(maturites)
        out[nom] = float(np.sqrt(np.mean(residus[:, mask] ** 2)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Comparaison lambda1 vs lambda2")
    parser.add_argument("--input", required=True)
    parser.add_argument("--lambda2", type=float, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")

    for nom, lambda_ in [("lambda1 (Diebold-Li)", LAMBDA1), ("lambda2 (estimé)", args.lambda2)]:
        residus, maturites, dates = calibrer_toutes_dates(df, lambda_)
        rmse_global = float(np.sqrt(np.mean(residus ** 2)))
        rmse_seg = rmse_par_segment(residus, maturites)

        print(f"\n=== {nom} = {lambda_:.4f} ===")
        print(f"RMSE global (toutes dates, toutes maturités) : {rmse_global:.5f}")
        for seg, val in rmse_seg.items():
            print(f"  RMSE segment {seg:20s} : {val:.5f}")
        print(f"Résidu moyen (biais) : {residus.mean():.5f}")
        print(f"Résidu max (valeur absolue) : {np.abs(residus).max():.5f}")


if __name__ == "__main__":
    main()