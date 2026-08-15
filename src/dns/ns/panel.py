"""
src/dns/ns/panel.py

Phase 4 : calibration OLS à lambda FIXÉ sur un panel complet de courbes
mensuelles, pour produire la série temporelle des facteurs latents
(beta0, beta1, beta2) -- l'entrée de la Phase 5 (VAR).
"""

import pandas as pd

from .calibration import calibrate_ns_ols


def calibrer_panel(
    df: pd.DataFrame,
    lambda_: float,
    date_col: str = "date_courbe",
    tau_col: str = "maturite_annees",
    taux_col: str = "taux_zc",
) -> pd.DataFrame:
    """
    Calibre (beta0, beta1, beta2) pour CHAQUE date du panel, à lambda fixé.

    Retourne un DataFrame trié par date avec colonnes :
    date_courbe, beta0, beta1, beta2, rmse.
    """
    lignes = []
    for d, day_df in df.groupby(date_col, sort=True):
        day_df = day_df.sort_values(tau_col)
        res = calibrate_ns_ols(day_df[tau_col].values, day_df[taux_col].values, lambda_)
        lignes.append({
            date_col: d,
            "beta0": res.beta0,
            "beta1": res.beta1,
            "beta2": res.beta2,
            "rmse": res.rmse,
        })
    return pd.DataFrame(lignes).sort_values(date_col).reset_index(drop=True)