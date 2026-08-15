"""
generaliser_zc.py

Applique le bootstrap ZC (coupon-stripping) sur TOUS les jours de la base,
pour construire la série temporelle complète de courbes ZC.

Réutilise la même logique que quatre_methodes_courbe.py (interpolation
linéaire vers une grille annuelle, puis bootstrap récursif), mais sans
génération de graphes (1854 jours -> on ne veut que le CSV final).

Usage :
    python generaliser_zc.py --input data/courbe_bdt_actuariel.csv --output data/zc_timeseries.csv
"""

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("generaliser_zc")

MIN_POINTS_REQUIS = 4  # en dessous, le fit/bootstrap n'a pas de sens

# Maturités pleines telles que définies dans le document méthodologique
# (nombre de jours exact repris de l'interface VBA du rapport).
MATURITES_PLEINES_JOURS = {
    "13_sem": 91,
    "26_sem": 182,
    "52_sem": 364,
    "2_ans": 730,
    "5_ans": 1826,
    "10_ans": 3652,
    "15_ans": 5478,
    "20_ans": 7305,
    "30_ans": 10957,
}
ANNEES_BOOTSTRAP = list(range(1, 31))  # 1 an à 30 ans


def jours_annee_entiere(k: int) -> int:
    """Nombre de jours pour k années entières (base 365.25, cohérent avec
    les valeurs données pour les maturités pleines : 2 ans=730, 5 ans=1826, ...)."""
    return int(np.floor(k * 365.25))


def interp_lineaire(x_connus: np.ndarray, y_connus: np.ndarray, x_cibles: np.ndarray) -> np.ndarray:
    f = interp1d(x_connus, y_connus, kind="linear", bounds_error=False,
                 fill_value=(y_connus[0], y_connus[-1]))
    return f(x_cibles)


def bootstrap_zc_pair(annees: np.ndarray, taux_pair_pct: np.ndarray):
    c = taux_pair_pct / 100
    n = len(annees)
    DF = np.zeros(n)
    cum_DF = 0.0
    for i in range(n):
        if annees[i] == 1:
            DF[i] = 1.0 / (1.0 + c[i])
        else:
            DF[i] = (1.0 - c[i] * cum_DF) / (1.0 + c[i])
        cum_DF += DF[i]
    taux_zc_pct = (DF ** (-1.0 / annees) - 1.0) * 100
    return DF, taux_zc_pct


def process_one_day(day_df: pd.DataFrame, target_date) -> pd.DataFrame:
    jours = day_df["n_jours"].values
    taux = day_df["taux_actuariel"].values

    # --- Étape A : courbe brute -> 9 maturités pleines (interpolation linéaire) ---
    # Inclut les 3 maturités courtes (13/26/52 sem) : déjà zéro-coupon, donc
    # aucun bootstrap sur elles, mais elles passent par la même réduction à un
    # jeu de maturités fixes que le segment long (cf. interface du rapport).
    noms_pleines = list(MATURITES_PLEINES_JOURS.keys())  # 13_sem, 26_sem, 52_sem, 2_ans, ...
    jours_pleines = np.array(list(MATURITES_PLEINES_JOURS.values()), dtype=float)
    taux_pleines = interp_lineaire(jours, taux, jours_pleines)

    court_jours = jours_pleines[:3]   # 13 sem, 26 sem, 52 sem
    court_taux = taux_pleines[:3]     # déjà ZC, telles quelles

    # --- Étape B : maturités pleines -> grille annuelle entière (2e interpolation) ---
    jours_annees = np.array([jours_annee_entiere(k) for k in ANNEES_BOOTSTRAP], dtype=float)
    taux_annees = interp_lineaire(jours_pleines, taux_pleines, jours_annees)

    # --- Étape C : bootstrap récursif sur la grille annuelle ---
    _, taux_zc_annees = bootstrap_zc_pair(np.array(ANNEES_BOOTSTRAP), taux_annees)

    zc_jours = np.concatenate([court_jours, jours_annees])
    zc_taux = np.concatenate([court_taux, taux_zc_annees])
    ordre = np.argsort(zc_jours)
    zc_jours, zc_taux = zc_jours[ordre], zc_taux[ordre]

    return pd.DataFrame({
        "date_courbe": target_date,
        "maturite_jours": zc_jours,
        "maturite_annees": zc_jours / 365.25,
        "taux_zc": zc_taux,
    })


def main():
    parser = argparse.ArgumentParser(description="Bootstrap ZC sur toute la série temporelle")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")
    all_dates = sorted(df["date_courbe"].unique())
    logger.info("Jours à traiter : %d", len(all_dates))

    resultats = []
    n_ok, n_skipped = 0, 0

    for i, d in enumerate(all_dates, start=1):
        day_df = df[df["date_courbe"] == d].drop_duplicates(subset=["n_jours"]).sort_values("n_jours")

        if len(day_df) < MIN_POINTS_REQUIS:
            n_skipped += 1
            continue

        try:
            zc_day = process_one_day(day_df, d)
            resultats.append(zc_day)
            n_ok += 1
        except Exception as e:
            logger.warning("Échec pour %s : %s", d, e)
            n_skipped += 1

        if i % 200 == 0 or i == len(all_dates):
            logger.info("Progression : %d/%d (ok=%d, skip=%d)", i, len(all_dates), n_ok, n_skipped)

    zc_full = pd.concat(resultats, ignore_index=True)
    zc_full.to_csv(args.output, index=False, sep=";")

    logger.info("Terminé. %d jours traités, %d ignorés (< %d points)", n_ok, n_skipped, MIN_POINTS_REQUIS)
    logger.info("Fichier de sortie : %s (%d lignes)", args.output, len(zc_full))


if __name__ == "__main__":
    main()