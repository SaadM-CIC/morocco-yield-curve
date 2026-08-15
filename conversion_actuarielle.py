"""
conversion_actuarielle.py

Convertit les taux monétaires (maturité < 365 jours, convention actuelle/360,
taux simple) en taux actuariels équivalents (base annuelle composée), pour
avoir une base homogène sur toute la courbe avant le bootstrap ZC.

Formule :
    1 + r_m * n/360 = (1 + r_a)^(n/365)
    => r_a = (1 + r_m * n/360)^(365/n) - 1

Pour maturité >= 365 jours : taux déjà actuariel, inchangé.

Usage :
    python conversion_actuarielle.py --input data/clean/courbe_bdt_clean.csv \
        --output data/zc/courbe_bdt_actuariel.csv
"""

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("conversion")


def taux_monetaire_vers_actuariel(r_m: float, n_jours: float) -> float:
    """
    r_m : taux monétaire simple, en fraction décimale (ex: 0.02 pour 2%)
    n_jours : nombre de jours jusqu'à l'échéance
    Retourne le taux actuariel équivalent, en fraction décimale.
    """
    facteur_capitalisation = 1 + r_m * n_jours / 360
    return facteur_capitalisation ** (365 / n_jours) - 1


def convert(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_courbe"] = pd.to_datetime(df["date_courbe"])
    df["maturite_dt"] = pd.to_datetime(df["maturite"])
    df["n_jours"] = (df["maturite_dt"] - df["date_courbe"]).dt.days

    is_monetaire = df["n_jours"] <= 365
    n_monetaire = int(is_monetaire.sum())
    logger.info("Points en taux monétaire (< 365j) à convertir : %d / %d", n_monetaire, len(df))

    # taux est stocké en % dans le CSV (ex: 2.332 pour 2.332%) -> fraction décimale pour la formule
    r_m = df.loc[is_monetaire, "taux"] / 100
    n = df.loc[is_monetaire, "n_jours"]
    r_a = taux_monetaire_vers_actuariel(r_m, n)

    df["taux_actuariel"] = df["taux"]  # par défaut : déjà actuariel (>= 365j)
    df.loc[is_monetaire, "taux_actuariel"] = r_a.values * 100  # retour en %

    # Sanity check : l'écart doit rester faible (quelques points de base à quelques
    # dizaines de bps pour les toutes petites maturités), sinon quelque chose cloche
    ecart = (df.loc[is_monetaire, "taux_actuariel"] - df.loc[is_monetaire, "taux"]).abs()
    if len(ecart) > 0:
        logger.info("Écart taux monétaire -> actuariel : moyenne=%.4f pts, max=%.4f pts",
                    ecart.mean(), ecart.max())

    df = df.drop(columns=["maturite_dt"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Conversion taux monétaire -> actuariel")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")
    out = convert(df)
    out.to_csv(args.output, index=False, sep=";")
    logger.info("Fichier sauvegardé : %s (%d lignes)", args.output, len(out))


if __name__ == "__main__":
    main()