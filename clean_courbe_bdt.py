"""
clean_courbe_bdt.py

Nettoyage et prétraitement de la base brute scrapée depuis bkam.ma, avant
le fit NS/NSS quotidien.

Traitements appliqués :
  1. Parsing des dates et calcul de la maturité résiduelle en années
  2. Suppression des lignes avec maturité résiduelle <= 0 (obligations déjà
     échues encore affichées par erreur sur le site quelques jours/semaines)
  3. Conversion robuste de la colonne "transaction" (volume) en numérique,
     '-' devient NaN (pas une erreur : signifie simplement pas de transaction
     récente sur cette ligne)
  4. Agrégation des doublons (date_courbe, maturite) par moyenne pondérée du
     taux par le volume (fallback en moyenne simple si les volumes sont NaN)

Usage :
    python clean_courbe_bdt.py --input data/courbe_bdt.csv --output data/courbe_bdt_clean.csv
"""

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("clean")


def parse_volume(x) -> float:
    """Convertit '1 102,19' -> 1102.19, '-' -> NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "-" or s == "":
        return np.nan
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_v2(raw: pd.DataFrame):
    """Version robuste (indépendante de la version de pandas pour le groupby.apply)."""
    n0 = len(raw)
    df = raw.copy()

    df["date_courbe"] = pd.to_datetime(df["date_courbe"])
    df["maturite_dt"] = pd.to_datetime(df["maturite"])
    df["date_valeur_dt"] = pd.to_datetime(df["date_valeur"], errors="coerce")
    df["volume"] = df["transaction"].apply(parse_volume)
    # Maturité résiduelle en années -- base 365.25 utilisée ICI uniquement pour
    # détecter les obligations échues (le signe importe, pas la précision).
    # La convention exacte (actual/360, actual/365...) est traitée dans
    # conversion_actuarielle.py à partir de n_jours, pas de ce champ.
    df["maturite_annees"] = (df["maturite_dt"] - df["date_courbe"]).dt.days / 365.25

    n_expired = int((df["maturite_annees"] <= 0).sum())
    df = df[df["maturite_annees"] > 0].copy()
    logger.info("Lignes supprimées (maturité résiduelle <= 0, échues) : %d", n_expired)

    n_before_dedup = len(df)
    n_groups_dup = df.duplicated(subset=["date_courbe", "maturite"]).sum()
    logger.info("Lignes concernées par des doublons (date_courbe, maturité) : %d", n_groups_dup)

    rows = []
    for (dc, mat), g in df.groupby(["date_courbe", "maturite"], sort=False):
        if g["volume"].notna().any() and g["volume"].fillna(0).sum() > 0:
            w = g["volume"].fillna(0).values
            taux_agg = float(np.average(g["taux"].values, weights=w))
        else:
            taux_agg = float(g["taux"].mean())

        rows.append({
            "date_courbe": dc,
            "maturite": mat,
            "maturite_annees": g["maturite_annees"].iloc[0],
            "taux": taux_agg,
            "volume": g["volume"].sum(skipna=True) if g["volume"].notna().any() else np.nan,
            "date_valeur": g["date_valeur_dt"].max(),
        })

    clean_df = pd.DataFrame(rows).sort_values(["date_courbe", "maturite_annees"]).reset_index(drop=True)
    return clean_df, n0, n_expired, n_before_dedup


def main():
    parser = argparse.ArgumentParser(description="Nettoie la base brute courbe_bdt.csv")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input, sep=";")
    clean_df, n0, n_expired, n_before_dedup = clean_v2(raw)

    n_final = len(clean_df)
    logger.info("Résumé : %d lignes brutes -> %d après suppression échues -> %d après dédoublonnage",
                n0, n0 - n_expired, n_final)

    pts_per_day = clean_df.groupby("date_courbe").size()
    logger.info("Points par jour après nettoyage : min=%d, median=%.0f, max=%d",
                pts_per_day.min(), pts_per_day.median(), pts_per_day.max())

    clean_df.to_csv(args.output, index=False, sep=";")
    logger.info("Fichier nettoyé sauvegardé : %s", args.output)


if __name__ == "__main__":
    main()