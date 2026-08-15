"""
scripts/comparer_deux_etapes_vs_kalman.py

Phase 8 : reproduit, sur nos données, la structure des comparaisons du
rapport (chapitre 2, section V) entre l'approche deux étapes (OLS + VAR)
et l'approche une étape (SSM + filtre de Kalman) :
  - comparaison des matrices A (transition)
  - comparaison des matrices Q (covariance des chocs)
  - comparaison des mu (moyennes de long terme)
  - facteurs niveau/pente/courbure superposés dans le temps
  - RMSE par échéance pour les deux approches (tableau + figure)

Usage :
    python scripts/comparer_deux_etapes_vs_kalman.py \
        --zc data/zc_timeseries.csv --beta data/beta_timeseries.csv --lambda_ 0.0605
"""

import argparse
import sys
from pathlib import Path

OUTPUT_DIR = ROOT = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dns.dynamics.var_model import estimer_var1
from src.dns.kalman.filter import filtrer, lisser
from src.dns.kalman.mle import estimer_mle
from src.dns.kalman.state_space import deflater_observations, matrice_observation


def charger_Y(zc_df: pd.DataFrame):
    dates = sorted(zc_df["date_courbe"].unique())
    tau_ref = zc_df[zc_df["date_courbe"] == dates[0]].sort_values("maturite_annees")["maturite_annees"].values
    Y = np.array([
        zc_df[zc_df["date_courbe"] == d].sort_values("maturite_annees")["taux_zc"].values
        for d in dates
    ])
    return Y, tau_ref, dates


def main():
    parser = argparse.ArgumentParser(description="Comparaison deux-étapes vs Kalman (Phase 8)")
    parser.add_argument("--zc", required=True)
    parser.add_argument("--beta", required=True)
    parser.add_argument("--lambda_", type=float, default=0.0605)
    args = parser.parse_args()

    zc_path = Path(args.zc)
    beta_path = Path(args.beta)
    if not zc_path.is_absolute():
        zc_path = next((p for p in [zc_path, ROOT / zc_path, ROOT / "data" / zc_path] if p.exists()), zc_path)
    if not beta_path.is_absolute():
        beta_path = next((p for p in [beta_path, ROOT / beta_path, ROOT / "data" / beta_path, ROOT / "src" / "dns" / "data" / beta_path.name] if p.exists()), beta_path)

    zc_df = pd.read_csv(zc_path, sep=";")
    beta_df = pd.read_csv(beta_path, sep=";")
    Y, tau_ref, dates = charger_Y(zc_df)
    C = matrice_observation(tau_ref, args.lambda_)

    # --- Approche deux étapes (déjà estimée, Phases 4-5) ---
    res_var = estimer_var1(beta_df)

    # --- Approche une étape (Kalman, Phase 7), initialisée par le VAR ---
    res_mle = estimer_mle(Y, C, res_var.A, res_var.mu, res_var.Q, sigma2_H_init=0.065 ** 2)
    H_mle = np.eye(C.shape[0]) * res_mle.sigma2_H
    Yd = deflater_observations(Y, C, res_mle.mu)
    res_filtre = filtrer(Yd, C, res_mle.A, res_mle.Q, H_mle)
    z_lisse, _ = lisser(res_filtre.z_filtres, res_filtre.P_filtres, res_mle.A, res_mle.Q)
    beta_kalman = z_lisse + res_mle.mu

    np.set_printoptions(precision=4, suppress=True)

    # --- Comparaison A ---
    print("=== Matrice de transition A ===")
    print("Deux étapes (VAR) :\n", res_var.A)
    print("Kalman (MLE)      :\n", res_mle.A)

    # --- Comparaison Q ---
    print("\n=== Matrice de covariance Q ===")
    print("Deux étapes (VAR) :\n", res_var.Q)
    print("Kalman (MLE)      :\n", res_mle.Q)

    # --- Comparaison mu ---
    print("\n=== Moyennes de long terme mu ===")
    tableau_mu = pd.DataFrame({
        "facteur": ["beta0 (niveau)", "beta1 (pente)", "beta2 (courbure)"],
        "deux_etapes": res_var.mu,
        "kalman": res_mle.mu,
    })
    print(tableau_mu.to_string(index=False))

    # --- Figures : facteurs superposés ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    noms = ["niveau (beta0)", "pente (beta1)", "courbure (beta2)"]
    for i, ax in enumerate(axes):
        ax.plot(range(len(dates)), beta_df.iloc[:, i + 1].values, label="Deux étapes (OLS)", color="#4C72B0")
        ax.plot(range(len(dates)), beta_kalman[:, i], label="Kalman (lissé)", color="#DD8452", linestyle="--")
        ax.set_title(f"Facteur {noms[i]}")
        ax.legend()
    axes[-1].set_xlabel("index temporel (mois)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparaison_facteurs.png", dpi=150)
    print(f"\nFigure sauvegardée : {OUTPUT_DIR / 'comparaison_facteurs.png'}")

    # --- RMSE par échéance pour les deux approches ---
    fitted_deux_etapes = beta_df[["beta0", "beta1", "beta2"]].values @ C.T
    fitted_kalman = beta_kalman @ C.T
    residus_deux_etapes = Y - fitted_deux_etapes
    residus_kalman = Y - fitted_kalman

    rmse_deux_etapes = np.sqrt(np.mean(residus_deux_etapes ** 2, axis=0))
    rmse_kalman = np.sqrt(np.mean(residus_kalman ** 2, axis=0))

    tableau_rmse = pd.DataFrame({
        "maturite_annees": tau_ref,
        "rmse_deux_etapes": rmse_deux_etapes,
        "rmse_kalman": rmse_kalman,
    })
    tableau_rmse["kalman_meilleur"] = tableau_rmse["rmse_kalman"] < tableau_rmse["rmse_deux_etapes"]
    print("\n=== RMSE par échéance ===")
    print(tableau_rmse.to_string(index=False))
    tableau_rmse.to_csv(OUTPUT_DIR / "rmse_par_echeance_comparaison.csv", index=False, sep=";")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(tau_ref, rmse_deux_etapes, marker="o", label="Deux étapes (OLS)")
    ax2.plot(tau_ref, rmse_kalman, marker="o", label="Kalman")
    ax2.set_xlabel("maturité (années)")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE par échéance : deux étapes vs Kalman")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rmse_par_echeance.png", dpi=150)
    print(f"Figure sauvegardée : {OUTPUT_DIR / 'rmse_par_echeance.png'}")

    n_kalman_meilleur = tableau_rmse["kalman_meilleur"].sum()
    print(f"\nKalman meilleur sur {n_kalman_meilleur}/{len(tableau_rmse)} maturités")


if __name__ == "__main__":
    main()