import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path if needed to use DNS tools
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def matrice_facteurs(tau, lambda_):
    tau = np.atleast_1d(tau)
    C = np.ones((len(tau), 3))
    idx = tau > 0
    l_tau = lambda_ * tau[idx]
    C[idx, 1] = (1 - np.exp(-l_tau)) / l_tau
    C[idx, 2] = C[idx, 1] - np.exp(-l_tau)
    return C

# Load data
zc_df = pd.read_csv(DATA_DIR / "zc_timeseries.csv", sep=";")
zc_df["date_courbe"] = pd.to_datetime(zc_df["date_courbe"])
dates = sorted(zc_df["date_courbe"].unique())
maturites = sorted(zc_df["maturite_annees"].unique())

beta_df = pd.read_csv(DATA_DIR / "beta_timeseries.csv", sep=";")
beta_df["date_courbe"] = pd.to_datetime(beta_df["date_courbe"]) if "date_courbe" in beta_df.columns else dates

# Figure 1: 4 representative curves
def fig1_4curves():
    plt.figure(figsize=(10, 6))
    sample_dates = [dates[0], dates[len(dates)//3], dates[2*len(dates)//3], dates[-1]]
    for d in sample_dates:
        sub = zc_df[zc_df["date_courbe"] == d].sort_values("maturite_annees")
        plt.plot(sub["maturite_annees"], sub["taux_zc"], marker='.', label=str(d.date()))
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux zéro-coupon (%)")
    plt.title("Courbes zéro-coupon à quatre dates représentatives")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig1_4curves.png", dpi=300, bbox_inches="tight")
    plt.close()

# Figure 2: Average curve with +/- 1 std
def fig2_avg_curve():
    grouped = zc_df.groupby("maturite_annees")["taux_zc"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(grouped["maturite_annees"], grouped["mean"], label="Moyenne", color="blue")
    plt.fill_between(grouped["maturite_annees"], 
                     grouped["mean"] - grouped["std"], 
                     grouped["mean"] + grouped["std"], 
                     alpha=0.2, color="blue", label="± 1 écart-type")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux zéro-coupon (%)")
    plt.title(f"Courbe zéro-coupon moyenne ({dates[0].year} - {dates[-1].year})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig2_avg_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

# Figure 3: Nelson-Siegel factor loadings
def fig3_loadings():
    lambda_ = 0.0632
    tau = np.linspace(0.1, 30, 300)
    C = matrice_facteurs(tau, lambda_)
    plt.figure(figsize=(10, 6))
    plt.plot(tau, C[:, 0], label="Niveau (β0)")
    plt.plot(tau, C[:, 1], label="Pente (β1)")
    plt.plot(tau, C[:, 2], label="Courbure (β2)")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Facteur de charge")
    plt.title(f"Facteurs de charge du modèle NS (lambda = {lambda_})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig3_loadings.png", dpi=300, bbox_inches="tight")
    plt.close()

# Figure 4: Evolution of 4 maturities
def fig4_evolution():
    # 13 sem (~0.25), 2 ans (2.0), 10 ans (10.0), 30 ans (30.0)
    mats = [0.25, 2.0, 10.0, 30.0]
    labels = ["13 sem", "2 ans", "10 ans", "30 ans"]
    plt.figure(figsize=(12, 6))
    for m, l in zip(mats, labels):
        # find closest maturity
        closest_m = min(maturites, key=lambda x: abs(x - m))
        sub = zc_df[zc_df["maturite_annees"] == closest_m].sort_values("date_courbe")
        plt.plot(sub["date_courbe"], sub["taux_zc"], label=l)
    plt.xlabel("Date")
    plt.ylabel("Taux zéro-coupon (%)")
    plt.title("Évolution des taux zéro-coupon marocains par maturité")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig4_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

# Figure 5: Histogram of lambda
def fig5_lambda_hist():
    lambda_file = ROOT / "lambda_par_date.csv"
    if lambda_file.exists():
        df_l = pd.read_csv(lambda_file, sep=";")
        plt.figure(figsize=(10, 6))
        plt.hist(df_l["lambda_opt"], bins=40, edgecolor="black", alpha=0.7)
        plt.axvline(df_l["lambda_opt"].mean(), color='r', linestyle='dashed', linewidth=2, label=f"Moyenne = {df_l['lambda_opt'].mean():.4f}")
        plt.axvline(df_l["lambda_opt"].median(), color='orange', linestyle='dashed', linewidth=2, label=f"Médiane = {df_l['lambda_opt'].median():.4f}")
        plt.xlabel("Lambda estimé par date")
        plt.ylabel("Fréquence")
        plt.title("Distribution des valeurs de lambda estimées individuellement")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / "fig5_lambda_hist.png", dpi=300, bbox_inches="tight")
        plt.close()

# Figure 6: Evolution of beta factors
def fig6_betas():
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(["beta0", "beta1", "beta2"]):
        plt.subplot(3, 1, i+1)
        plt.plot(dates, beta_df[col], label=col, color=f"C{i}")
        plt.axhline(beta_df[col].mean(), color="gray", linestyle="dashed", alpha=0.5)
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
    plt.xlabel("Date")
    plt.suptitle("Évolution des facteurs β0, β1 et β2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_betas.png", dpi=300, bbox_inches="tight")
    plt.close()

# Figures 7 and 8: Forecasting
def fig7_8_forecasts():
    var_file = DATA_DIR / "var_params.npz"
    if not var_file.exists():
        return
    with np.load(var_file) as data:
        A = data["A"]
        mu = data["mu"]
        
    last_beta = beta_df[["beta0", "beta1", "beta2"]].iloc[-1].values
    
    horizons = np.arange(1, 13)
    forecasts = []
    
    # Predict factors
    for h in horizons:
        # E[B_{t+h}] = mu + A^h (B_t - mu)
        Ah = np.linalg.matrix_power(A, h)
        pred = mu + Ah @ (last_beta - mu)
        forecasts.append(pred)
        
    forecasts = np.array(forecasts)
    
    # Fig 7: beta forecasts
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(["beta0 (niveau)", "beta1 (pente)", "beta2 (courbure)"]):
        plt.subplot(3, 1, i+1)
        plt.plot(horizons, forecasts[:, i], marker='o', color=f"C{i}")
        plt.title(name)
        plt.grid(True, alpha=0.3)
    plt.xlabel("Horizon (mois)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_forecast_betas.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Fig 8: curve forecasts
    lambda_ = 0.0632
    mats_ref = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    C_ref = matrice_facteurs(mats_ref, lambda_)
    
    plt.figure(figsize=(10, 6))
    for h_idx, h in enumerate([1, 3, 6, 12]):
        curve_h = C_ref @ forecasts[h-1]
        plt.plot(mats_ref, curve_h, marker='o', label=f"h={h} mois")
    plt.xlabel("Maturité (années)")
    plt.ylabel("Taux zéro-coupon (%)")
    plt.title("Courbes des taux prévues aux horizons 1, 3, 6 et 12 mois")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "fig8_forecast_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    print("Generating Figure 1...")
    fig1_4curves()
    print("Generating Figure 2...")
    fig2_avg_curve()
    print("Generating Figure 3...")
    fig3_loadings()
    print("Generating Figure 4...")
    fig4_evolution()
    print("Generating Figure 5...")
    fig5_lambda_hist()
    print("Generating Figure 6...")
    fig6_betas()
    print("Generating Figures 7 and 8...")
    fig7_8_forecasts()
    print("Done! All figures are in the 'outputs' directory.")
