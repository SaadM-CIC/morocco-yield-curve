import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

print("--- TABLE 2: STATS DESCRIPTIVES TAUX ---")
zc_df = pd.read_csv(DATA_DIR / "zc_timeseries.csv", sep=";")
for mat in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]:
    closest_m = min(zc_df["maturite_annees"].unique(), key=lambda x: abs(x - mat))
    sub = zc_df[zc_df["maturite_annees"] == closest_m]["taux_zc"]
    print(f"Mat {mat} ans: Moy={sub.mean():.2f} Std={sub.std():.2f} Min={sub.min():.2f} Max={sub.max():.2f}")

print("\n--- TABLE 4: STATS FACTEURS ---")
beta_df = pd.read_csv(DATA_DIR / "beta_timeseries.csv", sep=";")
for col in ["beta0", "beta1", "beta2", "rmse"]:
    sub = beta_df[col]
    print(f"{col}: Moy={sub.mean():.2f} Std={sub.std():.2f} Min={sub.min():.2f} Max={sub.max():.2f}")

print("\n--- LAMBDA STATS ---")
l_df = pd.read_csv(DATA_DIR / "lambda_par_date.csv")
print(f"Max lambda: {l_df['lambda_opt'].max():.4f}")
bornes = len(l_df[l_df["lambda_opt"] <= 0.021])
print(f"Bornes basses (0.02): {bornes} / {len(l_df)} ({bornes/len(l_df)*100:.1f}%)")

