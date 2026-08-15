"""
scripts/calibrer_panel_final.py

Phase 4 : calibration OLS finale à lambda fixé, sur les vraies données
(zc_timeseries.csv issu de la Phase 1), pour produire la série temporelle
des facteurs beta0, beta1, beta2 -- entrée de la Phase 5 (VAR).

Usage :
    python scripts/calibrer_panel_final.py --input data/zc_timeseries.csv \
        --output data/beta_timeseries.csv --lambda_ 0.0605
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dns.ns.panel import calibrer_panel


def main():
    parser = argparse.ArgumentParser(description="Calibration OLS finale (Phase 4)")
    parser.add_argument("--input", required=True, help="CSV des courbes ZC (Phase 1), séparateur ;")
    parser.add_argument("--output", required=True, help="CSV de sortie (date_courbe, beta0, beta1, beta2, rmse)")
    parser.add_argument("--lambda_", type=float, default=0.0605, help="lambda retenu en Phase 3")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        candidate_paths = [input_path, ROOT / input_path, ROOT / "data" / input_path]
        input_path = next((p for p in candidate_paths if p.exists()), input_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path if not (ROOT / output_path).exists() and not output_path.parts else output_path

    df = pd.read_csv(input_path, sep=";")
    n_dates = df["date_courbe"].nunique()
    print(f"Dates à calibrer : {n_dates}")

    res = calibrer_panel(df, lambda_=args.lambda_)

    print(f"Lignes produites : {len(res)}")
    print(res[["beta0", "beta1", "beta2", "rmse"]].describe())

    res.to_csv(output_path, index=False, sep=";")
    print(f"\nFichier sauvegardé : {output_path}")


if __name__ == "__main__":
    main()