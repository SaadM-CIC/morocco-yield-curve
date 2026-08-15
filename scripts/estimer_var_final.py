"""
scripts/estimer_var_final.py

Phase 5 : estime le VAR(1) sur la série temporelle des facteurs
(beta_timeseries.csv, issu de la Phase 4), et affiche/sauvegarde
A, mu, Q ainsi que les diagnostics de stationnarité.

Usage :
    python scripts/estimer_var_final.py --input data/beta_timeseries.csv \
        --output data/var_params.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dns.dynamics.var_model import estimer_var1


def main():
    parser = argparse.ArgumentParser(description="Estimation VAR(1) (Phase 5)")
    parser.add_argument("--input", required=True, help="CSV beta_timeseries.csv (Phase 4), séparateur ;")
    parser.add_argument("--output", required=True, help="Fichier .npz de sortie (A, mu, c, Q)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        candidate_paths = [input_path, ROOT / input_path, ROOT / "data" / input_path]
        input_path = next((p for p in candidate_paths if p.exists()), input_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path if not (ROOT / output_path).exists() and not output_path.parts else output_path

    df = pd.read_csv(input_path, sep=";")
    print(f"Observations : {len(df)}")

    res = estimer_var1(df)

    np.set_printoptions(precision=4, suppress=True)
    print("\nA =")
    print(res.A)
    print("\nmu =", res.mu)
    print("\nQ =")
    print(res.Q)
    print("\nvaleurs propres |A| :", res.valeurs_propres_A)
    print("Stationnaire :", res.stationnaire)

    if not res.stationnaire:
        print("\nATTENTION : au moins une valeur propre de A a un module >= 1 "
              "(VAR non stationnaire) -- la prévision à horizon long ne "
              "convergera pas vers mu.")

    np.savez(output_path, A=res.A, mu=res.mu, c=res.c, Q=res.Q,
             valeurs_propres_A=res.valeurs_propres_A)
    print(f"\nParamètres sauvegardés : {output_path}")


if __name__ == "__main__":
    main()