import numpy as np
import pandas as pd
from datetime import datetime
import os
from scipy.interpolate import interp1d

# Importing our predictive pipelines
try:
    from backend.models_pipeline import (
        prepare_data_for_models, 
        reconstruct_ns_curve, 
        get_vecm_prediction_for_date,
        get_lstm_prediction_for_date,
        get_latest_lambda
    )
except ImportError:
    pass

def est_bissextile(annee):
    return annee % 4 == 0 and (annee % 100 != 0 or annee % 400 == 0)

def calculer_A(date_eval):
    return 366 if est_bissextile(date_eval.year) else 365

def pricer_obligation_ammc(date_eval, date_emission, date_echeance, date_jouissance, nominal, tf, tr):
    # Maturité initiale (Mi) et Maturité résiduelle (Mr) en jours
    Mi = (date_echeance - date_emission).days
    Mr = (date_echeance - date_eval).days
    
    if Mr < 0:
        return 0.0  # L'obligation est déjà échue
        
    A = calculer_A(date_eval)
    
    # 1. Évaluation des titres de créances de maturité initiale <= 1 an
    if Mi <= 365:
        prix = nominal * ((1 + tf * (Mi / 360)) / (1 + tr * (Mr / 360)))
        return prix
        
    # 2. Évaluation des titres de créances de maturité initiale > 1 an
    if Mi > 365:
        if Mr <= 365:
            prix = nominal * ((1 + tf) / (1 + tr * (Mr / 360)))
            return prix
        else:
            annee_prochain = date_eval.year
            prochain_coupon = datetime(annee_prochain, date_jouissance.month, date_jouissance.day)
            
            if prochain_coupon <= date_eval:
                prochain_coupon = datetime(annee_prochain + 1, date_jouissance.month, date_jouissance.day)
                
            nj = (prochain_coupon - date_eval).days
            n = (date_echeance.year - prochain_coupon.year) + 1
            
            somme_flux = 0
            for i in range(1, n + 1):
                somme_flux += tf / ((1 + tr) ** (i - 1))
            somme_flux += 1 / ((1 + tr) ** (n - 1))
            
            prix = nominal * (1 / ((1 + tr) ** (nj / A))) * somme_flux
            return prix

def valoriser_portefeuille(df_portefeuille, date_eval, get_taux_rendement_func):
    """
    Valorise tout le portefeuille contenu dans un DataFrame.
    """
    df = df_portefeuille.copy()
    valeurs = []
    
    for idx, row in df.iterrows():
        dt_emission = pd.to_datetime(row['Date_emission'])
        dt_echeance = pd.to_datetime(row['Date_echeance'])
        dt_jouissance = pd.to_datetime(row['Date_jouissance'])
        
        mr_jours = (dt_echeance - date_eval).days
        mr_annees = mr_jours / 365.25
        
        tr = get_taux_rendement_func(mr_annees)
        
        prix = pricer_obligation_ammc(
            date_eval=date_eval,
            date_emission=dt_emission,
            date_echeance=dt_echeance,
            date_jouissance=dt_jouissance,
            nominal=row['Nominal'],
            tf=row['Taux_facial'],
            tr=tr
        )
        valeurs.append(prix)
        
    df['Valeur_MAD'] = valeurs
    return df

def run_full_backtest(df_portefeuille, date_eval_str):
    """
    Fonction principale de backtesting pour le Dashboard.
    1. Récupère la vraie courbe à la date d'évaluation
    2. Prédit les facteurs NS (VECM et LSTM)
    3. Reconstruit les courbes ZC
    4. Valorise le portefeuille avec les 3 courbes
    """
    date_eval = pd.to_datetime(date_eval_str)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # --- 1. COURBE RÉELLE ---
    zc_df = pd.read_csv(os.path.join(project_root, "YieldCurve_VECM", "data", "processed", "zc_timeseries.csv"), sep=";")
    zc_eval = zc_df[zc_df['date_courbe'] == date_eval_str]
    if zc_eval.empty:
        raise ValueError(f"Pas de courbe réelle trouvée pour {date_eval_str}")
    
    interp_real = interp1d(zc_eval['maturite_annees'], zc_eval['taux_zc'] / 100, bounds_error=False, fill_value="extrapolate")
    def get_tr_real(mr): return float(interp_real(mr))
    
    # --- 2. COURBE VECM ---
    df_betas = prepare_data_for_models(os.path.join(project_root, "YieldCurve_VECM", "data", "processed", "beta_factors.csv"))
    b0_v, b1_v, b2_v = get_vecm_prediction_for_date(date_eval_str, df_betas)
    
    lam = get_latest_lambda()
    tau_array = np.array([0.25, 0.5, 1.0] + list(range(2, 31)))
    vecm_curve = reconstruct_ns_curve(b0_v, b1_v, b2_v, lambda_val=lam, tau_array=tau_array)
    
    interp_vecm = interp1d(tau_array, vecm_curve / 100, bounds_error=False, fill_value="extrapolate")
    def get_tr_vecm(mr): return float(interp_vecm(mr))
    
    # --- 3. COURBE LSTM ---
    try:
        b0_l, b1_l, b2_l = get_lstm_prediction_for_date(date_eval_str)
        lstm_curve = reconstruct_ns_curve(b0_l, b1_l, b2_l, lambda_val=lam, tau_array=tau_array)
        interp_lstm = interp1d(tau_array, lstm_curve / 100, bounds_error=False, fill_value="extrapolate")
        def get_tr_lstm(mr): return float(interp_lstm(mr))
        has_lstm = True
    except Exception as e:
        has_lstm = False
        lstm_error_msg = str(e)
    
    # --- VALORISATIONS ---
    df_exact = valoriser_portefeuille(df_portefeuille, date_eval, get_tr_real)
    df_vecm = valoriser_portefeuille(df_portefeuille, date_eval, get_tr_vecm)
    
    df_final = pd.DataFrame({
        'Code ISIN': df_portefeuille['Code_ISIN'],
        'Prix Exact': df_exact['Valeur_MAD'],
        'Prix VECM': df_vecm['Valeur_MAD']
    })
    
    df_final['Erreur VECM'] = (df_final['Prix Exact'] - df_final['Prix VECM']).abs()
    
    if has_lstm:
        df_lstm = valoriser_portefeuille(df_portefeuille, date_eval, get_tr_lstm)
        df_final['Prix LSTM'] = df_lstm['Valeur_MAD']
        df_final['Erreur LSTM'] = (df_final['Prix Exact'] - df_final['Prix LSTM']).abs()
    else:
        df_final['Prix LSTM'] = np.nan
        df_final['Erreur LSTM'] = np.nan
        
    return df_final, has_lstm
