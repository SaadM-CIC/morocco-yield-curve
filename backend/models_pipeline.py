import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.api import VAR

import os

def prepare_data_for_models(file_path=None):
    """
    Charge les facteurs de Nelson-Siegel (Beta0, Beta1, Beta2) 
    estimés au préalable pour la modélisation.
    """
    if file_path is None:
        # Pointer vers le vrai fichier des betas
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, "YieldCurve_VECM", "data", "processed", "beta_factors.csv")
        
    df = pd.read_csv(file_path, sep=";")
    
    # On gère le nom de la colonne date qui peut varier ('date' ou 'date_courbe')
    if 'date_courbe' in df.columns:
        df['date'] = pd.to_datetime(df['date_courbe'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df[['beta0', 'beta1', 'beta2']].dropna()

def get_latest_lambda():
    """
    Récupère le dernier lambda optimisé depuis lambda_par_date.csv
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, "lambda_par_date.csv")
    try:
        df = pd.read_csv(file_path, sep=";")
        return df['lambda_opt'].iloc[-1]
    except:
        return 0.0609 # Default fallback

def reconstruct_ns_curve(beta0, beta1, beta2, lambda_val=None, tau_array=None):
    """
    Reconstruit la courbe Nelson-Siegel pour un ensemble de maturités.
    Si tau_array n'est pas fourni, utilise les 33 maturités standard.
    """
    if lambda_val is None:
        lambda_val = get_latest_lambda()
        
    if tau_array is None:
        # 13s, 26s, 52s (approx 0.25, 0.5, 1.0) et années 2 à 30
        tau_array = np.array([0.25, 0.5, 1.0] + list(range(2, 31)))
        
    term1 = beta0
    term2 = beta1 * ((1 - np.exp(-lambda_val * tau_array)) / (lambda_val * tau_array))
    term3 = beta2 * (((1 - np.exp(-lambda_val * tau_array)) / (lambda_val * tau_array)) - np.exp(-lambda_val * tau_array))
    
    return pd.Series(term1 + term2 + term3, index=tau_array)

def forecast_var(data, lags=1, steps=6):
    """
    Prévision avec modèle VAR sur les facteurs Beta.
    """
    model = VAR(data)
    results = model.fit(lags)
    forecast = results.forecast(data.values[-lags:], steps=steps)
    return pd.DataFrame(forecast, columns=['beta0', 'beta1', 'beta2'])

def forecast_vecm(data, lags=2, steps=6, rank=1):
    from statsmodels.tsa.vector_ar.vecm import select_order
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    vecm_res = VECM(data, k_ar_diff=lags, coint_rank=rank, deterministic="ci").fit()
    forecast = vecm_res.predict(steps=steps)
    return pd.DataFrame(forecast, columns=['beta0', 'beta1', 'beta2'])


def get_vecm_prediction_for_date(target_date_str, df_betas, lags=2, rank=1):
    """
    Simule une prédiction VECM à 1 mois (M+1) pour la date cible.
    On coupe les données just avant la date cible et on prédit 1 step.
    """
    target_date = pd.to_datetime(target_date_str)
    # Filtrer les données jusqu'au mois précédent
    train_data = df_betas[df_betas.index < target_date]
    if len(train_data) < lags + 5:
        raise ValueError(f"Pas assez de données avant {target_date} pour entraîner le VECM.")
    
    forecast_df = forecast_vecm(train_data, lags=lags, steps=1, rank=rank)
    pred_beta0 = forecast_df.iloc[0]['beta0']
    pred_beta1 = forecast_df.iloc[0]['beta1']
    pred_beta2 = forecast_df.iloc[0]['beta2']
    
    return pred_beta0, pred_beta1, pred_beta2


def get_lstm_prediction_for_date(target_date_str):
    """
    Charge les prédictions LSTM sauvegardées par le script run_lstm.py pour la date cible.
    """
    target_date = pd.to_datetime(target_date_str)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preds_path = os.path.join(project_root, "outputs", "lstm", "lstm_beta_predictions.csv")
    
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Le fichier des prédictions LSTM n'existe pas : {preds_path}. Lancez l'entraînement d'abord.")
        
    df_preds = pd.read_csv(preds_path, sep=";")
    df_preds['date'] = pd.to_datetime(df_preds['date_courbe'])
    
    # Chercher la date la plus proche ou exacte
    match = df_preds[df_preds['date'] == target_date]
    if match.empty:
        # Essayer de trouver le même mois/année
        match = df_preds[(df_preds['date'].dt.year == target_date.year) & (df_preds['date'].dt.month == target_date.month)]
        if match.empty:
            raise ValueError(f"Aucune prédiction LSTM trouvée pour le mois de {target_date.strftime('%Y-%m')}")
            
    row = match.iloc[-1]
    return row['beta0_pred'], row['beta1_pred'], row['beta2_pred']

def forecast_lstm(data, steps=6):
    """
    Appelle le modèle LSTM. 
    Entraîne le modèle sur les données actuelles et fait une prévision récursive.
    """
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense
    
    # Préparation des données (différences premières)
    raw_data = data.values
    diff_data = np.diff(raw_data, axis=0)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    diff_scaled = scaler.fit_transform(diff_data)
    
    LOOK_BACK = 3
    X, y = [], []
    for i in range(len(diff_scaled) - LOOK_BACK):
        X.append(diff_scaled[i:i + LOOK_BACK, :])
        y.append(diff_scaled[i + LOOK_BACK, :])
    X, y = np.array(X), np.array(y)
    
    # Entraînement rapide du modèle
    model = Sequential([
        Input(shape=(LOOK_BACK, 3)),
        LSTM(16, dropout=0.1),
        Dense(3, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    
    # Prévision récursive
    preds_diff_scaled = []
    current_input = diff_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 3)
    
    for _ in range(steps):
        pred_scaled = model.predict(current_input, verbose=0)
        preds_diff_scaled.append(pred_scaled[0])
        # Décale la fenêtre
        current_input = np.append(current_input[:, 1:, :], pred_scaled.reshape(1, 1, 3), axis=1)
        
    preds_diff = scaler.inverse_transform(preds_diff_scaled)
    
    # Reconstruction des niveaux
    preds = np.zeros((steps, 3))
    preds[0] = raw_data[-1] + preds_diff[0]
    for i in range(1, steps):
        preds[i] = preds[i-1] + preds_diff[i]
        
    return pd.DataFrame(preds, columns=['beta0', 'beta1', 'beta2'])
