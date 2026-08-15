import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import time

# --- Parameters ---
DATA_PATH = "YieldCurve_VECM/data/processed/beta_factors.csv"
ZC_PATH = "YieldCurve_VECM/data/processed/zc_timeseries.csv"
LOOK_BACK = 3  # Reduced look_back based on AIC=3 from VAR
INITIAL_WINDOW = 180
N_ORIGINS = 67
MAX_HORIZON = 12
LAMBDA = 0.0632 # Updated to lambda from utils_loadings
beta_cols = ['beta0', 'beta1', 'beta2']

# Make output dir
os.makedirs("outputs/lstm", exist_ok=True)

# --- Load Data ---
df = pd.read_csv(DATA_PATH, sep=';', parse_dates=['date_courbe'])
df = df.sort_values('date_courbe').reset_index(drop=True)
betas_all = df[beta_cols].values
dates_all = df['date_courbe'].values

zc_data = pd.read_csv(ZC_PATH, sep=';')
maturities = np.sort(zc_data['maturite_annees'].unique())

def build_loadings_matrix(maturities, lam):
    tau = np.asarray(maturities, dtype='float64')
    term = (1 - np.exp(-lam * tau)) / (lam * tau)
    C = np.column_stack([np.ones_like(tau), term, term - np.exp(-lam * tau)])
    return C

C_matrix = build_loadings_matrix(maturities, LAMBDA)

# =============================================================================
# PART 1: Multivariate LSTM Train/Test Split (For Figures & Validation Error)
# =============================================================================
print("=== PART 1: Train/Test Split & Figures ===")

train_raw = betas_all[:INITIAL_WINDOW]
test_raw = betas_all[INITIAL_WINDOW:]

# Differencing (Stationarization)
train_diff = np.diff(train_raw, axis=0)
test_diff = np.diff(np.vstack((train_raw[-1:], test_raw)), axis=0)

scaler = MinMaxScaler(feature_range=(-1, 1))
train_diff_scaled = scaler.fit_transform(train_diff)
test_diff_scaled = scaler.transform(test_diff)

def make_windows(series, look_back):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i + look_back, :])
        y.append(series[i + look_back, :])
    return np.array(X), np.array(y)

X_train, y_train = make_windows(train_diff_scaled, LOOK_BACK)
X_test, y_test = make_windows(np.vstack((train_diff_scaled[-LOOK_BACK:], test_diff_scaled)), LOOK_BACK)

# Build Multivariate LSTM
model = Sequential([
    Input(shape=(LOOK_BACK, 3)),
    LSTM(16, dropout=0.1),
    Dense(3, activation='linear')
])
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=200,
    batch_size=16,
    callbacks=[es],
    verbose=0
)

# Predict differences
pred_train_diff_scaled = model.predict(X_train, verbose=0)
pred_test_diff_scaled = model.predict(X_test, verbose=0)

pred_train_diff = scaler.inverse_transform(pred_train_diff_scaled)
pred_test_diff = scaler.inverse_transform(pred_test_diff_scaled)

# Reconstruct levels
pred_train = np.zeros_like(pred_train_diff)
pred_train[0] = train_raw[LOOK_BACK] + pred_train_diff[0]
for i in range(1, len(pred_train)):
    pred_train[i] = train_raw[LOOK_BACK + i] + pred_train_diff[i]

pred_test = np.zeros_like(pred_test_diff)
pred_test[0] = train_raw[-1] + pred_test_diff[0]
for i in range(1, len(pred_test)):
    pred_test[i] = test_raw[i - 1] + pred_test_diff[i]

# Real aligned arrays
train_aligned = train_raw[LOOK_BACK + 1 : ]
test_aligned = test_raw

val_errors = []
test_errors = []

factors_names = ['Niveau (Beta 0)', 'Pente (Beta 1)', 'Courbure (Beta 2)']
for j, name in enumerate(factors_names):
    # Calculate errors on levels
    err_val = np.mean((pred_train[-int(len(pred_train)*0.1):, j] - train_aligned[-int(len(train_aligned)*0.1):, j])**2)
    err_test = np.mean((pred_test[:, j] - test_aligned[:, j])**2)
    
    val_errors.append(err_val)
    test_errors.append(err_test)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(betas_all)), betas_all[:, j], label="Variation réelle du facteur", color="steelblue", linewidth=1.5)
    
    # Train pred
    train_x = range(LOOK_BACK + 1, INITIAL_WINDOW)
    plt.plot(train_x, pred_train[:, j], label="Modèle LSTM (training data)", color="darkorange", linewidth=1.5, alpha=0.8)
    
    # Test pred
    test_x = range(INITIAL_WINDOW, len(betas_all))
    plt.plot(test_x, pred_test[:, j], label="Test", color="forestgreen", linewidth=1.5, alpha=0.9)
    
    plt.title(f"Tracé du facteur {name.split(' ')[0].lower()}")
    plt.legend()
    plt.savefig(f"outputs/lstm/plot_beta{j}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{name}: Val Error={err_val:.6f}, Test Error={err_test:.6f}")

# Save errors to table
pd.DataFrame({
    'Facteur': factors_names,
    'Erreur de validation': val_errors,
    'Erreur de prédiction (test)': test_errors
}).to_csv("outputs/lstm/lstm_factor_errors.csv", index=False, sep=';')

# Save Beta predictions to CSV for Dashboard integration
lstm_preds_df = pd.DataFrame({
    'date_courbe': dates_all[INITIAL_WINDOW:],
    'beta0_pred': pred_test[:, 0],
    'beta1_pred': pred_test[:, 1],
    'beta2_pred': pred_test[:, 2]
})
lstm_preds_df.to_csv("outputs/lstm/lstm_beta_predictions.csv", index=False, sep=';')
print("Prévisions des betas sauvegardées dans outputs/lstm/lstm_beta_predictions.csv")

# =============================================================================
# PART 2: Recursive Backtest (To compare with VECM)
# =============================================================================
print("\n=== PART 2: Recursive Backtest ===")

records = []
t0 = time.time()

for k in range(N_ORIGINS):
    t = INITIAL_WINDOW + k
    train_data = betas_all[:t]
    origin_date = dates_all[t - 1]
    
    # Stationarize
    train_diff = np.diff(train_data, axis=0)
    scaler_rt = MinMaxScaler(feature_range=(-1, 1))
    train_diff_scaled = scaler_rt.fit_transform(train_diff)
    
    X_tr, y_tr = make_windows(train_diff_scaled, LOOK_BACK)
    X_tr = X_tr.reshape((X_tr.shape[0], LOOK_BACK, 3))
    
    model_rt = Sequential([
        Input(shape=(LOOK_BACK, 3)),
        LSTM(16, dropout=0.1),
        Dense(3, activation='linear')
    ])
    model_rt.compile(loss='mse', optimizer='adam')
    
    es_rt = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_rt.fit(X_tr, y_tr, validation_split=0.1, epochs=150, batch_size=16, callbacks=[es_rt], verbose=0)
    
    # Forecast differences recursively
    beta_forecast_path = np.zeros((MAX_HORIZON, 3))
    window = train_diff_scaled[-LOOK_BACK:].tolist()
    last_beta = train_data[-1].copy()
    
    for h in range(MAX_HORIZON):
        x_input = np.array(window[-LOOK_BACK:]).reshape((1, LOOK_BACK, 3))
        pred_diff_scaled = model_rt.predict(x_input, verbose=0)[0]
        window.append(pred_diff_scaled.tolist())
        
        pred_diff = scaler_rt.inverse_transform([pred_diff_scaled])[0]
        next_beta = last_beta + pred_diff
        beta_forecast_path[h] = next_beta
        last_beta = next_beta
        
    for h in range(1, MAX_HORIZON + 1):
        idx_real = t - 1 + h
        if idx_real >= len(betas_all):
            continue
        curve_true = C_matrix @ betas_all[idx_real]
        curve_pred = C_matrix @ beta_forecast_path[h - 1]
        for m_idx, mat in enumerate(maturities):
            records.append({
                'origin_date': origin_date, 'horizon': h,
                'maturite': mat,
                'sq_error': (curve_pred[m_idx] - curve_true[m_idx]) ** 2
            })

    elapsed = time.time() - t0
    print(f"Origine {k+1}/{N_ORIGINS} ({origin_date}) - {elapsed:.0f}s")

results_df = pd.DataFrame(records)
rmse_by_horizon = results_df.groupby('horizon')['sq_error'].mean().apply(np.sqrt)
print("\nRMSE LSTM par horizon :")
print(rmse_by_horizon)
rmse_by_horizon.to_csv("outputs/lstm/lstm_rmse_by_horizon.csv", sep=';')
print("=== DONE ===")
