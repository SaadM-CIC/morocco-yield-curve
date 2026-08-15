import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from scipy.interpolate import interp1d

# Ajouter le répertoire racine au sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(
    page_title="Morocco Sovereign Yield Curve Analytics", 
    page_icon="🇲🇦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: #F4F7F6;
    --bg-panel: #FFFFFF;
    --border-soft: rgba(0, 0, 0, 0.05);
    --text-main: #1A3B5C;
    --text-muted: #64748B;
    --morocco-red: #A71930;
    --morocco-green: #007A3D;
    --gold: #D6A84F;
    --sidebar-bg: #1A3B5C;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Masquer le menu Streamlit (les trois points) et le footer, mais garder le header pour le bouton de la sidebar */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
}
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Titres */
h1, h2, h3 {
    color: var(--text-main) !important;
    font-weight: 700 !important;
}
h1 {
    font-size: 2.2rem !important;
    border-bottom: 2px solid var(--gold);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Boutons classiques */
.stButton > button {
    background: var(--text-main) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: var(--morocco-green) !important;
    transform: translateY(-2px);
}

/* Métriques (KPI) */
[data-testid="stMetric"] {
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-left: 4px solid var(--morocco-red);
    border-radius: 8px;
    padding: 1.2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}
div[data-testid="stMetricValue"] {
    color: var(--text-main) !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
}

/* Tableaux et Graphiques */
[data-testid="stDataFrame"], [data-testid="stChart"], [data-testid="stPlotlyChart"] {
    background: var(--bg-panel);
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}

/* Pipeline Box */
.pipeline-box {
    background-color: var(--bg-panel);
    border: 1px solid var(--gold);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    font-weight: 600;
    color: var(--text-main);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
    margin-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# -----------------
# NAVIGATION SIDEBAR
# -----------------
st.sidebar.image("logo.jpg", use_container_width=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Principale", 
    [
        "🏠 Accueil (Market Snapshot)", 
        "🗄️ DATA : Update Market Data", 
        "📈 CURVE ANALYTICS : Visualisation", 
        "🧠 MODELING : Prévisions & Modèles",
        "💰 PRICING : Valorisation Portefeuille",
        "⚠️ RISK : Stress Testing"
    ]
)

# -----------------
# PAGES LOGIC
# -----------------

if page == "🏠 Accueil (Market Snapshot)":
    st.title("Morocco Sovereign Yield Curve Analytics")
    st.subheader("Analyse, modélisation et prévision de la courbe des taux souverains marocains")
    
    st.markdown("""
    <div class='pipeline-box'>
         Market Data ➔  Zero-Coupon Calibration ➔  Statistical Modeling ➔  Forecasting ➔  Valuation ➔  Stress Testing
    </div>
    """, unsafe_allow_html=True)
    
    # Load latest curve for snapshot
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        zc_path = os.path.join(project_root, "data", "zc_timeseries.csv")
        df_zc = pd.read_csv(zc_path, sep=";")
        last_date = df_zc['date_courbe'].max()
        df_last = df_zc[df_zc['date_courbe'] == last_date]
        
        st.success(f"Dernière mise à jour des données du marché : **{last_date}**")
        
        # Calculate key yields via interpolation
        interp_yield = interp1d(df_last['maturite_annees'], df_last['taux_zc'], bounds_error=False, fill_value="extrapolate")
        y2 = float(interp_yield(2.0))
        y5 = float(interp_yield(5.0))
        y10 = float(interp_yield(10.0))
        y15 = float(interp_yield(15.0))
        y20 = float(interp_yield(20.0))
        
        slope = y10 - y2
        curvature = 2*y5 - (y2 + y10)
        
        st.write("### Market Snapshot (Key Maturities)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("2Y Yield", f"{y2:.3f}%")
        c2.metric("5Y Yield", f"{y5:.3f}%")
        c3.metric("10Y Yield", f"{y10:.3f}%")
        c4.metric("15Y Yield", f"{y15:.3f}%")
        c5.metric("20Y Yield", f"{y20:.3f}%")
        
        st.write("### Zero-Coupon Yield Curve")
        c_chart, c_stats = st.columns([3, 1])
        
        with c_chart:
            st.line_chart(data=df_last.set_index("maturite_annees")["taux_zc"])
            
        with c_stats:
            st.markdown("#### Curve Statistics")
            st.metric("10Y-2Y Slope", f"{slope*100:.1f} bps")
            st.metric("Curvature", f"{curvature*100:.1f} bps")
            st.metric("Max Yield", f"{df_last['taux_zc'].max():.3f}%")
            
    except Exception as e:
        st.warning("Impossible de charger les statistiques du marché. Veuillez vérifier la disponibilité des données ZC.")

elif page == "🗄️ DATA : Update Market Data":
    st.header("Data Pipeline : Bank Al-Maghrib")
    st.write("Processus d'ingestion et de nettoyage des données brutes vers la base de données Zero-Coupon.")
    
    st.markdown("""
    **Pipeline d'Ingestion :**
    1. 🌐 `Scraping` : Connexion aux serveurs de BAM.
    2. 🧹 `Cleaning` : Traitement des valeurs manquantes et anomalies.
    3. 🧮 `Calibration` : Extraction de la courbe Zero-Coupon (Bootstrapping).
    4. 💾 `Database` : Mise à jour de `zc_timeseries.csv`.
    """)
    
    if st.button("↻ Exécuter le Data Pipeline (Actualiser)"):
        with st.spinner("Exécution du pipeline en cours..."):
            try:
                from backend.data_pipeline import run_update_pipeline
                run_update_pipeline()
                st.success("✓ Scraping completed · ✓ Missing values processed · ✓ Zero-coupon curve recalibrated")
            except Exception as e:
                st.error(f"Une erreur est survenue : {e}")

elif page == "📈 CURVE ANALYTICS : Visualisation":
    st.header("Curve Dynamics & Visualisation")
    st.write("Visualisation historique et structure par terme des taux.")
    
    try:
        df_zc = pd.read_csv("data/zc_timeseries.csv", sep=";")
        last_date = df_zc['date_courbe'].max()
        st.info(f"Dernière courbe disponible : {last_date}")
        
        df_last = df_zc[df_zc['date_courbe'] == last_date]
        st.line_chart(data=df_last.set_index("maturite_annees")["taux_zc"])
    except FileNotFoundError:
        st.warning("Aucune donnée disponible. Exécutez le Data Pipeline d'abord.")

elif page == "🧠 MODELING : Prévisions & Modèles":
    st.header("Statistical Modeling & Forecasting")
    st.write("Prévision des facteurs de Nelson-Siegel avec des modèles économétriques et d'apprentissage profond.")
    
    modele = st.selectbox("Modèle quantitatif", ["VAR", "VECM (Traduit de R)", "LSTM"])
    horizon = st.slider("Horizon de prévision (Mois)", 1, 12, 6)
    
    if st.button("Lancer l'Inférence"):
        st.info(f"Exécution du modèle {modele} pour {horizon} mois...")
        try:
            from backend.models_pipeline import prepare_data_for_models, forecast_var, forecast_vecm, forecast_lstm, reconstruct_ns_curve, get_latest_lambda
            data = prepare_data_for_models()
            lambda_val = get_latest_lambda()
            
            if modele == "VAR":
                preds = forecast_var(data, steps=horizon)
            elif modele == "VECM (Traduit de R)":
                preds = forecast_vecm(data, steps=horizon, rank=1)
            elif modele == "LSTM":
                preds = forecast_lstm(data, steps=horizon)
            else:
                st.warning(f"Le modèle {modele} n'a pas encore implémenté sa fonction d'inférence directe.")
                st.stop()
                
            derniere_date = data.index[-1]
            tau_array = np.array([0.25, 0.5, 1.0] + list(range(2, 31)))
            
            beta0_actuel, beta1_actuel, beta2_actuel = data.iloc[-1]
            courbe_actuelle = reconstruct_ns_curve(beta0_actuel, beta1_actuel, beta2_actuel, lambda_val, tau_array)
            
            beta0_prevu = preds.iloc[-1]['beta0']
            beta1_prevu = preds.iloc[-1]['beta1']
            beta2_prevu = preds.iloc[-1]['beta2']
            courbe_prevue = reconstruct_ns_curve(beta0_prevu, beta1_prevu, beta2_prevu, lambda_val, tau_array)
            
            df_compare = pd.DataFrame({
                "Maturité (Années)": tau_array,
                "Taux Actuel (%)": courbe_actuelle,
                f"Taux Prévu à M+{horizon} (%)": courbe_prevue
            })
            df_compare["Variation (bps)"] = (df_compare[f"Taux Prévu à M+{horizon} (%)"] - df_compare["Taux Actuel (%)"]) * 100
            
            tab1, tab2 = st.tabs(["📉 Yield Curve Forecast", "🧮 Latent Factors"])
            
            with tab1:
                st.line_chart(df_compare.set_index("Maturité (Années)")[[ "Taux Actuel (%)", f"Taux Prévu à M+{horizon} (%)"]])
                st.dataframe(df_compare.style.format({"Taux Actuel (%)": "{:.3f}", f"Taux Prévu à M+{horizon} (%)": "{:.3f}", "Variation (bps)": "{:.1f}"}))
            
            with tab2:
                st.dataframe(preds)
                st.line_chart(preds)

            # Export Excel
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_compare.to_excel(writer, sheet_name='Comparaison Courbes', index=False)
                preds.to_excel(writer, sheet_name='Facteurs Beta Predits')
            
            st.download_button(
                label="📥 Exporter le Rapport Complet (Excel)",
                data=output.getvalue(),
                file_name=f'Rapport_Previsions_{modele}_M{horizon}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        except Exception as e:
            st.error(f"Erreur d'exécution du modèle : {e}")

elif page == "💰 PRICING : Valorisation Portefeuille":
    st.header("Portfolio Valuation & Backtesting")
    st.write("Évaluation PnL et tracking error par rapport aux prix du marché selon la circulaire AMMC.")
    
    st.subheader("1. Ingestion du Portefeuille")
    uploaded_file = st.file_uploader("Fichier CSV (Code_ISIN, Maturite_initiale, Date_emission, Date_jouissance, Date_echeance, Nominal, Taux_facial)", type=["csv"])
    
    if uploaded_file is not None:
        df_port = pd.read_csv(uploaded_file, sep=';')
    else:
        st.caption("Utilisation du portefeuille de test par défaut.")
        df_port = pd.read_csv("data/portefeuille_test.csv", sep=';')
        
    st.dataframe(df_port.head(5))
    
    st.subheader("2. Exécution du Pricer (Reelle vs Modèles)")
    date_eval = st.date_input("Date de valorisation (ex: 31/05/2023)", value=pd.to_datetime("2023-05-31"))
    
    if st.button("Lancer la Valorisation"):
        with st.spinner("Actualisation des flux financiers..."):
            try:
                from backend.pricer import run_full_backtest
                df_results, has_lstm = run_full_backtest(df_port, str(date_eval))
                
                valeur_reelle = df_results['Prix Exact'].sum()
                valeur_vecm = df_results['Prix VECM'].sum()
                err_vecm = abs(valeur_reelle - valeur_vecm)
                err_vecm_pct = (err_vecm / valeur_reelle) * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Valeur Réelle (Market)", f"{valeur_reelle:,.2f} MAD")
                c2.metric("Erreur Tracking VECM", f"{err_vecm:,.2f} MAD", f"{err_vecm_pct:.3f}%", delta_color="inverse")
                
                if has_lstm:
                    valeur_lstm = df_results['Prix LSTM'].sum()
                    err_lstm = abs(valeur_reelle - valeur_lstm)
                    err_lstm_pct = (err_lstm / valeur_reelle) * 100
                    c3.metric("Erreur Tracking LSTM", f"{err_lstm:,.2f} MAD", f"{err_lstm_pct:.3f}%", delta_color="inverse")
                else:
                    c3.metric("Erreur Tracking LSTM", "Non disponible")
                
                st.dataframe(df_results.style.format({
                    "Prix Exact": "{:,.2f}", "Prix VECM": "{:,.2f}", "Erreur VECM": "{:,.2f}",
                    "Prix LSTM": "{:,.2f}", "Erreur LSTM": "{:,.2f}"
                }))
                
                # Export Excel
                from io import BytesIO
                out_pricer = BytesIO()
                with pd.ExcelWriter(out_pricer, engine='xlsxwriter') as writer:
                    df_results.to_excel(writer, sheet_name='Valorisation', index=False)
                
                st.download_button(
                    label="📥 Exporter les Résultats de Valorisation (Excel)",
                    data=out_pricer.getvalue(),
                    file_name=f'Resultats_Valorisation_{date_eval}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            except Exception as e:
                st.error(f"Erreur du Pricer : {e}")

elif page == "⚠️ RISK : Stress Testing":
    st.header("Stress Testing & Scenario Analysis")
    st.write("Évaluation de la résilience du portefeuille obligataire face aux chocs macroéconomiques.")
    
    try:
        from backend.models_pipeline import prepare_data_for_models, reconstruct_ns_curve, get_latest_lambda
        from scipy.interpolate import interp1d
        from backend.pricer import valoriser_portefeuille
        
        df_betas = prepare_data_for_models()
        derniere_date = df_betas.index[-1]
        b0, b1, b2 = df_betas.iloc[-1]
        lam = get_latest_lambda()
        
        st.info(f"Courbe de Base : {derniere_date.strftime('%Y-%m-%d')} (β0={b0:.2f}, β1={b1:.2f}, β2={b2:.2f})")
        
        c1, c2, c3 = st.columns(3)
        delta_b0 = c1.slider("Choc Inflation / Taux Long Terme (Δβ0) pb", -200, 200, 0, step=5)
        delta_b1 = c2.slider("Choc Pol. Monétaire / Taux Court (Δβ1) pb", -200, 200, 0, step=5)
        delta_b2 = c3.slider("Choc Volatilité / Prime de Risque (Δβ2) pb", -200, 200, 0, step=5)
        
        # Reconstruction
        tau_array = np.array([0.25, 0.5, 1.0] + list(range(2, 31)))
        courbe_reelle = reconstruct_ns_curve(b0, b1, b2, lam, tau_array)
        courbe_choc = reconstruct_ns_curve(b0 + delta_b0/100, b1 + delta_b1/100, b2 + delta_b2/100, lam, tau_array)
        
        df_plot = pd.DataFrame({"Maturité": tau_array, "Base": courbe_reelle, "Stress": courbe_choc}).set_index("Maturité")
        st.line_chart(df_plot)
        
        st.subheader("Impact PnL (Portfolio Shock)")
        uploaded_file = st.file_uploader("Fichier CSV Portefeuille", type=["csv"])
        df_port = pd.read_csv(uploaded_file if uploaded_file else "data/portefeuille_test.csv", sep=';')
            
        interp_reel = interp1d(tau_array, courbe_reelle/100, bounds_error=False, fill_value="extrapolate")
        interp_choc = interp1d(tau_array, courbe_choc/100, bounds_error=False, fill_value="extrapolate")
        
        val_initiale = valoriser_portefeuille(df_port, derniere_date, lambda mr: float(interp_reel(mr)))['Valeur_MAD'].sum()
        val_stresse = valoriser_portefeuille(df_port, derniere_date, lambda mr: float(interp_choc(mr)))['Valeur_MAD'].sum()
        
        pnl = val_stresse - val_initiale
        
        cm1, cm2 = st.columns(2)
        cm1.metric("Valeur Portefeuille (Base)", f"{val_initiale:,.2f} MAD")
        cm2.metric("P&L (Scénario Stress)", f"{pnl:,.2f} MAD", f"{(pnl/val_initiale)*100:.3f}%", delta_color="normal")
        
    except Exception as e:
        st.error(f"Erreur du moteur de Stress Testing : {e}")
