library(dlm)
library(dplyr)
library(tidyr)
library(vars)
library(expm)

# --- Fonction : matrice Y (T x 33) des taux zéro-coupon, ordonnée par maturité ---
build_Y_matrix <- function(zc_data, dates_subset, maturities) {
  wide <- zc_data %>%
    dplyr::filter(date_courbe %in% dates_subset) %>%
    dplyr::select(date_courbe, maturite_annees, taux_zc) %>%
    tidyr::pivot_wider(names_from = maturite_annees, values_from = taux_zc) %>%
    dplyr::arrange(date_courbe)
  
  wide <- wide %>% dplyr::select(-date_courbe)
  wide <- wide[, order(as.numeric(names(wide)))]   
  as.matrix(wide)
}

# --- Boucle complète : backtest Kalman (Two-Step State-Space) ---

results_list_kalman <- list()
t0_total <- Sys.time()

cat("\nLancement du Backtest DNS-Kalman (Méthode 2-Step)...\n")

for (t in origins) {
  cat(sprintf("-> [Origin t = %d / %d] Calibrating and Filtering...\n", t, max(origins)))
  flush.console()
  
  # 1. Préparation des données jusqu'à t
  train_dates <- unique(zc_data$date_courbe)[1:t]
  Y_train <- build_Y_matrix(zc_data, train_dates, maturities)
  
  # Facteurs empiriques estimés par OLS (déjà disponibles dans beta_mat)
  mu_hat <- colMeans(beta_mat[1:t, ])
  
  # Centrage des rendements par rapport à la moyenne des facteurs
  Y_centered <- Y_train - matrix(C_matrix %*% mu_hat, nrow(Y_train), 33, byrow = TRUE)
  
  # 2. Calibration empirique (sans optimisation numérique)
  # Dynamique de transition (A) et variance des chocs (Q) via VAR(1)
  var_init <- vars::VAR(beta_mat[1:t, ], p = 1, type = "none")
  A_empirical <- t(sapply(coef(var_init), function(eq) eq[1:3, "Estimate"]))
  Q_empirical <- cov(residuals(var_init))
  
  # Variance d'observation (V) empirique, moyennée sur les maturités
  resid_var <- mean(apply(Y_centered, 2, var)) * 0.1
  V_empirical <- resid_var * diag(33)
  
  # 3. Construction du Modèle dlm (Instantané et stable)
  mod <- dlm(FF = as.matrix(C_matrix), 
             V = as.matrix(V_empirical),
             GG = as.matrix(A_empirical), 
             W = as.matrix(Q_empirical),
             m0 = rep(0, 3), 
             C0 = cov(beta_mat[1:t, ])) # Prior basé sur la variance inconditionnelle
  
  # 4. Filtrage de Kalman
  filt <- dlmFilter(as.matrix(Y_centered), mod)
  z_last <- filt$m[nrow(filt$m), ]   # z_{t|t} : dernier état filtré
  
  # 5. Prévision à horizon h = 1..12
  for (h in 1:horizon_max) {
    # Prévision de l'état : z_{t+h} = A^h * z_last
    z_h <- as.numeric(A_empirical %^% h %*% z_last)
    
    # Reconstitution de la courbe : Y_{t+h} = C * (z_h + mu_hat)
    curve_fc <- as.numeric(C_matrix %*% (z_h + mu_hat))
    
    # Comparaison avec la vraie courbe
    real_index <- t + h
    real_date <- unique(zc_data$date_courbe)[real_index]
    
    curve_real <- zc_data %>%
      filter(date_courbe == real_date) %>%
      arrange(maturite_annees) %>%
      pull(taux_zc)
    
    error_vec <- curve_fc - curve_real
    
    results_list_kalman[[length(results_list_kalman) + 1]] <- data.frame(
      origin = t,
      horizon = h,
      maturite = maturities,
      erreur = error_vec
    )
  }
}

cat("\nTemps total d'exécution :", round(Sys.time() - t0_total, 2), "secondes.\n\n")

# --- Agrégration et calcul du RMSE ---

forecast_errors_kalman <- do.call(rbind, results_list_kalman)

rmse_by_horizon_kalman <- forecast_errors_kalman %>%
  group_by(horizon) %>%
  summarise(
    mean_error = mean(erreur),
    sd_error   = sd(erreur),
    rmse       = sqrt(mean(erreur^2)),
    .groups = "drop"
  )

cat("=== PERFORMANCES DNS-KALMAN (RMSE par horizon) ===\n")
print(rmse_by_horizon_kalman)
