library(urca)
library(dplyr)
# --- Sélection automatique de la fenêtre d'estimation initiale ---

N <- nrow(beta_mat)
horizon_max <- 12

min_test_origins <- 50

n_test_origins <- function(W, N, horizon_max) {
  N - W - (horizon_max - 1)
}

candidate_years <- 8:15
candidate_W <- candidate_years * 12
test_origins <- sapply(candidate_W, n_test_origins, N = N, horizon_max = horizon_max)

selection_table <- data.frame(
  years = candidate_years,
  W_months = candidate_W,
  test_origins = test_origins,
  valid = test_origins >= min_test_origins
)

valid_windows <- selection_table[selection_table$valid, ]
W_selected <- max(valid_windows$W_months)

origins <- W_selected:(N - horizon_max)

cat("Fenêtre initiale retenue :", W_selected, "mois (",
    W_selected / 12, "ans ), avec",
    n_test_origins(W_selected, N, horizon_max), "origines de test\n")



horizon_max <- 12
origins <- W_selected:(N - horizon_max)

# Stockage des résultats : erreurs par origine, horizon, maturité
results_list <- list()

for (t in origins) {
  
  train_data <- beta_mat[1:t, ]
  
  # Réestimation Johansen + VECM (mêmes specs que Phase 3-4)
  jo_t   <- ca.jo(train_data, type = "trace", ecdet = "const", K = 3, spec = "transitory")
  vecm_t <- cajorls(jo_t, r = 1)
  
  # Conversion en objet VAR pour prévision à horizon h (via vec2var)
  var_t <- vec2var(jo_t, r = 1)
  
  fc <- predict(var_t, n.ahead = horizon_max)
  
  for (h in 1:horizon_max) {
    beta0_fc <- fc$fcst$beta0[h, "fcst"]
    beta1_fc <- fc$fcst$beta1[h, "fcst"]
    beta2_fc <- fc$fcst$beta2[h, "fcst"]
    
    curve_fc <- reconstruct_curve(beta0_fc, beta1_fc, beta2_fc, C_matrix)
    
    # Date réalisée correspondante
    real_index <- t + h
    real_date  <- unique(zc_data$date_courbe)[real_index]
    curve_real <- zc_data %>%
      filter(date_courbe == real_date) %>%
      arrange(maturite_annees) %>%
      pull(taux_zc)
    
    error_vec <- curve_fc - curve_real
    
    results_list[[length(results_list) + 1]] <- data.frame(
      origin = t,
      horizon = h,
      maturite = maturities,
      erreur = error_vec
    )
  }
}

forecast_errors <- do.call(rbind, results_list)
str(forecast_errors)
head(forecast_errors)


rmse_by_horizon <- forecast_errors %>%
  group_by(horizon) %>%
  summarise(
    mean_error = mean(erreur),
    sd_error   = sd(erreur),
    rmse       = sqrt(mean(erreur^2)),
    .groups = "drop"
  )

print(rmse_by_horizon)
