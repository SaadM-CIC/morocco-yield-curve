library(vars)
library(dplyr)

results_list_var <- list()


for (t in origins) {   # origins déjà défini en Phase 6 (132:182)
  
  train_data <- beta_mat[1:t, ]
  
  # Estimation VAR(1) par OLS
  var_t <- VAR(train_data, p = 1, type = "const")
  
  # Prévision récursive à horizon 12
  fc <- predict(var_t, n.ahead = horizon_max)
  
  for (h in 1:horizon_max) {
    beta0_fc <- fc$fcst$beta0[h, "fcst"]
    beta1_fc <- fc$fcst$beta1[h, "fcst"]
    beta2_fc <- fc$fcst$beta2[h, "fcst"]
    
    curve_fc <- reconstruct_curve(beta0_fc, beta1_fc, beta2_fc, C_matrix)
    
    real_index <- t + h
    real_date  <- unique(zc_data$date_courbe)[real_index]
    curve_real <- zc_data %>%
      filter(date_courbe == real_date) %>%
      arrange(maturite_annees) %>%
      pull(taux_zc)
    
    error_vec <- curve_fc - curve_real
    
    results_list_var[[length(results_list_var) + 1]] <- data.frame(
      origin = t,
      horizon = h,
      maturite = maturities,
      erreur = error_vec
    )
  }
}

forecast_errors_var <- do.call(rbind, results_list_var)
str(forecast_errors_var)

rmse_by_horizon_var <- forecast_errors_var %>%
  group_by(horizon) %>%
  summarise(
    mean_error = mean(erreur),
    sd_error   = sd(erreur),
    rmse       = sqrt(mean(erreur^2)),
    .groups = "drop"
  )

print(rmse_by_horizon_var)
