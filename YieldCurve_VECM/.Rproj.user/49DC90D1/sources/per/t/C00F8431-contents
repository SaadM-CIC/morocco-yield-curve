library(vars)

# Construction de la matrice multivariée des niveaux (β0, β1, β2)
beta_mat <- cbind(beta0_ts, beta1_ts, beta2_ts)
colnames(beta_mat) <- c("beta0", "beta1", "beta2")

# Sélection du nombre de retards
lag_selection <- VARselect(beta_mat, lag.max = 12, type = "const")

print(lag_selection$selection)
print(lag_selection$criteria)
