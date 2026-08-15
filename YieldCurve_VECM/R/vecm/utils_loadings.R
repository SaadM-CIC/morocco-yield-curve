library(readr)
library(dplyr)

# --- Extraction du vecteur des 33 maturités (en années) ---
zc_data <- read_delim("data/processed/zc_timeseries.csv", delim = ";",
                      locale = locale(decimal_mark = "."))


maturities <- sort(unique(zc_data$maturite_annees))

stopifnot(length(maturities) == 33)  # vérification de sécurité

# --- Construction de la matrice de facteurs de charge C (33 x 3) ---
build_loadings_matrix <- function(maturities, lambda) {
  tau <- maturities
  term <- (1 - exp(-lambda * tau)) / (lambda * tau)
  C <- cbind(
    level = rep(1, length(tau)),
    slope = term,
    curvature = term - exp(-lambda * tau)
  )
  rownames(C) <- paste0("mat_", round(tau, 3))
  return(C)
}

lambda_hat <- 0.0632
C_matrix <- build_loadings_matrix(maturities, lambda_hat)

print(C_matrix)

# --- Reconstruction d'une courbe de taux à partir de (beta0, beta1, beta2) ---
reconstruct_curve <- function(beta0, beta1, beta2, C_matrix) {
  beta_vec <- c(beta0, beta1, beta2)
  as.numeric(C_matrix %*% beta_vec)   # vecteur de 33 taux
}

