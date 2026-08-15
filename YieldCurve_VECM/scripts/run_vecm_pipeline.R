setwd("c:/Users/Saad/Documents/YieldCurve_VECM")

cat("=== SOURCING utils_loadings.R ===\n")
source("R/vecm/utils_loadings.R")

cat("=== SOURCING 01_data_import.R ===\n")
source("R/vecm/01_data_import.R")

cat("\n=== SOURCING 02_stationarity.R ===\n")
source("R/vecm/02_stationarity.R")

cat("\n=== SOURCING 03_lag_selection.R ===\n")
source("R/vecm/03_lag_selection.R")

cat("\n=== SOURCING 04_johansen.R ===\n")
source("R/vecm/04_johansen.R")

cat("\n=== SOURCING 05_vecm_estimation.R ===\n")
source("R/vecm/05_vecm_estimation.R")

cat("\n=== SOURCING 06_diagnostics.R ===\n")
source("R/vecm/06_diagnostics.R")

cat("\n=== SOURCING 07_forecast.R ===\n")
source("R/vecm/07_forecast.R")

cat("\n=== SOURCING 08_backtest_var.R ===\n")
source("R/vecm/08_backtest_var.R")

cat("\n=== SOURCING 09_backtest_kalman.R ===\n")
source("R/vecm/09_backtest_kalman.R")

cat("\n=== SOURCING 10_comparison.R ===\n")
source("R/vecm/10_comparison.R")

cat("\n=== PIPELINE COMPLETE ===\n")
