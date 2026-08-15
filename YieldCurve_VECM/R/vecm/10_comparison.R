library(dplyr)
library(ggplot2)

# --- Assemblage du tableau comparatif ---
comparison_table <- dplyr::bind_rows(
  rmse_by_horizon_var    %>% dplyr::mutate(model = "VAR"),
  rmse_by_horizon        %>% dplyr::mutate(model = "VECM"),
  rmse_by_horizon_kalman %>% dplyr::mutate(model = "DNS-Kalman (2-step)")
) %>%
  dplyr::select(model, horizon, mean_error, sd_error, rmse) %>%
  dplyr::arrange(horizon, model)

print(comparison_table, n = 36)

# --- Tableau large (un modèle par colonne, RMSE uniquement) ---
comparison_wide <- comparison_table %>%
  dplyr::select(model, horizon, rmse) %>%
  tidyr::pivot_wider(names_from = model, values_from = rmse)

print(comparison_wide)

# --- Modèle gagnant par horizon ---
best_model_by_horizon <- comparison_table %>%
  dplyr::group_by(horizon) %>%
  dplyr::slice_min(rmse, n = 1) %>%
  dplyr::select(horizon, model, rmse)

print(best_model_by_horizon)

# --- Figure comparative (RMSE vs horizon, 3 courbes) ---
ggplot(comparison_table, aes(x = horizon, y = rmse, color = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = 1:12) +
  labs(
    title = "Comparaison des modèles DNS — RMSE hors échantillon par horizon",
    x = "Horizon de prévision (mois)",
    y = "RMSE (points de %)",
    color = "Modèle"
  ) +
  theme_minimal(base_size = 12)

ggsave("outputs/vecm/comparison_rmse_horizon.png", width = 8, height = 5, dpi = 300)
