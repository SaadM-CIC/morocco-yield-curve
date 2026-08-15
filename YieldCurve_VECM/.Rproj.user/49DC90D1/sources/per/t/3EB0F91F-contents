library(ggplot2)
library(tidyr)
library(dplyr)

beta_long <- beta_data %>%
  dplyr::select(date_courbe, beta0, beta1, beta2) %>%
  tidyr::pivot_longer(-date_courbe, names_to = "facteur", values_to = "valeur")

ggplot(beta_long, aes(x = date_courbe, y = valeur, color = facteur)) +
  geom_line(linewidth = 0.7) +
  labs(title = "Évolution des facteurs de Nelson-Siegel",
       x = "Date", y = "Valeur", color = "Facteur") +
  theme_minimal()

ggsave("outputs/vecm/factors_timeseries.png", width = 8, height = 4.5, dpi = 300)

crit_long <- as.data.frame(t(lag_selection$criteria)) %>%
  mutate(lag = 1:12) %>%
  pivot_longer(-lag, names_to = "critere", values_to = "valeur")

ggplot(crit_long, aes(x = lag, y = valeur, color = critere)) +
  geom_line() + geom_point() +
  geom_vline(xintercept = 3, linetype = "dashed", color = "gray40") +
  labs(title = "Critères d'information par nombre de retards",
       x = "Nombre de retards", y = "Valeur du critère") +
  theme_minimal()

ggsave("outputs/vecm/lag_selection.png", width = 7, height = 4.5, dpi = 300)


png("outputs/vecm/residuals_acf.png", width = 900, height = 350)
par(mfrow = c(1,3))
acf(residuals(eq_beta0), main = expression(Delta*beta[0]))
acf(residuals(eq_beta1), main = expression(Delta*beta[1]))
acf(residuals(eq_beta2), main = expression(Delta*beta[2]))
dev.off()


ect_series <- as.numeric(regressors$ect1)
dates_ect <- beta_data$date_courbe[-(1:3)]  # ajuster selon nb obs perdues (K=3)

ect_df <- data.frame(date = dates_ect, ect = ect_series)

ggplot(ect_df, aes(x = date, y = ect)) +
  geom_line(color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Terme de correction d'erreur (ECT) dans le temps",
       x = "Date", y = "ECT") +
  theme_minimal()

ggsave("outputs/vecm/ect_series.png", width = 8, height = 4, dpi = 300)

example_origin <- 160  # à choisir
example_h <- 6

train_data <- beta_mat[1:example_origin, ]
var_ex <- VAR(train_data, p = 1, type = "const")
fc_ex <- predict(var_ex, n.ahead = example_h)

beta_fc <- c(fc_ex$fcst$beta0[example_h,"fcst"],
             fc_ex$fcst$beta1[example_h,"fcst"],
             fc_ex$fcst$beta2[example_h,"fcst"])
curve_fc_ex <- reconstruct_curve(beta_fc[1], beta_fc[2], beta_fc[3], C_matrix)

real_date_ex <- unique(zc_data$date_courbe)[example_origin + example_h]
curve_real_ex <- zc_data %>%
  filter(date_courbe == real_date_ex) %>%
  arrange(maturite_annees) %>% pull(taux_zc)

df_curve <- data.frame(maturite = rep(maturities, 2),
                       taux = c(curve_real_ex, curve_fc_ex),
                       type = rep(c("Réalisée","Prévue (VAR)"), each = 33))

ggplot(df_curve, aes(x = maturite, y = taux, color = type)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.5) +
  labs(title = paste("Courbe réalisée vs prévue à", example_h, "mois"),
       x = "Maturité (années)", y = "Taux (%)", color = "") +
  theme_minimal()

ggsave("outputs/vecm/example_curve_forecast.png", width = 7, height = 4.5, dpi = 300)


library(dplyr)
library(ggplot2)
library(urca)

example_origin <- 160   # même origine que l'exemple VAR
example_h <- 6           # même horizon que l'exemple VAR

# --- Réestimation du VECM sur la même fenêtre d'entraînement ---
train_data <- beta_mat[1:example_origin, ]

jo_ex   <- ca.jo(train_data, type = "trace", ecdet = "const", K = 3, spec = "transitory")
var_ex  <- vec2var(jo_ex, r = 1)

fc_ex <- predict(var_ex, n.ahead = example_h)

beta_fc <- c(fc_ex$fcst$beta0[example_h, "fcst"],
             fc_ex$fcst$beta1[example_h, "fcst"],
             fc_ex$fcst$beta2[example_h, "fcst"])

curve_fc_vecm <- reconstruct_curve(beta_fc[1], beta_fc[2], beta_fc[3], C_matrix)

# --- Courbe réalisée (identique à l'exemple VAR) ---
real_date_ex <- unique(zc_data$date_courbe)[example_origin + example_h]
curve_real_ex <- zc_data %>%
  dplyr::filter(date_courbe == real_date_ex) %>%
  dplyr::arrange(maturite_annees) %>%
  dplyr::pull(taux_zc)

# --- Graphique ---
df_curve_vecm <- data.frame(maturite = rep(maturities, 2),
                            taux = c(curve_real_ex, curve_fc_vecm),
                            type = rep(c("Réalisée", "Prévue (VECM)"), each = 33))

ggplot(df_curve_vecm, aes(x = maturite, y = taux, color = type)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.5) +
  labs(title = paste("Courbe réalisée vs prévue (VECM) à", example_h, "mois"),
       x = "Maturité (années)", y = "Taux (%)", color = "") +
  theme_minimal()

ggsave("outputs/vecm/example_curve_forecast_vecm.png", width = 7, height = 4.5, dpi = 300)
