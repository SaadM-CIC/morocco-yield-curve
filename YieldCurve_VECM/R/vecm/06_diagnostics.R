library(lmtest)
data_mat  <- vecm_model$rlm$model
Z0 <- data_mat$`z@Z0`   # matrice des 3 variables dépendantes en différence

regressors <- data_mat[, c("ect1", "beta0.dl1", "beta1.dl1", "beta2.dl1",
                           "beta0.dl2", "beta1.dl2", "beta2.dl2")]

eq_beta0 <- lm(Z0[, "beta0.d"] ~ . - 1, data = regressors)
eq_beta1 <- lm(Z0[, "beta1.d"] ~ . - 1, data = regressors)
eq_beta2 <- lm(Z0[, "beta2.d"] ~ . - 1, data = regressors)

# --- Durbin-Watson (autocorrélation) ---
dwtest(eq_beta0)
dwtest(eq_beta1)
dwtest(eq_beta2)

# --- Breusch-Pagan (homoscédasticité) ---
bptest(eq_beta0)
bptest(eq_beta1)
bptest(eq_beta2)
summary(eq_beta0)  # comparer ect1 ≈ -0.232
