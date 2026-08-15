library(urca)

# --- Test ADF en niveau ---
adf_beta0 <- ur.df(beta0_ts, type = "trend", selectlags = "AIC")
adf_beta1 <- ur.df(beta1_ts, type = "trend", selectlags = "AIC")
adf_beta2 <- ur.df(beta2_ts, type = "trend", selectlags = "AIC")

summary(adf_beta0)
summary(adf_beta1)
summary(adf_beta2)

# --- Test ADF en différence première ---
dbeta0_ts <- diff(beta0_ts)
dbeta1_ts <- diff(beta1_ts)
dbeta2_ts <- diff(beta2_ts)

adf_dbeta0 <- ur.df(dbeta0_ts, type = "drift", selectlags = "AIC")
adf_dbeta1 <- ur.df(dbeta1_ts, type = "drift", selectlags = "AIC")
adf_dbeta2 <- ur.df(dbeta2_ts, type = "drift", selectlags = "AIC")

summary(adf_dbeta0)
summary(adf_dbeta1)
summary(adf_dbeta2)
