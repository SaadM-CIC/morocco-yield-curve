library(urca)

# Estimation du VECM à partir du test de Johansen (r = 1)
vecm_model <- cajorls(johansen_test, r = 1)

summary(vecm_model$rlm)
print(vecm_model$beta)   # coefficients de la relation de cointégration
