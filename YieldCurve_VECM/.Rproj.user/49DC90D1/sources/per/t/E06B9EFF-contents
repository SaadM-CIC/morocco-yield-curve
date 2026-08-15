library(urca)

# Test de Johansen (trace statistic), K = retard du VAR (p = 3)
johansen_test <- ca.jo(beta_mat, type = "trace", ecdet = "const", K = 3, spec = "transitory")

summary(johansen_test)
