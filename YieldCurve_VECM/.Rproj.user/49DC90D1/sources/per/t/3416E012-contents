library(readr)
library(dplyr)

# Import
beta_data <- read_delim("data/processed/beta_factors.csv", delim = ";",
                        locale = locale(decimal_mark = "."))

# Vérification structure
str(beta_data)

# Conversion date
beta_data$date_courbe <- as.Date(beta_data$date_courbe)
beta_data <- beta_data %>% arrange(date_courbe)

# Construction des séries temporelles (fréquence mensuelle)
start_year <- as.numeric(format(beta_data$date_courbe[1], "%Y"))
start_month <- as.numeric(format(beta_data$date_courbe[1], "%m"))

beta0_ts <- ts(beta_data$beta0, start = c(start_year, start_month), frequency = 12)
beta1_ts <- ts(beta_data$beta1, start = c(start_year, start_month), frequency = 12)
beta2_ts <- ts(beta_data$beta2, start = c(start_year, start_month), frequency = 12)

