library(cmdstanr)
set_cmdstan_path("C:/Users/godli/.cmdstan/cmdstan-2.36.0/cmdstan-2.36.0")
library(posterior)
library(tidyverse)
library(arrow)

set.seed(123)

df_all_daily <- read_parquet("data/all_locations_daily_weather_inflation.parquet")

df_loc1 <- df_all_daily %>% filter(location_id == "SRQS8F7JWA9MZ")
df_loc1_train <- df_loc1 %>% filter(date < "2022-08-01")
df_loc1_test <- df_loc1 %>% filter(date >= "2022-08-01")

# 1. Get names of numeric columns (excluding outcome and date)
numeric_cols <- df_loc1_train %>%
  select(-date, -location_id, -contains("cat"), -season, -weekend, -nonvegan_outcome) %>%
  select(where(is.numeric)) %>%
  colnames()

# 2. Define a function to scale using training mean and sd
scale_with_train_stats <- function(train_df, test_df, cols) {
  means <- map_dbl(train_df[cols], mean, na.rm = TRUE)
  sds   <- map_dbl(train_df[cols], sd, na.rm = TRUE)
  
  scale_fn <- function(df) {
    df[cols] <- map2_dfc(df[cols], cols, ~ (.x - means[.y]) / sds[.y])
    df
  }
  
  list(
    train = scale_fn(train_df),
    test  = scale_fn(test_df)
  )
}

# 3. Apply scaling
scaled_data <- scale_with_train_stats(df_loc1_train, df_loc1_test, numeric_cols)
df_loc1_train <- scaled_data$train
df_loc1_test  <- scaled_data$test

outcome <- "nonvegan_outcome"

predictors <- c(
  "vegan_price_real",
  "meat_price_real",
  "day_of_week_cat",
  "weekend",
  "month_cat",
  "season",
  "year",
  "inflation",
  "temp",
  "precip"
)

formula_var <- ~ 1 + vegan_price_real + meat_price_real +
  day_of_week_cat + weekend +
  month_cat + season + year +
  inflation + temp + precip

X_train <- model.matrix(formula_var, data = df_loc1_train)
X_test  <- model.matrix(formula_var, data = df_loc1_test)
y_train <- df_loc1_train[[outcome]]
y_test  <- df_loc1_test[[outcome]]

# Compile the Stan model using cmdstanr
mod <- cmdstan_model("model.stan")

# Define dimensions and lag orders
N_train <- X_train %>% nrow()
N_test  <- X_test %>% nrow()
K       <- X_train %>% ncol()
p <- 56   # lag order for counts
q <- 56   # lag order for intensities
p_effective <- 14
q_effective <- 14
effective_lags_alpha <- c(1,2,3,4,5,6,7,14,21,28,35,42,49,56)
effective_lags_delta <- c(1,2,3,4,5,6,7,14,21,28,35,42,49,56)

# Hyperprior parameter inputs
beta_scale_input <- rep(1, K)
alpha_scale_input <- seq(0.03,0.03,length.out=p)
delta_scale_input <- seq(0.03,0.03,length.out=q)

# Prepare data list for Stan
data_list <- list(
  K = K,
  p = p,
  q = q,
  p_effective = p_effective,
  q_effective = q_effective,
  effective_lags_alpha = effective_lags_alpha,
  effective_lags_delta = effective_lags_delta,
  N_train = N_train,
  X_train = X_train,
  y_train = y_train,
  N_test = N_test,
  X_test = X_test,
  y_test = y_test,
  # beta_scale = ,
  # alpha_scale = ,
  # delta_scale = ,
  beta_hyper = 10,
  alpha_hyper = 10,
  delta_hyper = 10
)

alpha_init <- rep(0, p_effective)
if (p_effective >= 1) alpha_init[1] <- 0.1
if (p_effective >= 2) alpha_init[2] <- 0
if (p_effective >= 3) alpha_init[3] <- 0
if (p_effective >= 4) alpha_init[4] <- 0
if (p_effective >= 5) alpha_init[5] <- 0
if (p_effective >= 6) alpha_init[6] <- 0
if (p_effective >= 7) alpha_init[7] <- 0.1

delta_init <- rep(0, q_effective)
if (q_effective >= 1) delta_init[1] <- 0.1
if (q_effective >= 2) delta_init[2] <- 0
if (q_effective >= 3) delta_init[3] <- 0
if (q_effective >= 4) delta_init[4] <- 0
if (q_effective >= 5) delta_init[5] <- 0
if (q_effective >= 6) delta_init[6] <- 0
if (q_effective >= 7) delta_init[7] <- 0.1

init_fn <- function() {
  list(
    beta = rep(0.5, K),
    alpha = alpha_init,
    delta = delta_init, 
    beta_scale = rep(1, K),
    alpha_scale = rep(0.03, p_effective),
    delta_scale = rep(0.03, q_effective),
    phi = 5
  )
}

# Fit the model
fit <- mod$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  parallel_chains = 15,
  iter_warmup = 500,
  iter_sampling = 1500,
  init = init_fn
)

# Print a summary of the results
print(fit$summary(), n=40)

# Extract posterior predictive draws and log likelihood
y_test_rep_df <- as_draws_df(fit$draws("y_test_rep"))

# Remove metadata columns (.chain, .iteration, .draw)
y_test_rep_df <- y_test_rep_df %>% select(starts_with("y_test_rep"))

# Compute the posterior predictive mean for each test time point
y_test_pred_mean <- colMeans(as_draws_matrix(fit$draws("y_test_rep")))

# Save
saveRDS(fit, "model_fits/empirical/fit_loc1_high_lags.rds")

weekly_data <- df_loc1_test %>%
  dplyr::select(date) %>%
  cbind(y_test, y_test_pred_mean) %>%
  group_by(week = floor_date(date, "week")) %>% 
  summarize(y_test = sum(y_test),
            y_test_pred_mean = sum(y_test_pred_mean), .groups = "drop")

n_test_weekly <- weekly_data$y_test %>% length()

# Plot observed test counts vs. posterior predictive mean
plot(1:n_test_weekly, weekly_data$y_test, type = "l", col = "black", lwd = 2,
     main = "Test Data: Observed vs. Predicted Counts",
     xlab = "Test Time Point", ylab = "Count")
lines(1:n_test_weekly, weekly_data$y_test_pred_mean, col = "red", lwd = 2)
legend("topright", legend = c("Observed", "Predicted"),
       col = c("black", "red"), lwd = 2)