## ===== Imports =====

# Processing and visualizing
library(fpp3) # tibble, dplyr, tidyr, lubridate, ggplot2, tsibble, tsibbledata, feasts, fable
library(arrow) # parquet
library(skimr) # data summary
library(grid) # visuals
library(gridExtra) # visuals
library(shiny) # dashboards
library(R.utils) # safe executions if errors

# Modeling
library(tscount) # tsglm
library(MASS) # glm.nb
library(bayesforecast) # ACF, PACF

## ===== Main Modeling Functions: INGARCH =====

# 1. Fits and returns a NB model with AR terms and Mean lag teams
fit_nbar_model <- function(df, outcome, predictors, ar_lags=c(), mean_lags=c()) {

  # tsglm takes time series
  outcome_ts <- df[[outcome]]
  # Check if predictors are provided
  xreg <- if (!is.null(predictors)) {model.matrix(~ . - 1, data = df[, predictors])} else {NULL}

  # Skip the model if it fails
  result <- tryCatch(
    {tsglm(outcome_ts,
           model = list(past_obs = ar_lags,
                       past_mean = mean_lags),
           xreg = xreg,
           link = "log",
           distr = "nbinom")},
    error = function(e)
      {message("Skipping due to error: ", e$message)
      return(NULL)})
}

# 2. Generates and returns rolling forecasts for testing data using nbar
rolling_forecast_nbar <- function(test_df, model, outcome, predictors) {

  # Carry up any existing error from the inner model
  if (is.null(model)) {
    message("rolling_forecast_nbar: Model is NULL, returning NA forecasts")
    return(rep(NA, nrow(test_df)))
  }

  # Set up vector to store forecasts
  n_test <- nrow(test_df)
  forecasts <- numeric(n_test)
  for (i in 1:n_test) {

    # If predictors exist, create the design matrix for the first i test observations
    newxreg <- NULL
    if (!is.null(predictors) && length(predictors) > 0) {
      newxreg <- model.matrix(~ . - 1, data=test_df[1:i, predictors, drop=FALSE])}

    # Pass the first i-1 actual outcomes as newobs and get forecasts for step i
    newobs <- NULL
    if (1 < i) {
      newobs <- test_df[1:(i-1), outcome, drop=TRUE]}

    # Predict using tscount's predict and extract the i-th observation
    pred_obj <- tryCatch(
      {withTimeout(predict(model,
                           n.ahead = i,
                           newobs = newobs,
                           newxreg = newxreg),
                   timeout = 3,
                   onTimeout = "silent")},
      error = function(e)
        {message("Predict error at iteration ", i, ": ", e$message)
        return(NULL)})

    # Convert NULL model to NA forecasts
    if (is.null(pred_obj)) {
      message("Prediction timed out or failed at observation ", i, " in the fold; setting forecast to NA.")
      forecasts[i] <- NA}
    else {
      forecasts[i] <- pred_obj$pred[i]
      # cat("Forecast", i, "in the fold completed.\n")
      }

  }
  # Return
  forecasts
}

# 3. Performs walk-forward (expanding) K-fold cross-validation with early stopping
walk_forward_cv_nbar <- function(df,
                                 loc,
                                 outcome,
                                 predictors,
                                 initial_train_days,
                                 test_days = 30,
                                 ar_lags=c(),
                                 mean_lags=c(),
                                 sample=FALSE,
                                 bad_fold_threshold = Inf) { # <-- Added threshold parameter

  # message("Starting walk_forward_cv_nbar with threshold: ", bad_fold_threshold)

  fold_counter <- 1

  # Filter to restaurant (do this once at the start)
  df_loc <- df %>% dplyr::filter(location_id == loc)
  if(nrow(df_loc) == 0) {
      message("walk_forward_cv_nbar: No data found for location_id: ", loc)
      return(NULL)
  }


  # Set default sample size to the entire dataset
  if (sample) {df_loc <- df_loc %>% slice(1:sample)}

  # Define training fold end dates:
  all_dates <- df_loc$date
  first_train_end <- min(all_dates) + days(initial_train_days - 1)
  last_possible_train_end <- max(all_dates) - days(test_days)

  # Check if any valid folds are possible
  if (first_train_end > last_possible_train_end) {
      message("walk_forward_cv_nbar: Initial train days + test days exceeds data range. No folds possible.")
      return(NULL)
  }

  # print(first_train_end) # Keep if useful
  # print(last_possible_train_end)

  # Move forward by test_days number each time, i.e., move forward one fold
  current_train_end <- first_train_end

  cv_results <- list()
  while (current_train_end <= last_possible_train_end) {

    fold_data <- NULL # To store the results of the current fold if successful

    fold_status <- tryCatch({
      # Define the training and testing sets for the current fold
      train_fold <- df_loc %>%
        dplyr::filter(date <= current_train_end)
      test_fold <- df_loc %>%
        dplyr::filter(date > current_train_end & date <= current_train_end + days(test_days))

      # Ensure the test fold has the expected number of days
      if (nrow(test_fold) < test_days) {
          # This might happen near the end if dates are missing
          message("walk_forward_cv_nbar: Fold ", fold_counter, " skipped. Not enough test days (found ", nrow(test_fold), ", need ", test_days, "). Train end: ", current_train_end)
          return("skip_fold") # Signal to skip this iteration
      }

      # --- Fit and Predict ---
      model <- fit_nbar_model(train_fold, outcome, predictors, ar_lags, mean_lags)
      if (is.null(model)) {
        message("walk_forward_cv_nbar: Model fitting failed for fold ", fold_counter, ". Skipping fold.")
        return("skip_fold") # Signal to skip
      }
      pred <- rolling_forecast_nbar(test_fold, model, outcome, predictors)
      # Basic check on prediction output
      if (is.null(pred) || length(pred) != nrow(test_fold)) {
           message("walk_forward_cv_nbar: Prediction failed or returned incorrect length for fold ", fold_counter, ". Skipping fold.")
           return("skip_fold")
      }
      if (anyNA(pred)) {
           message("walk_forward_cv_nbar: Predictions contain NA for fold ", fold_counter, ". Skipping fold.")
           return("skip_fold")
      }


      # --- Calculate Fold MSE and Check Threshold ---
      actual_values <- test_fold[[outcome]]
      fold_mse <- mean((actual_values - pred)^2)

      if (!is.finite(fold_mse)) { # Check for NA, NaN, Inf MSE
          message(sprintf("walk_forward_cv_nbar: Fold %d resulted in non-finite MSE. Skipping fold.", fold_counter))
          return("skip_fold")
      }

      if (fold_mse > bad_fold_threshold) {
        message(sprintf("walk_forward_cv_nbar: Fold %d failed MSE check (MSE=%.4f > Threshold=%.4f). Stopping CV.",
                        fold_counter, fold_mse, bad_fold_threshold))
        return("stop_cv") # Signal to stop entire CV
      }

      # --- Store Fold Results ---
      # Use <<- to assign to fold_data outside the tryCatch scope
      fold_data <<- tibble(
        fold = fold_counter,
        train_end = current_train_end,
        date = test_fold$date,
        horizon = 1:nrow(test_fold),
        actual = actual_values,
        forecast = pred
        # Optionally add fold_mse here if needed later:
        # fold_mse = fold_mse
      )
      "success" # Signal success for this fold

    }, error = function(e) {
      # Catch errors *within* a specific fold's processing
      message("walk_forward_cv_nbar: Error processing fold ", fold_counter, ": ", e$message)
      # Decide whether an error should skip the fold or stop the whole CV
      # Defaulting to skip here, but could return "stop_cv" if errors are critical
      return("skip_fold")
    }) # End tryCatch

    # --- Process fold status ---
    if (identical(fold_status, "stop_cv")) {
      return(NULL) # Stop processing and signal failure to fit_and_cv
    } else if (identical(fold_status, "success") && !is.null(fold_data)) {
      cv_results[[as.character(fold_counter)]] <- fold_data # Use fold_counter as name
      message("walk_forward_cv_nbar: Finished processing fold ", fold_counter, " (MSE: ", sprintf("%.4f", fold_mse), ")")
    } else {
      # Fold was skipped (due to error, not enough data, or failed fit/pred)
      # Message already printed within tryCatch or the initial check
    }

    # --- Update for next iteration ---
    # Move to the next potential training end date
    current_train_end <- current_train_end + days(test_days)
    fold_counter <- fold_counter + 1

  } # End while loop

  # --- Final Return ---
  if (length(cv_results) > 0) {
    # Combine results from all *successful* folds
    return(bind_rows(cv_results))
  } else {
    message("walk_forward_cv_nbar: No successful folds completed.")
    return(NULL) # Signal failure if no folds worked
  }
}


## ===== Auxiliary Functions =====

fill_gaps <- function(df_daily) {
  df_daily %>%
    mutate(date = as.Date(created_at)) %>% 
    # complete missing weeks from first to last date; fill outcome with 0 (use NA in fill if desired)
    complete(date = seq.Date(min(date), max(date), by = "day"),
             fill = list(vegan_outcome = 0))
}

# Split data into train and test sets
split_data <- function(df, train_frac) {
  
  # Sort by date and split by time
  unique_dates <- sort(unique(df$date))
  cut_date <- unique_dates[floor(length(unique_dates) * train_frac)]
  train <- df %>% dplyr::filter(date <= cut_date)
  test  <- df %>% dplyr::filter(date > cut_date)
  list(train = train, test = test)
}

# Scale variables
standardize_data <- function(df) {
  df %>% mutate(
    meat_window_avg       = as.numeric(scale(meat_window_avg)[,1]),
    vegetarian_window_avg = as.numeric(scale(vegetarian_window_avg)[,1]),
    vegan_window_avg      = as.numeric(scale(vegan_window_avg)[,1]))
}

# Preprocess predictors
process_predictors <- function(df) {
  df %>% mutate(
    date = as.Date(created_at),
    day_of_week_cat = as.factor(day_of_week_cat#, 
                             #levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
                             ),
    season = as.factor(season#, 
                    #levels = c("Spring", "Summer", "Autumn", "Winter")
                    ),
    month = lubridate::month(created_at),
    month_cat = as.factor(month_cat#, 
                       #levels = month.abb
                       ),
    year = lubridate::year(created_at),
    year_cat = as.factor(year_cat),
    year_num = as.numeric(year)
  )
}

# Fits and returns a NB model with given outcome and predictors
fit_nb_model <- function(df, outcome, predictors) {
  formula_str <- paste(outcome, "~", paste(predictors, collapse = " + "))
  MASS::glm.nb(as.formula(formula_str), data = df)
}

# Generate fitted values for training data
training_predictions <- function(train_df, model) {
  fitted_vals <- NULL
  if (inherits(model, "tsglm")) {fitted_vals <- fitted(model)}
  else {fitted_vals <- predict(model, newdata = train_df, type = "response")}
  if (length(fitted_vals) < nrow(train_df)) {fitted_vals <- c(rep(NA, nrow(train_df) - length(fitted_vals)), fitted_vals)} else {fitted_vals <- fitted_vals[1:nrow(train_df)]}
  fitted_vals
}

# Append to training dataframe and returns dataframe
append_train_pred <- function(train_df, model) {
  train_df %>% mutate(pred = training_predictions(train_df, model))
}

# Appends to testing dataframe and returns dataframe
append_test_pred <- function(test_df, model, outcome, predictors) {
  test_df %>% mutate(pred = rolling_forecast_nbar(test_df, model, outcome, predictors))
}

# Aggregate daily data to weekly sums
agg_weekly <- function(df, outcome) {
  df %>% 
    mutate(week = floor_date(created_at, unit = "week")) %>% 
    group_by(week) %>% 
    summarise(obs  = sum(!!sym(outcome)),
              pred = ifelse(exists("pred"), sum(pred, na.rm = TRUE), NA_real_)) %>% 
    ungroup()
}

# Visualizes diagnostic plots: ACF of train residuals, PACF of train residuals, weekly train and test obs vs pred
diag_plots <- function(loc, train_weekly, test_weekly, ar_label, mean_label) {
  
  # Append residuals
  train_resid_df <- train_weekly %>% 
    mutate(resid = obs - pred, 
           week = as.Date(week)
           ) %>% 
    as_tsibble(index = week)
  test_resid_df <- test_weekly %>% 
    mutate(resid = obs - pred, 
           week = as.Date(week)
           ) %>% 
    as_tsibble(index = week)
  
  # ACF plot using ggacf
  p_acf <- train_resid_df %>% ACF(resid, lag_max=30) %>% autoplot() + ggtitle("ACF of train residuals") + theme_minimal()
  
  # PACF plot using ggpacf
  p_pacf <- train_resid_df %>% PACF(resid, lag_max=30) %>% autoplot() + ggtitle("PACF of train residuals") + xlim(0,30) + theme_minimal()
  
  # Plot of train residuals
  p_train_res <- ggplot(train_resid_df, aes(sample = resid)) +
    geom_point(aes(x=pred, y=resid), alpha=0.3) +
    ggtitle("Plot of train residuals") +
    theme_minimal()
  
  # Plot of test residuals
  p_test_res <- ggplot(test_resid_df, aes(sample = resid)) +
    geom_point(aes(x=pred, y=resid), alpha=0.3) +
    ggtitle("Plot of test residuals") +
    theme_minimal()
  
  # Train weekly obs vs pred plot
  p_train <- ggplot(train_weekly, aes(x = week)) +
    geom_line(aes(y = obs, color = "obs")) +
    geom_line(aes(y = pred, color = "pred")) +
    ggtitle("Train weekly: obs vs pred") +
    labs(x = "Day", y = "Count") +
    scale_color_manual(values = c("obs" = "blue", "pred" = "red")) +
    #ylim(0, 90) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Test weekly obs vs pred plot
  p_test <- ggplot(test_weekly, aes(x = week)) +
    geom_line(aes(y = obs, color = "obs")) +
    geom_line(aes(y = pred, color = "pred")) +
    ggtitle("Test weekly: obs vs pred") +
    labs(x = "Day", y = "Count") +
    scale_color_manual(values = c("obs" = "blue", "pred" = "red")) +
    #ylim(0, 90) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Arrange the plots together with a header that shows location and AR lags.
  gridExtra::grid.arrange(
    gridExtra::arrangeGrob(p_acf, p_train_res, p_train, p_pacf, p_test_res, p_test, ncol = 3),
    top = grid::textGrob(paste("Diagnostic Plots: Restaurant", loc, "- AR lags:", ar_label, "- Mean lags:", mean_label), 
                         gp = grid::gpar(fontsize = 16, fontface = "bold"))
  )
}

plot_train_test_side_by_side <- function(loc, train_weekly, test_weekly, ar_label, mean_label) {
  
  # Create the plot
  p <- ggplot() +
    # Plot observations from both training and testing (same color)
    geom_line(data = train_weekly, aes(x = week, y = obs), color = "blue", size = 1) +
    geom_line(data = test_weekly, aes(x = week, y = obs), color = "blue", size = 1) +
    # Plot training predictions (red) and testing predictions (green)
    geom_line(data = train_weekly, aes(x = week, y = pred, color = "Train Prediction"), size = 1) +
    geom_line(data = test_weekly, aes(x = week, y = pred, color = "Test Prediction"), size = 1) +
    labs(title = paste("Training and Testing Predictions: Restaurant", loc, "- AR lags:", ar_label, "- Mean lags:", mean_label),
         x = "Day", y = "Count", color = "Legend") +
    theme_minimal() +
    scale_color_manual(values = c("Train Prediction" = "red", "Test Prediction" = "orange"))
  
  p
}

# Primary function for putting together all visualizations to later be used in a Shiny app
process_models <- function(df, loc, outcome, predictors, model_type="nb", ar_lags=c(), mean_lags=c(), sample=FALSE, standardize=TRUE, train_frac=0.5) {
  
  # model_dir <- file.path("modeling_results")
  # model_file <- file.path(model_dir, paste0(loc, "_model.rds"))
  # # Check if the model file exists
  # if (file.exists(model_file)) {
  #   message("Loading existing model from: ", model_file)
  #   model <- readRDS(model_file)
  # } else {
  #   message("No existing model found. Training a new model...")
  
  # Filter to restaurant
  df <- df %>% dplyr::filter(location_id == loc)
  
  # Fill gaps
  df <- fill_gaps(df)
  
  # Set default sample size to entire dataset
  if (sample) {df <- df %>% slice_sample(n = sample)}
  
  # Standardize if opted for
  if (standardize) {df <- df %>% standardize_data()}
  
  # Train test split
  splits <- split_data(df, train_frac)
  train_df <- splits$train
  test_df  <- splits$test
  
  # Fit the model on training data:
  model <- NULL
  if (model_type == "nb") {
    model <- fit_nb_model(train_df, outcome, predictors)
  }
  if (model_type == "nbar") {
    model <- fit_nbar_model(train_df, outcome, predictors, ar_lags, mean_lags)
  }
  
  # Append predictions to training and testing data
  train_df <- append_train_pred(train_df, model)
  test_df  <- append_test_pred(test_df, model, outcome, predictors)
  
  # Aggregate to weekly sums for train and test (use dates from original df)
  train_weekly <- agg_weekly(train_df, outcome)
  test_weekly  <- agg_weekly(test_df, outcome)
  
  glimpse(train_weekly)
  
  # Show diagnostic plots: ACF, train weekly obs vs pred, test weekly obs vs pred
  diag_grob <- diag_plots(loc, train_weekly, test_weekly, paste(ar_lags, collapse = ","), paste(mean_lags, collapse = ","))
  
  # Show prediction plot, train and test combined
  pred_plot <- plot_train_test_side_by_side(loc, train_weekly, test_weekly, paste(ar_lags, collapse = ","), paste(mean_lags, collapse = ","))
  
  return(list(model = model, diag_plot = diag_grob, pred_plot = pred_plot))
}

# Aggregates the forecasts from cross-validation using given metrics
aggregate_cv_results <- function(cv) {
  cv %>%
    group_by(fold) %>%
    # Calculate fold-specific metrics
    summarize(mae = mean(abs(actual - forecast)),
              mse = mean((actual - forecast)^2),
              .groups = 'drop') %>% # Ensure ungrouping after summarizing by fold
    # Calculate overall average metrics across successful folds
    summarize(mae = mean(mae, na.rm = TRUE), # Use na.rm just in case
              mse = mean(mse, na.rm = TRUE)) %>%
    pull('mse') %>%
    identity()
}

