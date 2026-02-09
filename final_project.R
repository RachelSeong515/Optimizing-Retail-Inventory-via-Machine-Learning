# Load Relevant Libraries
library(readxl)
library(dplyr)
library(tidyr)
library(glmnet)
library(rpart)
library(ggplot2)
library(reshape2)
library(randomForest)
library(Matching)
library(caret)
library(scales)
require(lubridate)
require(magrittr)
require(forecast)
library(purrr) 
library(tibble) 
library(doParallel)
library(ranger)

# Import Dataset
df <- read.csv(file.choose())

######### Data Cleaning (STAGE 1)

# Sums of NA variables
colSums(is.na(df)) # 0 NA's

# Check whether there are duplicated rows
sum(duplicated(df)) # None

# Check for any explicit 'None' columns 
sapply(df, function(x) sum(x %in% c("", "NA", "None", "null")))

# Check Data's Skew
hist(df$Weekly_Sales, main="Weekly_Sales Distribution")

# Check for multicollinearity for linear regression (linear, unregularized)
cor(df[sapply(df, is.numeric)], use = "complete.obs") # CPI & Unemployment have moderate multicollinearity

# TESTING DATE COLUMN (1-6)

#1. Parse as day-month-year
df <- df %>%
  mutate(Date_raw = as.character(Date), Date = dmy(Date_raw))

#2. Did parsing work?
sum(is.na(df$Date)) #if 0 then yep

#3 Here i convert the parsed date back to string, and compare it with the original string. This ensures
#  that the order of dmy is correct
sum(format(df$Date, "%d-%m-%Y") != df$Date_raw, na.rm = TRUE)   # should be 0

#4. Checking range of dates to see if it isn't odd
range(year(df$Date), na.rm = TRUE)

#5 Is each store date combo unique?
sum(duplicated(df[, c("Store","Date")]))   # should be 0


#6 add calendar fields to df
df <- df %>%
  mutate(
    Date  = as.Date(Date),                     
    Year  = year(Date),
    Month = month(Date, label = TRUE, abbr = TRUE),
    Week  = isoweek(Date)
  )

#### EDA --------------------------------------------------------

# Total sales over time (overall)

df %>%
  group_by(Date) %>%
  summarise(Total_Sales = sum(Weekly_Sales), .groups = "drop") %>%
  ggplot(aes(Date, Total_Sales)) +
  geom_line(color = "steelblue") +
  scale_y_continuous(labels = label_number(scale_cut = cut_si(" "))) +
  labs(title = "Total Weekly Sales Over Time", y = "Total Sales") +
  theme_minimal()

# Average sales by month
df %>%
  group_by(Month) %>%
  summarise(Avg_Sales = mean(Weekly_Sales), .groups = "drop") %>%
  ggplot(aes(Month, Avg_Sales, group = 1)) +
  geom_line() + geom_point() +
  labs(title = "Average Weekly Sales by Month", y = "Average Sales")

#we can see the data spiking around the same time 

# Boxplot + effect size label
ggplot(df, aes(x = factor(Holiday_Flag, labels = c("No","Yes")), y = Weekly_Sales)) +
  geom_boxplot(outlier.alpha = .25) +
  labs(title = "Holiday vs Non-Holiday Sales", x = "Holiday Week", y = "Weekly Sales")

# Density (holiday vs non-holiday)
ggplot(df, aes(Weekly_Sales, fill = factor(Holiday_Flag, labels = c("No","Yes")))) +
  geom_density(alpha = .35) +
  labs(title = "Sales Density: Holiday vs Non-Holiday", x = "Weekly Sales", fill = "Holiday")

# Temperature vs sales (overall shape)
ggplot(df, aes(Temperature, Weekly_Sales)) +
  geom_point(alpha = .12) +
  geom_smooth(se = FALSE) +
  labs(title = "Sales vs Temperature", y = "Weekly Sales")

# Fuel price, CPI, Unemployment
for (v in c("Fuel_Price","CPI","Unemployment")) {
  print(
    ggplot(df, aes(.data[[v]], Weekly_Sales)) +
      geom_point(alpha = .12) +
      geom_smooth(se = FALSE) +
      labs(title = paste("Sales vs", v), y = "Weekly Sales", x = v)
  )
}

#Temperature x Holiday interaction
ggplot(df, aes(Temperature, Weekly_Sales, color = factor(Holiday_Flag, labels = c("No","Yes")))) +
  geom_point(alpha = .15) +
  geom_smooth(se = FALSE) +
  labs(title = "Sales vs Temperature (Holiday vs Non-Holiday)", color = "Holiday")


# Average by month - how seasonal
df %>%
  group_by(Month) %>%
  summarise(Avg_Sales = mean(Weekly_Sales), .groups = "drop") %>%
  ggplot(aes(Month, Avg_Sales, group = 1)) +
  geom_line() + geom_point() +
  labs(title = "Average Weekly Sales by Month", y = "Average Sales")

#-----------------------------------------------------------------------------------------------------------------------
######### Data split (STAGE 2)

# Set Randomized Seed & Include Unique Dates
set.seed(100)  
unique_dates <- sort(unique(df$Date))

# Split by 80/20 split on unique dates (not store, preventing data leakage)
n_train_dates <- floor(0.8 * length(unique_dates))
train_dates <- unique_dates[1:n_train_dates]          
train_df <- df %>% filter(Date %in% train_dates)
test_with_sales <- df %>% filter(!Date %in% train_dates)
test_df <- test_with_sales %>% dplyr::select(-Weekly_Sales)

# Data Checks
stopifnot(nrow(train_df) > 0)     
stopifnot(nrow(test_df) > 0)      

# Ensure Store Factor levels are consistent
all_stores <- sort(unique(df$Store))
train_df$Store <- factor(train_df$Store, levels = all_stores)
test_df$Store  <- factor(test_df$Store, levels = all_stores)
test_with_sales$Store <- factor(test_with_sales$Store, levels = all_stores)

######### Model Creation (STAGE 3)

# Cross-Validation (CV) Setup 
ctrl <- trainControl(method = "cv", number = 10)
predictors <- c("Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag") # Defining Predictors

#  Linear Regression with Store Fixed Effects (Model 1)
lm_formula <- as.formula(paste("Weekly_Sales ~", paste(c(predictors, "Store"), collapse = " + ")))
lm_fit <- train(lm_formula, data = train_df, method = "lm", trControl = ctrl, preProcess = c("center","scale"))

# CART (Model 2)
ctrl_cart <- trainControl(method = "cv", number = 10, savePredictions = "final")
cp_grid <- expand.grid(cp = seq(0.001, 0.05, by = 0.003))
cart_fit <- train(lm_formula, data = train_df, method = "rpart",
                  trControl = ctrl_cart, tuneGrid = cp_grid,
                  control = rpart.control(minsplit = 20, minbucket = 7),
                  na.action = na.omit)


# Random Forest (Model 3)
cores <- max(1, parallel::detectCores() - 1)
cl <- makePSOCKcluster(cores); registerDoParallel(cl)
ctrl_fast <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

# Creating Tuning Grid
tune_grid <- expand.grid(
  mtry = c(4, 8, 12),                    # number of variables to try at each split
  splitrule = c("variance","extratrees"),# valid split rules for regression
  min.node.size = c(5, 10)               # minimum node size
)

# Fit Model
rf_fit <- train(
  lm_formula,
  data = train_df,
  method = "ranger",
  trControl = ctrl_fast,
  tuneGrid = tune_grid,
  num.trees = 300,
  importance = "impurity",
  respect.unordered.factors = "partition"
)

stopCluster(cl); registerDoSEQ()
print(rf_fit)

# Lasso (Model 4)
glmnet_grid <- expand.grid(alpha = 1, lambda = 10^seq(-5, 2, length = 200))

lasso_fit <- train(
  as.formula(paste0("Weekly_Sales ~ (", paste(predictors, collapse = " + "), ")^2 + Store")),
  data = train_df,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = glmnet_grid,
  preProcess = c("center", "scale")
)

# Visualize Lasso Fitting, Extract coefficients across lambdas
coef_mat <- as.matrix(coef(lasso_fit$finalModel, s = lasso_fit$finalModel$lambda))
coef_df <- data.frame(t(coef_mat))
coef_df$lambda <- lasso_fit$finalModel$lambda
coef_long <- reshape2::melt(coef_df, id.vars = "lambda",
                            variable.name = "Predictor", value.name = "Coefficient") # Long format for ease when using ggplot
coef_long <- subset(coef_long, !grepl("Intercept", Predictor, ignore.case = TRUE))

# Find minimum lambda
best_lambda <- lasso_fit$bestTune$lambda

ggplot(coef_long, aes(x = log10(lambda), y = Coefficient, color = Predictor)) +
  geom_line(size = 0.8) +
  geom_vline(xintercept = log10(best_lambda),
             color = "red", linetype = "dashed") +
  # Label predictors at smallest lambda
  geom_text(
    data = coef_long |> dplyr::filter(lambda == min(lambda)),
    aes(label = Predictor),
    hjust = -1.5, size = 3, show.legend = FALSE
  ) +
  labs(title = "Lasso Coefficient Shrinkage",
       x = "log10(Lambda)",
       y = "Coefficient") +
  theme_minimal()

# Post-Lasso (Model 5)
x <- model.matrix(~ . - 1, data = train_df[, predictors])
y <- train_df$Weekly_Sales
cv_lasso <- cv.glmnet(x, y, alpha = 1, nfolds = 10)
coefs <- coef(cv_lasso, s = "lambda.min")
selected_vars <- setdiff(rownames(coefs)[which(as.numeric(coefs) != 0)], "(Intercept)")
selected_vars <- intersect(selected_vars, predictors)

# Choosing Predictors from Lasso (Previous 5)
post_vars <- if (length(selected_vars) == 0) predictors else selected_vars
post_formula <- as.formula(paste0("Weekly_Sales ~ (", paste(post_vars, collapse = " + "), ")^2 + Store"))

post_lasso_fit <- train(
  post_formula,
  data = train_df,
  method = "lm",
  trControl = ctrl,
  preProcess = c("center", "scale")
)

# Visualize Post-Lasso coefficients
post_coefs <- coef(post_lasso_fit$finalModel)
post_coef_df <- data.frame(
  Predictor = names(post_coefs),
  Coefficient = as.numeric(post_coefs)
)
post_coef_df <- subset(post_coef_df, !grepl("Intercept", Predictor, ignore.case = TRUE))

ggplot(post_coef_df, aes(x = reorder(Predictor, Coefficient), y = Coefficient, fill = Coefficient)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Post-Lasso Coefficients (OLS Refit)",
    x = "Predictor",
    y = "Coefficient"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


######### Model Analysis (STAGE 4)

# Compare Model RMSEs
get_cv_rmse <- function(m) {
  if (!is.null(m$results$RMSE)) min(m$results$RMSE) else NA
}

cv_rmses <- data.frame(
  Model = c("Linear", "CART", "Random Forest", "Lasso", "Post-Lasso"),
  CV_RMSE = c(get_cv_rmse(lm_fit),
              get_cv_rmse(cart_fit),
              get_cv_rmse(rf_fit),
              get_cv_rmse(lasso_fit),
              get_cv_rmse(post_lasso_fit))
)
print(cv_rmses)

# Best Model Selected
best_model <- cv_rmses$Model[which.min(cv_rmses$CV_RMSE)]
cat("Best model by CV RMSE:", best_model, "\n")

# Making prediction for each model
pred_lm       <- predict(lm_fit, newdata = test_df)
pred_cart     <- predict(cart_fit, newdata = test_df)
pred_rf       <- predict(rf_fit, newdata = test_df)
pred_lasso    <- predict(lasso_fit, newdata = test_df)
pred_plasso   <- predict(post_lasso_fit, newdata = test_df)

# Make final predictions according to the best model
final_preds <- switch(best_model,
                      "Linear" = pred_lm,
                      "CART" = pred_cart,
                      "Random Forest" = pred_rf,
                      "Lasso" = pred_lasso,
                      "Post-Lasso" = pred_plasso,
                      default = pred_rf) # Default Random Forest model

# Creating a new column by the name of Predicted_Weekly_Sales
out_preds <- test_df %>%
  mutate(Predicted_Weekly_Sales = round(as.numeric(final_preds), 2))
out_preds$Store <- factor(out_preds$Store, levels = all_stores)
out_preds$Date <- as.Date(out_preds$Date, format = "%d-%m-%Y")

eval_df <- test_with_sales %>%
  left_join(
    out_preds %>% dplyr::select(Store, Date, Predicted_Weekly_Sales),
    by = c("Store", "Date")
  )

if(any(!is.na(eval_df$Weekly_Sales) & !is.na(eval_df$Predicted_Weekly_Sales))) {
  test_rmse <- sqrt(mean((eval_df$Weekly_Sales - eval_df$Predicted_Weekly_Sales)^2, na.rm = TRUE))
  cat("Test RMSE (best model on held-out dates):", round(test_rmse, 2), "\n")
} else {
  cat("No actual Weekly_Sales available in test_with_sales to compute test RMSE.\n")
}

# Saving the test prediction 
out_file <- file.path(getwd(), "Walmart_Test_Predictions.csv")
write.csv(out_preds, out_file, row.names = FALSE)

# Visualize Results comparing the actual vs. predicted OOS Weekly_Sales
ggplot(eval_df, aes(x = Weekly_Sales, y = Predicted_Weekly_Sales)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Actual vs Predicted Weekly Sales",
    x = "Actual Weekly Sales",
    y = "Predicted Weekly Sales"
  ) +
  theme_minimal()

######################## SUPPLEMENTAL MODEL CHECKS

############ Causal Modeling: Holiday Weeks and Sales

### Model 1: Simple Model: just looking at holiday effects and weekly sales
t.test(Weekly_Sales ~ Holiday_Flag, data = train_df)


### Model 2: Freakonomics Model: look at all othe factors while maintaining CIA
lm_causal <- lm(Weekly_Sales ~ Holiday_Flag + Temperature + Fuel_Price + CPI + Unemployment + Store,
                data = train_df)
summary(lm_causal)
# For Model 2: Estimate the propensity score for each observation
ps_model <- glm(Holiday_Flag ~ Temperature + Fuel_Price + CPI + Unemployment,
                family = binomial, data = train_df)
# For Model 2: Perform matching based on the estimated propensity scores
match_res <- Match(Y = train_df$Weekly_Sales,
                   Tr = train_df$Holiday_Flag,
                   X = ps_model$fitted)
# Display the matching summary and estimated treatment effect
summary(match_res)
match_res$est


### Model 3: Double Selection: taking advantage of alhgorithm to only keep the useful ones
# Build x matrix of controls (drop intercept)
x <- model.matrix(~ Temperature + Fuel_Price + CPI + Unemployment + Store, data = train_df)[, -1, drop = FALSE]
y <- train_df$Weekly_Sales
d <- train_df$Holiday_Flag
# helper: support extractor returning column indices of x
support <- function(beta_mat, x_names){
  rn <- rownames(beta_mat)
  nz <- which(as.numeric(beta_mat) != 0)
  # drop intercept if present
  if("(Intercept)" %in% rn){
    nz <- nz[rn[nz] != "(Intercept)"]
  }
  vars <- rn[nz]
  # match to x column positions
  pos <- match(vars, x_names)
  pos <- pos[!is.na(pos)]
  return(pos)
}
num.features <- ncol(x)
num.n <- nrow(x)
# Step 1: Run Lasso for y on x, control(y)
supp1 <- which.max(abs(cor(x, y)))
res <- glm(y ~ x[, supp1])
w <- sd(res$residuals)
lambda.theory1 <- w * qnorm(1 - (0.01 / num.features)) * sqrt(1 / num.n)
# Call Lasso
lassoTheory1 <- glmnet(x, y, lambda = lambda.theory1, standardize = TRUE)
supp1 <- support(lassoTheory1$beta, colnames(x))
length(supp1)
colnames(x[, supp1, drop = FALSE])
# Step 2: Ru Lasso for d on x, control(d)
supp2 <- which.max(abs(cor(x, d)))
res <- glm(d ~ x[, supp2], family = binomial)
w <- sd(res$residuals)
lambda.theory2 <- w * qnorm(1 - (0.01 / num.features)) * sqrt(1 / num.n)
# Call Lasso
lassoTheory2 <- glmnet(x, d, family = "binomial", lambda = lambda.theory2, standardize = TRUE)
supp2 <- support(lassoTheory2$beta, colnames(x))
length(supp2)
colnames(x[, supp2, drop = FALSE])
# Step 3: Run linear regression of Y ~ d + control(d) + control(y)
inthemodel <- unique(c(supp1, supp2))
selectdata <- cbind(d, x[, inthemodel, drop = FALSE])
selectdata <- as.data.frame(as.matrix(selectdata))
dim(selectdata)
# Generating Double Selection results
causal_glm <- glm(y ~ ., data = selectdata)
summary(causal_glm)$coef["d", , drop = FALSE]


#-----------------------------------------------------------------------------------------------------------
# SARIMAX - a time series analysis - we wanted to do an additional seasonal ARIMA with exogenous variables

#Exogenous variables and cutoff from earlier test/train split
xvars  <- c("Holiday_Flag","Temperature","Fuel_Price","CPI","Unemployment")
cutoff <- max(train_dates)

#one store SARIMAX with IS + OOS RMSE (check)
sarimax_one_store <- function(dat, cutoff_date, h_future = 12) {
  tab <- dat %>%
    arrange(Date) %>%
    group_by(Date) %>%
    summarise(across(c(Weekly_Sales, all_of(xvars)), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    complete(Date = seq(min(Date), max(Date), by = "week")) %>%
    arrange(Date)
  
#Fill gaps so y and xreg align
  tab$Weekly_Sales <- forecast::na.interp(tab$Weekly_Sales)
  for (v in xvars) {
    tab[[v]] <- if (v == "Holiday_Flag") tidyr::replace_na(tab[[v]], 0) else forecast::na.interp(tab[[v]])
  }
  
#Train/test boundary - same as earlier models
  tab <- tab %>% mutate(is_train = Date <= cutoff_date)
  n_tr <- sum(tab$is_train); n_te <- sum(!tab$is_train)
  if (n_tr < 20 || n_te < 1) {
    return(list(
      metrics = tibble(RMSE_OOS = NA_real_, RMSE_IS = NA_real_,
                       n_train = n_tr, n_test = n_te, order = NA_character_),
      future_forecast = tibble(Date = as.Date(character()), Predicted_Weekly_Sales = numeric(),
                               Lo80 = numeric(), Hi80 = numeric(), Lo95 = numeric(), Hi95 = numeric())
    ))
  }
  
  y_tr <- ts(tab$Weekly_Sales[tab$is_train], frequency = 52)
  x_tr <- as.matrix(tab[tab$is_train, xvars, drop = FALSE])
  y_te <- tab$Weekly_Sales[!tab$is_train]
  x_te <- as.matrix(tab[!tab$is_train, xvars, drop = FALSE])
  
  fit <- forecast::auto.arima(y_tr, xreg = x_tr, seasonal = TRUE,
                              stepwise = FALSE, approximation = FALSE)
  
# In sample RMSE 
  yhat_tr <- as.numeric(fitted(fit))
  rmse_is <- sqrt(mean((as.numeric(y_tr) - yhat_tr)^2, na.rm = TRUE))
  
# OOS RMSE
  fc_te   <- forecast::forecast(fit, xreg = x_te, h = length(y_te))
  rmse_oos <- sqrt(mean((y_te - as.numeric(fc_te$mean))^2, na.rm = TRUE))
  
#Simple future xreg placeholder FoR Predictions
  last_x   <- as.matrix(tab[nrow(tab), xvars, drop = FALSE])
  x_future <- last_x[rep(1, h_future), , drop = FALSE]
  x_future[, "Holiday_Flag"] <- 0
  fc_future <- forecast::forecast(fit, xreg = x_future, h = h_future)
  
  list(
    metrics = tibble(RMSE_OOS = rmse_oos, RMSE_IS = rmse_is,
                     n_train = n_tr, n_test = n_te, order = capture.output(fit)[1]),
    future_forecast = tibble(
      Date = seq(max(tab$Date) + 7, by = "week", length.out = h_future),
      Predicted_Weekly_Sales = as.numeric(fc_future$mean),
      Lo80 = as.numeric(fc_future$lower[, "80%"]),  Hi80 = as.numeric(fc_future$upper[, "80%"]),
      Lo95 = as.numeric(fc_future$lower[, "95%"]),  Hi95 = as.numeric(fc_future$upper[, "95%"])
    )
  )
}

# Run for all stores
stores <- if (is.factor(df$Store)) levels(df$Store) else unique(df$Store)

sarimax_results <- purrr::map(stores, function(s) {
  out <- sarimax_one_store(dat = dplyr::filter(df, Store == s), cutoff_date = cutoff)
  list(
    metrics = dplyr::mutate(out$metrics, Store = s),
    fc      = dplyr::mutate(out$future_forecast, Store = s)
  )
})

#Bind results
sarimax_metrics <- purrr::map_dfr(sarimax_results, "metrics") %>%
  dplyr::mutate(Store = as.character(Store)) %>%
  # Backward compatibility in case an old run produced "RMSE"
  { if ("RMSE" %in% names(.)) dplyr::rename(., RMSE_OOS = RMSE) else . } %>%
  dplyr::select(Store, RMSE_OOS, RMSE_IS, n_train, n_test, order, dplyr::everything())

sarimax_fc <- purrr::map_dfr(sarimax_results, "fc") %>%
  dplyr::select(Store, Date, dplyr::everything())

#Summary (both IS and OOS)
sarimax_metrics %>%
  summarise(
    Stores = dplyr::n(),
    Mean_RMSE_OOS   = mean(RMSE_OOS, na.rm = TRUE),
    Median_RMSE_OOS = median(RMSE_OOS, na.rm = TRUE),
    Mean_RMSE_IS    = mean(RMSE_IS,  na.rm = TRUE),
    Median_RMSE_IS  = median(RMSE_IS, na.rm = TRUE)
  ) %>% print()

# These are top/bottom stores by OOS RMSE
sarimax_metrics %>%
  arrange(RMSE_OOS) %>%
  dplyr::slice_head(n = 10) %>%
  print(n = 10)

mean(df$Weekly_Sales) #to check % RMSE versus mean

# ------------------------- end ------------------------------------------------
