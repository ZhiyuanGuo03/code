# ==============================
# LASSO Feature Selection (English)
# ==============================

library(tidyverse)
library(broom)
library(glmnet)

set.seed(66666)

# ------------------------------
# Input files (use relative paths)
# ------------------------------
# X: expression matrix (rows = samples, columns = metabolites/features)
#    The first column in the CSV should be sample IDs (used as row names).
x_file <- "data/expression_matrix.csv"

# y: class labels (binary outcome)
#    The first column in the CSV should be sample IDs (used as row names).
#    The label should be 0/1 (0 = control, 1 = ischemic stroke).
y_file <- "data/class_labels.csv"

# ------------------------------
# Load data
# ------------------------------
x_df <- read.csv(x_file, row.names = 1, check.names = FALSE)
y_df <- read.csv(y_file, row.names = 1, check.names = FALSE)

# Convert to matrices for glmnet
x <- as.matrix(x_df)
y <- as.matrix(y_df)

# Ensure y is a numeric vector (0/1) for binomial modeling
# If y has one column, convert it to a vector
if (ncol(y) == 1) {
  y <- as.numeric(y[, 1])
} else {
  stop("The class label file should contain exactly one label column (0/1).")
}

# ------------------------------
# Cross-validated LASSO
# ------------------------------
# Notes:
# - family = "binomial" for binary classification
# - alpha = 1 for LASSO
# - nfolds = 5 to match the current implementation
# - type.measure = "mse" to match your current pipeline
cvfit <- cv.glmnet(
  x = x,
  y = y,
  family = "binomial",
  type.measure = "mse",
  nfolds = 5,
  alpha = 1
)

# Plot cross-validation curve
plot(cvfit)

# Report selected lambdas
lambda_min <- cvfit$lambda.min
lambda_1se <- cvfit$lambda.1se
print(c(lambda_min, lambda_1se))

# ------------------------------
# Fit full LASSO path (binomial)
# ------------------------------
lasso_fit <- glmnet(
  x = x,
  y = y,
  family = "binomial",
  alpha = 1,
  nlambda = 100
)

# Plot coefficient paths
plot(lasso_fit, xvar = "lambda", label = FALSE)

# ------------------------------
# Extract non-zero coefficients at lambda.min
# ------------------------------
coef_min <- coef(lasso_fit, s = lambda_min)

nonzero_idx <- which(coef_min != 0)
active_coef <- coef_min[nonzero_idx]
active_features <- rownames(coef_min)[nonzero_idx]

selected_table <- data.frame(
  Feature = active_features,
  Coefficient = as.numeric(active_coef),
  row.names = NULL
)

# Optional: remove the intercept row if present
selected_table <- selected_table %>%
  filter(Feature != "(Intercept)")

# ------------------------------
# Save selected features
# ------------------------------
out_file <- "results/lasso_selected_features.csv"
dir.create("results", showWarnings = FALSE)

write.csv(selected_table, out_file, row.names = FALSE)

message("Saved LASSO-selected features to: ", out_file)
message("lambda.min = ", lambda_min, " ; lambda.1se = ", lambda_1se)
message("Number of selected features (excluding intercept): ", nrow(selected_table))
