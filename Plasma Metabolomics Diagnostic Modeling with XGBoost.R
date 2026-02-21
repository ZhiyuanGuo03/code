# ============================================================
# Plasma Metabolomics Diagnostic Modeling with XGBoost
# - Preprocessing
# - 10-fold CV on training set
# - Final model training
# - Internal test + external validation (Suzhou, Ningbo)
# - ROC curve plots (CV / train / test / external)
# - Learning curve analysis (clinical / metabolites / combined)
# ============================================================

# ---------------------------
# Load required packages
# ---------------------------
library(pROC)
library(caret)
library(ggplot2)
library(dplyr)
library(xgboost)
library(ggrepel)
library(RColorBrewer)

set.seed(42)

# ---------------------------
# Step 1: Load data
# ---------------------------
discovery <- read.csv("haerbin.csv", stringsAsFactors = FALSE)
external1 <- read.csv("suzhou.csv", stringsAsFactors = FALSE)
external2 <- read.csv("ningbo.csv", stringsAsFactors = FALSE)

# ---------------------------
# Step 2: Data preprocessing
# ---------------------------
preprocess_data <- function(df) {
  # outcome: 0 = control, 1 = case
  df$outcome <- factor(df$outcome, levels = c(0, 1))
  
  # Remove sample_id if present
  if ("sample_id" %in% names(df)) df <- df %>% select(-sample_id)
  
  return(df)
}

discovery <- preprocess_data(discovery)
external1 <- preprocess_data(external1)
external2 <- preprocess_data(external2)

# ---------------------------
# Step 3: Define feature sets
# ---------------------------
clinical_cols <- c("sex", "age")
metab_cols <- names(discovery)[!(names(discovery) %in% c("outcome", clinical_cols))]

feature_sets <- list(
  clinical    = clinical_cols,
  metabolites = metab_cols,
  combined    = c(clinical_cols, metab_cols)
)

# ---------------------------
# Step 4: Split discovery cohort into training and test sets
# ---------------------------
set.seed(42)
split_index <- createDataPartition(discovery$outcome, p = 0.8, list = FALSE)
train_data <- discovery[split_index, ]
test_data  <- discovery[-split_index, ]

cat("Dataset split summary:\n")
cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
cat("External validation set 1 (Suzhou) size:", nrow(external1), "\n")
cat("External validation set 2 (Ningbo) size:", nrow(external2), "\n")

# ---------------------------
# Step 5: Prepare data for XGBoost
# ---------------------------
# NOTE: xgboost expects numeric matrix input.
# If 'sex' is a character/factor, we convert it into numeric codes.
make_numeric_matrix <- function(df, features) {
  X <- df[, features, drop = FALSE]
  
  # Convert non-numeric columns (e.g., sex) to numeric codes
  for (nm in names(X)) {
    if (!is.numeric(X[[nm]])) {
      X[[nm]] <- as.numeric(as.factor(X[[nm]]))
    }
  }
  as.matrix(X)
}

prepare_xgb_data <- function(df, features) {
  X <- make_numeric_matrix(df, features)
  y <- as.numeric(df$outcome) - 1
  return(list(X = X, y = y))
}

# ---------------------------
# Step 6: 10-fold cross-validation
# ---------------------------
perform_cv <- function(data, features, n_folds = 10, n_rounds = 100) {
  folds <- createFolds(data$outcome, k = n_folds)
  
  cv_results <- list(
    auc = numeric(n_folds),
    accuracy = numeric(n_folds),
    sensitivity = numeric(n_folds),
    specificity = numeric(n_folds),
    probs = vector("list", n_folds),
    true_labels = vector("list", n_folds)
  )
  
  for (fold in 1:n_folds) {
    val_idx   <- folds[[fold]]
    train_idx <- setdiff(1:nrow(data), val_idx)
    
    cv_train <- data[train_idx, ]
    cv_val   <- data[val_idx, ]
    
    # Skip fold if validation has only one class (rare but possible)
    if (length(unique(cv_val$outcome)) < 2) next
    
    train_prep <- prepare_xgb_data(cv_train, features)
    val_prep   <- prepare_xgb_data(cv_val, features)
    
    dtrain <- xgb.DMatrix(data = train_prep$X, label = train_prep$y)
    dval   <- xgb.DMatrix(data = val_prep$X, label = val_prep$y)
    
    model <- xgb.train(
      params = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = 0.1,
        max_depth = 3,
        subsample = 0.8,
        colsample_bytree = 0.8
      ),
      data = dtrain,
      nrounds = n_rounds,
      verbose = 0
    )
    
    probs <- predict(model, dval)
    preds <- ifelse(probs > 0.5, 1, 0)
    
    conf_matrix <- confusionMatrix(
      factor(preds, levels = c(0, 1)),
      factor(val_prep$y, levels = c(0, 1))
    )
    
    roc_obj <- roc(cv_val$outcome, probs, levels = c(0, 1), direction = "<", quiet = TRUE)
    
    cv_results$auc[fold]         <- as.numeric(auc(roc_obj))
    cv_results$accuracy[fold]    <- as.numeric(conf_matrix$overall["Accuracy"])
    cv_results$sensitivity[fold] <- as.numeric(conf_matrix$byClass["Sensitivity"])
    cv_results$specificity[fold] <- as.numeric(conf_matrix$byClass["Specificity"])
    cv_results$probs[[fold]]     <- probs
    cv_results$true_labels[[fold]] <- as.numeric(as.character(cv_val$outcome))
  }
  
  return(cv_results)
}

# ---------------------------
# Step 7: Run 10-fold CV on the training set
# ---------------------------
cat("\n=== 10-Fold Cross-Validation on Training Set ===\n")

cv_results_list <- list()

for (set_name in names(feature_sets)) {
  cat("\n---", set_name, "features ---\n")
  
  cv_results <- perform_cv(train_data, feature_sets[[set_name]])
  cv_results_list[[set_name]] <- cv_results
  
  avg_auc  <- mean(cv_results$auc, na.rm = TRUE)
  avg_acc  <- mean(cv_results$accuracy, na.rm = TRUE)
  avg_sens <- mean(cv_results$sensitivity, na.rm = TRUE)
  avg_spec <- mean(cv_results$specificity, na.rm = TRUE)
  
  cat("Mean AUC:", round(avg_auc, 3), "\n")
  cat("Mean Accuracy:", round(avg_acc, 3), "\n")
  cat("Mean Sensitivity:", round(avg_sens, 3), "\n")
  cat("Mean Specificity:", round(avg_spec, 3), "\n")
  cat("AUC SD:", round(sd(cv_results$auc, na.rm = TRUE), 3), "\n")
}

# ---------------------------
# Step 8: Train final models on the full training set
# ---------------------------
cat("\n=== Training Final Models on Full Training Set ===\n")

final_models <- list()

for (set_name in names(feature_sets)) {
  cat("\nTraining", set_name, "model...\n")
  
  train_prep <- prepare_xgb_data(train_data, feature_sets[[set_name]])
  dtrain <- xgb.DMatrix(data = train_prep$X, label = train_prep$y)
  
  final_model <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.1,
      max_depth = 3,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    data = dtrain,
    nrounds = 100,
    verbose = 0
  )
  
  final_models[[set_name]] <- final_model
}

# ---------------------------
# Step 9: Model evaluation function
# ---------------------------
evaluate_model <- function(model, data, features) {
  data_prep <- prepare_xgb_data(data, features)
  ddata <- xgb.DMatrix(data = data_prep$X)
  
  probs <- predict(model, ddata)
  preds <- ifelse(probs > 0.5, 1, 0)
  
  conf_matrix <- confusionMatrix(
    factor(preds, levels = c(0, 1)),
    factor(data_prep$y, levels = c(0, 1))
  )
  
  roc_obj <- roc(data$outcome, probs, levels = c(0, 1), direction = "<", quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  return(list(
    roc = roc_obj,
    auc = auc_val,
    accuracy = as.numeric(conf_matrix$overall["Accuracy"]),
    sensitivity = as.numeric(conf_matrix$byClass["Sensitivity"]),
    specificity = as.numeric(conf_matrix$byClass["Specificity"]),
    ppv = as.numeric(conf_matrix$byClass["Pos Pred Value"]),
    npv = as.numeric(conf_matrix$byClass["Neg Pred Value"]),
    probs = probs,
    labels = data_prep$y
  ))
}

# ---------------------------
# Step 10: Evaluate models on train/test/external sets
# ---------------------------
cat("\n=== Model Evaluation ===\n")

evaluation_results <- list()

for (set_name in names(feature_sets)) {
  cat("\n---", set_name, "features ---\n")
  
  model <- final_models[[set_name]]
  features <- feature_sets[[set_name]]
  
  train_results <- evaluate_model(model, train_data, features)
  test_results  <- evaluate_model(model, test_data, features)
  ext1_results  <- evaluate_model(model, external1, features)  # Suzhou
  ext2_results  <- evaluate_model(model, external2, features)  # Ningbo
  
  cat("Train AUC:",  round(train_results$auc, 3), "\n")
  cat("Test AUC:",   round(test_results$auc, 3), "\n")
  cat("Suzhou AUC:", round(ext1_results$auc, 3), "\n")
  cat("Ningbo AUC:", round(ext2_results$auc, 3), "\n")
  
  evaluation_results[[set_name]] <- list(
    train  = train_results,
    test   = test_results,
    suzhou = ext1_results,
    ningbo = ext2_results
  )
}

# ---------------------------
# Step 11: ROC curve plotting
# ---------------------------
cat("\n=== Generating and Saving ROC Curves ===\n")

roc_to_df <- function(roc_obj) {
  data.frame(
    sensitivity = rev(roc_obj$sensitivities),
    one_minus_specificity = 1 - rev(roc_obj$specificities)
  )
}

create_roc_plot <- function(results_list, title, out_file = NULL, colors = NULL) {
  plot_data <- data.frame()
  
  for (set_name in names(results_list)) {
    roc_obj <- results_list[[set_name]]$roc
    auc_val <- results_list[[set_name]]$auc
    
    roc_df <- roc_to_df(roc_obj)
    roc_df$Set <- set_name
    roc_df$AUC <- auc_val
    
    plot_data <- rbind(plot_data, roc_df)
  }
  
  # Sample a few points per curve for optional point markers
  plot_data <- do.call(rbind, lapply(split(plot_data, plot_data$Set), function(df) {
    n <- nrow(df)
    k <- min(8, n)
    idx <- unique(round(seq(1, n, length.out = k)))
    df$point_flag <- FALSE
    df$point_flag[idx] <- TRUE
    df
  }))
  
  if (is.null(colors)) {
    uniq_sets <- unique(plot_data$Set)
    cols <- brewer.pal(max(3, length(uniq_sets)), "Set2")
    colors <- setNames(cols[1:length(uniq_sets)], uniq_sets)
  }
  
  p <- ggplot(plot_data, aes(x = one_minus_specificity, y = sensitivity, color = Set, group = Set)) +
    geom_line(size = 1.4, alpha = 0.95) +
    geom_point(data = subset(plot_data, point_flag), size = 1.8, alpha = 0.9) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_manual(values = colors) +
    labs(title = title, x = "1 - Specificity", y = "Sensitivity", color = "Model") +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.title = element_text(face = "bold")
    ) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1))
  
  # AUC labels
  auc_labels <- sapply(names(results_list), function(nm) sprintf("%s: AUC=%.3f", nm, results_list[[nm]]$auc))
  xs <- rep(0.62, length(auc_labels))
  ys <- seq(0.35, 0.15, length.out = length(auc_labels))
  
  p <- p + annotate(
    "text", x = xs, y = ys, label = auc_labels,
    hjust = 0, color = colors[names(results_list)], size = 4
  )
  
  if (!is.null(out_file)) ggsave(out_file, p, width = 7.5, height = 6.5, dpi = 300)
  return(p)
}

# Build CV ROC list by merging fold predictions
cv_roc_list <- list()
for (set_name in names(cv_results_list)) {
  all_probs  <- unlist(cv_results_list[[set_name]]$probs)
  all_labels <- unlist(cv_results_list[[set_name]]$true_labels)
  
  if (length(all_probs) > 0) {
    roc_obj <- roc(all_labels, all_probs, levels = c(0, 1), direction = "<", quiet = TRUE)
    cv_roc_list[[set_name]] <- list(roc = roc_obj, auc = as.numeric(auc(roc_obj)))
  }
}

p_cv <- create_roc_plot(cv_roc_list, "ROC Curves - 10-Fold Cross Validation (Training Set)", out_file = "cv_roc_curves.png")
print(p_cv)

train_roc_list <- lapply(evaluation_results, function(x) list(roc = x$train$roc, auc = x$train$auc))
p_train <- create_roc_plot(train_roc_list, "ROC Curves - Discovery Training Set", out_file = "train_roc_curves.png")
print(p_train)

test_roc_list <- lapply(evaluation_results, function(x) list(roc = x$test$roc, auc = x$test$auc))
p_test <- create_roc_plot(test_roc_list, "ROC Curves - Test Set", out_file = "test_roc_curves.png")
print(p_test)

suzhou_roc_list <- lapply(evaluation_results, function(x) list(roc = x$suzhou$roc, auc = x$suzhou$auc))
p_suzhou <- create_roc_plot(suzhou_roc_list, "ROC Curves - Suzhou External Validation", out_file = "suzhou_roc_curves.png")
print(p_suzhou)

ningbo_roc_list <- lapply(evaluation_results, function(x) list(roc = x$ningbo$roc, auc = x$ningbo$auc))
p_ningbo <- create_roc_plot(ningbo_roc_list, "ROC Curves - Ningbo External Validation", out_file = "ningbo_roc_curves.png")
print(p_ningbo)

# ============================================================
# Learning Curve Analysis
# NOTE: Your original script called run_learning_curve() but did not define it.
# The function below implements what your calls expect.
# ============================================================

cat("\n=== Complete Learning Curve Analysis for All Feature Sets ===\n")

run_learning_curve <- function(train_source,
                               eval_data,
                               feature_set_name,
                               proportions = seq(0.05, 1.0, by = 0.05),
                               n_repeats = 5,
                               min_n = 20,
                               nrounds = 100) {
  
  # Determine features for this model
  feats <- feature_sets[[feature_set_name]]
  if (is.null(feats)) stop("Unknown feature_set_name: ", feature_set_name)
  
  # Prepare fixed evaluation matrix once
  eval_prep <- prepare_xgb_data(eval_data, feats)
  deval <- xgb.DMatrix(data = eval_prep$X, label = eval_prep$y)
  
  out <- data.frame(
    proportion = proportions,
    n_samples = NA_integer_,
    mean_auc = NA_real_,
    sd_auc = NA_real_
  )
  
  N <- nrow(train_source)
  
  for (i in seq_along(proportions)) {
    p <- proportions[i]
    n_sub <- max(min_n, floor(N * p))
    out$n_samples[i] <- n_sub
    
    aucs <- numeric(n_repeats)
    
    for (r in 1:n_repeats) {
      set.seed(42 + i * 100 + r)
      idx <- sample(1:N, size = n_sub, replace = FALSE)
      sub_train <- train_source[idx, ]
      
      train_prep <- prepare_xgb_data(sub_train, feats)
      dtrain <- xgb.DMatrix(data = train_prep$X, label = train_prep$y)
      
      model <- xgb.train(
        params = list(
          objective = "binary:logistic",
          eval_metric = "auc",
          eta = 0.1,
          max_depth = 3,
          subsample = 0.8,
          colsample_bytree = 0.8
        ),
        data = dtrain,
        nrounds = nrounds,
        verbose = 0
      )
      
      probs <- predict(model, deval)
      roc_obj <- roc(eval_prep$y, probs, levels = c(0, 1), direction = "<", quiet = TRUE)
      aucs[r] <- as.numeric(auc(roc_obj))
    }
    
    out$mean_auc[i] <- mean(aucs, na.rm = TRUE)
    out$sd_auc[i]   <- sd(aucs, na.rm = TRUE)
  }
  
  return(out)
}

feature_sets_names <- c("clinical", "metabolites", "combined")

eval_sets <- list(
  internal = list(data = test_data, name = "Internal test (Harbin)"),
  suzhou   = list(data = external1, name = "External Suzhou"),
  ningbo   = list(data = external2, name = "External Ningbo")
)

lc_results <- list()

for (feat in feature_sets_names) {
  cat(sprintf("\nRunning learning curves for %s features...\n", feat))
  lc_results[[feat]] <- list()
  
  for (set_key in names(eval_sets)) {
    cat(sprintf("  -> %s\n", eval_sets[[set_key]]$name))
    
    lc_results[[feat]][[set_key]] <- run_learning_curve(
      train_source     = train_data,
      eval_data        = eval_sets[[set_key]]$data,
      feature_set_name = feat,
      proportions      = seq(0.05, 1.0, by = 0.05),
      n_repeats        = 5,
      min_n            = 20
    ) %>% mutate(Eval_set = eval_sets[[set_key]]$name)
  }
}

cat("\n=== Generating Learning Curve Plots ===\n")

for (feat in feature_sets_names) {
  plot_data <- bind_rows(lc_results[[feat]][["internal"]],
                         lc_results[[feat]][["suzhou"]],
                         lc_results[[feat]][["ningbo"]])
  
  p <- ggplot(plot_data, aes(x = n_samples, y = mean_auc, color = Eval_set, fill = Eval_set)) +
    geom_line(linewidth = 1.1) +
    geom_ribbon(aes(ymin = mean_auc - sd_auc, ymax = mean_auc + sd_auc),
                alpha = 0.12, color = NA) +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1") +
    scale_y_continuous(limits = c(0.4, 1.05), breaks = seq(0.4, 1.05, by = 0.05)) +
    labs(
      title    = paste("Learning Curves -", toupper(feat), "Model"),
      subtitle = "AUC on different evaluation sets vs. training sample size",
      x        = "Number of training samples (from discovery cohort)",
      y        = "AUC (mean ± SD)",
      caption  = "Repeated 5 times per point • Fixed evaluation sets"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "bottom",
      plot.title      = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle   = element_text(hjust = 0.5)
    )
  
  print(p)
  
  filename <- paste0("learning_curve_", feat, "_compare.png")
  ggsave(filename, p, width = 9, height = 6.5, dpi = 300)
  cat(sprintf("Saved: %s\n", filename))
}

cat("\nAll learning curve plots have been generated and saved.\n")
