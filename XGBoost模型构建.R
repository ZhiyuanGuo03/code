# 加载所需的包
library(pROC)
library(caret)
library(ggplot2)
library(dplyr)
library(xgboost)
library(ggrepel)
library(RColorBrewer)
# 设置随机种子
set.seed(42)
# ---------- 第一步：加载数据 ----------
discovery <- read.csv("haerbin.csv", stringsAsFactors = FALSE)
external1 <- read.csv("suzhou.csv", stringsAsFactors = FALSE)
external2 <- read.csv("ningbo.csv", stringsAsFactors = FALSE)
# ---------- 第二步：数据预处理 ----------
preprocess_data <- function(df) {
  df$outcome <- factor(df$outcome, levels = c(0, 1))
  if ("sample_id" %in% names(df)) df <- df %>% select(-sample_id)
  return(df)
}
discovery <- preprocess_data(discovery)
external1 <- preprocess_data(external1)
external2 <- preprocess_data(external2)
# ---------- 第三步：定义特征组 ----------
clinical_cols <- c("sex", "age")
metab_cols <- names(discovery)[!(names(discovery) %in% c("outcome", clinical_cols))]
feature_sets <- list(
  clinical = clinical_cols,
  metabolites = metab_cols,
  combined = c(clinical_cols, metab_cols)
)
# ---------- 第四步：划分发现集为训练集和测试集 ----------
set.seed(42)
split_index <- createDataPartition(discovery$outcome, p = 0.8, list = FALSE)
train_data <- discovery[split_index, ]
test_data <- discovery[-split_index, ]
cat("数据划分结果：\n")
cat(paste("训练集样本数:", nrow(train_data), "\n"))
cat(paste("测试集样本数:", nrow(test_data), "\n"))
cat(paste("外部验证集1（苏州）样本数:", nrow(external1), "\n"))
cat(paste("外部验证集2（宁波）样本数:", nrow(external2), "\n"))
# ---------- 第五步：准备XGBoost数据 ----------
prepare_xgb_data <- function(df, features) {
  X <- as.matrix(df[, features])
  y <- as.numeric(df$outcome) - 1
  return(list(X = X, y = y))
}
# ---------- 第六步：10折交叉验证 ----------
perform_cv <- function(data, features, n_folds = 10, n_rounds = 100) {
  folds <- createFolds(data$outcome, k = n_folds)
  
  cv_results <- list(
    auc = numeric(n_folds),
    accuracy = numeric(n_folds),
    sensitivity = numeric(n_folds),
    specificity = numeric(n_folds),
    probs = list(),
    true_labels = list()
  )
  
  for (fold in 1:n_folds) {
    val_idx <- folds[[fold]]
    train_idx <- setdiff(1:nrow(data), val_idx)
    
    cv_train <- data[train_idx, ]
    cv_val <- data[val_idx, ]
    
    if (length(unique(cv_val$outcome)) < 2) {
      next
    }
    
    train_prep <- prepare_xgb_data(cv_train, features)
    val_prep <- prepare_xgb_data(cv_val, features)
    
    dtrain <- xgb.DMatrix(data = train_prep$X, label = train_prep$y)
    dval <- xgb.DMatrix(data = val_prep$X, label = val_prep$y)
    
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
    
    cv_results$auc[fold] <- auc(roc_obj)
    cv_results$accuracy[fold] <- conf_matrix$overall["Accuracy"]
    cv_results$sensitivity[fold] <- conf_matrix$byClass["Sensitivity"]
    cv_results$specificity[fold] <- conf_matrix$byClass["Specificity"]
    cv_results$probs[[fold]] <- probs
    cv_results$true_labels[[fold]] <- as.numeric(as.character(cv_val$outcome))
  }
  
  return(cv_results)
}
# ---------- 第七步：在训练集上进行10折交叉验证 ----------
cat("\n=== 10-Fold Cross-Validation on Training Set ===\n")
cv_results_list <- list()
for (set_name in names(feature_sets)) {
  cat(paste("\n---", set_name, "Features ---\n"))
  
  cv_results <- perform_cv(train_data, feature_sets[[set_name]])
  cv_results_list[[set_name]] <- cv_results
  
  avg_auc <- mean(cv_results$auc, na.rm = TRUE)
  avg_acc <- mean(cv_results$accuracy, na.rm = TRUE)
  avg_sens <- mean(cv_results$sensitivity, na.rm = TRUE)
  avg_spec <- mean(cv_results$specificity, na.rm = TRUE)
  
  cat(paste("Mean AUC:", round(avg_auc, 3), "\n"))
  cat(paste("Mean Accuracy:", round(avg_acc, 3), "\n"))
  cat(paste("Mean Sensitivity:", round(avg_sens, 3), "\n"))
  cat(paste("Mean Specificity:", round(avg_spec, 3), "\n"))
  cat(paste("AUC SD:", round(sd(cv_results$auc, na.rm = TRUE), 3), "\n"))
}
# ---------- 第八步：训练最终模型 ----------
cat("\n=== Training Final Models on Full Training Set ===\n")
final_models <- list()
for (set_name in names(feature_sets)) {
  cat(paste("\nTraining", set_name, "model...\n"))
  
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
# ---------- 第九步：评估模型 ----------
cat("\n=== Model Evaluation ===\n")
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
  auc_val <- auc(roc_obj)
  
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
evaluation_results <- list()
for (set_name in names(feature_sets)) {
  cat(paste("\n---", set_name, "Features ---\n"))
  
  model <- final_models[[set_name]]
  features <- feature_sets[[set_name]]
  
  # 在训练集上评估（为了绘制训练集ROC）
  train_results <- evaluate_model(model, train_data, features)
  
  # 在测试集上评估
  test_results <- evaluate_model(model, test_data, features)
  
  # 在外部验证集1上评估
  ext1_results <- evaluate_model(model, external1, features)
  
  # 在外部验证集2上评估
  ext2_results <- evaluate_model(model, external2, features)
  
  cat("Train AUC:", round(train_results$auc, 3), "\n")
  cat("Test AUC:", round(test_results$auc, 3), "\n")
  cat("Suzhou AUC:", round(ext1_results$auc, 3), "\n")
  cat("Ningbo AUC:", round(ext2_results$auc, 3), "\n")
  
  evaluation_results[[set_name]] <- list(
    train = train_results,
    test = test_results,
    suzhou = ext1_results,
    ningbo = ext2_results
  )
}
# ---------- 第十步：绘制ROC曲线（使用第二个代码的绘图函数） ----------
cat("\n=== 绘制并保存ROC曲线（使用第二个代码的绘图函数）===\n")
# 辅助函数：将ROC对象转换为数据框
roc_to_df <- function(roc_obj) {
  data.frame(
    sensitivity = rev(roc_obj$sensitivities),
    one_minus_specificity = 1 - rev(roc_obj$specificities)
  )
}
# 主绘图函数
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
  
  # 对每个Set选取几个点用于标注
  plot_data <- do.call(rbind, lapply(split(plot_data, plot_data$Set), function(df) {
    n <- nrow(df)
    k <- min(8, n)
    idx <- unique(round(seq(1, n, length.out = k)))
    df$point_flag <- FALSE
    df$point_flag[idx] <- TRUE
    return(df)
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
    theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.title = element_text(face = "bold")) +
    scale_x_continuous(limits = c(0,1), breaks = seq(0,1,0.1)) +
    scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.1))
  
  # 添加AUC标签
  auc_labels <- sapply(names(results_list), function(nm) sprintf("%s: AUC=%.3f", nm, results_list[[nm]]$auc))
  xs <- rep(0.62, length(auc_labels))
  ys <- seq(0.35, 0.15, length.out = length(auc_labels))
  p <- p + annotate("text", x = xs, y = ys, label = auc_labels, hjust = 0, color = colors[names(results_list)], size = 4)
  
  if (!is.null(out_file)) ggsave(out_file, p, width = 7.5, height = 6.5, dpi = 300)
  return(p)
}
# 准备交叉验证的ROC列表（合并所有折的预测）
cv_roc_list <- list()
for (set_name in names(cv_results_list)) {
  all_probs <- unlist(cv_results_list[[set_name]]$probs)
  all_labels <- unlist(cv_results_list[[set_name]]$true_labels)
  
  if (length(all_probs) > 0) {
    roc_obj <- roc(all_labels, all_probs, levels = c(0, 1), direction = "<", quiet = TRUE)
    cv_roc_list[[set_name]] <- list(
      roc = roc_obj,
      auc = auc(roc_obj)
    )
  }
}
# 绘制交叉验证ROC
p_cv <- create_roc_plot(cv_roc_list, "ROC Curves - 10-Fold Cross Validation (Training Set)", out_file = "cv_roc_curves.png")
print(p_cv)
# 绘制训练集ROC
train_roc_list <- lapply(evaluation_results, function(x) list(roc = x$train$roc, auc = x$train$auc))
p_train <- create_roc_plot(train_roc_list, "ROC Curves - Discovery Training Set", out_file = "train_roc_curves.png")
print(p_train)
# 绘制测试集ROC
test_roc_list <- lapply(evaluation_results, function(x) list(roc = x$test$roc, auc = x$test$auc))
p_test <- create_roc_plot(test_roc_list, "ROC Curves - Test Set", out_file = "test_roc_curves.png")
print(p_test)
# 外部验证ROC
suzhou_roc_list <- lapply(evaluation_results, function(x) list(roc = x$suzhou$roc, auc = x$suzhou$auc))
p_suzhou <- create_roc_plot(suzhou_roc_list, "ROC Curves - Suzhou External Validation", out_file = "suzhou_roc_curves.png")
print(p_suzhou)
ningbo_roc_list <- lapply(evaluation_results, function(x) list(roc = x$ningbo$roc, auc = x$ningbo$auc))
p_ningbo <- create_roc_plot(ningbo_roc_list, "ROC Curves - Ningbo External Validation", out_file = "ningbo_roc_curves.png")
print(p_ningbo)


# 运行三种情况

# ======================
cat("\n=== Complete Learning Curve Analysis for All Feature Sets ===\n")

# 1. 定义要分析的特征集
feature_sets_names <- c("clinical", "metabolites", "combined")

# 2. 定义评估集（方便循环）
eval_sets <- list(
  internal = list(data = test_data,     name = "Internal test (Harbin)"),
  suzhou   = list(data = external1,     name = "External Suzhou"),
  ningbo   = list(data = external2,     name = "External Ningbo")
)

# 3. 存储所有结果的嵌套列表
lc_results <- list()

# 4. 循环每个特征集
for (feat in feature_sets_names) {
  
  cat(sprintf("\nRunning learning curves for %s features...\n", feat))
  lc_results[[feat]] <- list()
  
  # 对每个评估集运行
  for (set_key in names(eval_sets)) {
    
    cat(sprintf("  → %s\n", eval_sets[[set_key]]$name))
    
    lc_results[[feat]][[set_key]] <- run_learning_curve(
      train_source     = train_data,
      eval_data        = eval_sets[[set_key]]$data,
      feature_set_name = feat,
      proportions      = seq(0.05, 1.0, by = 0.05),
      n_repeats        = 5,
      min_n            = 20
    )
  }
}

# 5. 绘制三张独立的对比图（每张图一个特征集）
cat("\n=== Generating Learning Curve Plots ===\n")

for (feat in feature_sets_names) {
  
  # 准备绘图数据
  plot_data <- bind_rows(
    lapply(names(eval_sets), function(set_key) {
      lc_results[[feat]][[set_key]] %>%
        mutate(Eval_set = eval_sets[[set_key]]$name)
    })
  )
  
  # 绘图
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
      caption  = sprintf("Repeated %d times per point • Fixed evaluation sets", 5)
    ) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "bottom",
      plot.title      = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle   = element_text(hjust = 0.5)
    )
  
  # 打印并保存
  print(p)
  
  filename <- paste0("learning_curve_", feat, "_compare.png")
  ggsave(filename, p, width = 9, height = 6.5, dpi = 300)
  cat(sprintf("Saved: %s\n", filename))
}

cat("\nAll learning curve plots have been generated and saved.\n")