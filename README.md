Plasma Metabolomics Diagnostic Modeling Using LASSO and XGBoost

This repository contains the complete machine learning workflow for the manuscript:

"Plasma metabolic signature and single-cell regulatory network for ischemic stroke"

The script performs LASSO-based feature selection, preprocessing, model training, internal validation, external validation, ROC visualization, and learning curve analysis.

Software Environment

R version: 4.5.2

Required R packages:

pROC

caret

ggplot2

dplyr

xgboost

ggrepel

RColorBrewer

glmnet

tidyverse

broom

To install required packages in R, run:
install.packages(c("pROC","caret","ggplot2","dplyr","xgboost","ggrepel","RColorBrewer","glmnet","tidyverse","broom"))

Random seed is fixed at 42 (XGBoost) and 66666 (LASSO) to ensure reproducibility.

Required Input Files

For XGBoost modeling, place the following CSV files in the working directory:

haerbin.csv (Discovery cohort – Harbin)

suzhou.csv (External validation cohort 1)

ningbo.csv (External validation cohort 2)

Each dataset must contain:

outcome column
0 = control
1 = ischemic stroke

Clinical variables:
sex
age

Metabolite variables (all remaining columns)

Optional column:

sample_id (automatically removed if present)

Additional files required for LASSO feature selection:

expression_matrix.csv
Rows = samples
Columns = metabolite features
First column = sample IDs (used as row names)

class_labels.csv
Rows = samples
Single label column coded as 0/1
First column = sample IDs (used as row names)
Sample order must match expression_matrix.csv

Analysis Workflow

The workflow consists of two major stages:

Stage A – LASSO Feature Selection

Step A1 – Data Loading

Expression matrix and class labels are read into R

Data are converted to matrix format

Step A2 – Cross-Validated LASSO

Logistic regression (binomial family) implemented using glmnet

5-fold cross-validation

Performance metric: Mean Squared Error (MSE)

Optimal penalty parameter selected at λ_min

Step A3 – Feature Extraction

Non-zero coefficients at λ_min are extracted

Intercept term is removed

Selected metabolites are saved to:
lasso_selected_features.csv

LASSO is used exclusively for feature selection. No classification performance is derived from the LASSO model itself.

Stage B – XGBoost Modeling

Step 1 – Data Preprocessing

Convert outcome to factor

Remove sample_id if present

Convert non-numeric predictors (e.g., sex) into numeric codes

Step 2 – Train/Test Split

Discovery cohort is split into:
80% training set
20% internal test set

Stratified sampling is applied using caret::createDataPartition

Step 3 – Feature Sets
Three models are constructed:

Clinical model (sex, age)

Metabolite model (all metabolite variables or LASSO-selected metabolites)

Combined model (clinical + metabolites)

Step 4 – 10-Fold Cross-Validation

Performed within the training set

Reports mean AUC, accuracy, sensitivity, and specificity

Used for internal model stability assessment

Step 5 – Final Model Training
Final XGBoost models are trained on the full training set.

Hyperparameters:

objective = binary:logistic

eval_metric = auc

eta = 0.1

max_depth = 3

subsample = 0.8

colsample_bytree = 0.8

nrounds = 100

Step 6 – Model Evaluation
Models are evaluated on:

Training set

Internal test set (Harbin)

External validation cohort 1 (Suzhou)

External validation cohort 2 (Ningbo)

No retraining, hyperparameter tuning, or recalibration is performed on external cohorts.

Step 7 – ROC Curve Visualization
ROC curves are generated and saved for:

10-fold cross-validation

Discovery training set

Internal test set

Suzhou validation cohort

Ningbo validation cohort

Step 8 – Learning Curve Analysis
Learning curves are generated for:

Clinical model

Metabolite model

Combined model

For each model:

Training subsets range from 5% to 100%

Each point is repeated 5 times

AUC is evaluated on:
Internal test set
Suzhou cohort
Ningbo cohort

Plots are automatically saved in the working directory.

Reproducibility

To reproduce the full analysis:

Place all required CSV files in the working directory.

Run the LASSO script to obtain selected metabolites (optional if using all metabolites).

Run the XGBoost modeling script.

ROC and learning curve plots will be generated automatically.

All modeling steps are fully scripted and reproducible.

External Validation Policy

External validation cohorts (Suzhou and Ningbo) are evaluated using models trained exclusively on the discovery cohort.

No retraining, recalibration, or hyperparameter tuning is performed on external datasets.

Notes

All code comments are written in English.

The repository contains both LASSO feature selection and XGBoost modeling workflows.

LASSO is used only for feature selection; diagnostic performance is evaluated using XGBoost.

This framework is intended for research reproducibility and hypothesis-generating analysis.
