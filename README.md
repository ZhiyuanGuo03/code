# Plasma Metabolomics Diagnostic Modeling Using XGBoost

This repository contains the complete machine learning workflow for the manuscript:

**"Plasma metabolic signature and single-cell regulatory network for ischemic stroke"**

The script performs preprocessing, model training, internal validation, external validation, ROC visualization, and learning curve analysis using XGBoost.

---

## 1. Software Environment

R version: 4.0.2  

### Required R packages

- pROC  
- caret  
- ggplot2  
- dplyr  
- xgboost  
- ggrepel  
- RColorBrewer  

Install missing packages using:

```r
install.packages(c(
  "pROC",
  "caret",
  "ggplot2",
  "dplyr",
  "xgboost",
  "ggrepel",
  "RColorBrewer"
))
Random seed is fixed at 42 to ensure reproducibility.

2. Required Input Files
Place the following CSV files in the working directory:

haerbin.csv – Discovery cohort (Harbin)

suzhou.csv – External validation cohort 1

ningbo.csv – External validation cohort 2

Each dataset must contain:

outcome column

0 = control

1 = ischemic stroke

Clinical variables:

sex

age

Metabolite features (remaining columns)

Optional column:

sample_id (automatically removed if present)

3. Analysis Workflow
The script executes the following steps:

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

Metabolite model (all metabolite variables)

Combined model (clinical + metabolites)

Step 4 – 10-Fold Cross-Validation (Training Set Only)
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

No retraining, hyperparameter tuning, or model updating is performed on external cohorts.

Step 7 – ROC Curve Visualization
ROC curves are generated and saved for:

10-fold cross-validation

Discovery training set

Internal test set

Suzhou validation cohort

Ningbo validation cohort

Figures are saved as PNG files in the working directory.

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

Learning curve plots are saved as PNG files.

4. Reproducibility
To reproduce the analysis:

Place all CSV files in the working directory.

Open the R script.

Run the entire script sequentially.

ROC and learning curve plots will be automatically saved.

All modeling steps are fully scripted and reproducible.

5. External Validation Policy
External validation cohorts (Suzhou and Ningbo) are evaluated using models trained exclusively on the discovery cohort.

No retraining, recalibration, or hyperparameter tuning is performed on external datasets.

6. Notes
All code comments are written in English to ensure transparency.

The repository contains the complete modeling workflow.

This modeling framework is intended for research reproducibility and hypothesis-generating analysis.
