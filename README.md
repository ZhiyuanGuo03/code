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
