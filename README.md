# 📊 Measuring Firm Performance Using Financial Ratios: An XGBoost Approach
### A Comparative Study of U.S. and Chinese Technology Firms

> **Course:** FIE453 — Big Data with Applications to Finance | NHH Norwegian School of Economics  
> **Author:** Promotosh Barua  
> **Date:** December 2022

---

## 🗂️ Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

This project applies **XGBoost (eXtreme Gradient Boosting)** to predict and compare the financial performance of **63 U.S.** and **63 Chinese** technology companies listed on major U.S. stock exchanges. Financial ratios across six categories — Capitalization, Efficiency, Financial Soundness, Liquidity, Profitability, and Valuation — are used as predictive features, with stockholder equity as the target variable.

The study finds that **U.S. tech firms consistently outperform their Chinese counterparts** across most financial ratio categories.

---

## Research Question

> *Can financial ratios predict the financial performance of technology firms, and do U.S. firms outperform Chinese firms when evaluated using machine learning?*

Performance is measured by classifying firms as **High** or **Low** on stockholder equity — a proxy for financial health representing total assets minus total liabilities.

---

## Dataset

| | U.S. Firms | Chinese Firms |
|---|---|---|
| **Number of firms** | 63 | 63 |
| **Total observations** | ~7,000 (6,452 usable) | 654 |
| **Time period** | January 2010 – December 2021 (11 years) |
| **Source** | [WRDS — Wharton Research Data Service](https://wrds-www.wharton.upenn.edu/) |
| **Firm list (China)** | [U.S.-China Economic and Security Review Commission](https://www.uscc.gov/research/chinese-companies-listed-major-us-stock-exchanges) |

### Financial Ratio Categories

| Category | Description | Example Variables |
|---|---|---|
| **Capitalization** | Debt portion of total capital structure | `totdebt_invcap`, `equity_invcap`, `capital_ratio` |
| **Efficiency** | Effectiveness of asset/liability use | `at_turn`, `rect_turn`, `sale_equity`, `inv_turn` |
| **Financial Soundness** | Ability to meet long-term obligations | `cfm`, `cash_lt`, `debt_ebitda`, `ocf_lct` |
| **Liquidity** | Ability to meet short-term obligations | `curr_ratio`, `quick_ratio`, `cash_ratio`, `cash_conversion` |
| **Profitability** | Ability to generate profit | `roa`, `roe`, `npm`, `gpm`, `aftret_invcapx` |
| **Valuation** | Attractiveness of a company's stock | `CAPEI`, `pe_op_basic`, `bm`, `evm`, `ptb` |

> The full variable list (70+ ratios) is included in the [Appendix of the paper](FIE453_TERM_PAPER.pdf).

### Data Preprocessing

- Missing values were **deleted** where feasible (especially for U.S. data).
- For the Chinese dataset, some missing values were **replaced with 0**, assuming no meaningful outcome occurred that year.
- **Highly correlated variables** (correlation > 0.80) were removed to reduce redundancy.
- The target variable was **discretized into two classes**: `High` and `Low` using equal-frequency binning.
- Data was split **70% training / 30% test**.

---

## Methodology

### Model: XGBoost

XGBoost (eXtreme Gradient Boosting) is an optimized implementation of Gradient Boosting Decision Trees (GBDT). Key advantages used in this study:

- Handles missing data natively via directional splits
- Uses **second-order Taylor expansion** of the loss function for faster, more accurate optimization than standard GBDT
- Applies **regularization** to control overfitting
- Scalable and memory-efficient

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (`eta`) | 0.01 |
| Max tree depth | 6 |
| Subsample | 1 |
| Column sample by tree | 1 |
| Number of rounds | 200 |
| Threads (`nthread`) | 4 |

Hyperparameter optimization was performed using **5-fold cross-validation grid search** over: `eta ∈ {0.01, 0.1, 0.2, 0.3}`, `max_depth ∈ {2, 3, 4, 5}`, `nrounds ∈ {30, 50, 60, 100}`.

### Evaluation Metrics

| Metric | Formula |
|---|---|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) |

---

## Results

### Category-Level Performance (Test Set)

| Category | US Recall | US Precision | US Accuracy | CN Recall | CN Precision | CN Accuracy |
|---|---|---|---|---|---|---|
| Capitalization | 0.908 | 0.938 | **0.923** | 0.896 | 0.780 | 0.843 |
| Efficiency | 0.882 | 0.959 | **0.919** | 0.926 | 0.933 | 0.929 |
| Financial Soundness | 0.976 | 0.941 | **0.959** | 1.000 | 0.967 | **0.984** |
| Liquidity | 0.836 | 0.879 | **0.857** | 0.770 | 0.928 | 0.847 |
| Profitability | 0.801 | 0.942 | **0.872** | 0.942 | 0.818 | 0.885 |
| Valuation | 0.934 | 0.943 | **0.938** | 0.973 | 0.954 | **0.965** |

### Final Full-Dataset Performance

| Dataset | Recall | Precision | Accuracy |
|---|---|---|---|
| **U.S.** | 0.993 | **1.000** | **0.997** |
| **China** | 0.981 | 0.988 | 0.984 |

### Key Findings

- The **U.S. model achieves near-perfect accuracy (99.7%)** on the full dataset, with a precision of 1.0 — indicating no false positives.
- The **Chinese model also performs strongly (98.4%)**, though with slightly more variance, likely due to the smaller dataset (654 vs. 6,452 observations).
- **U.S. tech firms outperform Chinese tech firms** overall when financial ratios are used as predictors.
- **Hyperparameter tuning** with grid search confirmed the robustness of both models, with the Chinese model reaching 99.85% cross-validated accuracy and the U.S. model achieving a perfect Kappa score of 1.0.

---

## Repository Structure

```
├── README.md
├── USA_ALL.R          # Full XGBoost pipeline for U.S. dataset
├── china_ALL.R        # Full XGBoost pipeline for Chinese dataset
└── FIE453_TERM_PAPER.pdf  # Full paper with tables, figures, and appendix
```

### Pipeline Overview

```
Raw WRDS Data
    │
    ▼
Data Cleaning & Preprocessing
(delete irrelevant vars, handle missing values)
    │
    ▼
Feature Engineering
(correlation analysis, remove vars > 0.80 corr.)
    │
    ▼
Train/Test Split (70% / 30%)
    │
    ▼
XGBoost Model Training
    │
    ▼
Confusion Matrix & Evaluation Metrics
    │
    ▼
Hyperparameter Grid Search (5-fold CV)
```

---

## How to Run

### Requirements

```r
install.packages(c("tidyverse", "readxl", "mlbench", "caret",
                   "corrplot", "xgboost", "arules", "dplyr"))
```

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/promotosh/Measuring-Firm-Performance-Using-Financial-Ratios-An-XGBoost-Approach.git
   cd Measuring-Firm-Performance-Using-Financial-Ratios-An-XGBoost-Approach
   ```

2. **Obtain the data** from [WRDS](https://wrds-www.wharton.upenn.edu/) using the GV keys listed in the paper appendix. Save as:
   - `data_usa_all.csv`
   - `data_china_all.csv`

3. **Run the U.S. analysis**
   ```r
   source("USA_ALL.R")
   ```

4. **Run the Chinese analysis**
   ```r
   source("china_ALL.R")
   ```

Each script outputs:
- Correlation matrix plot
- Training and test confusion matrices (accuracy, precision, recall)
- Hyperparameter grid search results and plot

---

## Limitations

- **Small Chinese sample size** (654 observations vs. 7,000 for U.S.) may introduce bias and limit generalizability.
- Firms were **not filtered by market capitalization** or other size characteristics, so the samples may not be fully comparable.
- Missing data handling (replacing NAs with 0) is a simplifying assumption that may distort certain ratio categories.
- The study does not account for **macroeconomic moderators** (e.g., interest rates, geopolitical factors) that may affect results differently across the two markets.
- High accuracy scores suggest potential **overfitting** risks, particularly for the U.S. model (precision = 1.0), warranting further validation on out-of-sample data.

---

## References

- Chen, T., et al. (2015). XGBoost: Extreme Gradient Boosting. *R Package Version 0.4-2*.
- Delen, D., Kuzey, C., & Uyar, A. (2013). Measuring firm performance using financial ratios: A decision tree approach. *Expert Systems with Applications, 40*(10), 3970–3983.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*.
- Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive logistic regression: A statistical view of boosting. *The Annals of Statistics, 28*(2), 337–407.
- Hayes, A. (2022). Stockholders' Equity. *Investopedia*.
- WRDS Research Team. (2016). WRDS Industry Financial Ratio. Wharton Research Data Service.

---

<p align="center">
  Made at <strong>NHH Norwegian School of Economics</strong> · FIE453 Big Data with Applications to Finance · Fall 2022
</p>
