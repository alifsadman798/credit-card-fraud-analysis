# Credit Card Fraud Detection — EDA & Linear Regression

## Overview
Exploratory Data Analysis and Linear Regression modeling on a real-world 
European credit card transaction dataset to understand fraud patterns and 
build predictive baselines.

## Dataset
- **Source:** Kaggle — MLG-ULB (Université Libre de Bruxelles)
- **Size:** 284,807 transactions | 31 columns
- **Fraud Rate:** Only 492 fraudulent cases (0.1727%) — heavily imbalanced
- **Note:** 28 features anonymized via PCA (V1–V28). Only `Time` and `Amount` 
  retain original meaning.
- Dataset not included due to size. Download from 
  [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Project Structure
credit-card-fraud-analysis/
├── creditcard_fraud_detect_EDA-linear-regression.ipynb
├── linear-regression.ipynb
├── requirements.txt
└── .gitignore

## Part 1 — Exploratory Data Analysis

### Key Findings
- **Class Imbalance:** 99.827% legitimate vs 0.173% fraud
- **Transaction Amount:** Highly right-skewed (skewness: 16.98). Median €22 vs Mean €88.35. 31,904 transactions (11.20%) flagged as outliers
- **Transaction Time:** Bimodal pattern reflecting day/night cycles. Fraud spikes at 2:00 AM (1.71%) and 4:00 AM
- **Fraud Behavior:** Fraudulent transactions cluster at lower-to-moderate amounts to avoid bank triggers
- **Top Fraud Indicators:**

| Feature | Correlation |
|---------|-------------|
| V17 | -0.326 |
| V14 | -0.303 |
| V12 | -0.261 |
| V10 | -0.216 |
| V16 | -0.197 |

- **Data Quality:** 0 missing values | 1,081 duplicates detected

## Part 2 — Linear Regression Modeling

Built using the Normal Equation (no ML libraries):

### Model Results

| Model | Test RMSE | R² |
|-------|-----------|-----|
| Simple (Hour + Amount) | ~28.51 | ~0.370 |
| Duplicates Removed | — | ~0.387 |
| Outliers Removed | ~7.76 | ~0.320 |

### Final Regression Equation
AmountPerHour = 16.5295 - 1.1608(Hour) + 0.0943(Amount) - 0.0478(V17)

## Limitations & Future Work
- Normal Equation not viable beyond ~1M records
- Linear Regression inappropriate for binary fraud classification
- **Next steps:** Logistic Regression, Random Forest, or XGBoost with SMOTE

## Libraries Used
See requirements.txt
## Acknowledgements
Special thanks to the following friends for their support and contribution in completing this project:
- [Md. Asif Khan](https://github.com/asifkhan06)
- [Sudeepta Mandal](https://github.com/Sudeepta-MANDAL743)
- [MD Sirajul Islam](https://github.com/SHOJIB-80)