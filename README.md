# Customer Churn Prediction — End-to-End ML Project

Predicting which telecom customers are likely to churn, using the IBM/Kaggle Telco Customer Churn dataset. This project covers the full pipeline: data cleaning, EDA, feature engineering, model training, threshold tuning, and model interpretation.

## Problem Statement

Customer churn — customers leaving for a competitor — is costly for subscription-based businesses, since acquiring a new customer is typically far more expensive than retaining an existing one. This project builds a classification model to predict which customers are at risk of churning, so a retention team could proactively intervene.

**Target variable:** `Churn` (Yes/No)
**Success metric:** Given the class imbalance (~26.5% churn), recall on the churn class matters more than raw accuracy — missing a customer who's about to leave is costlier than a false alarm on someone who was staying anyway.

## Dataset

- **Source:** [IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Size:** 7,043 customers, 21 columns
- **Target distribution:** 73.5% stayed, 26.5% churned (imbalanced)

## Project Structure

customer-churn-prediction/
├── data/
│ ├── raw/ # Original, unmodified dataset
│ └── processed/ # Cleaned dataset
├── notebooks/
│ └── 01_data_exploration.ipynb
├── models/
│ ├── logistic_regression_baseline.pkl
│ └── random_forest_final.pkl
├── reports/figures/ # Saved chart images
├── src/
│ └── clean_data.py
└── README.md



## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
```

## Data Cleaning

- **`TotalCharges`** loaded as text instead of numeric, due to 11 blank-string values. Investigation showed all 11 affected rows had `tenure == 0` and `Churn == 'No'` — these are brand-new customers with no billing history yet, not missing-at-random data. Filled with `0` rather than the column median, since the median would fabricate a charge that never happened for these specific customers.
- No duplicate customer IDs found.

## Exploratory Data Analysis — Key Findings

**1. Contract type is one of the strongest churn predictors.**
| Contract | Churn Rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

Month-to-month customers churn roughly 15x more than two-year customers.

**2. Churn risk is concentrated in the first 1–2 months of tenure**, then drops off sharply. This points to onboarding as the highest-risk window for retention efforts, rather than churn being spread evenly across a customer's lifetime.

**3. Churn rate increases with monthly charges**, roughly tripling from the cheapest pricing bucket (~10% churn, $0–30/month) to mid-to-high brackets (~34%, $60–90/month), before flattening slightly at the top end ($90–120) — possibly reflecting more committed customers in premium bundles.

**4. Correlation with `Churn`:** strongest negative relationships were `tenure` (-0.35) and `Contract_Two year` (-0.30); strongest positive were `InternetService_Fiber optic` (+0.31) and `PaymentMethod_Electronic check` (+0.30).

## Feature Engineering

- Dropped `customerID` (identifier, not predictive).
- Binary-mapped true Yes/No columns (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) to 1/0.
- One-hot encoded multi-category columns (`Contract`, `InternetService`, `PaymentMethod`, `MultipleLines`, and the six add-on service columns) with `drop_first=True` to avoid the dummy variable trap.
- Result: 30 features after encoding.

## Modeling

- **Split:** 80/20 train/test, stratified on `Churn` to preserve class balance in both sets.
- **Scaling:** `StandardScaler` applied to `tenure`, `MonthlyCharges`, `TotalCharges` for Logistic Regression (fit on train only, to avoid data leakage). Not needed for Random Forest, which splits on thresholds rather than distances.

### Model Comparison

| Model | ROC-AUC | Threshold | Recall (churn) | Precision (churn) | Missed churners |
|---|---|---|---|---|---|
| Logistic Regression | 0.842 | 0.45 | 0.61 | 0.60 | 144 / 374 |
| Random Forest (tuned) | 0.844 | 0.35 | 0.88 | 0.45 | 44 / 374 |

Both models achieve essentially identical ROC-AUC (~0.84), meaning they're equally good at *ranking* customers by churn risk overall. However, tuned Random Forest (hyperparameters selected via 5-fold `GridSearchCV`, optimizing ROC-AUC) reaches a much higher recall ceiling — at threshold 0.35, it catches 88% of churners, a level Logistic Regression can't reach without precision degrading similarly.

**Final model: Random Forest (tuned), threshold = 0.35.** Chosen for its stronger recall on the churn class, prioritizing catching at-risk customers over minimizing false alarms — a reasonable trade-off when a missed churner (lost recurring revenue) is assumed costlier than a low-cost retention outreach to a customer who was staying anyway.

**Random Forest hyperparameters (via GridSearchCV):** `n_estimators=300`, `max_depth=None`, `min_samples_leaf=10`, `class_weight='balanced'`.

## Model Interpretation — Feature Importance

Top features driving Random Forest's predictions:

| Feature | Importance |
|---|---|
| tenure | 0.184 |
| TotalCharges | 0.132 |
| Contract_Two year | 0.111 |
| InternetService_Fiber optic | 0.079 |
| MonthlyCharges | 0.078 |
| PaymentMethod_Electronic check | 0.059 |

These align closely with the EDA and correlation findings above — two independent methods (simple correlation and a trained model) converge on the same top drivers of churn, reinforcing that these relationships are real rather than coincidental.

## Business Recommendations

1. **Prioritize retention outreach for month-to-month customers in their first 1–2 months** — this is both the highest-churn segment and the highest-risk time window.
2. **Incentivize longer contract commitments** (e.g. discounts for 1–2 year plans) — churn drops from 42.7% to 2.8% between month-to-month and two-year contracts.
3. **Investigate the fiber optic customer experience** — this segment shows a disproportionately high churn rate, worth checking for pricing or service-quality issues.
4. **Promote add-on services like Online Security and Tech Support** — these correlate with reduced churn and may increase switching costs / perceived value.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Author

[Redwan Ahmed] — [https://www.linkedin.com/in/redwan-ahmed-5b807938a/]
