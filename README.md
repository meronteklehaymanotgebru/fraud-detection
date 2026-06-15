# Fraud Detection for E‑commerce and Bank Transactions

**Adey Innovations Inc. – 10 Academy Week 5‑6 Challenge**

---

## Business Need
Adey Innovations serves e‑commerce and banking clients. Fraud losses harm both revenue and customer trust. This project builds classification models to detect fraudulent transactions in two streams:
- **E‑commerce transactions** (rich user, device, IP, browser, and behavioral context)
- **Bank credit card transactions** (anonymized PCA features)

The goal is to balance **false positives** (customer friction) and **false negatives** (financial loss) using metrics tailored for imbalanced data.

---

## 1. Data Analysis & Preprocessing (Task 1)

### 1.1 Data Cleaning
- **E‑commerce data** (`Fraud_Data.csv`): 151,112 rows, 11 columns. No missing values. Duplicates dropped.
- IP addresses were converted from mixed format to integer and enriched with **country** via `merge_asof` using the provided IP‑to‑country range mapping.
- **Credit card data** (`creditcard.csv`): 284,807 rows; no missing values; 1,081 duplicates removed.

### 1.2 Exploratory Data Analysis
- **Class imbalance**: E‑commerce fraud ~10%, credit card fraud ~0.17%.
- **Temporal patterns**: Fraud rate peaks in early morning hours (2‑5 AM).
- **Geolocation**: Certain countries show significantly higher fraud rates.
- **Browser & source**: Some browsers and marketing sources are more prone to fraud.
- **Purchase value**: Fraudulent transactions often involve smaller amounts.

*(All EDA visualizations are available in `reports/images/` and can be reproduced with `scripts/generate_eda_images.py`.)*

### 1.3 Feature Engineering (E‑commerce)
New features were created from raw timestamps and user history:
- `hour_of_day`, `day_of_week`
- `time_since_signup`: hours between account creation and purchase
- `user_txn_count`: total transactions per user
- `txn_last_24h`: rolling count of transactions in the preceding 24 hours

### 1.4 Class Imbalance Handling
- **SMOTE** (Synthetic Minority Over‑sampling Technique) applied to the **training set only**, after a stratified train‑test split.
- This retains all real data while generating synthetic minority examples, avoiding information loss from undersampling.
- Class distribution before and after resampling is documented in the notebooks.

---

## 2. Model Building & Training (Task 2)

### 2.1 Models Trained
Three classifiers were trained for each dataset:
- Logistic Regression (interpretable baseline)
- Random Forest (with class‑weight balancing)
- XGBoost (with `scale_pos_weight`)

Hyper‑parameters were tuned using **RandomizedSearchCV** (F1 scoring) to accommodate hardware constraints; the code is structured for easy upgrade to a full grid search on more powerful machines.

### 2.2 Evaluation Metrics
Because the data is highly imbalanced, **Accuracy is not used as a primary metric**. Instead, models are evaluated on:
- **F1‑score**
- **AUC‑PR** (Area Under the Precision‑Recall Curve)
- **Confusion matrices** (True/False Positives/Negatives)

All metrics are validated with **5‑fold stratified cross‑validation**, reporting mean and standard deviation.

### 2.3 Results (Test Set)

**E‑commerce models**
| Model               | F1    | AUC‑PR | TN     | FP   | FN   | TP    |
|---------------------|-------|--------|--------|------|------|-------|
| Logistic Regression | 0.2776| 0.3911 | 17,927 | 9,466| 848  | 1,982 |
| Random Forest       | 0.6941| 0.6293 | 27,346 | 47   | 1,301| 1,529 |
| **XGBoost**         | **0.6946**| **0.6229** | 27,344 | 49   | 1,298| 1,532 |

**Credit card models**
| Model               | F1    | AUC‑PR | TN     | FP  | FN  | TP  |
|---------------------|-------|--------|--------|-----|-----|-----|
| Logistic Regression | 0.1002| 0.6768 | 55,172 | 1,479| 12  | 83  |
| **Random Forest**   | **0.8229**| **0.8133** | 56,643 | 8   | 23  | 72  |
| XGBoost             | 0.7600| 0.8105 | 56,622 | 29  | 19  | 76  |

**Key insight**: The ensemble models dramatically reduce false positives compared to Logistic Regression – from thousands to just tens – directly addressing the business need of minimising customer friction.

Selected best models:
- **E‑commerce**: XGBoost
- **Credit card**: Random Forest

Both models are saved in the `models/` directory.

---

## 3. Model Explainability (Task 3)

### 3.1 SHAP Global Importance
SHAP analysis was performed on the e‑commerce XGBoost model (the richer dataset for interpretability). The summary plot identified the top fraud drivers:
1. `time_since_signup`
2. `hour_of_day`
3. `txn_last_24h`
4. `purchase_value`
5. High‑risk country indicators

### 3.2 SHAP vs. Built‑in Feature Importance
We compared the top 10 features from SHAP (mean absolute SHAP value) with XGBoost’s built‑in `feature_importances_`. The two methods agree on the most important features, but SHAP gives more weight to categorical country variables because it captures both magnitude and direction of impact. This comparison validates the robustness of the top drivers and demonstrates why SHAP is preferred for regulatory explainability.

*(Comparison chart available in `reports/images/importance_comparison.png`)*

### 3.3 Force Plots
Individual force plots were generated for:
- A true positive (correctly identified fraud)
- A false positive (legitimate flagged as fraud)
- A false negative (missed fraud)

These plots illustrate exactly which factors pushed the model’s decision, aiding in root‑cause analysis and rule refinement.

---

## 4. Business Recommendations
1. **Flag transactions within 1 hour of signup for manual review** – `time_since_signup` is the strongest fraud predictor.
2. **Require step‑up authentication for late‑night purchases (0–5 AM)** – `hour_of_day` drives risk during low‑traffic hours.
3. **Monitor transaction velocity: 3+ transactions in 24 hours triggers a review** – `txn_last_24h` captures account takeover.
4. **Weight fraud scores by country‑specific risk** – high‑risk countries identified by SHAP should receive higher baseline scores.

Each recommendation is directly linked to a SHAP insight, making them fully explainable to risk and compliance teams.

---

## 5. Limitations & Future Work
- **Credit card interpretability**: PCA features limit business‑level explanations; obtaining raw features is recommended.
- **Hardware constraints**: Tuning was lightweight (RandomizedSearchCV, 2‑fold CV); full grid search on cloud resources could improve performance.
- **Potential overfitting on credit card CV**: Near‑perfect CV scores suggest high separability after SMOTE; test‑set metrics are more realistic.
- **Temporal drift**: A monitoring dashboard should track prediction distributions and trigger retraining when drift is detected.
- **Production deployment**: The pipeline is batch‑oriented; a streaming API with preprocessing integration is the next step.

---

## 6. Repository Structure
fraud-detection/
├── .github/workflows/ # CI/CD (linting, unit tests)
├── data/
│ ├── raw/ # Original datasets (not committed)
│ └── processed/ # Pickle files after preprocessing
├── notebooks/
│ ├── eda-fraud-data.ipynb # EDA, geolocation, feature engineering
│ ├── eda-creditcard.ipynb # EDA, scaling
│ ├── modeling.ipynb # Model training, tuning, comparison
│ └── shap-explainability.ipynb # SHAP plots, force plots, recommendations
├── src/
│ └── utils.py # Reusable functions (evaluation, IP conversion)
├── scripts/
│ ├── generate_eda_images.py # Reproduces all report EDA figures
│ └── train_model.py # Standalone training script
├── models/ # Saved model artifacts (.pkl)
├── reports/
│ └── images/ # All generated visualizations
├── tests/ # Unit tests
├── requirements.txt
├── README.md
└── .gitignore

---

## 7. Setup & Reproducibility
```bash
git clone <your-repo-url>
cd fraud-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
Run the full pipeline:

    notebooks/eda-fraud-data.ipynb

    notebooks/eda-creditcard.ipynb

    notebooks/modeling.ipynb

    notebooks/shap-explainability.ipynb

Regenerate all report figures:
bash

python scripts/generate_eda_images.py

8. Development Workflow

This project followed a branch‑per‑task strategy:

    task-1 – Data cleaning, EDA, geolocation, and feature engineering.

    task-2 – Model training, tuning, and evaluation.

    task-3 – SHAP explainability and business recommendations.


