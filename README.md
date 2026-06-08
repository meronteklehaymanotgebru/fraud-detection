# Fraud Detection for E‑commerce and Bank Transactions

**Adey Innovations Inc. – Week 5‑6 Challenge**

## Business Need
Adey Innovations serves e‑commerce and banking clients. Fraud losses harm both revenue and customer trust. This project builds classification models to detect fraudulent transactions in two streams:
- **E‑commerce transactions** (rich user, device, and geolocation context)
- **Bank credit card transactions** (anonymized PCA features)

The goal is to balance false positives (customer friction) and false negatives (financial loss) using metrics tailored for imbalanced data.

---

## Interim‑1 Report: Data Analysis and Preprocessing

### 1. Data Cleaning
- E‑commerce data: 151,112 rows, 11 columns. No missing values. Duplicates dropped.
- IP addresses converted from mixed format to integer and enriched with country via `merge_asof` using the IP‑to‑country range mapping.
- Credit card data: 284,807 rows; no missing values; duplicates removed.

### 2. Exploratory Data Analysis
- **Class imbalance**: E‑commerce fraud ~10%, credit card fraud ~0.17%.
- **Temporal patterns**: Fraud rate peaks in early morning hours.
- **Geolocation**: Certain countries exhibit significantly higher fraud rates.
- **Browser & source**: Some browsers and marketing sources are more prone to fraud.
- **Purchase value**: Fraudulent transactions often involve smaller amounts.

*(See `reports/images/` for all EDA visualisations.)*

### 3. Feature Engineering (E‑commerce)
- `hour_of_day`, `day_of_week`
- `time_since_signup`: hours between account creation and purchase
- `user_txn_count`: total transactions per user
- `txn_last_24h`: rolling count of transactions in the preceding 24 hours

### 4. Class Imbalance Handling
- **SMOTE** applied to the training set only, after train‑test split.
- Reason: retains all real data while generating synthetic minority examples, avoiding information loss from undersampling.
- Class distribution before/after resampling is logged in the notebooks.

---

## Repository Structure
fraud-detection/
├── .github/workflows/unittests.yml
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Pickle files after preprocessing
├── notebooks/
│ ├── eda-fraud-data.ipynb
│ └── eda-creditcard.ipynb
├── src/ # Production code (future tasks)
├── tests/
├── models/
├── scripts/
│ └── generate_eda_images.py # Reproduces all report figures
├── reports/images/ # Saved EDA figures
├── requirements.txt
└── README.md

## Setup & Reproducibility
```bash
git clone <your-repo-url>
cd fraud-detection
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_eda_images.py
jupyter notebook
