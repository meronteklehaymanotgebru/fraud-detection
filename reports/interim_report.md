## Interim-1 Report: Data Analysis and Preprocessing

### Data Cleaning
- E‑commerce data: no missing values, no duplicates after drop_duplicates(). Data types corrected (signup_time, purchase_time parsed as datetime).
- IP mapping: converted IP addresses to integers and used `merge_asof` to attach country based on IP range.
- Credit card data: no missing values, X duplicates removed. 'Time' and 'Amount' scaled with StandardScaler.

### EDA Insights
- Both datasets heavily imbalanced: e‑commerce ~10% fraud, credit card ~0.17% fraud.
- E‑commerce: fraud rate peaks at night (early morning hours) and varies significantly by country and browser.
- Credit card: fraudulent transactions have a slightly different distribution of Amount; V‑features show separation between classes in PCA space.

### Feature Engineering (E‑commerce)
- `hour_of_day`, `day_of_week`: capture temporal patterns.
- `time_since_signup`: measures time between account creation and transaction (rapid purchases may indicate fraud).
- `user_txn_count`: total transactions per user.
- `txn_last_24h`: rolling count of transactions in the last 24 hours — a velocity feature.

### Class Imbalance Handling
- Chose SMOTE over undersampling to avoid losing majority‑class information.
- SMOTE applied **only on the training set** after train/test split.
- Resulting training set has balanced classes, test set remains imbalanced as in real‑world scenarios.

### Deliverables
- Cleaned datasets saved in `data/processed/` as pickle files.
- EDA notebooks: `notebooks/eda-fraud-data.ipynb`, `notebooks/eda-creditcard.ipynb`.
- Processing pipeline ready for modeling in Task 2.