"""Train and save fraud detection models from processed data."""
import sys, os, pickle, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils import evaluate_model, cv_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    # Load processed data
    try:
        with open("data/processed/fraud_train.pkl", "rb") as f:
            X_train_f, y_train_f = pickle.load(f)
        with open("data/processed/fraud_test.pkl", "rb") as f:
            X_test_f, y_test_f, _ = pickle.load(f)
        with open("data/processed/creditcard_train.pkl", "rb") as f:
            X_train_c, y_train_c = pickle.load(f)
        with open("data/processed/creditcard_test.pkl", "rb") as f:
            X_test_c, y_test_c = pickle.load(f)
    except FileNotFoundError as e:
        logging.error(f"Processed data not found. Run eda-fraud-data.ipynb and eda-creditcard.ipynb first. {e}")
        return

    # E‑commerce: XGBoost
    logging.info("Training XGBoost for e‑commerce...")
    xgb_f = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.2,
                          scale_pos_weight=(len(y_train_f)-sum(y_train_f))/sum(y_train_f),
                          random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_f.fit(X_train_f, y_train_f)
    evaluate_model("XGBoost (E‑comm)", xgb_f, X_test_f, y_test_f)
    cv_evaluate("XGBoost (E‑comm)", xgb_f, X_train_f, y_train_f, cv=5)

    os.makedirs("models", exist_ok=True)
    with open("models/fraud_xgb.pkl", "wb") as f:
        pickle.dump(xgb_f, f)

    # Credit card: Random Forest
    logging.info("Training Random Forest for credit card...")
    rf_c = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42, class_weight='balanced')
    rf_c.fit(X_train_c, y_train_c)
    evaluate_model("Random Forest (Credit)", rf_c, X_test_c, y_test_c)
    cv_evaluate("Random Forest (Credit)", rf_c, X_train_c, y_train_c, cv=5)

    with open("models/credit_rf.pkl", "wb") as f:
        pickle.dump(rf_c, f)

    logging.info("Models saved to models/")

if __name__ == "__main__":
    main()