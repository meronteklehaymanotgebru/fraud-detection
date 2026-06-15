"""Reusable utility functions for fraud detection."""

import numpy as np
import ipaddress
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import cross_validate

def ip_to_int(ip_val):
    """Robustly convert an IP address (string or numeric) to integer."""
    if isinstance(ip_val, str):
        try:
            return int(ipaddress.ip_address(ip_val))
        except ValueError:
            pass
    try:
        return int(float(ip_val))
    except (ValueError, TypeError):
        return None

def evaluate_model(name, model, X_test, y_test):
    """Return F1, AUC-PR, and confusion matrix for a fitted model."""
    try:
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, "predict_proba")
                  else model.decision_function(X_test))
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        print(f"{name} | F1: {f1:.4f} | AUC-PR: {ap:.4f}")
        print(cm)
        return f1, ap, cm
    except Exception as e:
        print(f"Error evaluating {name}: {e}")
        return None, None, None

def cv_evaluate(name, model, X, y, cv=5):
    """Stratified K‑Fold cross‑validation with F1 and AUC-PR."""
    try:
        scoring = {'f1': 'f1', 'avg_precision': 'average_precision'}
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        f1_mean = np.mean(cv_results['test_f1'])
        f1_std = np.std(cv_results['test_f1'])
        ap_mean = np.mean(cv_results['test_avg_precision'])
        ap_std = np.std(cv_results['test_avg_precision'])
        print(f"{name} CV | F1: {f1_mean:.4f} ± {f1_std:.4f} | AUC-PR: {ap_mean:.4f} ± {ap_std:.4f}")
        return f1_mean, f1_std, ap_mean, ap_std
    except Exception as e:
        print(f"Error in CV for {name}: {e}")
        return None, None, None, None