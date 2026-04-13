"""
AI Panchayat – ML Bias Auditing Engine
Trains a LogisticRegression model and computes Fairlearn bias metrics.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from typing import Dict, Any


def run_audit(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
) -> Dict[str, Any]:
    """
    Train a LogisticRegression on *df*, compute accuracy & Fairlearn bias
    metrics, and return a structured dictionary of results.
    """

    # --- Encode categoricals ------------------------------------------------
    label_encoders: Dict[str, LabelEncoder] = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    # --- Features / target ---------------------------------------------------
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col].values
    sensitive = df_encoded[sensitive_col].values

    # --- Train / test split --------------------------------------------------
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- Model ---------------------------------------------------------------
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    # --- Accuracy metrics ----------------------------------------------------
    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    precision = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    recall = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

    # --- Fairlearn bias metrics ----------------------------------------------
    dp_diff = round(float(demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)), 4)
    eo_diff = round(float(equalized_odds_difference(y_test, y_pred, sensitive_features=sens_test)), 4)

    # --- Per-group acceptance rates ------------------------------------------
    unique_groups = np.unique(sens_test)
    group_rates = {}
    for g in unique_groups:
        mask = sens_test == g
        rate = float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0
        # Map back to original label if encoded
        if sensitive_col in label_encoders:
            label = label_encoders[sensitive_col].inverse_transform([g])[0]
        else:
            label = str(g)
        group_rates[label] = round(rate, 4)

    return {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_column": target_col,
        "sensitive_column": sensitive_col,
        "model_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "bias_metrics": {
            "demographic_parity_difference": dp_diff,
            "equalized_odds_difference": eo_diff,
        },
        "group_acceptance_rates": group_rates,
    }
