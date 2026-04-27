"""
AI Panchayat – ML Bias Auditing Engine  (REFACTORED)
────────────────────────────────────────────────────
Mitigation methods now supported:
  • reweighing          – Pre-processing: re-weights training samples so that
                          each (sensitive_group × label) cell contributes equally
                          to the loss function. Pure data-level fix; the base
                          estimator is unchanged.
  • exponentiated_gradient – In-processing: iteratively reweights the loss
                          using Lagrangian multipliers to satisfy a
                          DemographicParity constraint. Fairlearn canonical.
  • threshold_optimizer – Post-processing: fits group-specific decision
                          thresholds after an unconstrained model is trained,
                          deriving the thresholds that minimise accuracy loss
                          while satisfying DemographicParity.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from typing import Dict, Any, Literal


# ── Helpers ──────────────────────────────────────────────────────────────────

def _encode_dataframe(df: pd.DataFrame, target_col: str):
    """Label-encode every categorical column; return encoded df + encoders."""
    label_encoders: Dict[str, LabelEncoder] = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    return df_encoded, label_encoders


def _compute_metrics(y_test, y_pred, sens_test, sensitive_col, label_encoders):
    """Return accuracy + Fairlearn fairness metrics in a structured dict."""
    accuracy  = round(float(accuracy_score(y_test, y_pred)), 4)
    precision = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    recall    = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
    f1        = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

    dp_diff = round(float(demographic_parity_difference(
        y_test, y_pred, sensitive_features=sens_test)), 4)
    eo_diff = round(float(equalized_odds_difference(
        y_test, y_pred, sensitive_features=sens_test)), 4)

    group_rates = {}
    for g in np.unique(sens_test):
        mask = sens_test == g
        rate = float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0
        label = (label_encoders[sensitive_col].inverse_transform([g])[0]
                 if sensitive_col in label_encoders else str(g))
        group_rates[label] = round(rate, 4)

    return {
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


# ── Pre-processing: Reweighing ────────────────────────────────────────────────

def _compute_reweighing_weights(y_train: np.ndarray, sens_train: np.ndarray) -> np.ndarray:
    """
    Reweighing (Kamiran & Calders 2012).

    Algorithm
    ---------
    Expected weight  W_exp(x) = P(S=s) · P(Y=y)
    Observed weight  W_obs(x) = P(S=s, Y=y)
    Sample weight    w(x)     = W_exp(x) / W_obs(x)

    This makes the joint distribution P(S, Y) look independent by construction,
    so a model trained on these weights cannot exploit the S→Y correlation.
    Crucially the feature matrix X is *not* touched – only the loss weights.
    """
    n = len(y_train)
    weights = np.ones(n, dtype=float)

    unique_s = np.unique(sens_train)
    unique_y = np.unique(y_train)

    for s in unique_s:
        for y in unique_y:
            mask_sy = (sens_train == s) & (y_train == y)
            mask_s  = (sens_train == s)
            mask_y  = (y_train == y)

            p_s  = mask_s.sum()  / n
            p_y  = mask_y.sum()  / n
            p_sy = mask_sy.sum() / n

            if p_sy > 0:
                weights[mask_sy] = (p_s * p_y) / p_sy

    return weights


# ── Audit (unmitigated) ───────────────────────────────────────────────────────

def run_audit(df: pd.DataFrame, target_col: str, sensitive_col: str) -> Dict[str, Any]:
    """
    Train a LogisticRegression on *df*, compute accuracy & Fairlearn bias
    metrics, and return a structured dictionary of results.
    """
    df_encoded, label_encoders = _encode_dataframe(df, target_col)

    X         = df_encoded.drop(columns=[target_col])
    y         = df_encoded[target_col].values
    sensitive = df_encoded[sensitive_col].values

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    metrics = _compute_metrics(y_test, y_pred, sens_test, sensitive_col, label_encoders)

    return {
        "dataset_shape":  {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_column":  target_col,
        "sensitive_column": sensitive_col,
        **metrics,
    }


# ── Mitigated Audit ───────────────────────────────────────────────────────────

MitigationMethod = Literal["reweighing", "exponentiated_gradient", "threshold_optimizer"]

METHOD_DESCRIPTIONS = {
    "reweighing": {
        "name": "Reweighing (Pre-processing)",
        "description": (
            "Assigns higher loss weights to under-represented (group, label) "
            "pairs so the model treats all demographic groups equally during "
            "training. Zero architectural change; works with any base learner."
        ),
        "trade_off": "Minimal accuracy cost; best when bias is in training distribution."
    },
    "exponentiated_gradient": {
        "name": "Exponentiated Gradient + Demographic Parity (In-processing)",
        "description": (
            "Converts the fairness constraint into a Lagrangian saddle-point "
            "problem and trains a mixture of classifiers whose convex combination "
            "satisfies Demographic Parity. Uses multiplicative weight updates — "
            "analogous to AdaBoost but for fairness."
        ),
        "trade_off": "Moderate accuracy-fairness trade-off; powerful for hard constraints."
    },
    "threshold_optimizer": {
        "name": "Threshold Optimizer (Post-processing)",
        "description": (
            "Trains an unconstrained base model, then finds per-group decision "
            "thresholds on the ROC curve that jointly minimise accuracy loss "
            "while satisfying Demographic Parity. No retraining required."
        ),
        "trade_off": "Requires access to sensitive attribute at inference time."
    },
}


def run_mitigated_audit(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    method: MitigationMethod = "exponentiated_gradient",
) -> Dict[str, Any]:
    """
    Run both an unmitigated and a mitigated audit and return side-by-side
    results with improvement metrics.

    Parameters
    ----------
    df             : Input dataframe
    target_col     : Name of the binary target column
    sensitive_col  : Name of the protected attribute column
    method         : One of 'reweighing' | 'exponentiated_gradient' |
                     'threshold_optimizer'
    """
    df_encoded, label_encoders = _encode_dataframe(df, target_col)

    X         = df_encoded.drop(columns=[target_col])
    y         = df_encoded[target_col].values
    sensitive = df_encoded[sensitive_col].values

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 1. Unmitigated baseline ───────────────────────────────────────────────
    base_model = LogisticRegression(max_iter=500, random_state=42)
    base_model.fit(X_train_sc, y_train)
    y_pred_original = base_model.predict(X_test_sc)
    original_metrics = _compute_metrics(
        y_test, y_pred_original, sens_test, sensitive_col, label_encoders)

    # ── 2. Mitigated model ────────────────────────────────────────────────────
    if method == "reweighing":
        # Pre-processing: compute sample weights, retrain the same architecture
        sample_weights = _compute_reweighing_weights(y_train, sens_train)
        mitigated_model = LogisticRegression(max_iter=500, random_state=42)
        mitigated_model.fit(X_train_sc, y_train, sample_weight=sample_weights)
        y_pred_mitigated = mitigated_model.predict(X_test_sc)

    elif method == "threshold_optimizer":
        # Post-processing: train unconstrained, then fit per-group thresholds
        unconstrained = LogisticRegression(max_iter=500, random_state=42)
        unconstrained.fit(X_train_sc, y_train)

        optimizer = ThresholdOptimizer(
            estimator=unconstrained,
            constraints="demographic_parity",
            objective="accuracy_score",
            predict_method="predict_proba",
        )
        optimizer.fit(X_train_sc, y_train, sensitive_features=sens_train)
        y_pred_mitigated = optimizer.predict(X_test_sc, sensitive_features=sens_test)

    else:  # default: exponentiated_gradient
        constraint = DemographicParity()
        mitigator = ExponentiatedGradient(
            estimator=LogisticRegression(max_iter=500, random_state=42),
            constraints=constraint,
        )
        mitigator.fit(X_train_sc, y_train, sensitive_features=sens_train)
        y_pred_mitigated = mitigator.predict(X_test_sc)

    mitigated_metrics = _compute_metrics(
        y_test, y_pred_mitigated, sens_test, sensitive_col, label_encoders)

    # ── 3. Improvement statistics ─────────────────────────────────────────────
    dp_before = abs(original_metrics["bias_metrics"]["demographic_parity_difference"])
    dp_after  = abs(mitigated_metrics["bias_metrics"]["demographic_parity_difference"])
    dp_reduction = round((dp_before - dp_after) / dp_before * 100, 1) if dp_before > 0 else 0.0

    eo_before = abs(original_metrics["bias_metrics"]["equalized_odds_difference"])
    eo_after  = abs(mitigated_metrics["bias_metrics"]["equalized_odds_difference"])
    eo_reduction = round((eo_before - eo_after) / eo_before * 100, 1) if eo_before > 0 else 0.0

    acc_change = round(
        (mitigated_metrics["model_metrics"]["accuracy"] -
         original_metrics["model_metrics"]["accuracy"]) * 100, 2)

    method_info = METHOD_DESCRIPTIONS[method]

    return {
        "dataset_shape":     {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_column":     target_col,
        "sensitive_column":  sensitive_col,
        "original":          original_metrics,
        "mitigated":         mitigated_metrics,
        "improvement": {
            "dp_reduction_pct":   dp_reduction,
            "eo_reduction_pct":   eo_reduction,
            "accuracy_change_pct": acc_change,
            "method":             method_info["name"],
            "method_key":         method,
            "description":        method_info["description"],
            "trade_off":          method_info["trade_off"],
        },
    }

def run_mitigation(df: pd.DataFrame, target_col: str, sensitive_col: str) -> Dict[str, Any]:
    """
    Apply mathematical reweighing, retrain, and return metrics + CSV string.
    """
    df_encoded, label_encoders = _encode_dataframe(df, target_col)

    X         = df_encoded.drop(columns=[target_col])
    y         = df_encoded[target_col].values
    sensitive = df_encoded[sensitive_col].values

    sample_weights = _compute_reweighing_weights(y, sensitive)

    X_train, X_test, y_train, y_test, sens_train, sens_test, w_train, w_test = train_test_split(
        X, y, sensitive, sample_weights, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 1. Unmitigated baseline
    base_model = LogisticRegression(max_iter=500, random_state=42)
    base_model.fit(X_train_sc, y_train)
    y_pred_original = base_model.predict(X_test_sc)
    original_metrics = _compute_metrics(
        y_test, y_pred_original, sens_test, sensitive_col, label_encoders)

    # 2. Mitigated model
    mitigated_model = LogisticRegression(max_iter=500, random_state=42)
    mitigated_model.fit(X_train_sc, y_train, sample_weight=w_train)
    y_pred_mitigated = mitigated_model.predict(X_test_sc)
    
    mitigated_metrics = _compute_metrics(
        y_test, y_pred_mitigated, sens_test, sensitive_col, label_encoders)

    dp_before = abs(original_metrics["bias_metrics"]["demographic_parity_difference"])
    dp_after  = abs(mitigated_metrics["bias_metrics"]["demographic_parity_difference"])
    dp_reduction = round((dp_before - dp_after) / dp_before * 100, 1) if dp_before > 0 else 0.0

    eo_before = abs(original_metrics["bias_metrics"]["equalized_odds_difference"])
    eo_after  = abs(mitigated_metrics["bias_metrics"]["equalized_odds_difference"])
    eo_reduction = round((eo_before - eo_after) / eo_before * 100, 1) if eo_before > 0 else 0.0

    acc_change = round(
        (mitigated_metrics["model_metrics"]["accuracy"] -
         original_metrics["model_metrics"]["accuracy"]) * 100, 2)

    # Mitigated dataset
    df_mitigated = df.copy()
    df_mitigated["mitigation_weight"] = sample_weights
    csv_string = df_mitigated.to_csv(index=False)
    
    return {
        "metrics": {
            "dataset_shape":     {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "target_column":     target_col,
            "sensitive_column":  sensitive_col,
            "original":          original_metrics,
            "mitigated":         mitigated_metrics,
            "improvement": {
                "dp_reduction_pct":   dp_reduction,
                "eo_reduction_pct":   eo_reduction,
                "accuracy_change_pct": acc_change,
                "method":             "Reweighing (Pre-processing)",
                "method_key":         "reweighing",
                "description":        "Assigns higher loss weights to under-represented (group, label) pairs.",
                "trade_off":          "Minimal accuracy cost; best when bias is in training distribution.",
            },
        },
        "csv_string": csv_string
    }
