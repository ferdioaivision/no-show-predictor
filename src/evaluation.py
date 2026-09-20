"""
no-show-predictor - evaluation helpers

Ranking metrics do not depend on a threshold; operating-point metrics do.
The operating threshold must be chosen on data that is NOT the test set
(here: out-of-fold predictions on the training set).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def ranking_metrics(y_true, proba) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flagged_rate": float(pred.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def select_threshold(y_true, proba, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target_recall (i.e. the most selective one)."""
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = recall[:-1] >= target_recall
    if not ok.any():
        raise ValueError("Target recall cannot be reached.")
    return float(thresholds[ok].max())


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs observed no-show rate in quantile bins."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True)
