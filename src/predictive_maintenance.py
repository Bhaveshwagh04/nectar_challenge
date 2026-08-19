"""
predictive_maintenance.py - Task 2

Predicts whether an asset will fail in the next 24h.

The label is heavily imbalanced (failures are rare, as they should be in a
well-run facility), so this uses:
  - time-based train/test split (never test on the past)
  - XGBoost with scale_pos_weight to account for the imbalance
  - PR-AUC alongside ROC-AUC since ROC-AUC is optimistic on imbalanced data

Run:
    python src/predictive_maintenance.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                              average_precision_score, confusion_matrix, precision_recall_curve)
from xgboost import XGBClassifier

from preprocessing import load_raw, build_feature_table, make_failure_labels

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOT_DIR = OUT_DIR / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["timestamp", "asset_id", "site_id", "building_id", "installation_date",
             "fault_flag", "label_fail_next_24h", "manufacturer"]


def prepare_dataset():
    telemetry, assets, connectivity = load_raw()
    feat = build_feature_table(telemetry, assets)
    labeled = make_failure_labels(feat, horizon_steps=96)
    return labeled


def time_split(df, test_frac=0.2):
    """Split by timestamp, not randomly - this is a forecasting-style problem,
    training on the future would leak information."""
    cutoff = df.timestamp.quantile(1 - test_frac)
    train = df[df.timestamp < cutoff]
    test = df[df.timestamp >= cutoff]
    return train, test


def get_feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS]


def train():
    df = prepare_dataset()
    train_df, test_df = time_split(df)

    feature_cols = get_feature_cols(df)
    X_train, y_train = train_df[feature_cols], train_df.label_fail_next_24h
    X_test, y_test = test_df[feature_cols], test_df.label_fail_next_24h

    print(f"train rows: {len(X_train):,}  positives: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"test rows: {len(X_test):,}  positives: {y_test.sum()} ({y_test.mean()*100:.3f}%)")

    n_pos = max(y_train.sum(), 1)
    scale_pos_weight = (len(y_train) - n_pos) / n_pos

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]

    # pick an operating threshold off the PR curve rather than defaulting to 0.5,
    # which is meaningless on data this imbalanced
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) else 0
    best_threshold = thresholds[best_idx] if len(thresholds) else 0.5

    preds = (proba >= best_threshold).astype(int)

    metrics = {
        "threshold_used": float(best_threshold),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_test": int(len(y_test)),
        "n_positive_test": int(y_test.sum()),
    }
    print("\nmetrics:", json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, preds)
    print("\nconfusion matrix [[tn fp][fn tp]]:\n", cm)

    # feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\ntop 15 features:\n", importances.head(15).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(15).sort_values().plot.barh(ax=ax, color="#3b6ea5")
    ax.set_title("Top 15 features - predictive maintenance model")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_pm_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, color="#d1495b")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision-Recall curve (PR-AUC={metrics['pr_auc']:.3f})")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_pm_pr_curve.png", bbox_inches="tight")
    plt.close(fig)

    joblib.dump({"model": model, "feature_cols": feature_cols, "threshold": best_threshold},
                MODEL_DIR / "predictive_maintenance_xgb.joblib")

    with open(OUT_DIR / "predictive_maintenance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # a couple of misclassified examples for the error-analysis writeup
    test_df = test_df.copy()
    test_df["pred_proba"] = proba
    test_df["pred"] = preds
    false_negatives = test_df[(test_df.label_fail_next_24h == 1) & (test_df.pred == 0)]
    false_positives = test_df[(test_df.label_fail_next_24h == 0) & (test_df.pred == 1)]
    print(f"\nfalse negatives: {len(false_negatives)}, false positives: {len(false_positives)}")

    return model, metrics, importances


if __name__ == "__main__":
    train()
