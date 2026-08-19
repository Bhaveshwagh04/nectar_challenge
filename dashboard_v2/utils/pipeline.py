"""
utils/pipeline.py

Thin caching layer over the existing src/ modules. The app never
re-implements EDA/model/graph logic - it just calls into src/ and caches the
results so switching between pages doesn't retrain a model or rebuild a
380k-row dataframe on every click.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import preprocessing as pp  # noqa: E402
import predictive_maintenance as pm  # noqa: E402
import forecasting as fc  # noqa: E402
import anomaly_detection as ad  # noqa: E402
import connectivity_analysis as ca  # noqa: E402


@st.cache_data(show_spinner="Loading telemetry, asset metadata and connectivity...")
def load_raw_data():
    telemetry, assets, connectivity = pp.load_raw()
    return telemetry, assets, connectivity


@st.cache_data(show_spinner="Cleaning telemetry...")
def get_clean_telemetry(_telemetry):
    # leading underscore on the arg tells streamlit not to hash the (large) dataframe itself
    return pp.clean_telemetry(_telemetry)


@st.cache_resource(show_spinner="Training predictive maintenance model - this takes a minute the first time...")
def get_pm_model():
    df = pm.prepare_dataset()
    train_df, test_df = pm.time_split(df)
    feature_cols = pm.get_feature_cols(df)

    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                                  average_precision_score, precision_recall_curve, confusion_matrix)

    X_train, y_train = train_df[feature_cols], train_df.label_fail_next_24h
    X_test, y_test = test_df[feature_cols], test_df.label_fail_next_24h

    n_pos = max(y_train.sum(), 1)
    scale_pos_weight = (len(y_train) - n_pos) / n_pos

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) else 0
    threshold = thresholds[best_idx] if len(thresholds) else 0.5
    preds = (proba >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    importances = None
    import pandas as pd
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    test_scored = test_df.copy()
    test_scored["pred_proba"] = proba
    test_scored["pred"] = preds

    return {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "metrics": metrics,
        "importances": importances,
        "test_scored": test_scored,
        "precisions": precisions,
        "recalls": recalls,
    }


@st.cache_resource(show_spinner="Training forecasting model...")
def get_forecast_model():
    telemetry, assets, connectivity = pp.load_raw()
    series = fc.build_building_series(telemetry)
    feat = fc.add_forecast_features(series)
    feat = feat.dropna(subset=["target"] + [f"lag_{l}" for l in fc.LAGS])

    drop_cols = ["timestamp", "target", "power_consumption"]
    feature_cols = [c for c in feat.columns if c not in drop_cols]
    train_df, test_df = fc.time_split(feat)

    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df.target)
    pred = model.predict(test_df[feature_cols])

    test_df = test_df.copy()
    test_df["pred"] = pred

    metrics = {
        "MAE": float(mean_absolute_error(test_df.target, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(test_df.target, pred))),
        "MAPE": fc.mape(test_df.target, pred),
    }

    bldg_cols = [c for c in feature_cols if c.startswith("bldg_")]
    buildings = [c.replace("bldg_", "") for c in bldg_cols]

    return {
        "model": model, "feature_cols": feature_cols, "metrics": metrics,
        "test_df": test_df, "bldg_cols": bldg_cols, "buildings": buildings,
    }


@st.cache_data(show_spinner="Running anomaly detection across all sensors...")
def get_anomalies():
    telemetry, assets, connectivity = pp.load_raw()
    df = pp.clean_telemetry(telemetry)
    df = pp.add_time_features(df)
    df = ad.statistical_anomalies(df)
    df, _, _ = ad.isolation_forest_anomalies(df)
    df["any_anomaly"] = df.stat_anomaly | df.iforest_anomaly
    df["both_methods_agree"] = df.stat_anomaly & df.iforest_anomaly
    return df


@st.cache_resource(show_spinner="Building the asset dependency graph...")
def get_graph():
    telemetry, assets, connectivity = pp.load_raw()
    G = ca.build_graph(assets, connectivity)
    dq = ca.data_quality_checks(assets, connectivity, G)
    return G, dq
