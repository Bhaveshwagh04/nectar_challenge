"""
forecasting.py - Task 3: Energy consumption forecasting

Forecasts next-24h energy consumption per building. Data is at 15-min
resolution so 24h ahead = 96 steps.

Went with gradient-boosted trees on lag/calendar features rather than
Prophet/ARIMA - with 6 buildings and clear daily/weekly seasonality plus a
slow trend, a single global XGBoost model with per-building features
generalizes better than fitting 6 separate ARIMA models, and it's much
faster to retrain as new data comes in. Noted as an assumption in the README.

Run:
    python src/forecasting.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from preprocessing import load_raw, clean_telemetry

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOT_DIR = OUT_DIR / "plots"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 96  # 24h at 15-min resolution
LAGS = [1, 4, 16, 96, 96 * 7]  # 15min, 1h, 4h, 1day, 1week


def build_building_series(telemetry):
    df = clean_telemetry(telemetry)
    building_power = (
        df.groupby(["building_id", "timestamp"]).power_consumption.sum().reset_index()
    )
    return building_power


def add_forecast_features(series_df):
    df = series_df.sort_values(["building_id", "timestamp"]).copy()
    g = df.groupby("building_id")["power_consumption"]

    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)

    df["rollmean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["rollmean_96"] = g.transform(lambda s: s.shift(1).rolling(96).mean())
    df["rollstd_96"] = g.transform(lambda s: s.shift(1).rolling(96).std())

    df["hour"] = df.timestamp.dt.hour
    df["dow"] = df.timestamp.dt.dayofweek
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)

    # the target: power_consumption HORIZON steps ahead, per building
    df["target"] = df.groupby("building_id")["power_consumption"].shift(-HORIZON)

    df = pd.get_dummies(df, columns=["building_id"], prefix="bldg")
    return df


def time_split(df, test_frac=0.2):
    cutoff = df.timestamp.quantile(1 - test_frac)
    return df[df.timestamp < cutoff], df[df.timestamp >= cutoff]


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_and_evaluate():
    telemetry, assets, connectivity = load_raw()
    series = build_building_series(telemetry)
    feat = add_forecast_features(series)
    feat = feat.dropna(subset=["target"] + [f"lag_{l}" for l in LAGS])

    drop_cols = ["timestamp", "target", "power_consumption"]
    feature_cols = [c for c in feat.columns if c not in drop_cols]

    train_df, test_df = time_split(feat)
    X_train, y_train = train_df[feature_cols], train_df.target
    X_test, y_test = test_df[feature_cols], test_df.target

    print(f"train rows: {len(X_train):,}, test rows: {len(X_test):,}")

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAPE_pct": mape(y_test, pred),
    }
    print("metrics:", json.dumps(metrics, indent=2))

    with open(OUT_DIR / "forecasting_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # plot actual vs predicted for one building over the test window
    bldg_col = [c for c in feature_cols if c.startswith("bldg_")][0]
    example_bldg = bldg_col.replace("bldg_", "")
    mask = test_df[bldg_col] == 1
    plot_df = test_df[mask].copy()
    plot_df["pred"] = pred[mask.values]
    plot_df = plot_df.sort_values("timestamp").tail(96 * 5)  # last 5 days of test window

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df.timestamp, plot_df.target, label="actual", color="#3b6ea5")
    ax.plot(plot_df.timestamp, plot_df.pred, label="forecast (24h ahead)", color="#d1495b", linestyle="--")
    ax.set_title(f"Energy forecast vs actual - {example_bldg}")
    ax.set_ylabel("power_consumption (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_forecast_vs_actual.png", bbox_inches="tight")
    plt.close(fig)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\ntop 10 features:\n", importances.head(10).to_string())

    # persist the model so the dashboard/API can reuse it without retraining
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_DIR / "forecasting_xgb.joblib")

    # export actual vs predicted for every building in the test window, not just
    # the one example plotted above - this is what the dashboard reads to let
    # someone pick any building interactively
    bldg_cols = [c for c in feature_cols if c.startswith("bldg_")]
    export_rows = []
    test_export = test_df.copy()
    test_export["pred"] = pred
    for col in bldg_cols:
        bname = col.replace("bldg_", "")
        sub = test_export[test_export[col] == 1][["timestamp", "target", "pred"]].copy()
        sub["building_id"] = bname
        export_rows.append(sub)
    all_preds = pd.concat(export_rows).rename(columns={"target": "actual"})
    all_preds = all_preds.sort_values(["building_id", "timestamp"])
    all_preds.to_csv(OUT_DIR / "forecast_test_predictions.csv", index=False)
    print(f"\nper-building test predictions written to outputs/forecast_test_predictions.csv "
          f"({len(all_preds):,} rows across {all_preds.building_id.nunique()} buildings)")

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
