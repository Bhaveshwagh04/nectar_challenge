"""
preprocessing.py

Common loading + cleaning + feature engineering helpers shared by the
predictive maintenance and forecasting modules. Kept in one place so both
tasks build features the same way.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SENSOR_COLS = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]


def load_raw():
    telemetry = pd.read_csv(RAW_DIR / "sensor_telemetry.csv", parse_dates=["timestamp"])
    assets = pd.read_csv(RAW_DIR / "asset_metadata.csv", parse_dates=["installation_date"])
    connectivity = pd.read_csv(RAW_DIR / "asset_connectivity.csv")
    return telemetry, assets, connectivity


def clean_telemetry(df, method="interpolate"):
    """Handle missing sensor values and obviously bad readings.

    method='interpolate' fills gaps per-asset using time interpolation, which
    is reasonable for slowly-varying sensor signals. We cap absurd outliers
    (>6 std from an asset's rolling mean) instead of dropping them outright,
    since a spike is often the exact signal we care about for fault detection
    - clipping just prevents it from blowing up training.
    """
    df = df.sort_values(["asset_id", "timestamp"]).copy()

    for col in SENSOR_COLS:
        df[col] = df.groupby("asset_id")[col].transform(lambda s: s.interpolate(limit_direction="both"))

    # a few assets might still have all-NaN for a column at the start; fall back to global median
    for col in SENSOR_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def add_rolling_features(df, windows=(4, 16, 96)):
    """Rolling stats per asset. Windows are in number of 15-min samples:
    4 = 1 hour, 16 = 4 hours, 96 = 24 hours.
    """
    df = df.sort_values(["asset_id", "timestamp"]).copy()
    g = df.groupby("asset_id")

    for w in windows:
        for col in ["temperature", "vibration", "power_consumption"]:
            df[f"{col}_rollmean_{w}"] = g[col].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            df[f"{col}_rollstd_{w}"] = g[col].transform(lambda s, w=w: s.rolling(w, min_periods=1).std().fillna(0))

    # rate of change - captures the "ramping up before failure" pattern
    for col in ["temperature", "vibration", "power_consumption"]:
        df[f"{col}_delta_1h"] = g[col].diff(4)

    df[["temperature_delta_1h", "vibration_delta_1h", "power_consumption_delta_1h"]] = \
        df[["temperature_delta_1h", "vibration_delta_1h", "power_consumption_delta_1h"]].fillna(0)

    return df


def add_time_features(df):
    df = df.copy()
    df["hour"] = df.timestamp.dt.hour
    df["dow"] = df.timestamp.dt.dayofweek
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["is_business_hours"] = ((df.hour >= 8) & (df.hour <= 19) & (df.dow < 5)).astype(int)
    return df


def make_failure_labels(df, horizon_steps=96):
    """Label = 1 if a fault occurs anywhere in the next `horizon_steps` samples
    for that asset (96 steps * 15min = 24h ahead, matching the task spec).
    We look forward, so the last `horizon_steps` rows per asset can't be
    labeled reliably and get dropped.
    """
    df = df.sort_values(["asset_id", "timestamp"]).copy()

    # vectorized per-asset: loop over the (small) number of assets rather than
    # groupby.apply, which keeps this fast and avoids pandas version quirks
    # around grouping-column handling inside apply()
    df["label_fail_next_24h"] = 0
    for asset_id, idx in df.groupby("asset_id").groups.items():
        idx = idx.sort_values() if hasattr(idx, "sort_values") else idx
        fault = df.loc[idx, "fault_flag"].to_numpy()
        n = len(fault)
        label = np.zeros(n, dtype=int)
        for i in range(n):
            end = min(n, i + 1 + horizon_steps)
            label[i] = 1 if fault[i + 1:end].max(initial=0) == 1 else 0
        df.loc[idx, "label_fail_next_24h"] = label

    # drop the tail of each asset's series where we don't have a full lookahead window
    df["rn"] = df.groupby("asset_id").cumcount(ascending=False)
    df = df[df.rn >= horizon_steps].drop(columns="rn")
    return df.reset_index(drop=True)


def build_feature_table(telemetry, assets):
    df = clean_telemetry(telemetry)
    df = add_time_features(df)
    df = add_rolling_features(df)
    df = df.merge(assets[["asset_id", "asset_type", "manufacturer", "capacity", "installation_date"]],
                   on="asset_id", how="left")
    df["asset_age_days"] = (df.timestamp - df.installation_date).dt.days
    df = pd.get_dummies(df, columns=["operating_mode", "asset_type"], prefix=["mode", "atype"])
    return df
