"""
anomaly_detection.py - Task 4

Combines two approaches on purpose - they catch different things:

1. Statistical thresholding (rolling z-score per asset): cheap, explainable,
   good at catching sudden spikes (power surge, vibration jump). Ops teams
   can understand "6 std above this asset's own recent baseline" without
   needing to trust a black box.

2. Isolation Forest on a multivariate feature set: catches the subtler stuff
   a single-sensor threshold would miss - e.g. temperature and vibration each
   look "normal" in isolation but the combination is unusual (sensor drift,
   early-stage degradation).

Run:
    python src/anomaly_detection.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from preprocessing import load_raw, clean_telemetry, add_time_features, SENSOR_COLS

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOT_DIR = OUT_DIR / "plots"

Z_THRESHOLD = 4.0
ROLLING_WINDOW = 96  # 24h, for computing each asset's own baseline


def statistical_anomalies(df):
    df = df.sort_values(["asset_id", "timestamp"]).copy()
    flags = pd.DataFrame(index=df.index)

    for col in SENSOR_COLS:
        roll_mean = df.groupby("asset_id")[col].transform(
            lambda s: s.rolling(ROLLING_WINDOW, min_periods=20).mean())
        roll_std = df.groupby("asset_id")[col].transform(
            lambda s: s.rolling(ROLLING_WINDOW, min_periods=20).std())
        z = (df[col] - roll_mean) / roll_std.replace(0, np.nan)
        flags[f"{col}_zscore"] = z
        flags[f"{col}_anomaly"] = (z.abs() > Z_THRESHOLD).fillna(False)

    df["stat_anomaly"] = flags[[c for c in flags.columns if c.endswith("_anomaly")]].any(axis=1)
    df = pd.concat([df, flags[[c for c in flags.columns if c.endswith("_zscore")]]], axis=1)
    return df


def isolation_forest_anomalies(df, contamination=0.005):
    feature_cols = SENSOR_COLS + ["occupancy_count"]
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    pred = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)  # lower = more anomalous

    df = df.copy()
    df["iforest_anomaly"] = pred == -1
    df["iforest_score"] = scores
    return df, model, scaler


def run():
    telemetry, assets, connectivity = load_raw()
    df = clean_telemetry(telemetry)
    df = add_time_features(df)

    df = statistical_anomalies(df)
    df, iforest_model, scaler = isolation_forest_anomalies(df)

    df["any_anomaly"] = df.stat_anomaly | df.iforest_anomaly
    df["both_methods_agree"] = df.stat_anomaly & df.iforest_anomaly

    n_stat = int(df.stat_anomaly.sum())
    n_iforest = int(df.iforest_anomaly.sum())
    n_both = int(df.both_methods_agree.sum())
    n_any = int(df.any_anomaly.sum())

    summary = {
        "total_rows": int(len(df)),
        "statistical_threshold_flags": n_stat,
        "isolation_forest_flags": n_iforest,
        "flagged_by_both": n_both,
        "flagged_by_either": n_any,
        "pct_flagged": round(n_any / len(df) * 100, 3),
    }
    print(json.dumps(summary, indent=2))

    # how well do the flags line up with actual recorded faults? (sanity check,
    # not a formal eval since anomalies != faults by definition - most anomalies
    # are near-misses or sensor issues rather than outright failures)
    overlap = df.groupby("fault_flag").any_anomaly.mean()
    print("\nanomaly rate by fault_flag (0=normal,1=fault day):\n", overlap)

    # top anomalous events for the report
    top_iforest = df.sort_values("iforest_score").head(10)[
        ["timestamp", "asset_id", "temperature", "vibration", "power_consumption", "iforest_score"]]
    print("\nmost anomalous readings (isolation forest):\n", top_iforest.to_string(index=False))

    # timeline plot for one asset that has anomalies flagged
    assets_with_anomalies = df[df.any_anomaly].asset_id.value_counts()
    if len(assets_with_anomalies):
        example_asset = assets_with_anomalies.index[0]
        sub = df[df.asset_id == example_asset].sort_values("timestamp")

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(sub.timestamp, sub.vibration, color="#3b6ea5", label="vibration", linewidth=1)
        flagged = sub[sub.any_anomaly]
        ax.scatter(flagged.timestamp, flagged.vibration, color="#d1495b", s=25, zorder=5, label="flagged anomaly")
        ax.set_title(f"Vibration timeline with flagged anomalies - {example_asset}")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "11_anomaly_timeline.png", bbox_inches="tight")
        plt.close(fig)

    # anomaly counts by asset type - which equipment is misbehaving most
    merged = df.merge(assets[["asset_id", "asset_type"]], on="asset_id", how="left")
    by_type = merged.groupby("asset_type").any_anomaly.agg(["sum", "mean"]).rename(
        columns={"sum": "anomaly_count", "mean": "anomaly_rate"})
    print("\nanomalies by asset type:\n", by_type.round(4).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(by_type.index, by_type.anomaly_rate * 100, color="#c17f3e")
    ax.set_ylabel("% of readings flagged anomalous")
    ax.set_title("Anomaly rate by asset type")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "12_anomaly_by_asset_type.png", bbox_inches="tight")
    plt.close(fig)

    with open(OUT_DIR / "anomaly_detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    flagged_export = df[df.any_anomaly][
        ["timestamp", "asset_id", "site_id", "building_id", "temperature", "humidity",
         "pressure", "vibration", "power_consumption", "stat_anomaly", "iforest_anomaly", "iforest_score"]
    ].sort_values("timestamp")
    flagged_export.to_csv(OUT_DIR / "flagged_anomalies.csv", index=False)
    print(f"\n{len(flagged_export)} flagged rows written to outputs/flagged_anomalies.csv")

    return df, summary


if __name__ == "__main__":
    run()
