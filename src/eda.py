"""
eda.py - Task 1: Exploratory Data Analysis

Produces the plots + summary stats used in the report. Run directly:
    python src/eda.py
Outputs land in outputs/plots/ and outputs/eda_summary.txt
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from preprocessing import load_raw, clean_telemetry, SENSOR_COLS

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.style.use("seaborn-v0_8-whitegrid")


def missing_value_report(df):
    miss = df.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]
    return miss


def plot_distributions(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(SENSOR_COLS):
        axes[i].hist(df[col].dropna(), bins=60, color="#3b6ea5", edgecolor="none")
        axes[i].set_title(col)
    axes[-1].axis("off")
    fig.suptitle("Sensor value distributions (all assets, all time)", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_sensor_distributions.png", bbox_inches="tight")
    plt.close(fig)


def plot_daily_pattern(df):
    df = df.copy()
    df["hour"] = df.timestamp.dt.hour
    hourly = df.groupby("hour")[["power_consumption", "occupancy_count", "temperature"]].mean()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(hourly.index, hourly.power_consumption, marker="o", color="#d1495b", label="avg power (kWh)")
    ax1.set_xlabel("hour of day")
    ax1.set_ylabel("power_consumption", color="#d1495b")
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly.occupancy_count, marker="s", color="#2e8b57", label="avg occupancy")
    ax2.set_ylabel("occupancy_count", color="#2e8b57")
    ax1.set_title("Average power draw and occupancy by hour of day")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_daily_pattern.png", bbox_inches="tight")
    plt.close(fig)


def plot_weekday_pattern(df):
    df = df.copy()
    df["dow"] = df.timestamp.dt.dayofweek
    weekday_avg = df.groupby("dow").power_consumption.mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, weekday_avg.values, color="#3b6ea5")
    ax.set_ylabel("avg power_consumption (kWh)")
    ax.set_title("Average energy consumption by day of week")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_weekday_pattern.png", bbox_inches="tight")
    plt.close(fig)


def plot_asset_type_comparison(df, assets):
    merged = df.merge(assets[["asset_id", "asset_type"]], on="asset_id", how="left")
    summary = merged.groupby("asset_type")[["power_consumption", "vibration", "temperature"]].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col in zip(axes, ["power_consumption", "vibration", "temperature"]):
        ax.bar(summary.index, summary[col], color="#5b8c5a")
        ax.set_title(f"avg {col} by asset type")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_asset_type_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_site_comparison(df):
    site_summary = df.groupby("site_id").power_consumption.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(site_summary.index, site_summary.values, color="#8a5b8c")
    ax.set_title("Average power consumption by site")
    ax.set_ylabel("kWh")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_site_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return site_summary


def plot_failure_precursors(df):
    """Compare sensor readings in the 24h before a fault vs a random sample of
    normal operating windows - this is the core 'what drives failures' plot.
    """
    df = df.sort_values(["asset_id", "timestamp"]).copy()
    fault_times = df.loc[df.fault_flag == 1, ["asset_id", "timestamp"]]

    pre_fault_rows = []
    for _, row in fault_times.iterrows():
        window = df[(df.asset_id == row.asset_id) &
                     (df.timestamp <= row.timestamp) &
                     (df.timestamp > row.timestamp - pd.Timedelta(hours=24))]
        pre_fault_rows.append(window)
    pre_fault = pd.concat(pre_fault_rows) if pre_fault_rows else pd.DataFrame(columns=df.columns)

    normal_sample = df[df.fault_flag == 0].sample(min(len(pre_fault) * 5, len(df[df.fault_flag == 0])),
                                                    random_state=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col in zip(axes, ["temperature", "vibration", "power_consumption"]):
        ax.boxplot([normal_sample[col].dropna(), pre_fault[col].dropna()],
                   tick_labels=["normal", "24h pre-fault"])
        ax.set_title(col)
    fig.suptitle("Sensor readings: normal operation vs 24h before a fault", y=1.03)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_failure_precursors.png", bbox_inches="tight")
    plt.close(fig)

    return pre_fault, normal_sample


def plot_correlation_heatmap(df):
    corr = df[SENSOR_COLS + ["occupancy_count"]].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Sensor correlation matrix")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return corr


def run():
    telemetry, assets, connectivity = load_raw()

    lines = []
    lines.append(f"telemetry rows: {len(telemetry):,}")
    lines.append(f"assets: {telemetry.asset_id.nunique()}, sites: {telemetry.site_id.nunique()}, "
                 f"buildings: {telemetry.building_id.nunique()}")
    lines.append(f"date range: {telemetry.timestamp.min()} to {telemetry.timestamp.max()}")

    miss = missing_value_report(telemetry)
    lines.append("\nmissing value % by column:")
    lines.append(miss.to_string())

    clean = clean_telemetry(telemetry)

    plot_distributions(clean)
    plot_daily_pattern(clean)
    plot_weekday_pattern(clean)
    type_summary = plot_asset_type_comparison(clean, assets)
    site_summary = plot_site_comparison(clean)
    pre_fault, normal_sample = plot_failure_precursors(clean)
    corr = plot_correlation_heatmap(clean)

    lines.append("\naverage sensor values by asset type:")
    lines.append(type_summary.round(2).to_string())

    lines.append("\naverage power consumption by site:")
    lines.append(site_summary.round(2).to_string())

    lines.append("\nsensor correlation matrix:")
    lines.append(corr.round(2).to_string())

    if len(pre_fault):
        lines.append("\n24h-before-fault vs normal operation, mean values:")
        cmp = pd.DataFrame({
            "normal": normal_sample[["temperature", "vibration", "power_consumption"]].mean(),
            "pre_fault_24h": pre_fault[["temperature", "vibration", "power_consumption"]].mean(),
        })
        cmp["pct_increase"] = ((cmp.pre_fault_24h - cmp.normal) / cmp.normal * 100).round(1)
        lines.append(cmp.round(3).to_string())

    lines.append(f"\ntotal fault events: {int(telemetry.fault_flag.sum())}")
    lines.append(f"assets that experienced >=1 fault: {telemetry[telemetry.fault_flag==1].asset_id.nunique()} "
                 f"of {telemetry.asset_id.nunique()}")

    summary_text = "\n".join(lines)
    (OUT_DIR / "eda_summary.txt").write_text(summary_text)
    print(summary_text)
    print(f"\nplots saved to {PLOT_DIR}")


if __name__ == "__main__":
    run()
