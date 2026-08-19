"""
pages/1_EDA.py - interactive version of Task 1 (exploratory analysis).

Every chart here responds to the sidebar filters - unlike the static PNGs
in outputs/plots/, this recomputes on the filtered slice so you can drill
into a single site or asset type and see the pattern hold (or not).
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.ui import page_setup, ASSET_TYPE_COLORS
from utils.pipeline import load_raw_data, get_clean_telemetry

page_setup("Exploratory Data Analysis", icon="📊")

telemetry, assets, connectivity = load_raw_data()
clean = get_clean_telemetry(telemetry)
df = clean.merge(assets[["asset_id", "asset_type"]], on="asset_id", how="left")

# ---- filters ----
st.sidebar.header("Filters")
sites = st.sidebar.multiselect("Site", sorted(df.site_id.unique()), default=sorted(df.site_id.unique()))
types = st.sidebar.multiselect("Asset type", sorted(df.asset_type.dropna().unique()),
                                default=sorted(df.asset_type.dropna().unique()))

view = df[df.site_id.isin(sites) & df.asset_type.isin(types)]

if view.empty:
    st.warning("No data matches the current filters.")
    st.stop()

st.caption(f"{len(view):,} readings across {view.asset_id.nunique()} assets in the current filter.")

# ---- missing values ----
with st.expander("Data quality: missing values", expanded=False):
    raw_view = telemetry[telemetry.site_id.isin(sites)]
    miss = raw_view.isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]
    if len(miss):
        st.bar_chart(miss)
    else:
        st.write("No missing values in the current selection.")

# ---- distributions ----
st.subheader("Sensor distributions")
sensor_cols = ["temperature", "humidity", "pressure", "vibration", "power_consumption"]
sel_col = st.selectbox("Sensor", sensor_cols, index=3)
fig = px.histogram(view, x=sel_col, color="asset_type", nbins=60, opacity=0.75,
                    color_discrete_map=ASSET_TYPE_COLORS)
fig.update_layout(margin=dict(t=10, b=10), bargap=0.02)
st.plotly_chart(fig, use_container_width=True)

# ---- daily / weekly patterns ----
st.subheader("Temporal patterns")
c1, c2 = st.columns(2)

with c1:
    hourly = view.copy()
    hourly["hour"] = hourly.timestamp.dt.hour
    hourly_avg = hourly.groupby("hour")[["power_consumption", "occupancy_count"]].mean().reset_index()
    fig_h = px.line(hourly_avg, x="hour", y="power_consumption", markers=True,
                     title="Avg power draw by hour of day")
    fig_h.update_traces(line_color="#d1495b")
    fig_h.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig_h, use_container_width=True)

with c2:
    weekday = view.copy()
    weekday["dow"] = weekday.timestamp.dt.dayofweek
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_avg = weekday.groupby("dow").power_consumption.mean().reindex(range(7))
    fig_w = px.bar(x=labels, y=weekday_avg.values, title="Avg power draw by day of week",
                    labels={"x": "", "y": "power_consumption"})
    fig_w.update_traces(marker_color="#2E4B6B")
    fig_w.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig_w, use_container_width=True)

# ---- asset type comparison ----
st.subheader("Comparison across asset types")
type_summary = view.groupby("asset_type")[sensor_cols].mean().reset_index()
metric_choice = st.radio("Metric", sensor_cols, horizontal=True, index=4)
fig_t = px.bar(type_summary, x="asset_type", y=metric_choice, color="asset_type",
                color_discrete_map=ASSET_TYPE_COLORS)
fig_t.update_layout(margin=dict(t=10, b=10), showlegend=False)
st.plotly_chart(fig_t, use_container_width=True)

# ---- correlation heatmap ----
st.subheader("Sensor correlation matrix")
corr = view[sensor_cols + ["occupancy_count"]].corr()
fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
fig_c.update_layout(margin=dict(t=10, b=10))
st.plotly_chart(fig_c, use_container_width=True)

# ---- failure precursors ----
st.subheader("What does equipment look like right before it fails?")
fault_times = view.loc[view.fault_flag == 1, ["asset_id", "timestamp"]]

if len(fault_times):
    windows = []
    for _, row in fault_times.iterrows():
        w = view[(view.asset_id == row.asset_id) &
                 (view.timestamp <= row.timestamp) &
                 (view.timestamp > row.timestamp - pd.Timedelta(hours=24))]
        windows.append(w)
    pre_fault = pd.concat(windows) if windows else pd.DataFrame(columns=view.columns)
    pre_fault["group"] = "24h pre-fault"

    normal_n = min(len(pre_fault) * 5, len(view[view.fault_flag == 0]))
    normal_sample = view[view.fault_flag == 0].sample(max(normal_n, 1), random_state=1)
    normal_sample["group"] = "normal"

    combined = pd.concat([normal_sample, pre_fault])
    box_col = st.selectbox("Sensor to compare", ["temperature", "vibration", "power_consumption"], index=1)
    fig_box = px.box(combined, x="group", y=box_col, color="group",
                      color_discrete_map={"normal": "#3b6ea5", "24h pre-fault": "#d1495b"})
    fig_box.update_layout(margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    diff = pre_fault[box_col].mean() - normal_sample[box_col].mean()
    pct = diff / normal_sample[box_col].mean() * 100 if normal_sample[box_col].mean() else 0
    st.caption(f"Average {box_col} runs {pct:+.1f}% relative to normal in the 24h before a fault, "
               f"in the current filter selection.")
else:
    st.info("No fault events in the current filter selection to compare against.")
