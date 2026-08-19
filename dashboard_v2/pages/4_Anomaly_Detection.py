"""
pages/4_Anomaly_Detection.py

Filterable view over the flagged anomalies from both detection methods
(rolling z-score + Isolation Forest), plus a per-asset timeline so you can
see exactly where in the sensor trace an anomaly landed.
"""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.ui import page_setup
from utils.pipeline import get_anomalies, load_raw_data

page_setup("Anomaly Detection", icon="🚨")

st.caption(
    "Two methods run side by side: rolling per-asset z-score (fast, explainable, catches "
    "sudden spikes) and Isolation Forest across all sensors together (catches subtler "
    "multivariate drift). Readings flagged by both are the highest-confidence alerts."
)

df = get_anomalies()
telemetry, assets, connectivity = load_raw_data()
df = df.merge(assets[["asset_id", "asset_type", "asset_name"]], on="asset_id", how="left")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total readings", f"{len(df):,}")
c2.metric("Flagged (either method)", int(df.any_anomaly.sum()))
c3.metric("Flagged by both", int(df.both_methods_agree.sum()))
c4.metric("% of readings flagged", f"{df.any_anomaly.mean()*100:.2f}%")

st.divider()

st.sidebar.header("Filters")
site_filter = st.sidebar.multiselect("Site", sorted(df.site_id.unique()), default=sorted(df.site_id.unique()))
type_filter = st.sidebar.multiselect("Asset type", sorted(df.asset_type.dropna().unique()),
                                      default=sorted(df.asset_type.dropna().unique()))
confidence = st.sidebar.radio("Confidence", ["Either method", "Both methods only"], index=0)

flagged = df[df.any_anomaly & df.site_id.isin(site_filter) & df.asset_type.isin(type_filter)]
if confidence == "Both methods only":
    flagged = flagged[flagged.both_methods_agree]

tab1, tab2, tab3 = st.tabs(["Alert feed", "By asset type", "Asset timeline"])

with tab1:
    st.write(f"{len(flagged):,} flagged readings match the current filters.")
    show_cols = ["timestamp", "asset_id", "asset_name", "asset_type", "site_id",
                 "temperature", "vibration", "power_consumption",
                 "stat_anomaly", "iforest_anomaly"]
    st.dataframe(
        flagged[show_cols].sort_values("timestamp", ascending=False).head(200),
        use_container_width=True, hide_index=True,
    )

with tab2:
    by_type = df[df.site_id.isin(site_filter)].groupby("asset_type").any_anomaly.mean().reset_index()
    by_type.any_anomaly *= 100
    fig = px.bar(by_type, x="asset_type", y="any_anomaly", color="asset_type",
                 labels={"any_anomaly": "% of readings flagged"})
    fig.update_layout(margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fault_overlap = df.groupby("fault_flag").any_anomaly.mean() * 100
    st.caption(
        f"Sanity check: {fault_overlap.get(1, 0):.1f}% of readings on days an asset actually "
        f"faulted are flagged anomalous, vs {fault_overlap.get(0, 0):.1f}% on normal days - "
        f"anomaly flags correlate with real problems rather than just noise."
    )

with tab3:
    assets_with_anomalies = sorted(flagged.asset_id.unique())
    if not assets_with_anomalies:
        st.info("No flagged assets in the current filter selection.")
    else:
        chosen_asset = st.selectbox("Asset", assets_with_anomalies)
        sensor = st.radio("Sensor", ["vibration", "temperature", "power_consumption"], horizontal=True)

        sub = df[df.asset_id == chosen_asset].sort_values("timestamp")
        flagged_points = sub[sub.any_anomaly]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sub.timestamp, y=sub[sensor], mode="lines",
                                   line=dict(color="#3b6ea5", width=1), name=sensor))
        fig2.add_trace(go.Scatter(x=flagged_points.timestamp, y=flagged_points[sensor], mode="markers",
                                   marker=dict(color="#d1495b", size=7), name="flagged anomaly"))
        fig2.update_layout(title=f"{sensor} timeline - {chosen_asset}", margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)
