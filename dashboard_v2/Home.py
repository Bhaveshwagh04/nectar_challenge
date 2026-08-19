"""
Home.py - entry point for the Nectar Intelligent Facilities Platform app.

Run from the project root:
    streamlit run dashboard_v2/Home.py

The other five tasks live in pages/ as a native Streamlit multipage app -
use the sidebar to jump between them. Data and models are cached, so the
first click into a page (e.g. Predictive Maintenance) takes a bit to train,
then it's instant on every page revisit within the same session.
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.ui import page_setup
from utils.pipeline import load_raw_data, get_clean_telemetry

page_setup("Intelligent Facilities Platform", icon="🏢")

st.caption(
    "Coimbatore-based multi-site portfolio · AI-first predictive maintenance, "
    "energy forecasting, anomaly detection and asset connectivity in one place."
)

telemetry, assets, connectivity = load_raw_data()
clean = get_clean_telemetry(telemetry)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sites", telemetry.site_id.nunique())
col2.metric("Buildings", telemetry.building_id.nunique())
col3.metric("Assets monitored", telemetry.asset_id.nunique())
col4.metric("Telemetry rows", f"{len(telemetry):,}")
col5.metric("Faults logged", int(telemetry.fault_flag.sum()))

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Portfolio energy trend")
    daily = clean.set_index("timestamp").resample("D").power_consumption.sum().reset_index()
    fig = px.area(daily, x="timestamp", y="power_consumption",
                   labels={"power_consumption": "Total daily energy (kWh)", "timestamp": ""})
    fig.update_traces(line_color="#2E4B6B", fillcolor="rgba(46,75,107,0.15)")
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Assets by type")
    counts = assets.asset_type.value_counts().reset_index()
    counts.columns = ["asset_type", "count"]
    fig2 = px.pie(counts, names="asset_type", values="count", hole=0.45)
    fig2.update_layout(margin=dict(t=10, b=10), showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Where to go next")

nav1, nav2, nav3, nav4, nav5 = st.columns(5)
with nav1:
    st.markdown("**📊 EDA**")
    st.caption("Distributions, patterns, failure precursors")
with nav2:
    st.markdown("**🔧 Predictive Maintenance**")
    st.caption("24h-ahead failure risk, live scoring")
with nav3:
    st.markdown("**⚡ Energy Forecasting**")
    st.caption("24h-ahead building load forecast")
with nav4:
    st.markdown("**🚨 Anomaly Detection**")
    st.caption("Flagged sensor anomalies, live alerts")
with nav5:
    st.markdown("**🔗 Asset Connectivity**")
    st.caption("Dependency graph, failure impact queries")

st.info("Use the sidebar (top-left `>`) to navigate between pages.", icon="👈")

with st.expander("About this dataset"):
    st.write(
        "No dataset was attached to the original challenge brief, so this app runs on a "
        "synthetic-but-realistic dataset generated from the provided schema — see "
        "`src/generate_data.py` and the project README for details and assumptions. "
        "Point `data/raw/` at real telemetry with the same column names and everything "
        "here runs unchanged."
    )
