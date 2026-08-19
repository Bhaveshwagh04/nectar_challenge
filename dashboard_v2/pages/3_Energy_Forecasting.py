"""
pages/3_Energy_Forecasting.py

Lets you pick a building and see the 24h-ahead forecast against what
actually happened in the test window, plus the metrics and which features
the model leans on.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.ui import page_setup
from utils.pipeline import get_forecast_model

page_setup("Energy Consumption Forecasting", icon="⚡")

st.caption("XGBoost forecast of building-level power consumption, 24 hours ahead.")

bundle = get_forecast_model()
metrics = bundle["metrics"]

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{metrics['MAE']:.1f} kWh")
c2.metric("RMSE", f"{metrics['RMSE']:.1f} kWh")
c3.metric("MAPE", f"{metrics['MAPE']:.1f} %")

st.divider()

selected_building = st.selectbox("Building", bundle["buildings"])
bldg_col = f"bldg_{selected_building}"

test_df = bundle["test_df"]
sub = test_df[test_df[bldg_col] == 1].sort_values("timestamp")

days_back = st.slider("Days to show", 1, 14, 5)
sub = sub.tail(days_back * 96)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sub.timestamp, y=sub.target, name="actual",
                          line=dict(color="#2E4B6B", width=2)))
fig.add_trace(go.Scatter(x=sub.timestamp, y=sub.pred, name="forecast (24h ahead)",
                          line=dict(color="#d1495b", width=2, dash="dash")))
fig.update_layout(title=f"Actual vs forecast - {selected_building}",
                   yaxis_title="power_consumption (kWh)", margin=dict(t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

err = (sub.pred - sub.target).abs()
st.caption(f"Mean absolute error for {selected_building} over this window: {err.mean():.1f} kWh "
           f"({(err.mean() / sub.target.mean() * 100):.1f}% of average load).")

st.divider()

st.subheader("What drives the forecast")
top_feats = 10
importances = None
try:
    import pandas as pd
    importances = pd.Series(bundle["model"].feature_importances_, index=bundle["feature_cols"]) \
        .sort_values(ascending=False).head(top_feats).sort_values()
except Exception:
    pass

if importances is not None:
    fig2 = px.bar(x=importances.values, y=importances.index, orientation="h",
                  labels={"x": "importance", "y": ""})
    fig2.update_traces(marker_color="#2E4B6B")
    fig2.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Day-of-week and the same-hour-last-week lag dominate - building energy demand tracks "
        "weekly occupancy cycles more than short-term trend, which is why a 24h-ahead forecast "
        "is achievable without needing minute-by-minute lookback."
    )
