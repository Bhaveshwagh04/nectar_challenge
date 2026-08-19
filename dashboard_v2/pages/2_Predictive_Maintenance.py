"""
pages/2_Predictive_Maintenance.py

Shows the trained model's performance, then two practical views on top of
it: which currently-live assets look highest-risk right now, and a
what-if form where you can punch in sensor readings and get a failure
probability back - the same thing the FastAPI /predict_failure endpoint
does, just without leaving the browser.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.ui import page_setup
from utils.pipeline import get_pm_model, load_raw_data

page_setup("Predictive Maintenance", icon="🔧")

st.caption("XGBoost classifier predicting whether an asset will fail within the next 24 hours.")

bundle = get_pm_model()
metrics = bundle["metrics"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Precision", f"{metrics['precision']:.2f}")
c2.metric("Recall", f"{metrics['recall']:.2f}")
c3.metric("F1", f"{metrics['f1']:.2f}")
c4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
c5.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

st.divider()

tab1, tab2, tab3 = st.tabs(["Model insights", "Current risk board", "Score a reading"])

# ---- tab 1: feature importance + PR curve ----
with tab1:
    left, right = st.columns(2)

    with left:
        top_feats = bundle["importances"].head(15).sort_values()
        fig = px.bar(x=top_feats.values, y=top_feats.index, orientation="h",
                     title="Top 15 features driving predictions",
                     labels={"x": "importance", "y": ""})
        fig.update_traces(marker_color="#2E4B6B")
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bundle["recalls"], y=bundle["precisions"],
                                    mode="lines", line=dict(color="#d1495b", width=2)))
        fig2.update_layout(title=f"Precision-Recall curve (PR-AUC={metrics['pr_auc']:.3f})",
                            xaxis_title="recall", yaxis_title="precision", margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Rolling vibration volatility (16h/96h standard deviation) is the strongest signal - "
        "an asset's vibration becoming *unstable* is a clearer warning sign than any single "
        "sensor crossing a fixed threshold, which is exactly the pattern the EDA page shows "
        "in the 24h-before-failure comparison."
    )

# ---- tab 2: which live assets look risky right now ----
with tab2:
    st.write("Assets from the held-out test window, ranked by predicted failure probability.")
    scored = bundle["test_scored"]
    latest_per_asset = scored.sort_values("timestamp").groupby("asset_id").tail(1)
    latest_per_asset = latest_per_asset.sort_values("pred_proba", ascending=False)

    risk_view = latest_per_asset[["asset_id", "timestamp", "pred_proba", "pred",
                                    "temperature", "vibration", "power_consumption"]].head(20).copy()
    risk_view["pred_proba"] = (risk_view.pred_proba * 100).round(1)
    risk_view = risk_view.rename(columns={"pred_proba": "failure_risk_%", "pred": "flagged"})
    st.dataframe(risk_view, use_container_width=True, hide_index=True)

    n_flagged = int(latest_per_asset.pred.sum())
    st.caption(f"{n_flagged} of {len(latest_per_asset)} assets are currently flagged above the "
               f"model's operating threshold ({bundle['threshold']*100:.1f}% probability).")

# ---- tab 3: manual what-if scoring ----
with tab3:
    st.write("Enter a sensor reading to get a live failure-probability score from the trained model.")

    telemetry, assets, connectivity = load_raw_data()
    asset_types = sorted(assets.asset_type.dropna().unique())
    modes = ["Cooling", "Heating", "Idle"]

    with st.form("score_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            asset_type = st.selectbox("Asset type", asset_types)
            temperature = st.number_input("Temperature (°C)", value=25.0)
            humidity = st.number_input("Humidity (%)", value=50.0)
        with f2:
            pressure = st.number_input("Pressure", value=101.0)
            vibration = st.number_input("Vibration", value=8.0)
            power_consumption = st.number_input("Power consumption (kWh)", value=150.0)
        with f3:
            occupancy = st.number_input("Occupancy count", value=10, step=1)
            mode = st.selectbox("Operating mode", modes)
            asset_age_days = st.number_input("Asset age (days)", value=800, step=1)

        st.markdown("**Recent trend (optional - improves accuracy)**")
        g1, g2 = st.columns(2)
        with g1:
            vib_std_16 = st.number_input("Vibration rolling std, last 4h", value=2.0)
        with g2:
            vib_std_96 = st.number_input("Vibration rolling std, last 24h", value=1.5)

        submitted = st.form_submit_button("Score this reading", type="primary")

    if submitted:
        model, feature_cols, threshold = bundle["model"], bundle["feature_cols"], bundle["threshold"]
        row = {c: 0 for c in feature_cols}

        now = pd.Timestamp.now()
        row.update({
            "temperature": temperature, "humidity": humidity, "pressure": pressure,
            "vibration": vibration, "power_consumption": power_consumption,
            "occupancy_count": occupancy, "hour": now.hour, "dow": now.dayofweek,
            "is_weekend": int(now.dayofweek >= 5),
            "is_business_hours": int(8 <= now.hour <= 19 and now.dayofweek < 5),
            "asset_age_days": asset_age_days,
            "temperature_rollmean_4": temperature, "vibration_rollmean_4": vibration,
            "power_consumption_rollmean_4": power_consumption,
            "vibration_rollstd_16": vib_std_16, "vibration_rollstd_96": vib_std_96,
        })
        mode_col, atype_col = f"mode_{mode}", f"atype_{asset_type}"
        if mode_col in row:
            row[mode_col] = 1
        if atype_col in row:
            row[atype_col] = 1

        X = pd.DataFrame([row])[feature_cols]
        proba = float(model.predict_proba(X)[:, 1][0])
        flagged = proba >= threshold

        st.divider()
        result_col1, result_col2 = st.columns([1, 2])
        with result_col1:
            st.metric("Failure probability (next 24h)", f"{proba*100:.1f}%")
            if flagged:
                st.error("⚠️ Above operating threshold - flag for inspection", icon="🚨")
            else:
                st.success("Below operating threshold - normal", icon="✅")
        with result_col2:
            st.progress(min(proba, 1.0))
            st.caption(f"Operating threshold: {threshold*100:.1f}%. This mirrors the same "
                       f"`/predict_failure` logic exposed by the FastAPI service in `api/app.py`.")
