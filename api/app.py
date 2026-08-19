"""
api/app.py - Bonus: model deployment

Serves the trained predictive maintenance model behind a small FastAPI app.

Run (from project root, after src/predictive_maintenance.py has been run
once so the model file exists):
    uvicorn api.app:app --reload --port 8000

Then hit POST /predict_failure with a JSON body of raw telemetry readings
for a single asset - see TelemetryInput below for the fields.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "predictive_maintenance_xgb.joblib"

app = FastAPI(
    title="Nectar Predictive Maintenance API",
    description="Predicts probability of asset failure within the next 24 hours",
    version="1.0.0",
)

_bundle = None  # lazy-loaded on first request so the app can still start without a trained model present


def get_model_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python src/predictive_maintenance.py` first to train it.",
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class TelemetryInput(BaseModel):
    asset_id: str
    asset_type: str = Field(..., description="Chiller/AHU/Pump/EnergyMeter/EnvSensor")
    temperature: float
    humidity: float
    pressure: float
    vibration: float
    power_consumption: float
    occupancy_count: int = 0
    operating_mode: str = Field(default="Cooling", description="Cooling/Heating/Idle")
    hour: int = 12
    dow: int = 2
    asset_age_days: int = 365

    # recent history the model was trained with - callers should pass their
    # best available rolling stats; we default to point values if unknown
    temperature_rollmean_4: float | None = None
    vibration_rollmean_4: float | None = None
    power_consumption_rollmean_4: float | None = None
    vibration_rollstd_16: float | None = None
    vibration_rollstd_96: float | None = None


class PredictionOutput(BaseModel):
    asset_id: str
    failure_probability: float
    failure_predicted: bool
    threshold_used: float


def build_feature_row(payload: TelemetryInput, feature_cols: list[str]) -> pd.DataFrame:
    row = {c: 0 for c in feature_cols}

    row["temperature"] = payload.temperature
    row["humidity"] = payload.humidity
    row["pressure"] = payload.pressure
    row["vibration"] = payload.vibration
    row["power_consumption"] = payload.power_consumption
    row["occupancy_count"] = payload.occupancy_count
    row["hour"] = payload.hour
    row["dow"] = payload.dow
    row["is_weekend"] = int(payload.dow >= 5)
    row["is_business_hours"] = int(8 <= payload.hour <= 19 and payload.dow < 5)
    row["asset_age_days"] = payload.asset_age_days

    # rolling features fall back to the point reading if the caller doesn't have history yet
    row["temperature_rollmean_4"] = payload.temperature_rollmean_4 or payload.temperature
    row["vibration_rollmean_4"] = payload.vibration_rollmean_4 or payload.vibration
    row["power_consumption_rollmean_4"] = payload.power_consumption_rollmean_4 or payload.power_consumption
    row["vibration_rollstd_16"] = payload.vibration_rollstd_16 or 0.0
    row["vibration_rollstd_96"] = payload.vibration_rollstd_96 or 0.0

    mode_col = f"mode_{payload.operating_mode}"
    if mode_col in row:
        row[mode_col] = 1
    atype_col = f"atype_{payload.asset_type}"
    if atype_col in row:
        row[atype_col] = 1

    return pd.DataFrame([row])[feature_cols]


@app.get("/")
def health():
    return {"status": "ok", "service": "nectar-predictive-maintenance"}


@app.post("/predict_failure", response_model=PredictionOutput)
def predict_failure(payload: TelemetryInput):
    bundle = get_model_bundle()
    model, feature_cols, threshold = bundle["model"], bundle["feature_cols"], bundle["threshold"]

    X = build_feature_row(payload, feature_cols)
    proba = float(model.predict_proba(X)[:, 1][0])

    return PredictionOutput(
        asset_id=payload.asset_id,
        failure_probability=round(proba, 4),
        failure_predicted=proba >= threshold,
        threshold_used=round(float(threshold), 4),
    )
