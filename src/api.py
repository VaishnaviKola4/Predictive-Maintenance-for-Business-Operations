import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from recommender import MaintenanceRecommender
from preprocessor import load_and_merge_datasets, engineer_features

app = FastAPI(
    title="Predictive Maintenance for Business Operations API",
    description="Intelligent AI API for failure prediction, telemetry anomaly detection, and operational schedule-aligned maintenance recommendations.",
    version="1.0.0"
)

recommender = MaintenanceRecommender()

class SensorTelemetryInput(BaseModel):
    machine_id: str = Field(..., example="MACH-001")
    machine_type: str = Field("CNC-Milling", example="CNC-Milling")
    temperature_c: float = Field(..., example=88.5)
    vibration_mm_s: float = Field(..., example=5.8)
    pressure_psi: float = Field(..., example=85.0)
    rotational_speed_rpm: float = Field(..., example=3100.0)
    power_consumption_kw: float = Field(..., example=24.5)
    acoustic_db: float = Field(..., example=89.0)
    operational_hours: float = Field(..., example=340.0)

class ScheduleInput(BaseModel):
    date: Optional[str] = Field("2026-10-26", example="2026-10-26")
    shift: Optional[int] = Field(1, example=1)
    planned_downtime: Optional[int] = Field(0, example=0)
    target_units_per_hour: Optional[int] = Field(120, example=120)
    load_factor: Optional[float] = Field(0.95, example=0.95)
    next_planned_maintenance_window: Optional[str] = Field("Shift 3 (02:00-05:00 UTC)", example="Shift 3 (02:00-05:00 UTC)")

class PredictionRequest(BaseModel):
    telemetry: SensorTelemetryInput
    schedule: Optional[ScheduleInput] = None

@app.get("/")
def root():
    return {
        "service": "Predictive Maintenance for Business Operations API",
        "status": "online",
        "model_loaded": recommender.model is not None,
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "features_loaded": len(recommender.feature_names),
        "model_version": "v1.0-rf-ensemble"
    }

@app.get("/metrics")
def get_model_metrics():
    metrics_path = os.path.join("models", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Metrics file not found. Train model first.")

@app.post("/predict")
def predict_machine_health(payload: PredictionRequest):
    """
    Accepts real-time machine telemetry & production context, returning:
    - Failure risk probability
    - Severity tier & response timeframe
    - Root cause diagnosis
    - Preventive actions aligned with production schedules
    - Financial cost-avoidance estimation
    """
    if recommender.model is None:
        raise HTTPException(status_code=500, detail="Predictive model not initialized.")

    raw_tel = payload.telemetry.dict()
    sched = payload.schedule.dict() if payload.schedule else {}

    # Synthesize feature dictionary mapping to trained feature columns
    feat_dict = {}
    for col in recommender.feature_names:
        feat_dict[col] = 0.0

    # Fill primary telemetry
    for k in ["temperature_c", "vibration_mm_s", "pressure_psi", "rotational_speed_rpm", "power_consumption_kw", "acoustic_db", "operational_hours"]:
        if k in feat_dict:
            feat_dict[k] = raw_tel.get(k, 0.0)

    # Derived domain interactions
    vib = raw_tel.get("vibration_mm_s", 2.0)
    rpm = raw_tel.get("rotational_speed_rpm", 2000.0)
    press = raw_tel.get("pressure_psi", 100.0)
    power = raw_tel.get("power_consumption_kw", 20.0)
    temp = raw_tel.get("temperature_c", 60.0)

    if "vib_to_rpm_ratio" in feat_dict:
        feat_dict["vib_to_rpm_ratio"] = vib / (rpm + 1e-4)
    if "power_pressure_ratio" in feat_dict:
        feat_dict["power_pressure_ratio"] = power / (press + 1e-4)
    if "thermal_deviation" in feat_dict:
        feat_dict["thermal_deviation"] = temp - 60.0

    # Rolling window baselines approx
    for w in [6, 24]:
        for stat in ["mean", "std", "max"]:
            k = f"vibration_mm_s_roll_{stat}_{w}"
            if k in feat_dict:
                feat_dict[k] = vib if stat != "std" else 0.4
            k = f"temperature_c_roll_{stat}_{w}"
            if k in feat_dict:
                feat_dict[k] = temp if stat != "std" else 1.2
            k = f"pressure_psi_roll_{stat}_{w}"
            if k in feat_dict:
                feat_dict[k] = press if stat != "std" else 3.0

    # Categorical match
    m_type = f"machine_type_{raw_tel.get('machine_type')}"
    if m_type in feat_dict:
        feat_dict[m_type] = 1.0

    recommendation = recommender.recommend(feat_dict, raw_tel, sched)
    return recommendation

@app.get("/fleet-status")
def get_fleet_status(limit: int = Query(10, ge=1, le=50)):
    """
    Returns latest telemetry snapshot across fleet machines with real-time risk scores.
    """
    sensor_path = "data/machine_sensors.csv"
    if not os.path.exists(sensor_path):
        raise HTTPException(status_code=404, detail="Machine telemetry data not found.")

    df = pd.read_csv(sensor_path)
    latest_per_machine = df.sort_values("timestamp").groupby("machine_id").last().reset_index()

    fleet_summary = []
    for _, row in latest_per_machine.head(limit).iterrows():
        fleet_summary.append({
            "machine_id": row["machine_id"],
            "machine_type": row["machine_type"],
            "temperature_c": row["temperature_c"],
            "vibration_mm_s": row["vibration_mm_s"],
            "pressure_psi": row["pressure_psi"],
            "operational_hours": row["operational_hours"],
            "failure_imminent_status": bool(row["failure_imminent_48h"]),
            "fault_category": row["fault_category"]
        })

    return {"fleet_size": len(fleet_summary), "machines": fleet_summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

