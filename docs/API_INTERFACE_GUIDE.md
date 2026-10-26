# API & Interface Reference Guide

This document provides a technical specification for the **FastAPI REST Microservice** and the **Streamlit Web Application**.

---

## 1. REST API Specification

- **Base URL:** `http://localhost:8000`
- **Documentation UI:** `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc` (ReDoc)
- **Standard Protocol:** HTTP/1.1 / JSON

---

### Endpoint 1: Service Root & Health
#### `GET /`
Returns service status and indicates if the trained ML model is loaded.

**Response (200 OK):**
```json
{
  "service": "Predictive Maintenance for Business Operations API",
  "status": "online",
  "model_loaded": true,
  "docs_url": "/docs"
}
```

#### `GET /health`
Returns system health and metadata.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "features_loaded": 58,
  "model_version": "v1.0-rf-ensemble"
}
```

---

### Endpoint 2: Model Performance Metrics
#### `GET /metrics`
Retrieves the latest validation metrics, confusion matrix, and feature importances computed on the holdout test set.

**Response (200 OK):**
```json
{
  "model_type": "random_forest",
  "evaluation_timestamp": "2026-10-26T01:43:21.050123",
  "test_samples": 6912,
  "accuracy": 0.9889,
  "precision": 0.9925,
  "recall": 0.9024,
  "f1_score": 0.9454,
  "roc_auc": 0.9993,
  "confusion_matrix": {
    "true_negative": 6169,
    "false_positive": 5,
    "false_negative": 72,
    "true_positive": 666
  },
  "top_features": [
    { "feature": "operational_hours", "importance": 0.1852 },
    { "feature": "acoustic_db_roll_max_24", "importance": 0.1421 },
    { "feature": "acoustic_db_roll_max_6", "importance": 0.1105 }
  ]
}
```

---

### Endpoint 3: Real-Time Machine Prediction & Prescription
#### `POST /predict`
Evaluates live telemetry from an operational asset and provides a full failure risk diagnosis, production schedule alignment, and cost avoidance calculation.

**Request Headers:** `Content-Type: application/json`

**Request Body:**
```json
{
  "telemetry": {
    "machine_id": "MACH-002",
    "machine_type": "CNC-Milling",
    "temperature_c": 98.4,
    "vibration_mm_s": 7.8,
    "pressure_psi": 44.0,
    "rotational_speed_rpm": 3100.0,
    "power_consumption_kw": 28.5,
    "acoustic_db": 98.5,
    "operational_hours": 420.0
  },
  "schedule": {
    "date": "2026-10-26",
    "shift": 1,
    "planned_downtime": 0,
    "target_units_per_hour": 120,
    "load_factor": 0.95,
    "next_planned_maintenance_window": "Upcoming Shift 3 (02:00-05:00 UTC)"
  }
}
```

**Response (200 OK):**
```json
{
  "machine_id": "MACH-002",
  "timestamp": "2026-10-26T01:47:06.941362",
  "failure_probability": 0.9628,
  "prediction_binary": 1,
  "risk_tier": "CRITICAL",
  "priority_score": 1,
  "recommended_action_timeline": "< 24 Hours",
  "schedule_recommendation": "Emergency intervention required. Pause current production batch (Target: 120 u/h) to prevent catastrophic machine seizure. Re-route critical batch.",
  "diagnostics": [
    {
      "component": "Drive Shaft / Roller Bearings",
      "finding": "High vibration (7.8 mm/s) & acoustic emissions (98.5 dB) indicating bearing spalling or rotor imbalance.",
      "action": "Inspect mechanical coupling, dynamically balance rotor, replace bearings."
    },
    {
      "component": "Cooling System / Heat Exchanger",
      "finding": "Elevated temperature (98.4 °C) indicates coolant blockage or severe thermal dissipation loss.",
      "action": "Flush coolant circuit, test thermocouple, clean radiator fins."
    },
    {
      "component": "Hydraulic Line / Seals",
      "finding": "Low pressure (44.0 psi) indicates pneumatic/hydraulic valve leakage or seal deterioration.",
      "action": "Pressure-test lines, replace O-rings, check hydraulic fluid levels."
    }
  ],
  "business_impact": {
    "estimated_unplanned_failure_cost_usd": 12000.0,
    "preventive_service_cost_usd": 900.0,
    "net_cost_avoidance_usd": 11100.0,
    "production_risk": "High Batch Loss"
  }
}
```

---

### Endpoint 4: Fleet-Wide Status Snapshot
#### `GET /fleet-status`
Returns recent telemetry and failure warnings across all fleet machines.

**Query Parameters:**
- `limit` (integer, optional, default: 10, min: 1, max: 50): Number of machines to return.

**Response (200 OK):**
```json
{
  "fleet_size": 10,
  "machines": [
    {
      "machine_id": "MACH-001",
      "machine_type": "CNC-Milling",
      "temperature_c": 64.2,
      "vibration_mm_s": 2.65,
      "pressure_psi": 89.2,
      "operational_hours": 145.0,
      "failure_imminent_status": false,
      "fault_category": "None"
    }
  ]
}
```

---

## 2. Streamlit Web Dashboard Interface

The Streamlit UI is organized into four intuitive views:

```
+----------------------------------------------------------------------------------+
|                     AI PREDICTIVE MAINTENANCE COMMAND CENTER                    |
+-------------------+--------------------------------------------------------------+
| SIDEBAR           | MAIN VIEW AREA                                               |
|                   |                                                              |
| [Select View]     | 1. Executive Fleet Overview:                                 |
| - Fleet Overview  |    - Fleet KPIs: Machines Active, Risk Count, Avoidance ROI  |
| - Telemetry Dive  |    - Interactive Fleet Status Matrix (Color Coded)           |
| - AI Diagnostics  |    - Fault Distribution Pie Chart & Spend Bar Charts        |
| - Model Metrics   |                                                              |
|                   | 2. Machine Telemetry Deep-Dive:                              |
| [Select Asset]    |    - Sensor KPIs: Temp, Vibration, Pressure, Wear            |
| MACH-001          |    - Time-series progression curves (Plotly interactive)     |
| MACH-002...       |                                                              |
|                   | 3. Predictive Diagnostics & Recommendations:                 |
|                   |    - Gauge: Failure Probability & Assigned Risk Tier         |
|                   |    - Prescriptions: Schedule alignment & Root cause findings|
|                   |    - Interactive What-If Simulator: Sliders for Temp & Vib   |
|                   |                                                              |
|                   | 4. Model Performance & Metrics:                              |
|                   |    - Accuracy, Precision, Recall, F1 Score Cards             |
|                   |    - Confusion Matrix Heatmap & Feature Importance Rankings  |
+-------------------+--------------------------------------------------------------+
```

