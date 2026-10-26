# Predictive Maintenance for Business Operations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, data-driven AI system designed for **Predictive Maintenance (PdM)** across industrial machinery and business operations. The solution ingests continuous machine sensor telemetry, historical maintenance logs, and dynamic production schedules to detect failure patterns before breakdowns occur, optimize technician intervention windows, eliminate unscheduled downtime, and minimize operational costs.

---

## Problem Statement

In modern manufacturing, logistics, and capital-intensive enterprise facilities, unexpected mechanical breakdowns represent one of the single largest sources of operational inefficiency and margin erosion. When critical assets—such as CNC milling machines, high-pressure hydraulic presses, slurry pumps, or rotary kilns—experience catastrophic failure, businesses face severe cascading consequences:

- **Crippling Unscheduled Downtime:** Complete line shutdowns halt throughput, leading to missed customer delivery commitments, breached Service Level Agreements (SLAs), and severe contractual penalties.
- **Asymmetric Maintenance Costs:** Emergency corrective maintenance costs between **5x to 10x** more than planned preventive servicing due to expedited replacement parts freight, premium weekend or night-shift labor rates, and collateral secondary damage to neighboring machine components.
- **Flawed Traditional Maintenance Strategies:**
  - *Run-to-Failure (Reactive):* Results in unpredictable catastrophic seizures, safety hazards for line technicians, and total batch spoilage.
  - *Calendar-Based / Usage-Based (Preventive):* Leads to premature overhauls of healthy components (wasting remaining useful life) while still failing to intercept sudden load-induced or fatigue anomalies that develop between inspection intervals.
- **Disconnect Between Maintenance and Production Operations:** Existing monitoring tools isolate sensor anomalies from real-time operational context. Plant engineers often lack insight into production loads, shift targets, and planned changeovers, resulting in maintenance scheduled during peak revenue runs rather than planned idle windows.

### The Objective
To bridge this gap, this system delivers an intelligent, end-to-end Predictive Maintenance AI solution that:
1. Detects incipient mechanical degradation across multi-variate sensor streams up to **48–72 hours prior to functional failure**.
2. Identifies specific root-cause fault modes (bearing wear, thermal runaway, hydraulic leaks, shaft imbalance, motor coil overload).
3. Harmonizes prescriptive maintenance recommendations with real-time **production schedules and shift load factors** to ensure repairs are scheduled with zero lost throughput.
4. Quantifies financial impact by computing exact **net cost avoidance** and downtime risk for plant managers.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Project Purpose](#project-purpose)
- [Core Features](#core-features)
- [Repository Structure](#repository-structure)
- [Algorithm Overview](#algorithm-overview)
- [High-Level Model Architecture](#high-level-model-architecture)
- [Data Requirements and Format](#data-requirements-and-format)
- [Installation and Setup](#installation-and-setup)
- [Usage Guide](#usage-guide)
  - [1. Synthetic Data Generation](#1-synthetic-data-generation)
  - [2. Model Training & Evaluation](#2-model-training--evaluation)
  - [3. Interactive Web Command Center (Streamlit)](#3-interactive-web-command-center-streamlit)
  - [4. Production REST API (FastAPI)](#4-production-rest-api-fastapi)
- [Evaluation Metrics & Performance](#evaluation-metrics--performance)
- [Recommendation Interpretation](#recommendation-interpretation)
- [Documentation Index](#documentation-index)
- [Contribution Guidelines](#contribution-guidelines)
- [License Information](#license-information)

---

## Project Purpose
In manufacturing, logistics, and capital-intensive business operations, unexpected machine breakdowns lead to substantial financial losses through interrupted production throughput, rush parts procurement, overtime labor fees, and downstream supply chain breaches. Conversely, rigid calendar-based maintenance often schedules costly overhauls prematurely or fails to catch sudden mechanical degradation.

This system provides:
1. **Early Anomaly & Failure Pattern Recognition:** Pinpoints microscopic indicators of wear, acoustic anomalies, thermal runaway, and pressure leaks up to 48–72 hours before failure occurs.
2. **Operational Schedule Synchronization:** Rather than dispatching technicians blindly, it pairs impending machine failures with planned shift downtime windows and production loads to schedule repairs with **zero lost throughput**.
3. **Financial Cost-Benefit Assessment:** Translates raw machine health into dollar figures (estimated breakdown cost vs. preventive intervention cost), giving plant managers actionable business intelligence.

---

## Core Features
- **Multi-Source Data Fusion:** Merges high-frequency physical telemetry with business logs and scheduling constraints.
- **Advanced Feature Engineering:** Computes rolling statistics (means, standard deviations, max envelopes), rates of change, acoustic-vibration ratios, and thermal deviation baselines.
- **Temporal Integrity:** Implements strictly time-ordered splits to guarantee zero data leakage between training and testing epochs.
- **Multi-Class Anomaly Diagnostics:** Categorizes specific physical fault modes:
  - Bearing Degradation & Spalling
  - Cooling Failure & Thermal Runaway
  - Hydraulic Pressure Drops & Valve Leakage
  - Shaft Imbalance & Harmonic Vibration
  - Motor Stator Coil Overload
- **Automated Prescriptive Actions:** Produces prioritized maintenance work orders, timeline constraints (`< 24h`, `24–48h`, `Next Shift`), and workload reduction policies.
- **Dual Interface:**
  - **Executive & Engineering UI:** An interactive multi-tab Streamlit dashboard with real-time what-if scenario simulations and fleet heatmaps.
  - **High-Performance REST API:** OpenAPI-compliant FastAPI microservice ready for industrial IoT and MES/SCADA integration.

---

## Repository Structure

```
Predictive Maintenance for Business Operations/
├── README.md                      # Comprehensive project guide and reference
├── requirements.txt               # Pinned Python package dependencies
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore patterns
│
├── config/                        # Configuration settings and hyperparameter schemas
│
├── data/                          # Dataset repository
│   ├── machine_sensors.csv        # Multi-sensor time-series telemetry (34,560 rows)
│   ├── maintenance_history.csv    # Historical intervention and repair logs (38 events)
│   ├── production_schedules.csv   # Planned shifts, downtime windows, and load factors (2,160 rows)
│   └── dataset_summary.json       # Metadata summary of generated datasets
│
├── docs/                          # Detailed project documentation
│   ├── USE_CASES.md               # Enterprise application scenarios and business impact
│   ├── HOW_TO_USE.md              # Installation, execution, and user guide
│   ├── API_INTERFACE_GUIDE.md     # OpenAPI specification and UI navigation guide
│   └── TECHNICAL_DEEP_DIVE.md     # Mathematical formulations, degradation physics, and architecture
│
├── models/                        # Serialized model artifacts and performance reports
│   ├── random_forest_model.joblib # Trained Cost-Balanced Random Forest Classifier
│   ├── feature_scaler.joblib      # Fitted StandardScaler instance
│   ├── feature_names.json         # Complete 58-feature schema
│   └── evaluation_metrics.json    # Holdout benchmark metrics, confusion matrix, top features
│
├── src/                           # Core source codebase
│   ├── data_generator.py          # Physics-informed synthetic dataset generation pipeline
│   ├── preprocessor.py            # Feature engineering, rolling transformations, temporal split
│   ├── train.py                   # Model training, hyperparameter configuration, evaluation
│   ├── recommender.py             # Prescriptive maintenance engine and cost-avoidance logic
│   ├── api.py                     # High-performance FastAPI REST endpoints
│   └── dashboard.py               # Streamlit multi-tab visual analytics command center
│
└── tests/                         # Automated test suite
    └── test_pipeline.py           # Unit tests for data, features, model inference, prescriptions
```

---

## Algorithm Overview

The system implements a multi-stage machine learning and decision intelligence pipeline that moves from raw time-series physical signals to actionable, schedule-optimized operational prescriptions:

```
  [Raw Sensor Signals]        [Historical Logs]       [Production Schedule]
           │                         │                         │
           ▼                         ▼                         ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 1. Time-Series Feature Engineering & Signal Transformation   │
   │    - Multi-scale rolling statistics (3h & 12h windows)        │
   │    - Physical domain ratios (Vib/RPM, Power/Pressure, ΔTemp)  │
   │    - Temporal alignment with shift load and planned downtime   │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 2. Cost-Balanced Ensemble Classification (Random Forest)      │
   │    - 120 decision trees with balanced class weighting         │
   │    - Temporal walk-forward split (Zero future data leakage)   │
   │    - Outputs continuous failure probability P(Failure in 48h)  │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 3. Multi-Tier Risk Stratification Engine                      │
   │    - CRITICAL  (P >= 0.75) ───► Emergency intervention (<24h) │
   │    - ELEVATED  (P >= 0.45) ───► Planned window repair (24-48h)│
   │    - MODERATE  (P >= 0.20) ───► Inspection watchlist (3-5d)  │
   │    - HEALTHY   (P < 0.20)  ───► Nominal baseline operation     │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 4. Root-Cause Diagnostic & Physics-Bound Isolation           │
   │    - Acoustic / Vibration spike ──► Bearing / Spindle wear    │
   │    - Thermal runaway ────────────► Coolant loop failure       │
   │    - Pressure drop + Power rise ──► Hydraulic seal bypass     │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 5. Schedule Optimization & Net Cost-Avoidance Quantifier     │
   │    - Matches repair window to next zero-production idle shift │
   │    - Dynamic de-rating of load factor to prevent seizure      │
   │    - Net Savings = C(Breakdown Downtime) - C(Planned Service) │
   └───────────────────────────────────────────────────────────────┘
```

### 1. Signal Processing & Domain Feature Transformations
Industrial sensors produce noisy, non-stationary signals. Rather than evaluating isolated observations, the preprocessor computes rolling temporal metrics across short (6-step / 3-hour) and medium (24-step / 12-hour) moving windows:
- **Rolling Envelopes:** Rolling Mean $\mu_k$, Rolling Standard Deviation $\sigma_k$, and Rolling Maximum $\mathcal{M}_k$ capture both the baseline drift and micro-transient spikes.
- **Specific Vibration per Revolution ($V_{\text{vib/rpm}}$):** Divides vibration amplitude by rotational spindle speed to detect mechanical eccentricity independent of rotational speed.
- **Energy Dissipation Ratio ($R_{\text{power/press}}$):** Identifies internal fluid leakage when power consumption surges while hydraulic pressure drops.
- **Thermal Drift ($\Delta T$):** Measures instantaneous temperature deviation against the 12-hour historical baseline.

### 2. Cost-Weighted Ensemble Classifier
In predictive maintenance, **False Negatives** (failing to predict an actual machine breakdown) are an order of magnitude more costly than **False Positives** (inspecting a machine that did not fail immediately).
- **Algorithm:** Random Forest Classifier consisting of 120 de-correlated decision trees with cost-sensitive class weights:
  $$w_c = \frac{N}{2 \cdot N_c}$$
- **Hyperparameters:** `n_estimators=120`, `max_depth=12`, `min_samples_split=6`, `class_weight='balanced'`.
- **Temporal Split Validation:** The model is strictly evaluated on the most recent 20% temporal slice to replicate real-world operational inference without lookahead bias.

### 3. Prescriptive Scheduling & Cost-Avoidance Optimizer
Predictions are ingested by the recommendation engine ([`src/recommender.py`](src/recommender.py)), which integrates live shift constraints:
- **Schedule Matching:** If the machine enters the `ELEVATED` risk tier and an upcoming planned downtime window exists within 48 hours, the system assigns maintenance to that window, eliminating lost production throughput.
- **Operational De-Rating:** When an asset is at critical risk and cannot be halted immediately without batch spoilage, the engine recommends de-rating the machine's load factor from 1.0 to 0.65 to safely bridge the operating window until the next shift change.
- **Financial Quantifier:** Translates failure probabilities into dollar savings by evaluating expected emergency downtime loss ($10\text{ hours} \times \$850/\text{hr} + \$3,500\text{ rush parts} = \$12,000$) against scheduled preventive service ($2\text{ hours} \times \$150/\text{hr} + \$600\text{ parts} = \$900$), returning a net cost avoidance of **\$11,100 per prevented failure event**.

---

## High-Level Model Architecture

```
 +-------------------------------------------------------------------------------+
 |                           Data Ingestion Layer                               |
 |   +------------------------+  +----------------------+  +------------------+  |
 |   | Machine Sensor Streams |  | Maintenance History  |  | Production Sched |  |
 |   |  (Temp, Vib, Press,    |  |  (Past faults, costs,|  |  (Shifts, load,  |  |
 |   |   Acoustics, Power)    |  |   hours, parts)      |  |   planned down)  |  |
 |   +-----------+------------+  +----------+-----------+  +--------+---------+  |
 +---------------|--------------------------|-----------------------|------------+
                 |                          |                       |
                 +--------------------------+-----------------------+
                                            v
 +-------------------------------------------------------------------------------+
 |                       Feature Engineering & Preprocessing                     |
 |  - Temporal synchronization by Machine ID & Timestamp                         |
 |  - Rolling Window Features: 3h & 12h Rolling Mean, Rolling Std, Rolling Max   |
 |  - Dynamic Physical Indicators: Vib-to-RPM Ratio, Power-to-Pressure Ratio    |
 |  - Standard Scaling & Categorical One-Hot Encodings                           |
 +------------------------------------------+------------------------------------+
                                            v
 +-------------------------------------------------------------------------------+
 |                          Predictive Model Layer                               |
 |  - Supervised Random Forest Classifier with Cost-Balanced Weights             |
 |  - Temporal Holdout Cross-Validation (No Future Leakage)                      |
 |  - Multi-tier Probability Thresholding (Healthy, Moderate, Elevated, Critical)|
 +------------------------------------------+------------------------------------+
                                            v
 +-------------------------------------------------------------------------------+
 |                   Prescriptive Decision & Scheduling Engine                   |
 |  - Root Cause Isolation (Vibration vs Thermal vs Hydraulic vs Electrical)     |
 |  - Operational Schedule Matching (Opportunity Windows vs Emergency Shutdown)  |
 |  - Financial Impact Quantifier: Cost Avoidance & ROI Calculation              |
 +-------------------+--------------------------------------+--------------------+
                     v                                      v
     +-------------------------------+      +-------------------------------+
     |   Interactive Web Dashboard   |      |      Production REST API      |
     |          (Streamlit)          |      |           (FastAPI)           |
     +-------------------------------+      +-------------------------------+
```

---

## Data Requirements and Format

The system expects three primary data feeds structured as CSV files or database streams:

### 1. Machine Sensors (`machine_sensors.csv`)
| Field | Type | Description |
|---|---|---|
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | Telemetry capture timestamp (e.g. 30-min interval) |
| `machine_id` | `String` | Asset identifier (e.g., `MACH-001`) |
| `machine_type` | `String` | Asset class (CNC-Milling, Hydraulic-Press, Industrial-Pump, Rotary-Kiln) |
| `temperature_c` | `Float` | Machine operating temperature in Celsius |
| `vibration_mm_s`| `Float` | Root-mean-square vibration velocity (mm/s) |
| `pressure_psi` | `Float` | Operating fluid or air pressure |
| `rotational_speed_rpm` | `Float` | Drive spindle speed (RPM) |
| `power_consumption_kw` | `Float` | Electrical power consumption (kW) |
| `acoustic_db` | `Float` | Ultrasonic/acoustic sensor emission (dB) |
| `operational_hours` | `Float` | Continuous running hours since last overhaul |
| `failure_imminent_48h` | `Integer (0 or 1)` | Target label: Failure expected within 48h |

### 2. Maintenance History (`maintenance_history.csv`)
| Field | Type | Description |
|---|---|---|
| `maintenance_id` | `String` | Unique work order ID (e.g., `MAINT-0012`) |
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | Date and time when maintenance was performed |
| `machine_id` | `String` | Serviced asset identifier |
| `maintenance_type` | `String` | Corrective / Breakdown vs Preventive / Scheduled Inspection |
| `fault_detected` | `String` | Mechanical fault identified (e.g., Bearing Degradation) |
| `downtime_hours` | `Float` | Total hours machine was out of service |
| `cost_incurred_usd` | `Float` | Total labor, parts, and logistics cost ($) |
| `parts_replaced` | `String` | Replaced component descriptions |

### 3. Production Schedules (`production_schedules.csv`)
| Field | Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` | Scheduled production date |
| `shift` | `Integer (1, 2, 3)` | Daily shift identifier |
| `machine_id` | `String` | Assigned asset identifier |
| `planned_downtime` | `Integer (0 or 1)` | 1 if line is scheduled for planned idle / tooling change |
| `target_units_per_hour` | `Integer` | Throughput target for the shift |
| `load_factor` | `Float` | Normalized operational load factor (0.40 - 1.25) |
| `batch_type` | `String` | Material tolerance category (e.g., High-Tolerance-Steel) |

---

## Installation and Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone https://github.com/your-org/predictive-maintenance-business-operations.git
cd "predictive-maintenance-business-operations"

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Synthetic Data Generation
Generate calibrated time-series data mirroring multi-machine industrial environments:
```bash
python src/data_generator.py
```
This produces `machine_sensors.csv`, `maintenance_history.csv`, `production_schedules.csv`, and a `dataset_summary.json` inside the `data/` directory.

### 2. Model Training & Evaluation
Train the predictive model and output serialized artifacts:
```bash
python src/train.py
```
Outputs saved in `models/`:
- `random_forest_model.joblib`: Serialized trained ensemble classifier
- `feature_scaler.joblib`: Standard scaler fit on the feature set
- `feature_names.json`: Schema of all 58 engineered features
- `evaluation_metrics.json`: Detailed validation report and feature importance

### 3. Interactive Web Command Center (Streamlit)
Launch the rich visual operational dashboard:
```bash
streamlit run src/dashboard.py
```
Navigate to `http://localhost:8501` to view:
- **Executive Fleet Overview:** Health status, critical machine counts, historical downtime costs.
- **Machine Telemetry Deep-Dive:** Interactive multi-sensor graphs and wear tracking per asset.
- **Predictive Diagnostics & Recommendations:** Root cause diagnosis, schedule alignment, and cost avoidance metrics.
- **Interactive What-If Scenario Simulator:** Adjust sliders for temperature, vibration, and pressure to observe real-time model predictions.
- **Model Performance & Metrics:** Confusion matrix and top feature weights.

### 4. Production REST API (FastAPI)
Launch the microservice API server:
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Direct Health Check: `http://localhost:8000/health`
- Live Fleet Status: `http://localhost:8000/fleet-status`

#### Sample API Prediction Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "telemetry": {
         "machine_id": "MACH-001",
         "machine_type": "CNC-Milling",
         "temperature_c": 96.5,
         "vibration_mm_s": 7.2,
         "pressure_psi": 42.0,
         "rotational_speed_rpm": 3100.0,
         "power_consumption_kw": 28.0,
         "acoustic_db": 98.0,
         "operational_hours": 410.0
       },
       "schedule": {
         "date": "2026-10-26",
         "shift": 1,
         "planned_downtime": 0,
         "target_units_per_hour": 120,
         "load_factor": 0.95,
         "next_planned_maintenance_window": "Upcoming Shift 3 (02:00-05:00 UTC)"
       }
     }'
```

---

## Evaluation Metrics & Performance

The predictive model is benchmarked on a strictly time-held-out test set (most recent 20% of operational timeline, 6,912 telemetry cycles) to prevent lookahead bias:

| Evaluation Metric | Benchmark Result | Operational Interpretation |
|---|---|---|
| **Accuracy** | **98.89%** | Overall correct health classifications across fleet. |
| **Precision** | **99.25%** | When an alert fires, 99.25% are true imminent degradation events, minimizing false technician dispatches. |
| **Recall (Sensitivity)** | **90.24%** | Captures over 9 out of 10 impending component failures before breakdown. |
| **F1 Score** | **94.54%** | High harmonic balance between sensitivity and precision. |
| **ROC-AUC** | **0.9993** | High discriminative separation across healthy and degraded machine states. |

### Top Predictive Feature Importances
1. `operational_hours`: Continuous service run-time
2. `acoustic_db_roll_max_24`: 12-hour peak acoustic signature
3. `vibration_mm_s_roll_max_24`: 12-hour peak vibration envelope
4. `temperature_c_roll_max_24`: 12-hour thermal peak
5. `vib_to_rpm_ratio`: Mechanical vibration relative to angular velocity

---

## Recommendation Interpretation

The recommendation engine categorizes assets into four actionable tiers:

| Risk Tier | Failure Probability | Action Timeline | Operational Protocol |
|---|---|---|---|
| **CRITICAL** | `> 75%` | `< 24 Hours` | Halt non-critical batch runs immediately or de-rate load factor. Dispatch technician with designated replacement parts. |
| **ELEVATED** | `45% - 75%` | `24 - 48 Hours` | Re-route high-stress jobs. Schedule preventive service in the nearest planned downtime window. |
| **MODERATE** | `20% - 45%` | `3 - 5 Days` | Add asset to upcoming shift inspection checklist. Monitor rolling acoustic/thermal drift. |
| **HEALTHY** | `< 20%` | `Routine` | Machine in nominal operating envelope. Maintain standard production schedule. |

---

## Documentation Index
Comprehensive documentation is available in the [`docs/`](docs/) directory:
- [**Use Cases Guide**](docs/USE_CASES.md): Real-world business scenarios including seasonal throughput peaks, scrap/waste reduction, and parts inventory optimization.
- [**How to Use**](docs/HOW_TO_USE.md): End-to-end setup, data onboarding, custom model training, and integration walk-throughs.
- [**API & Interface Guide**](docs/API_INTERFACE_GUIDE.md): Complete OpenAPI endpoint documentation, schema definitions, and dashboard navigation.
- [**Technical Deep Dive**](docs/TECHNICAL_DEEP_DIVE.md): Mathematical formulations, degradation curve physics, signal processing, and architectural trade-offs.

---

## Contribution Guidelines
1. Fork the repository and create your feature branch (`git checkout -b feature/dynamic-degradation`).
2. Run test suites locally before submitting pull requests:
   ```bash
   python -m unittest tests/test_pipeline.py
   ```
3. Commit clear, descriptive messages following conventional commit standards.
4. Open a Pull Request detailing the problem solved, test outcomes, and metric impact.

---

## License Information
Distributed under the **MIT License**. See [LICENSE](LICENSE) for full legal text.
