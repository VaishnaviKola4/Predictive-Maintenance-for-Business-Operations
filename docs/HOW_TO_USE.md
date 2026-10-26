# How to Use: Complete Operational Guide

This document provides step-by-step instructions for installing, configuring, training, running, and integrating the **AI Predictive Maintenance System**.

---

## 1. Environment Setup

### 1.1 Prerequisites
- Python 3.10, 3.11, or 3.12 installed on host system.
- 4GB+ RAM recommended for large time-series training sets.
- Recommended tools: PowerShell or Bash, modern browser (Chrome, Edge, Firefox).

### 1.2 Installation
Clone or navigate to the repository directory and set up a virtual environment:

```bash
# Navigate to project root
cd "Predictive Maintenance for Business Operations"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Generating & Managing Input Datasets

The system requires three synchronized datasets located in `data/`:
1. `machine_sensors.csv`
2. `maintenance_history.csv`
3. `production_schedules.csv`

### Generating Synthetic Data
To generate a new multi-machine dataset reflecting real-world degradation trajectories:
```bash
python src/data_generator.py
```
Parameters can be customized inside `src/data_generator.py`:
- `num_machines`: Number of distinct assets in fleet (default: `12`).
- `days`: Duration of observation in days (default: `60`).
- `sampling_rate_minutes`: Telemetry capture frequency (default: `30` minutes).

### Supplying Custom Industrial Data
If using your own factory data:
1. Ensure column headers match the specifications in [Data Requirements](../README.md#data-requirements-and-format).
2. Save sensor readings into `data/machine_sensors.csv`.
3. Save service logs into `data/maintenance_history.csv`.
4. Save planned shifts into `data/production_schedules.csv`.

---

## 3. Preprocessing & Feature Engineering

Run the feature preprocessor to inspect data merge transformations and feature outputs:
```bash
python src/preprocessor.py
```
What happens under the hood:
1. **Time-Series Grouping:** Groups observations by `machine_id` in chronological order.
2. **Rolling Statistics:** Computes short-term (3-hour / 6-step) and medium-term (12-hour / 24-step) rolling averages, standard deviations, and maximum envelopes.
3. **Domain Cross-Ratios:** Generates mechanical interaction features (`vib_to_rpm_ratio`, `power_pressure_ratio`, `thermal_deviation`).
4. **Temporal Split:** Partitions the first 80% chronologically for model training and the final 20% strictly as holdout validation data.

---

## 4. Model Training & Validation

Train the Random Forest ensemble and generate validation metrics:
```bash
python src/train.py
```

### Model Artifacts Generated
Upon completion, the following files are saved into the `models/` directory:
- `random_forest_model.joblib`: Trained ensemble model.
- `feature_scaler.joblib`: Preprocessing normalization object.
- `feature_names.json`: Ordered list of 58 feature identifiers.
- `evaluation_metrics.json`: Accuracy, precision, recall, F1, ROC-AUC, and feature rankings.

---

## 5. Running the Interactive Web Command Center

Launch the Streamlit graphical interface:
```bash
streamlit run src/dashboard.py
```
Open your browser to `http://localhost:8501`.

### Navigating the Dashboard:
- **📊 Executive Fleet Overview:** High-level KPIs, machine risk breakdown, fleet status matrix, and historical expenditure by intervention type.
- **🔍 Machine Telemetry Deep-Dive:** Filter by any asset in the dropdown (e.g. `MACH-001`) to inspect real-time sensor streams, vibration progression, and wear indices.
- **🧠 Predictive Diagnostics & Recommendations:** Real-time AI inference output including failure probability, scheduled maintenance optimization, and business cost avoidance figures.
- **🛠️ Interactive What-If Scenario Simulator:** Move physical sliders (temperature, vibration, pressure) to simulate stress conditions and observe immediate AI classification shifts.
- **📈 Model Performance & Metrics:** Inspect test set confusion matrix and feature importance rankings.

---

## 6. Running the Production REST API

Start the FastAPI application:
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation:** Open `http://localhost:8000/docs` in your browser.
- **Live Fleet Snapshot:** Send a GET request to `http://localhost:8000/fleet-status`.
- **Telemetry Inference:** Send a POST request to `http://localhost:8000/predict` with telemetry payloads.

---

## 7. Running the Automated Test Suite

Validate all system modules and pipeline integrity:
```bash
python -m unittest tests/test_pipeline.py
```
Expected output:
```
.....
----------------------------------------------------------------------
Ran 5 tests in 2.5s

OK
```

