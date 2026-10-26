import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

class MaintenanceRecommender:
    """
    Intelligent recommendation engine for predictive maintenance.
    Combines:
    - Failure probability predictions from ML model
    - Telemetry anomaly diagnostics (thermal, vibration, pressure, acoustic)
    - Production schedule alignment (prioritizing planned downtime vs active shifts)
    - Cost-benefit and operational risk prioritization
    """

    # Baseline cost estimates for business impact calculation
    UNPLANNED_DOWNTIME_COST_PER_HOUR = 850.0  # Lost production throughput & rush labor
    PLANNED_MAINTENANCE_COST_PER_HOUR = 150.0
    TYPICAL_UNPLANNED_REPAIR_HOURS = 10.0
    TYPICAL_PREVENTIVE_SERVICE_HOURS = 2.0

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.model_dir, "random_forest_model.joblib")
        scaler_path = os.path.join(self.model_dir, "feature_scaler.joblib")
        features_path = os.path.join(self.model_dir, "feature_names.json")

        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            with open(features_path, "r") as f:
                self.feature_names = json.load(f)
        else:
            print(f"Warning: Artifacts not found in {self.model_dir}. Please train model first.")

    def diagnose_failure_root_cause(self, raw_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pinpoints anomaly drivers by analyzing physical telemetry against nominal baselines.
        """
        temp = raw_telemetry.get("temperature_c", 60.0)
        vib = raw_telemetry.get("vibration_mm_s", 2.0)
        press = raw_telemetry.get("pressure_psi", 100.0)
        power = raw_telemetry.get("power_consumption_kw", 20.0)
        acoustic = raw_telemetry.get("acoustic_db", 60.0)

        diagnostics = []
        severity = "NORMAL"

        if vib > 4.5 or acoustic > 85.0:
            diagnostics.append({
                "component": "Drive Shaft / Roller Bearings",
                "finding": f"High vibration ({vib} mm/s) & acoustic emissions ({acoustic} dB) indicating bearing spalling or rotor imbalance.",
                "action": "Inspect mechanical coupling, dynamically balance rotor, replace bearings."
            })
            severity = "HIGH"

        if temp > 85.0:
            diagnostics.append({
                "component": "Cooling System / Heat Exchanger",
                "finding": f"Elevated temperature ({temp} °C) indicates coolant blockage or severe thermal dissipation loss.",
                "action": "Flush coolant circuit, test thermocouple, clean radiator fins."
            })
            severity = "CRITICAL" if temp > 95.0 else "HIGH"

        if press < 50.0:
            diagnostics.append({
                "component": "Hydraulic Line / Seals",
                "finding": f"Low pressure ({press} psi) indicates pneumatic/hydraulic valve leakage or seal deterioration.",
                "action": "Pressure-test lines, replace O-rings, check hydraulic fluid levels."
            })
            severity = "HIGH"

        if power > 35.0:
            diagnostics.append({
                "component": "Electric Drive / Stator Windings",
                "finding": f"Excessive power draw ({power} kW) points to high friction load or motor coil degradation.",
                "action": "Check winding insulation resistance (megger test), check gearbox resistance."
            })

        if not diagnostics:
            diagnostics.append({
                "component": "General System",
                "finding": "Telemetry parameters within nominal operating tolerance envelopes.",
                "action": "Continue regular autonomous monitoring."
            })

        return {"diagnostics": diagnostics, "root_cause_severity": severity}

    def recommend(
        self,
        features_dict: Dict[str, Any],
        raw_telemetry: Dict[str, Any],
        schedule_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates full predictive maintenance prescription:
        - Failure Probability & Risk Tier
        - Root cause diagnosis
        - Optimal maintenance window aligned with production schedules
        - Business cost savings calculation
        """
        # Build feature vector
        vector = []
        for feat in self.feature_names:
            vector.append(features_dict.get(feat, 0.0))

        X_df = pd.DataFrame([vector], columns=self.feature_names)
        X_scaled = self.scaler.transform(X_df)

        failure_prob = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = int(failure_prob >= 0.5)

        # Risk Classification
        if failure_prob >= 0.75:
            risk_tier = "CRITICAL"
            time_to_act = "< 24 Hours"
            priority_score = 1
        elif failure_prob >= 0.45:
            risk_tier = "ELEVATED"
            time_to_act = "24 - 48 Hours"
            priority_score = 2
        elif failure_prob >= 0.20:
            risk_tier = "MODERATE"
            time_to_act = "Next Scheduled Maintenance (3-5 Days)"
            priority_score = 3
        else:
            risk_tier = "HEALTHY"
            time_to_act = "Routine (Next Shift Check)"
            priority_score = 4

        # Root cause diagnosis
        diag = self.diagnose_failure_root_cause(raw_telemetry)

        # Production schedule matching
        is_planned_down = schedule_info.get("planned_downtime", 0)
        target_units = schedule_info.get("target_units_per_hour", 120)
        load_factor = schedule_info.get("load_factor", 0.85)
        upcoming_window = schedule_info.get("next_planned_maintenance_window", "Shift 3 (02:00 - 05:00 UTC)")

        if is_planned_down:
            schedule_recommendation = "Execute maintenance IMMEDIATELY during current planned downtime with zero production penalty."
        elif risk_tier == "CRITICAL":
            schedule_recommendation = (
                f"Emergency intervention required. Pause current production batch (Target: {target_units} u/h) "
                f"to prevent catastrophic machine seizure. Re-route critical batch."
            )
        elif risk_tier in ["ELEVATED", "MODERATE"]:
            schedule_recommendation = (
                f"Schedule preventive intervention during upcoming window: '{upcoming_window}'. "
                f"Reduce operational load factor from {load_factor} to 0.65 until serviced to avoid triggering breakdown."
            )
        else:
            schedule_recommendation = "Maintain standard production throughput. No schedule disruption needed."

        # Financial Impact & Cost Avoidance
        est_breakdown_cost = (self.TYPICAL_UNPLANNED_REPAIR_HOURS * self.UNPLANNED_DOWNTIME_COST_PER_HOUR) + 3500.0  # parts
        est_preventive_cost = (self.TYPICAL_PREVENTIVE_SERVICE_HOURS * self.PLANNED_MAINTENANCE_COST_PER_HOUR) + 600.0   # parts
        potential_cost_avoidance = round(max(0.0, est_breakdown_cost - est_preventive_cost), 2) if risk_tier in ["CRITICAL", "ELEVATED"] else 0.0

        return {
            "machine_id": raw_telemetry.get("machine_id", "MACH-UNKNOWN"),
            "timestamp": datetime.now().isoformat(),
            "failure_probability": round(failure_prob, 4),
            "prediction_binary": prediction,
            "risk_tier": risk_tier,
            "priority_score": priority_score,
            "recommended_action_timeline": time_to_act,
            "schedule_recommendation": schedule_recommendation,
            "diagnostics": diag["diagnostics"],
            "business_impact": {
                "estimated_unplanned_failure_cost_usd": est_breakdown_cost,
                "preventive_service_cost_usd": est_preventive_cost,
                "net_cost_avoidance_usd": potential_cost_avoidance,
                "production_risk": "High Batch Loss" if risk_tier == "CRITICAL" else "Managed"
            }
        }

if __name__ == "__main__":
    recommender = MaintenanceRecommender()
    print("Recommender initialized successfully.")

