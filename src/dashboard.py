import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Predictive Maintenance for Business Operations",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #00B4D8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-healthy { color: #2ECC71; font-weight: bold; }
    .status-critical { color: #E74C3C; font-weight: bold; }
    .status-elevated { color: #F39C12; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/machine_sensors.csv"
MAINT_PATH = "data/maintenance_history.csv"
SCHED_PATH = "data/production_schedules.csv"
METRICS_PATH = "models/evaluation_metrics.json"

@st.cache_data
def load_all_data():
    if not os.path.exists(DATA_PATH):
        return None, None, None, None
    sensors = pd.read_csv(DATA_PATH)
    sensors["timestamp"] = pd.to_datetime(sensors["timestamp"])
    maint = pd.read_csv(MAINT_PATH) if os.path.exists(MAINT_PATH) else pd.DataFrame()
    sched = pd.read_csv(SCHED_PATH) if os.path.exists(SCHED_PATH) else pd.DataFrame()
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    return sensors, maint, sched, metrics

sensors_df, maint_df, sched_df, metrics_data = load_all_data()

st.title("⚙️ AI Predictive Maintenance Command Center")
st.markdown("Automated Failure Pattern Recognition, Fleet Telemetry Diagnostics & Schedule-Optimized Preventive Actions")

if sensors_df is None:
    st.error("Data not found. Please run synthetic data generator first via `python src/data_generator.py`.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation & Controls")
nav_choice = st.sidebar.radio(
    "Select View:",
    ["📊 Executive Fleet Overview", "🔍 Machine Telemetry Deep-Dive", "🧠 Predictive Diagnostics & Recommendations", "📈 Model Performance & Metrics"]
)

unique_machines = sorted(sensors_df["machine_id"].unique())
selected_machine = st.sidebar.selectbox("Select Asset Machine:", unique_machines)

# ----------------- VIEW 1: EXECUTIVE FLEET OVERVIEW -----------------
if nav_choice == "📊 Executive Fleet Overview":
    st.subheader("Global Fleet Health & Real-Time Operational Status")

    # High level KPI cards
    latest_snapshot = sensors_df.sort_values("timestamp").groupby("machine_id").last().reset_index()
    critical_count = (latest_snapshot["failure_imminent_48h"] == 1).sum()
    healthy_count = len(latest_snapshot) - critical_count
    total_maint_cost = maint_df["cost_incurred_usd"].sum() if not maint_df.empty else 0.0
    est_savings = (critical_count * 5200.0) + (len(maint_df[maint_df['maintenance_type'].str.contains('Preventive', na=False)]) * 3800.0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Active Machines", len(unique_machines), "Fleet Operational")
    col2.metric("Assets Requiring Action", critical_count, f"{critical_count} In Risk Envelope", delta_color="inverse")
    col3.metric("Historical Maintenance Spend", f"${total_maint_cost:,.2f}", f"{len(maint_df)} Logged Events")
    col4.metric("Est. Downtime Cost Avoidance", f"${est_savings:,.2f}", "ROI: 4.8x")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.write("### 🚨 Fleet Status Matrix")
        display_snapshot = latest_snapshot[["machine_id", "machine_type", "temperature_c", "vibration_mm_s", "pressure_psi", "operational_hours", "failure_imminent_48h", "fault_category"]].copy()
        display_snapshot["Health Status"] = display_snapshot["failure_imminent_48h"].apply(lambda x: "CRITICAL RISK" if x == 1 else "HEALTHY")
        
        def highlight_status(val):
            color = 'background-color: #5c1414; color: white' if val == "CRITICAL RISK" else 'background-color: #144622; color: white'
            return color

        st.dataframe(display_snapshot.style.applymap(highlight_status, subset=["Health Status"]), use_container_width=True)

    with col_right:
        st.write("### 📉 Fault Distribution Across Fleet")
        fault_counts = sensors_df[sensors_df["fault_category"] != "None"]["fault_category"].value_counts()
        fig_donut = px.pie(
            values=fault_counts.values,
            names=fault_counts.index,
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Tealgrn_r
        )
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.write("### 📅 Production Load vs Historical Maintenance Events")
    fig_hist = px.bar(
        maint_df,
        x="timestamp",
        y="cost_incurred_usd",
        color="maintenance_type",
        hover_data=["machine_id", "downtime_hours", "fault_detected"],
        title="Historical Intervention Cost & Down-hours by Event Type",
        color_discrete_map={"Corrective / Breakdown": "#EF553B", "Preventive / Scheduled Inspection": "#00CC96"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ----------------- VIEW 2: MACHINE TELEMETRY DEEP-DIVE -----------------
elif nav_choice == "🔍 Machine Telemetry Deep-Dive":
    st.subheader(f"Machine Sensor Telemetry Stream: {selected_machine}")
    
    m_data = sensors_df[sensors_df["machine_id"] == selected_machine].sort_values("timestamp")
    latest = m_data.iloc[-1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Temperature", f"{latest['temperature_c']} °C", "Nominal 50-70°C")
    k2.metric("Vibration RMS", f"{latest['vibration_mm_s']} mm/s", "Alert > 4.5 mm/s")
    k3.metric("Hydraulic Pressure", f"{latest['pressure_psi']} PSI", "Nominal 80-140 PSI")
    k4.metric("Operational Wear", f"{latest['wear_index']*100:.1f} %", "Service at 70%")

    fig_temp_vib = go.Figure()
    fig_temp_vib.add_trace(go.Scatter(x=m_data["timestamp"], y=m_data["temperature_c"], name="Temperature (°C)", line=dict(color="#FFA15A", width=2)))
    fig_temp_vib.add_trace(go.Scatter(x=m_data["timestamp"], y=m_data["vibration_mm_s"]*10, name="Vibration x10 (mm/s)", line=dict(color="#636EFA", width=2)))
    fig_temp_vib.update_layout(title="Multi-sensor Telemetry Progression Over Time", xaxis_title="Timestamp", yaxis_title="Signal Value", height=380)
    st.plotly_chart(fig_temp_vib, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_press = px.line(m_data, x="timestamp", y="pressure_psi", title="Hydraulic Pressure Stability (PSI)", color_discrete_sequence=["#00CC96"])
        st.plotly_chart(fig_press, use_container_width=True)
    with col_b:
        fig_pow = px.line(m_data, x="timestamp", y="power_consumption_kw", title="Power Consumption Draw (kW)", color_discrete_sequence=["#AB63FA"])
        st.plotly_chart(fig_pow, use_container_width=True)

# ----------------- VIEW 3: PREDICTIVE DIAGNOSTICS & RECOMMENDATIONS -----------------
elif nav_choice == "🧠 Predictive Diagnostics & Recommendations":
    st.subheader(f"AI Failure Pattern Recognition & Action Engine: {selected_machine}")
    
    m_data = sensors_df[sensors_df["machine_id"] == selected_machine].sort_values("timestamp")
    latest = m_data.iloc[-1]

    # Model inference simulator
    from recommender import MaintenanceRecommender
    recommender = MaintenanceRecommender()
    
    # Extract latest features
    raw_tel = {
        "machine_id": selected_machine,
        "machine_type": latest["machine_type"],
        "temperature_c": float(latest["temperature_c"]),
        "vibration_mm_s": float(latest["vibration_mm_s"]),
        "pressure_psi": float(latest["pressure_psi"]),
        "rotational_speed_rpm": float(latest["rotational_speed_rpm"]),
        "power_consumption_kw": float(latest["power_consumption_kw"]),
        "acoustic_db": float(latest["acoustic_db"]),
        "operational_hours": float(latest["operational_hours"])
    }

    sched_info = {
        "planned_downtime": 0,
        "target_units_per_hour": 120,
        "load_factor": 0.95,
        "next_planned_maintenance_window": "Upcoming Shift 3 (02:00-05:00 UTC)"
    }

    # Synthesize feature map
    feat_dict = {col: 0.0 for col in recommender.feature_names}
    for k in ["temperature_c", "vibration_mm_s", "pressure_psi", "rotational_speed_rpm", "power_consumption_kw", "acoustic_db", "operational_hours"]:
        if k in feat_dict:
            feat_dict[k] = raw_tel[k]

    feat_dict["vib_to_rpm_ratio"] = raw_tel["vibration_mm_s"] / (raw_tel["rotational_speed_rpm"] + 1e-4)
    feat_dict["power_pressure_ratio"] = raw_tel["power_consumption_kw"] / (raw_tel["pressure_psi"] + 1e-4)
    feat_dict["thermal_deviation"] = raw_tel["temperature_c"] - 62.0

    for w in [6, 24]:
        for stat in ["mean", "std", "max"]:
            if f"vibration_mm_s_roll_{stat}_{w}" in feat_dict:
                feat_dict[f"vibration_mm_s_roll_{stat}_{w}"] = raw_tel["vibration_mm_s"]
            if f"temperature_c_roll_{stat}_{w}" in feat_dict:
                feat_dict[f"temperature_c_roll_{stat}_{w}"] = raw_tel["temperature_c"]

    m_key = f"machine_type_{raw_tel['machine_type']}"
    if m_key in feat_dict:
        feat_dict[m_key] = 1.0

    rec = recommender.recommend(feat_dict, raw_tel, sched_info)

    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    
    with col1:
        st.markdown("### Failure Probability")
        prob = rec["failure_probability"]
        color = "#E74C3C" if prob > 0.5 else ("#F39C12" if prob > 0.25 else "#2ECC71")
        st.markdown(f"<h1 style='color:{color}; font-size: 3.5rem;'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
        st.write(f"**Assigned Risk Tier:** `{rec['risk_tier']}`")
        st.write(f"**Priority Level:** P{rec['priority_score']}")
        st.write(f"**Action Window:** {rec['recommended_action_timeline']}")

    with col2:
        st.markdown("### Production Schedule Optimization")
        st.info(rec["schedule_recommendation"])
        st.markdown("### Root Cause Diagnostics")
        for diag in rec["diagnostics"]:
            st.warning(f"**{diag['component']}**: {diag['finding']}\n\n*Action:* {diag['action']}")

    with col3:
        st.markdown("### Business ROI Impact")
        impact = rec["business_impact"]
        st.metric("Potential Breakdown Cost", f"${impact['estimated_unplanned_failure_cost_usd']:,.2f}")
        st.metric("Preventive Service Cost", f"${impact['preventive_service_cost_usd']:,.2f}")
        st.metric("Net Cost Avoidance", f"${impact['net_cost_avoidance_usd']:,.2f}", delta="Net Profit Protection")

    st.markdown("---")
    st.write("### 🛠️ Interactive What-If Scenario Simulator")
    st.write("Simulate changing machine sensor readings to test real-time model sensitivity and response.")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    sim_temp = sim_col1.slider("Simulated Temperature (°C)", 40.0, 120.0, float(raw_tel["temperature_c"]))
    sim_vib = sim_col2.slider("Simulated Vibration (mm/s)", 0.5, 12.0, float(raw_tel["vibration_mm_s"]))
    sim_press = sim_col3.slider("Simulated Pressure (PSI)", 20.0, 250.0, float(raw_tel["pressure_psi"]))

    if st.button("Run Simulation Inference"):
        sim_tel = dict(raw_tel)
        sim_tel["temperature_c"] = sim_temp
        sim_tel["vibration_mm_s"] = sim_vib
        sim_tel["pressure_psi"] = sim_press
        
        sim_feats = dict(feat_dict)
        sim_feats["temperature_c"] = sim_temp
        sim_feats["vibration_mm_s"] = sim_vib
        sim_feats["pressure_psi"] = sim_press
        sim_feats["thermal_deviation"] = sim_temp - 62.0
        
        sim_rec = recommender.recommend(sim_feats, sim_tel, sched_info)
        st.success(f"Simulation Result: Risk Tier: {sim_rec['risk_tier']} | Failure Probability: {sim_rec['failure_probability']*100:.2f}%")

# ----------------- VIEW 4: MODEL PERFORMANCE & METRICS -----------------
elif nav_choice == "📈 Model Performance & Metrics":
    st.subheader("Model Evaluation, Validation Benchmarks & Feature Importance")
    
    if metrics_data:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Model Architecture", metrics_data.get("model_type", "Random Forest").upper())
        m2.metric("Test Accuracy", f"{metrics_data.get('accuracy', 0)*100:.2f}%")
        m3.metric("Precision", f"{metrics_data.get('precision', 0)*100:.2f}%")
        m4.metric("Recall (Sensitivity)", f"{metrics_data.get('recall', 0)*100:.2f}%")
        m5.metric("F1-Score", f"{metrics_data.get('f1_score', 0)*100:.2f}%")

        st.markdown("---")
        c_left, c_right = st.columns([1.5, 2])
        
        with c_left:
            st.write("### 📊 Confusion Matrix on Holdout Set")
            cm = metrics_data.get("confusion_matrix", {})
            cm_matrix = [
                [cm.get("true_negative", 0), cm.get("false_positive", 0)],
                [cm.get("false_negative", 0), cm.get("true_positive", 0)]
            ]
            fig_cm = px.imshow(
                cm_matrix,
                text_auto=True,
                labels=dict(x="Predicted Condition", y="True Condition"),
                x=["Normal (0)", "Failure Imminent (1)"],
                y=["Normal (0)", "Failure Imminent (1)"],
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=360)
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_right:
            st.write("### 🌟 Top Feature Importance Weights")
            top_feats = metrics_data.get("top_features", [])
            if top_feats:
                df_imp = pd.DataFrame(top_feats)
                fig_imp = px.bar(
                    df_imp,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Key Physical & Operational Drivers of Failure Risk",
                    color="importance",
                    color_continuous_scale="Viridis"
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=360)
                st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Model metrics not found. Please train model using `python src/train.py`.")

