import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

def generate_synthetic_data(
    num_machines: int = 12,
    days: int = 60,
    sampling_rate_minutes: int = 30,
    random_seed: int = 42,
    output_dir: str = "data"
):
    """
    Generates realistic, synchronized synthetic datasets for predictive maintenance:
    1. machine_sensors.csv: Time-series telemetry (temp, vibration, pressure, rotation, acoustic, power).
    2. maintenance_history.csv: Historical service records, faults, repair logs, parts replaced.
    3. production_schedules.csv: Operational load, batch targets, planned downtime windows.
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = datetime(2026, 8, 1, 0, 0, 0)
    total_steps = int((days * 24 * 60) / sampling_rate_minutes)
    timestamps = [start_time + timedelta(minutes=i * sampling_rate_minutes) for i in range(total_steps)]
    
    machine_types = {
        "CNC-Milling": {"base_temp": 65.0, "base_vib": 2.5, "base_press": 90.0, "base_rpm": 3200, "base_power": 18.5},
        "Hydraulic-Press": {"base_temp": 55.0, "base_vib": 1.8, "base_press": 210.0, "base_rpm": 1200, "base_power": 28.0},
        "Industrial-Pump": {"base_temp": 50.0, "base_vib": 3.0, "base_press": 120.0, "base_rpm": 2800, "base_power": 14.0},
        "Rotary-Kiln": {"base_temp": 110.0, "base_vib": 4.2, "base_press": 45.0, "base_rpm": 600, "base_power": 45.0}
    }
    
    type_keys = list(machine_types.keys())
    machines = []
    for m_id in range(1, num_machines + 1):
        m_type = type_keys[(m_id - 1) % len(type_keys)]
        machines.append({
            "machine_id": f"MACH-{m_id:03d}",
            "machine_type": m_type,
            "installation_date": (start_time - timedelta(days=np.random.randint(180, 1000))).strftime("%Y-%m-%d"),
            "location_zone": f"Zone-{chr(65 + (m_id % 4))}",
            "baseline": machine_types[m_type]
        })

    # 1. Generate Production Schedules
    schedule_records = []
    current_date = start_time.date()
    end_date = timestamps[-1].date()
    
    while current_date <= end_date:
        for m in machines:
            m_id = m["machine_id"]
            for shift in [1, 2, 3]:
                is_planned_downtime = (np.random.rand() < 0.07)
                target_throughput = 0 if is_planned_downtime else int(np.random.normal(120, 15))
                load_factor = 0.0 if is_planned_downtime else float(np.clip(np.random.normal(0.85, 0.1), 0.4, 1.25))
                
                schedule_records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "shift": shift,
                    "machine_id": m_id,
                    "planned_downtime": int(is_planned_downtime),
                    "target_units_per_hour": max(0, target_throughput),
                    "load_factor": round(load_factor, 2),
                    "batch_type": np.random.choice(["High-Tolerance-Steel", "Standard-Alloy", "Polymer-Composite", "Cleaning-Run"])
                })
        current_date += timedelta(days=1)
        
    df_schedules = pd.DataFrame(schedule_records)
    schedule_file = os.path.join(output_dir, "production_schedules.csv")
    df_schedules.to_csv(schedule_file, index=False)
    print(f"Saved {len(df_schedules)} production schedule records to {schedule_file}")

    # 2. Generate Sensor Readings & Degradation Curves
    sensor_records = []
    maintenance_records = []
    
    machine_states = {}
    for m in machines:
        machine_states[m["machine_id"]] = {
            "wear": np.random.uniform(0.1, 0.5),
            "service_hours": np.random.uniform(20.0, 400.0),
            "degradation_rate": np.random.uniform(0.0007, 0.0016),
            "active_fault": "None",
            "failure_countdown_steps": -1
        }
    
    fault_modes = [
        "Bearing Degradation",
        "Overheating & Thermal Runaway",
        "Hydraulic Pressure Drop",
        "Shaft Imbalance & High Vibration",
        "Motor Coil Overload"
    ]
    
    # Pre-calculate schedule lookup for quick indexing
    sched_lookup = {(r.date, r.shift, r.machine_id): (r.load_factor, r.planned_downtime) 
                    for r in df_schedules.itertuples()}

    for t_idx, ts in enumerate(timestamps):
        current_date_str = ts.strftime("%Y-%m-%d")
        hour = ts.hour
        shift = 1 if 6 <= hour < 14 else (2 if 14 <= hour < 22 else 3)
        ambient_temp = 22.0 + 4.0 * np.sin(2 * np.pi * hour / 24.0)
        
        for m in machines:
            m_id = m["machine_id"]
            base = m["baseline"]
            st = machine_states[m_id]
            
            load_factor, is_planned_down = sched_lookup.get((current_date_str, shift, m_id), (0.85, 0))
            
            operating = (1 - is_planned_down)
            st["service_hours"] += (sampling_rate_minutes / 60.0) * operating
            st["wear"] += st["degradation_rate"] * (load_factor if operating else 0.05)
            
            # If wear crosses 0.75, start failure countdown (imminent failure in 48-72h)
            if st["wear"] >= 0.75 and st["failure_countdown_steps"] < 0:
                st["failure_countdown_steps"] = np.random.randint(60, 110) # 30 to 55 hours
                st["active_fault"] = np.random.choice(fault_modes)
            
            is_failure_imminent = 0
            if st["failure_countdown_steps"] > 0:
                is_failure_imminent = 1
                st["failure_countdown_steps"] -= 1
            
            # Telemetry synthesis
            if is_planned_down:
                temp = ambient_temp + np.random.normal(0, 0.4)
                vib = np.random.exponential(0.08)
                press = 6.0 + np.random.normal(0, 0.4)
                rpm = 0.0
                power = 0.6 + np.random.normal(0, 0.05)
                acoustic = 28.0 + np.random.normal(0, 1.0)
            else:
                wear_factor = min(1.0, st["wear"])
                temp = base["base_temp"] + (wear_factor * 28.0) + (load_factor * 8.0) + np.random.normal(0, 1.2)
                vib = (base["base_vib"] * (1.0 + wear_factor * 2.0)) + (load_factor * 0.9) + np.random.normal(0, 0.3)
                press = base["base_press"] * (1.0 - wear_factor * 0.25) + np.random.normal(0, 2.5)
                rpm = (base["base_rpm"] * np.random.normal(1.0, 0.02)) * load_factor
                power = (base["base_power"] * load_factor * (1.0 + wear_factor * 0.7)) + np.random.normal(0, 0.8)
                acoustic = 58.0 + (wear_factor * 35.0) + (vib * 3.5) + np.random.normal(0, 2.0)
                
                # Apply active failure signatures
                if is_failure_imminent:
                    fault = st["active_fault"]
                    if "Overheating" in fault:
                        temp += np.random.uniform(18, 38)
                    elif "Bearing" in fault or "Shaft" in fault:
                        vib += np.random.uniform(4.0, 9.0)
                        acoustic += np.random.uniform(18.0, 32.0)
                    elif "Hydraulic" in fault:
                        press *= np.random.uniform(0.35, 0.65)
                    elif "Motor Coil" in fault:
                        power *= np.random.uniform(1.35, 1.9)
                        temp += np.random.uniform(12, 22)

            sensor_records.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "machine_id": m_id,
                "machine_type": m["machine_type"],
                "temperature_c": round(float(temp), 2),
                "vibration_mm_s": round(float(vib), 3),
                "pressure_psi": round(float(press), 2),
                "rotational_speed_rpm": round(float(rpm), 1),
                "power_consumption_kw": round(float(power), 2),
                "acoustic_db": round(float(acoustic), 2),
                "operational_hours": round(float(st["service_hours"]), 1),
                "wear_index": round(float(st["wear"]), 4),
                "fault_category": st["active_fault"] if is_failure_imminent else "None",
                "failure_imminent_48h": is_failure_imminent
            })

            # Check for maintenance event
            # 1. Catastrophic Breakdown (countdown reaches 0)
            if st["failure_countdown_steps"] == 0:
                maintenance_records.append({
                    "maintenance_id": f"MAINT-{len(maintenance_records)+1:04d}",
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "machine_id": m_id,
                    "maintenance_type": "Corrective / Breakdown",
                    "technician_notes": f"Unplanned breakdown occurred. Severe fault: {st['active_fault']}.",
                    "fault_detected": st["active_fault"],
                    "downtime_hours": round(float(np.random.uniform(6.0, 20.0)), 2),
                    "cost_incurred_usd": round(float(np.random.uniform(3200, 9500)), 2),
                    "parts_replaced": np.random.choice(["Roller Bearings & Seals", "Cooling Fan & Sensor Array", "Hydraulic Cylinder & Valve", "Electric Rotor & Wiring", "Gear Assembly"]),
                    "action_taken": "Complete unit overhaul and recalibration."
                })
                # Reset
                st["wear"] = float(np.random.uniform(0.04, 0.12))
                st["service_hours"] = 0.0
                st["failure_countdown_steps"] = -1
                st["active_fault"] = "None"

            # 2. Preventive maintenance when service hours high or wear is moderate
            elif st["service_hours"] > 550 and np.random.rand() < 0.08:
                maintenance_records.append({
                    "maintenance_id": f"MAINT-{len(maintenance_records)+1:04d}",
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "machine_id": m_id,
                    "maintenance_type": "Preventive / Scheduled Inspection",
                    "technician_notes": "Routine scheduled service completed based on predictive interval.",
                    "fault_detected": "Minor Wear Detected" if st["wear"] > 0.45 else "None",
                    "downtime_hours": round(float(np.random.uniform(1.2, 3.0)), 2),
                    "cost_incurred_usd": round(float(np.random.uniform(350, 900)), 2),
                    "parts_replaced": "Lubricant & Filter Pack",
                    "action_taken": "Lubrication, sensor calibration, and tension adjustment."
                })
                st["wear"] = float(np.random.uniform(0.04, 0.12))
                st["service_hours"] = 0.0
                st["failure_countdown_steps"] = -1
                st["active_fault"] = "None"

    df_sensors = pd.DataFrame(sensor_records)
    sensor_file = os.path.join(output_dir, "machine_sensors.csv")
    df_sensors.to_csv(sensor_file, index=False)
    print(f"Saved {len(df_sensors)} sensor telemetry records to {sensor_file}")
    
    df_maint = pd.DataFrame(maintenance_records)
    maint_file = os.path.join(output_dir, "maintenance_history.csv")
    df_maint.to_csv(maint_file, index=False)
    print(f"Saved {len(df_maint)} maintenance event records to {maint_file}")

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    summary = {
        "generated_timestamp": datetime.now().isoformat(),
        "total_sensor_records": len(df_sensors),
        "total_maintenance_records": len(df_maint),
        "total_schedule_records": len(df_schedules),
        "unique_machines": num_machines,
        "sampling_rate_minutes": sampling_rate_minutes,
        "failure_rate": float(df_sensors["failure_imminent_48h"].mean()),
        "breakdown_events": int(len(df_maint[df_maint["maintenance_type"].str.contains("Corrective", na=False)])),
        "preventive_events": int(len(df_maint[df_maint["maintenance_type"].str.contains("Preventive", na=False)])),
        "fault_distribution": df_sensors["fault_category"].value_counts().to_dict()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    generate_synthetic_data()
