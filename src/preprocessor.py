import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_merge_datasets(
    sensors_path: str = "data/machine_sensors.csv",
    maintenance_path: str = "data/maintenance_history.csv",
    schedules_path: str = "data/production_schedules.csv"
) -> pd.DataFrame:
    """
    Loads raw CSVs and merges them into a unified feature set:
    - Parses timestamps and shifts
    - Links sensor telemetry with prevailing production schedule load
    - Aggregates recent maintenance event metrics
    """
    df_sensors = pd.read_csv(sensors_path)
    df_maint = pd.read_csv(maintenance_path)
    df_sched = pd.read_csv(schedules_path)

    df_sensors["timestamp"] = pd.to_datetime(df_sensors["timestamp"])
    df_sensors["date"] = df_sensors["timestamp"].dt.strftime("%Y-%m-%d")
    
    # Calculate shift (1: 06:00-14:00, 2: 14:00-22:00, 3: 22:00-06:00)
    hour = df_sensors["timestamp"].dt.hour
    df_sensors["shift"] = np.where((hour >= 6) & (hour < 14), 1, np.where((hour >= 14) & (hour < 22), 2, 3))

    # Merge sensor data with production schedule
    df_sched["shift"] = df_sched["shift"].astype(int)
    merged = pd.merge(
        df_sensors,
        df_sched[["date", "shift", "machine_id", "planned_downtime", "target_units_per_hour", "load_factor", "batch_type"]],
        on=["date", "shift", "machine_id"],
        how="left"
    )
    merged["load_factor"] = merged["load_factor"].fillna(0.85)
    merged["planned_downtime"] = merged["planned_downtime"].fillna(0)
    merged["target_units_per_hour"] = merged["target_units_per_hour"].fillna(120)

    # Calculate days since last maintenance and cumulative downtime
    df_maint["timestamp"] = pd.to_datetime(df_maint["timestamp"])
    
    # Sort for time-series feature engineering
    merged = merged.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
    return merged, df_maint

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling statistics, rate-of-change, degradation gradients, and ratios.
    """
    df = df.copy()
    
    df["fault_category"] = df["fault_category"].fillna("None")

    # Group by machine to preserve time series integrity
    grouped = df.groupby("machine_id")

    # 1. Rolling window statistics (short-term & medium-term)
    windows = [6, 24]  # 6 steps = 3 hours, 24 steps = 12 hours
    features_to_roll = ["temperature_c", "vibration_mm_s", "pressure_psi", "power_consumption_kw", "acoustic_db"]

    for w in windows:
        for feat in features_to_roll:
            df[f"{feat}_roll_mean_{w}"] = grouped[feat].transform(lambda s: s.rolling(w, min_periods=1).mean())
            df[f"{feat}_roll_std_{w}"] = grouped[feat].transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
            df[f"{feat}_roll_max_{w}"] = grouped[feat].transform(lambda s: s.rolling(w, min_periods=1).max())

    # 2. Rate of Change / Differential Features
    for feat in ["temperature_c", "vibration_mm_s", "pressure_psi"]:
        df[f"{feat}_diff_1h"] = grouped[feat].transform(lambda s: s.diff(periods=2).fillna(0))
        df[f"{feat}_accel"] = grouped[f"{feat}_diff_1h"].transform(lambda s: s.diff(periods=2).fillna(0))

    # 3. Domain Interaction Indicators
    # Ratio of vibration to rpm (abnormal vibration per cycle)
    df["vib_to_rpm_ratio"] = df["vibration_mm_s"] / (df["rotational_speed_rpm"] + 1e-4)
    # Energy dissipation index: Power per unit pressure
    df["power_pressure_ratio"] = df["power_consumption_kw"] / (df["pressure_psi"] + 1e-4)
    # Thermal anomaly score (temperature relative to rolling 12h baseline)
    df["thermal_deviation"] = df["temperature_c"] - df["temperature_c_roll_mean_24"]

    # One-hot encode categorical features: machine_type, batch_type
    df = pd.get_dummies(df, columns=["machine_type", "batch_type"], drop_first=False)

    return df

def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = "failure_imminent_48h",
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, StandardScaler]:
    """
    Performs temporal split to avoid future lookahead bias in time series.
    """
    unique_timestamps = np.sort(df["timestamp"].unique())
    split_idx = int(len(unique_timestamps) * (1.0 - test_ratio))
    split_time = unique_timestamps[split_idx]

    train_mask = df["timestamp"] < split_time
    test_mask = df["timestamp"] >= split_time

    drop_cols = [
        "timestamp", "date", "machine_id", "fault_category", 
        "wear_index", target_col
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, target_col].values

    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

if __name__ == "__main__":
    merged, _ = load_and_merge_datasets()
    featured = engineer_features(merged)
    print(f"Engineered dataset shape: {featured.shape}")
    X_tr, X_te, y_tr, y_te, f_cols, scaler = prepare_train_test_split(featured)
    print(f"X_train: {X_tr.shape}, X_test: {X_te.shape}, Feature count: {len(f_cols)}")
    print(f"Positive failure rate in train: {y_tr.mean():.4f}, in test: {y_te.mean():.4f}")

