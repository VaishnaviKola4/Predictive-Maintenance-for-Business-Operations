import unittest
import os
import json
import numpy as np
import pandas as pd

from src.data_generator import generate_synthetic_data
from src.preprocessor import load_and_merge_datasets, engineer_features, prepare_train_test_split
from src.recommender import MaintenanceRecommender

class TestPredictiveMaintenanceSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "data"
        if not os.path.exists(os.path.join(cls.test_dir, "machine_sensors.csv")):
            generate_synthetic_data(num_machines=4, days=5, output_dir=cls.test_dir)
        cls.recommender = MaintenanceRecommender()
        cls.merged, cls.maint = load_and_merge_datasets()
        cls.featured = engineer_features(cls.merged)

    def test_datasets_exist_and_non_empty(self):
        sensors = pd.read_csv("data/machine_sensors.csv")
        maint = pd.read_csv("data/maintenance_history.csv")
        sched = pd.read_csv("data/production_schedules.csv")

        self.assertFalse(sensors.empty, "Sensors dataset should not be empty")
        self.assertFalse(sched.empty, "Production schedules dataset should not be empty")
        self.assertIn("vibration_mm_s", sensors.columns)
        self.assertIn("temperature_c", sensors.columns)
        self.assertIn("failure_imminent_48h", sensors.columns)

    def test_feature_engineering_pipeline(self):
        self.assertIn("vibration_mm_s_roll_mean_6", self.featured.columns)
        self.assertIn("temperature_c_roll_std_24", self.featured.columns)
        self.assertIn("vib_to_rpm_ratio", self.featured.columns)
        self.assertFalse(self.featured[self.recommender.feature_names].isnull().values.any(), "Engineered feature matrix should have no NaNs")

    def test_model_artifact_loading(self):
        self.assertIsNotNone(self.recommender.model, "Random Forest model should be loaded")
        self.assertIsNotNone(self.recommender.scaler, "Scaler should be loaded")
        self.assertGreater(len(self.recommender.feature_names), 10, "Feature list should be non-empty")

    def test_recommendation_healthy_state(self):
        healthy_row = self.featured[self.featured["failure_imminent_48h"] == 0].iloc[0]
        raw_tel = healthy_row[['machine_id', 'temperature_c', 'vibration_mm_s', 'pressure_psi', 'rotational_speed_rpm', 'power_consumption_kw', 'acoustic_db', 'operational_hours']].to_dict()
        feat_dict = healthy_row[self.recommender.feature_names].to_dict()
        sched = {
            "planned_downtime": 0,
            "target_units_per_hour": 120,
            "load_factor": 0.85
        }
        res = self.recommender.recommend(feat_dict, raw_tel, sched)
        
        self.assertIn("failure_probability", res)
        self.assertIn("risk_tier", res)
        self.assertIn("schedule_recommendation", res)
        self.assertIn("business_impact", res)
        self.assertLess(res["failure_probability"], 0.50)

    def test_recommendation_critical_state(self):
        fail_row = self.featured[self.featured["failure_imminent_48h"] == 1].iloc[0]
        raw_tel = fail_row[['machine_id', 'temperature_c', 'vibration_mm_s', 'pressure_psi', 'rotational_speed_rpm', 'power_consumption_kw', 'acoustic_db', 'operational_hours']].to_dict()
        feat_dict = fail_row[self.recommender.feature_names].to_dict()
        sched = {
            "planned_downtime": 0,
            "target_units_per_hour": 150,
            "load_factor": 1.15
        }

        res = self.recommender.recommend(feat_dict, raw_tel, sched)
        self.assertGreaterEqual(res["failure_probability"], 0.75)
        self.assertEqual(res["risk_tier"], "CRITICAL")
        self.assertGreater(res["business_impact"]["net_cost_avoidance_usd"], 0)

if __name__ == "__main__":
    unittest.main()
