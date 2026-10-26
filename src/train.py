import numpy as np
import pandas as pd
import json
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from preprocessor import load_and_merge_datasets, engineer_features, prepare_train_test_split

class PredictiveMaintenanceTrainer:
    def __init__(self, model_type: str = "random_forest", output_dir: str = "models"):
        self.model_type = model_type
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metrics = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list, scaler):
        self.feature_names = feature_names
        self.scaler = scaler

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        print(f"Training {self.model_type} on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        print("Evaluating model performance on holdout test set...")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Compute feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_imp = sorted(
                [{"feature": f, "importance": round(float(imp), 4)} 
                 for f, imp in zip(self.feature_names, importances)],
                key=lambda x: x["importance"],
                reverse=True
            )
        else:
            feature_imp = []

        self.metrics = {
            "model_type": self.model_type,
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_samples": len(y_test),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1]
            },
            "top_features": feature_imp[:12]
        }

        print("\n=== MODEL PERFORMANCE METRICS ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("Confusion Matrix [[TN, FP], [FN, TP]]:")
        print(cm)
        print("=================================\n")

        return self.metrics

    def save_artifacts(self):
        model_path = os.path.join(self.output_dir, f"{self.model_type}_model.joblib")
        scaler_path = os.path.join(self.output_dir, "feature_scaler.joblib")
        features_path = os.path.join(self.output_dir, "feature_names.json")
        metrics_path = os.path.join(self.output_dir, "evaluation_metrics.json")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)

        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Metrics saved to {metrics_path}")

def run_training_pipeline(model_type: str = "random_forest"):
    print("Step 1: Loading and merging multi-source datasets...")
    merged, maint_df = load_and_merge_datasets()

    print("Step 2: Engineering time-series & operational features...")
    featured = engineer_features(merged)

    print("Step 3: Performing temporal train/test split...")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_train_test_split(featured)

    trainer = PredictiveMaintenanceTrainer(model_type=model_type)
    trainer.train(X_train, y_train, feature_names, scaler)
    trainer.evaluate(X_test, y_test)
    trainer.save_artifacts()

if __name__ == "__main__":
    run_training_pipeline(model_type="random_forest")

