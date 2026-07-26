import os
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

FEATURE_COLS = [
    "hour_of_day",
    "day_of_week",
    "session_duration",
    "off_hours_score",
    "resource_sensitivity_weight",
    "auth_failure_flag",
    "auth_method_encoded",
    "geo_velocity",
    "location_change",
    "device_change",
    "new_resource_access",
    "new_device_flag",
    "failed_login_count",
    "historical_resource_score",
    "historical_login_score",
    "resource_diversity"
]

class ExplainabilityEngine:
    def __init__(self, classifier_model_path=None):
        if classifier_model_path is None:
            classifier_model_path = os.path.join(ROOT_DIR, "models", "attack_classifier.pkl")

        self.model_payload = None
        self.clf = None
        self.scaler = None
        self.explainer = None

        if os.path.exists(classifier_model_path):
            try:
                self.model_payload = joblib.load(classifier_model_path)
                self.clf = self.model_payload.get("model")
                self.scaler = self.model_payload.get("scaler")
                
                if SHAP_AVAILABLE and self.clf is not None:
                    self.explainer = shap.TreeExplainer(self.clf)
            except Exception as e:
                print(f"Error loading classifier model for SHAP: {e}")

    def get_shap_values(self, row_dict):
        if self.clf is None or self.scaler is None:
            return {f: 0.0 for f in FEATURE_COLS}

        vals = [float(row_dict.get(col, 0)) for col in FEATURE_COLS]
        X_vec = np.array(vals).reshape(1, -1)
        X_scaled = self.scaler.transform(X_vec)

        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    pred_cls = self.clf.predict(X_scaled)[0]
                    cls_shap = shap_vals[pred_cls][0]
                else:
                    cls_shap = shap_vals[0]
                return dict(zip(FEATURE_COLS, cls_shap))
            except Exception:
                pass

        importances = self.clf.feature_importances_
        contributions = {f: float(X_scaled[0][i] * importances[i]) for i, f in enumerate(FEATURE_COLS)}
        return contributions

    def get_event_explanation(self, row):
        factors = []
        
        dev_pts = int(row.get("factor_device", 0))
        if dev_pts > 0 or row.get("device_change", 0) == 1:
            pts = max(35, dev_pts if dev_pts > 0 else 35)
            fp = row.get("device_fingerprint", "Unknown")
            factors.append((pts, f"+{pts} New Device Fingerprint / OS Mismatch ({fp})"))

        geo_velocity = float(row.get("geo_velocity", 0))
        geo_pts = int(row.get("factor_geo", 0))
        if geo_velocity > 500 or geo_pts > 15:
            pts = max(30, geo_pts if geo_pts > 0 else 30)
            city = row.get("geo_location", "Unknown")
            factors.append((pts, f"+{pts} Impossible Travel Velocity ({geo_velocity:.0f} km/h to {city})"))

        off_hours_pts = int(row.get("factor_off_hours", 0))
        hour = int(row.get("hour_of_day", 12))
        if off_hours_pts > 5 or row.get("off_hours_score", 0) > 0.5:
            pts = max(15, off_hours_pts if off_hours_pts > 0 else 15)
            factors.append((pts, f"+{pts} Off-Hours Access ({hour:02d}:00 Local Time)"))

        res_name = row.get("resource_accessed", "System")
        res_sens = row.get("resource_sensitivity", "internal")
        sens_pts = int(row.get("factor_sensitivity", 0))
        if res_sens in ["critical", "restricted"]:
            pts = 15 if res_sens == "critical" else 10
            factors.append((pts, f"+{pts} Access to {res_sens.upper()} Resource ({res_name})"))

        fail_count = int(row.get("failed_login_count", 0))
        fail_pts = int(row.get("factor_failures", 0))
        if fail_count > 1:
            pts = max(20, fail_count * 3)
            factors.append((pts, f"+{pts} High Auth Failure Burst ({fail_count} Failed Attempts)"))

        if row.get("new_resource_access", 0) == 1:
            factors.append((6, f"+6 First-Time Novel Resource Access ({res_name})"))

        factors.sort(key=lambda x: x[0], reverse=True)

        if not factors:
            factors = [(10, "+10 Statistical Anomaly in Behavioral Pattern")]

        shap_dict = self.get_shap_values(row)

        return {
            "entity_id": row.get("entity_id", "N/A"),
            "risk_score": row.get("risk_score", 0),
            "severity": row.get("severity", "Low"),
            "predicted_attack": row.get("predicted_attack", "Unknown"),
            "contributing_factors": factors,
            "shap_values": shap_dict
        }

if __name__ == "__main__":
    engine = ExplainabilityEngine()
    alerts_path = os.path.join(ROOT_DIR, "outputs", "alerts.csv")
    if os.path.exists(alerts_path):
        df_alerts = pd.read_csv(alerts_path)
        if len(df_alerts) > 0:
            sample_alert = df_alerts.iloc[0]
            exp = engine.get_event_explanation(sample_alert)
            print("--- SAMPLE EXPLAINABLE AI OUTPUT ---")
            print(f"Alert Entity: {exp['entity_id']} | Risk Score: {exp['risk_score']} ({exp['severity']})")
            print(f"Predicted Attack Vector: {exp['predicted_attack']}")
            print("Contributing Factors:")
            for pts, reason in exp["contributing_factors"]:
                print(f"  - {reason}")
    else:
        print("outputs/alerts.csv not found.")
