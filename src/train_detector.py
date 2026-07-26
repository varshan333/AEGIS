import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

def train_anomaly_detector(df):
    print("Training Isolation Forest Anomaly Detector on NORMAL baseline traffic...")

    df_train = df[df["label"] == "normal"].copy()
    X_train = df_train[FEATURE_COLS].fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    iso_forest = IsolationForest(
        n_estimators=120,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)

    X_all = df[FEATURE_COLS].fillna(0)
    X_all_scaled = scaler.transform(X_all)

    raw_scores = iso_forest.score_samples(X_all_scaled)

    inv_scores = -raw_scores
    min_s, max_s = inv_scores.min(), inv_scores.max()
    anomaly_scores = (inv_scores - min_s) / (max_s - min_s + 1e-8)

    threshold = np.percentile(anomaly_scores, 97.5)
    is_anomaly = (anomaly_scores >= threshold).astype(int)

    df_result = df.copy()
    df_result["anomaly_score"] = anomaly_scores
    df_result["is_anomaly"] = is_anomaly

    model_payload = {
        "model": iso_forest,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "threshold": threshold,
        "min_score": min_s,
        "max_score": max_s
    }
    model_path = os.path.join(ROOT_DIR, "models", "isolation_forest.pkl")
    joblib.dump(model_payload, model_path)
    print(f"Isolation Forest model saved to '{model_path}'.")

    return df_result, model_payload

if __name__ == "__main__":
    feat_path = os.path.join(ROOT_DIR, "data", "engineered_features.csv")
    if not os.path.exists(feat_path):
        from feature_engineering import extract_features
        df_raw = pd.read_csv(os.path.join(ROOT_DIR, "data", "synthetic_logs.csv"))
        df_feat = extract_features(df_raw)
    else:
        df_feat = pd.read_csv(feat_path)

    df_scored, _ = train_anomaly_detector(df_feat)
    scored_path = os.path.join(ROOT_DIR, "data", "anomaly_scored_logs.csv")
    df_scored.to_csv(scored_path, index=False)
    print(f"Saved anomaly scored logs to '{scored_path}'.")
