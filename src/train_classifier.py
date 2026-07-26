import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

def train_attack_classifier(df):
    print("Training Random Forest Attack Classifier on anomalous events...")

    df_anomaly = df[df["label"] != "normal"].copy()
    if len(df_anomaly) < 10:
        df_anomaly = df.sample(min(1000, len(df)), random_state=42).copy()

    X = df_anomaly[FEATURE_COLS].fillna(0)
    y_raw = df_anomaly["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_scaled, y_encoded)

    X_all = df[FEATURE_COLS].fillna(0)
    X_all_scaled = scaler.transform(X_all)

    probs = clf.predict_proba(X_all_scaled)
    pred_indices = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    predicted_classes = label_encoder.inverse_transform(pred_indices)

    df_result = df.copy()
    df_result["predicted_attack"] = predicted_classes
    df_result["confidence_score"] = confidences

    model_payload = {
        "model": clf,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": FEATURE_COLS,
        "classes": label_encoder.classes_.tolist()
    }
    model_path = os.path.join(ROOT_DIR, "models", "attack_classifier.pkl")
    joblib.dump(model_payload, model_path)
    print(f"Attack classifier model saved to '{model_path}'.")

    return df_result, model_payload

if __name__ == "__main__":
    scored_path = os.path.join(ROOT_DIR, "data", "anomaly_scored_logs.csv")
    if not os.path.exists(scored_path):
        from train_detector import train_anomaly_detector
        df_feat = pd.read_csv(os.path.join(ROOT_DIR, "data", "engineered_features.csv"))
        df_feat, _ = train_anomaly_detector(df_feat)
    else:
        df_feat = pd.read_csv(scored_path)

    df_classified, _ = train_attack_classifier(df_feat)
    class_path = os.path.join(ROOT_DIR, "data", "classified_logs.csv")
    df_classified.to_csv(class_path, index=False)
    print(f"Saved classified logs to '{class_path}'.")
