import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def calculate_risk_scores(df):
    print("Executing Risk Engine scoring pipeline...")
    df_calc = df.copy()

    anomaly_part = df_calc["anomaly_score"].clip(0, 1) * 35.0
    geo_norm = (df_calc["geo_velocity"] / 800.0).clip(0, 1)
    geo_part = geo_norm * 25.0
    dev_norm = df_calc["device_change"].clip(0, 1)
    dev_part = dev_norm * 15.0
    off_hours_part = df_calc["off_hours_score"].clip(0, 1) * 15.0
    sens_part = df_calc["resource_sensitivity_weight"].clip(0, 1) * 10.0
    fail_bonus = (df_calc["failed_login_count"] / 10.0).clip(0, 1) * 15.0

    total_risk = anomaly_part + geo_part + dev_part + off_hours_part + sens_part + fail_bonus
    risk_scores = np.clip(np.round(total_risk), 0, 100).astype(int)
    df_calc["risk_score"] = risk_scores

    def assign_severity(score):
        if score <= 30:
            return "Low"
        elif score <= 60:
            return "Medium"
        elif score <= 80:
            return "High"
        else:
            return "Critical"

    df_calc["severity"] = df_calc["risk_score"].map(assign_severity)

    df_calc["factor_device"] = np.round(dev_part).astype(int)
    df_calc["factor_geo"] = np.round(geo_part).astype(int)
    df_calc["factor_off_hours"] = np.round(off_hours_part).astype(int)
    df_calc["factor_sensitivity"] = np.round(sens_part).astype(int)
    df_calc["factor_failures"] = np.round(fail_bonus).astype(int)

    alerts_df = df_calc[(df_calc["risk_score"] >= 35) | (df_calc["is_anomaly"] == 1) | (df_calc["label"] != "normal")].copy()
    alerts_df = alerts_df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)

    alerts_path = os.path.join(ROOT_DIR, "outputs", "alerts.csv")
    alerts_df.to_csv(alerts_path, index=False)
    
    full_scored_path = os.path.join(ROOT_DIR, "data", "full_risk_scored_logs.csv")
    df_calc.to_csv(full_scored_path, index=False)

    print(f"Risk calculation complete. Total records: {len(df_calc)}")
    print(f"Generated {len(alerts_df)} security alerts saved to '{alerts_path}'.")

    return df_calc, alerts_df

if __name__ == "__main__":
    class_path = os.path.join(ROOT_DIR, "data", "classified_logs.csv")
    if not os.path.exists(class_path):
        from train_classifier import train_attack_classifier
        from train_detector import train_anomaly_detector
        df_feat = pd.read_csv(os.path.join(ROOT_DIR, "data", "engineered_features.csv"))
        df_scored, _ = train_anomaly_detector(df_feat)
        df_classified, _ = train_attack_classifier(df_scored)
    else:
        df_classified = pd.read_csv(class_path)

    df_full, df_alerts = calculate_risk_scores(df_classified)
