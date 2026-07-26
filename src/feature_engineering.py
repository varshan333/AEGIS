import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def extract_features(df_input):
    print("Beginning feature engineering pipeline...")
    df = df_input.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(by="timestamp").reset_index(drop=True)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    df["off_hours_score"] = df["hour_of_day"].apply(
        lambda h: 1.0 if 1 <= h <= 5 else (0.5 if h in [0, 6, 7, 21, 22, 23] else 0.0)
    )

    sensitivity_map = {"public": 0.1, "internal": 0.3, "restricted": 0.7, "critical": 1.0}
    df["resource_sensitivity_weight"] = df["resource_sensitivity"].map(lambda s: sensitivity_map.get(str(s).lower(), 0.3))

    df["auth_failure_flag"] = (df["auth_status"] == "failure").astype(int)
    
    auth_enc = {"password": 1, "token": 2, "certificate": 3, "biometric": 4}
    df["auth_method_encoded"] = df["auth_method"].map(lambda m: auth_enc.get(str(m).lower(), 1))

    pop_fav_hours = df["hour_of_day"].mode()[0] if not df["hour_of_day"].empty else 12

    entity_history = {}
    ip_failed_logins = {}

    geo_velocity_list = []
    location_change_list = []
    device_change_list = []
    new_resource_list = []
    new_device_list = []
    failed_login_count_list = []
    historical_resource_score_list = []
    historical_login_score_list = []
    is_cold_start_list = []

    for idx, row in df.iterrows():
        entity = row["entity_id"]
        ip = row["source_ip"]
        ts = row["timestamp"]
        lat, lon = row["lat"], row["lon"]
        fp = row["device_fingerprint"]
        res = row["resource_accessed"]
        hour = row["hour_of_day"]
        auth_failed = row["auth_failure_flag"]

        if ip not in ip_failed_logins:
            ip_failed_logins[ip] = []
        ip_failed_logins[ip] = [t for t in ip_failed_logins[ip] if (ts - t).total_seconds() <= 3600]
        if auth_failed:
            ip_failed_logins[ip].append(ts)
        failed_count = len(ip_failed_logins[ip])
        failed_login_count_list.append(failed_count)

        if entity not in entity_history:
            is_cold_start_list.append(1)
            geo_velocity_list.append(0.0)
            location_change_list.append(0.0)
            device_change_list.append(0.0)
            new_resource_list.append(0)
            new_device_list.append(0)
            historical_resource_score_list.append(0.5)
            
            hour_diff = abs(hour - pop_fav_hours)
            historical_login_score_list.append(1.0 - (hour_diff / 12.0))

            entity_history[entity] = {
                "logs": [{
                    "ts": ts, "lat": lat, "lon": lon, "fp": fp, "res": res, "hour": hour
                }],
                "primary_fp": fp,
                "home_lat": lat,
                "home_lon": lon,
                "resources_seen": {res: 1},
                "devices_seen": {fp: 1},
                "hour_counts": {hour: 1}
            }
        else:
            hist = entity_history[entity]
            logs = hist["logs"]
            is_cold_start = 1 if len(logs) < 3 else 0
            is_cold_start_list.append(is_cold_start)

            cutoff_7d = ts - timedelta(days=7)
            recent_logs = [l for l in logs if l["ts"] >= cutoff_7d]
            
            last_log = logs[-1]
            time_delta_hrs = (ts - last_log["ts"]).total_seconds() / 3600.0
            
            dist_km = haversine_distance(last_log["lat"], last_log["lon"], lat, lon)
            if time_delta_hrs > 0:
                velocity = dist_km / time_delta_hrs
            else:
                velocity = dist_km * 60.0

            loc_change = haversine_distance(hist["home_lat"], hist["home_lon"], lat, lon)
            dev_change = 1.0 if fp != hist["primary_fp"] else 0.0
            new_dev = 1 if fp not in hist["devices_seen"] else 0
            new_res = 1 if res not in hist["resources_seen"] else 0
            
            hist_res_count = hist["resources_seen"].get(res, 0)
            recent_res_count = sum(1 for l in recent_logs if l["res"] == res)
            
            hist_weight = 0.7 * (hist_res_count / len(logs))
            recent_weight = 0.3 * (recent_res_count / max(1, len(recent_logs)))
            res_score = hist_weight + recent_weight
            
            hist_hour_count = hist["hour_counts"].get(hour, 0)
            recent_hour_count = sum(1 for l in recent_logs if l["hour"] == hour)
            login_score = 0.7 * (hist_hour_count / len(logs)) + 0.3 * (recent_hour_count / max(1, len(recent_logs)))

            geo_velocity_list.append(float(velocity))
            location_change_list.append(float(loc_change))
            device_change_list.append(dev_change)
            new_resource_list.append(new_res)
            new_device_list.append(new_dev)
            historical_resource_score_list.append(float(res_score))
            historical_login_score_list.append(float(login_score))

            logs.append({"ts": ts, "lat": lat, "lon": lon, "fp": fp, "res": res, "hour": hour})
            hist["resources_seen"][res] = hist["resources_seen"].get(res, 0) + 1
            hist["devices_seen"][fp] = hist["devices_seen"].get(fp, 0) + 1
            hist["hour_counts"][hour] = hist["hour_counts"].get(hour, 0) + 1

    df["geo_velocity"] = geo_velocity_list
    df["location_change"] = location_change_list
    df["device_change"] = device_change_list
    df["new_resource_access"] = new_resource_list
    df["new_device_flag"] = new_device_list
    df["failed_login_count"] = failed_login_count_list
    df["historical_resource_score"] = historical_resource_score_list
    df["historical_login_score"] = historical_login_score_list
    df["is_cold_start"] = is_cold_start_list

    df["resource_diversity"] = df.groupby("entity_id")["resource_accessed"].transform(lambda s: s.nunique())
    df["resource_frequency"] = df["historical_resource_score"]

    print("Feature engineering complete. Dataset shape:", df.shape)
    return df

if __name__ == "__main__":
    data_path = os.path.join(ROOT_DIR, "data", "synthetic_logs.csv")
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        df_feat = extract_features(df_raw)
        out_path = os.path.join(ROOT_DIR, "data", "engineered_features.csv")
        df_feat.to_csv(out_path, index=False)
        print(f"Saved engineered features to {out_path}")
    else:
        print("synthetic_logs.csv not found.")
