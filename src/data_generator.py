import os
import random
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for folder in ["data", "models", "outputs", "reports", "diagrams", "screenshots", "docs"]:
    os.makedirs(os.path.join(ROOT_DIR, folder), exist_ok=True)

GEO_LOCATIONS = {
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Tokyo": (35.6762, 139.6503),
    "Mumbai": (19.0760, 72.8777),
    "Sydney": (-33.8688, 151.2093),
    "Berlin": (52.5200, 13.4050),
    "Paris": (48.8566, 2.3522),
    "San Francisco": (37.7749, -122.4194),
    "Singapore": (1.3521, 103.8198),
    "Toronto": (43.6532, -79.3832)
}

RESOURCE_CATALOG = {
    "Public_API": "public",
    "Company_Wiki": "internal",
    "HR_Portal": "internal",
    "CRM_System": "internal",
    "Dev_Database": "restricted",
    "SSH_Bastion": "restricted",
    "Analytics_Lake": "restricted",
    "Payroll_DB": "critical",
    "Prod_Master_DB": "critical",
    "CEO_Vault": "critical",
    "Financial_Master": "critical"
}

AUTH_METHODS = ["password", "token", "certificate", "biometric"]

COMMAND_TEMPLATES = {
    "public": ["HTTP_GET /api/v1/health", "HTTP_GET /api/v1/catalog", "HTTP_GET /docs"],
    "internal": ["LOGIN", "READ_DOC", "UPDATE_PROFILE", "SEARCH_DIRECTORY"],
    "restricted": ["SSH_CONNECT", "QUERY_TABLE", "RUN_BUILD_SCRIPT", "GIT_PULL"],
    "critical": ["ADMIN_LOGIN", "EXPORT_DATABASE", "SELECT_ALL_CREDENTIALS", "SUDO_ELEVATE", "MODIFY_ROLES"]
}

def generate_mac_hash(name):
    return hashlib.md5(name.encode()).hexdigest()[:8]

def generate_synthetic_data(num_events=50000, days=30, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    print(f"Generating synthetic logs ({num_events} target events over {days} days)...")

    users = [f"user_{i:03d}" for i in range(1, 1001)]
    service_accounts = [f"sa_{i:03d}" for i in range(1, 101)]
    devices = [f"dev_{i:03d}" for i in range(1, 201)]

    entity_profiles = {}
    
    for u in users:
        city = random.choice(list(GEO_LOCATIONS.keys()))
        lat, lon = GEO_LOCATIONS[city]
        primary_os = random.choice(["Win11-x64", "macOS-ARM64", "Ubuntu-Linux"])
        fp = f"{primary_os}-{generate_mac_hash(u)}"
        entity_profiles[u] = {
            "entity_type": "user",
            "city": city,
            "lat": lat,
            "lon": lon,
            "ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "preferred_hours": (8, 18),
            "fav_resources": random.sample(["Company_Wiki", "HR_Portal", "CRM_System", "Public_API"], k=2),
            "auth_method": random.choice(["password", "biometric", "token"]),
            "device_fingerprint": fp,
            "avg_duration": random.randint(300, 1800)
        }

    for sa in service_accounts:
        city = random.choice(["New York", "London", "San Francisco"])
        lat, lon = GEO_LOCATIONS[city]
        fp = f"Linux-Server-{generate_mac_hash(sa)}"
        entity_profiles[sa] = {
            "entity_type": "service_account",
            "city": city,
            "lat": lat,
            "lon": lon,
            "ip": f"10.0.{random.randint(1, 50)}.{random.randint(1, 255)}",
            "preferred_hours": (0, 24),
            "fav_resources": random.sample(["Dev_Database", "Analytics_Lake", "SSH_Bastion"], k=2),
            "auth_method": random.choice(["token", "certificate"]),
            "device_fingerprint": fp,
            "avg_duration": random.randint(60, 600)
        }

    for dev in devices:
        city = random.choice(list(GEO_LOCATIONS.keys()))
        lat, lon = GEO_LOCATIONS[city]
        fp = f"IoT-Firmware-{generate_mac_hash(dev)}"
        entity_profiles[dev] = {
            "entity_type": "edge_device",
            "city": city,
            "lat": lat,
            "lon": lon,
            "ip": f"172.16.{random.randint(1, 100)}.{random.randint(1, 255)}",
            "preferred_hours": (0, 24),
            "fav_resources": ["Public_API"],
            "auth_method": "certificate",
            "device_fingerprint": fp,
            "avg_duration": random.randint(10, 120)
        }

    all_entities = users + service_accounts + devices
    start_time = datetime.now() - timedelta(days=days)

    records = []
    normal_count = int(num_events * 0.98)

    for _ in range(normal_count):
        entity = random.choice(all_entities)
        profile = entity_profiles[entity]

        day_offset = random.uniform(0, days)
        if profile["entity_type"] == "user" and random.random() > 0.08:
            hour = random.randint(profile["preferred_hours"][0], profile["preferred_hours"][1] - 1)
        else:
            hour = random.randint(0, 23)

        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = start_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        if random.random() < 0.05:
            res = random.choice(list(RESOURCE_CATALOG.keys()))
        else:
            res = random.choice(profile["fav_resources"])

        sens = RESOURCE_CATALOG[res]
        auth_st = "failure" if random.random() < 0.02 else "success"
        dur = max(10, int(np.random.normal(profile["avg_duration"], profile["avg_duration"] * 0.2)))
        cmds = random.sample(COMMAND_TEMPLATES[sens], k=random.randint(1, len(COMMAND_TEMPLATES[sens])))

        records.append({
            "entity_id": entity,
            "entity_type": profile["entity_type"],
            "timestamp": ts,
            "source_ip": profile["ip"],
            "geo_location": profile["city"],
            "lat": profile["lat"],
            "lon": profile["lon"],
            "resource_accessed": res,
            "resource_sensitivity": sens,
            "auth_method": profile["auth_method"],
            "auth_status": auth_st,
            "session_duration": dur,
            "command_sequence": "; ".join(cmds),
            "device_fingerprint": profile["device_fingerprint"],
            "label": "normal"
        })

    attack_types = [
        "brute_force", "impossible_travel", "credential_stuffing",
        "lateral_movement", "device_spoofing", "low_and_slow_exfiltration", "insider_drift"
    ]
    target_attacks = num_events - normal_count
    per_attack_target = target_attacks // len(attack_types)

    print(f"Injecting ~{target_attacks} attack events across 7 threat categories...")

    # Attack 1: Brute Force
    for _ in range(per_attack_target // 20):
        victim = random.choice(users)
        attacker_ip = f"45.142.{random.randint(1, 255)}.{random.randint(1, 255)}"
        base_ts = start_time + timedelta(days=random.uniform(1, days - 1))
        for i in range(20):
            ts = base_ts + timedelta(seconds=i * 4)
            records.append({
                "entity_id": victim,
                "entity_type": "user",
                "timestamp": ts,
                "source_ip": attacker_ip,
                "geo_location": "Moscow",
                "lat": 55.7558,
                "lon": 37.6173,
                "resource_accessed": "HR_Portal",
                "resource_sensitivity": "internal",
                "auth_method": "password",
                "auth_status": "failure" if i < 19 else "success",
                "session_duration": 5,
                "command_sequence": "AUTH_ATTEMPT_FAILED",
                "device_fingerprint": f"Unknown-Botnet-{random.randint(100, 999)}",
                "label": "brute_force"
            })

    # Attack 2: Impossible Travel
    for _ in range(per_attack_target // 2):
        victim = random.choice(users)
        profile = entity_profiles[victim]
        base_ts = start_time + timedelta(days=random.uniform(1, days - 1))

        records.append({
            "entity_id": victim,
            "entity_type": "user",
            "timestamp": base_ts,
            "source_ip": profile["ip"],
            "geo_location": profile["city"],
            "lat": profile["lat"],
            "lon": profile["lon"],
            "resource_accessed": profile["fav_resources"][0],
            "resource_sensitivity": RESOURCE_CATALOG[profile["fav_resources"][0]],
            "auth_method": profile["auth_method"],
            "auth_status": "success",
            "session_duration": 300,
            "command_sequence": "LOGIN; VIEW_DASHBOARD",
            "device_fingerprint": profile["device_fingerprint"],
            "label": "normal"
        })

        distant_city = "Tokyo" if profile["city"] != "Tokyo" else "Sydney"
        d_lat, d_lon = GEO_LOCATIONS[distant_city]
        records.append({
            "entity_id": victim,
            "entity_type": "user",
            "timestamp": base_ts + timedelta(minutes=6),
            "source_ip": f"103.22.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "geo_location": distant_city,
            "lat": d_lat,
            "lon": d_lon,
            "resource_accessed": "Payroll_DB",
            "resource_sensitivity": "critical",
            "auth_method": "password",
            "auth_status": "success",
            "session_duration": 1200,
            "command_sequence": "QUERY_TABLE; EXPORT_ALL",
            "device_fingerprint": f"Spoofed-MacBook-{random.randint(10, 99)}",
            "label": "impossible_travel"
        })

    # Attack 3: Credential Stuffing
    for _ in range(per_attack_target // 25):
        attacker_ip = f"185.220.{random.randint(1, 255)}.{random.randint(1, 255)}"
        base_ts = start_time + timedelta(days=random.uniform(1, days - 1))
        target_users = random.sample(users, k=25)
        for i, u in enumerate(target_users):
            ts = base_ts + timedelta(seconds=i * 3)
            records.append({
                "entity_id": u,
                "entity_type": "user",
                "timestamp": ts,
                "source_ip": attacker_ip,
                "geo_location": "Berlin",
                "lat": 52.5200,
                "lon": 13.4050,
                "resource_accessed": "Company_Wiki",
                "resource_sensitivity": "internal",
                "auth_method": "password",
                "auth_status": "failure",
                "session_duration": 2,
                "command_sequence": "CREDENTIAL_CHECK_FAIL",
                "device_fingerprint": "Automated-Python-Script",
                "label": "credential_stuffing"
            })

    # Attack 4: Lateral Movement
    for _ in range(per_attack_target):
        insider = random.choice(users)
        profile = entity_profiles[insider]
        base_ts = start_time + timedelta(days=random.uniform(1, days - 1))
        high_val_resources = ["SSH_Bastion", "Dev_Database", "Prod_Master_DB", "CEO_Vault"]
        res = random.choice(high_val_resources)
        records.append({
            "entity_id": insider,
            "entity_type": "user",
            "timestamp": base_ts,
            "source_ip": profile["ip"],
            "geo_location": profile["city"],
            "lat": profile["lat"],
            "lon": profile["lon"],
            "resource_accessed": res,
            "resource_sensitivity": RESOURCE_CATALOG[res],
            "auth_method": "password",
            "auth_status": "success",
            "session_duration": 1800,
            "command_sequence": "SSH_CONNECT; SUDO_ELEVATE; SCAN_PORT",
            "device_fingerprint": profile["device_fingerprint"],
            "label": "lateral_movement"
        })

    # Attack 5: Device Spoofing
    for _ in range(per_attack_target):
        victim = random.choice(users)
        profile = entity_profiles[victim]
        base_ts = start_time + timedelta(days=random.uniform(1, days - 1))
        records.append({
            "entity_id": victim,
            "entity_type": "user",
            "timestamp": base_ts,
            "source_ip": profile["ip"],
            "geo_location": profile["city"],
            "lat": profile["lat"],
            "lon": profile["lon"],
            "resource_accessed": profile["fav_resources"][0],
            "resource_sensitivity": RESOURCE_CATALOG[profile["fav_resources"][0]],
            "auth_method": "password",
            "auth_status": "success",
            "session_duration": 600,
            "command_sequence": "LOGIN; QUERY",
            "device_fingerprint": f"Android-Mobile-FakeHash-{random.randint(100, 999)}",
            "label": "device_spoofing"
        })

    # Attack 6: Low and Slow Exfiltration
    for _ in range(per_attack_target):
        suspect = random.choice(users)
        profile = entity_profiles[suspect]
        base_ts = start_time + timedelta(days=random.uniform(1, days - 1))
        off_hour_ts = base_ts.replace(hour=3, minute=random.randint(10, 45))
        records.append({
            "entity_id": suspect,
            "entity_type": "user",
            "timestamp": off_hour_ts,
            "source_ip": profile["ip"],
            "geo_location": profile["city"],
            "lat": profile["lat"],
            "lon": profile["lon"],
            "resource_accessed": "Financial_Master",
            "resource_sensitivity": "critical",
            "auth_method": "password",
            "auth_status": "success",
            "session_duration": 14400,
            "command_sequence": "SELECT_ALL_CREDENTIALS; DUMP_DB_SLICES",
            "device_fingerprint": profile["device_fingerprint"],
            "label": "low_and_slow_exfiltration"
        })

    # Attack 7: Insider Drift
    for _ in range(per_attack_target):
        user_drift = random.choice(users)
        profile = entity_profiles[user_drift]
        base_ts = start_time + timedelta(days=random.uniform(15, days - 1))
        records.append({
            "entity_id": user_drift,
            "entity_type": "user",
            "timestamp": base_ts,
            "source_ip": profile["ip"],
            "geo_location": profile["city"],
            "lat": profile["lat"],
            "lon": profile["lon"],
            "resource_accessed": "CEO_Vault",
            "resource_sensitivity": "critical",
            "auth_method": "password",
            "auth_status": "success",
            "session_duration": 2400,
            "command_sequence": "ADMIN_LOGIN; READ_CONFIDENTIAL_FILES",
            "device_fingerprint": profile["device_fingerprint"],
            "label": "insider_drift"
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    output_path = os.path.join(ROOT_DIR, "data", "synthetic_logs.csv")
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved successfully to '{output_path}'. Total rows: {len(df)}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
