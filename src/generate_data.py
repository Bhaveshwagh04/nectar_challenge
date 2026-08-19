"""
generate_data.py

The challenge doesn't ship an actual dataset, so this script builds a synthetic
but realistic one to develop and test the pipeline against. Assumptions used
to build the generator are documented in the README.

Run this once before anything else:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SITES = ["SITE_A", "SITE_B", "SITE_C"]
BUILDINGS_PER_SITE = 2
ASSET_TYPES = ["Chiller", "AHU", "Pump", "EnergyMeter", "EnvSensor"]
MANUFACTURERS = ["Carrier", "Trane", "Daikin", "Johnson Controls", "Siemens"]

# how many of each asset type live under a building
ASSET_COUNT_PER_BUILDING = {
    "Chiller": 1,
    "AHU": 3,
    "Pump": 2,
    "EnergyMeter": 1,
    "EnvSensor": 4,
}

DAYS_OF_HISTORY = 60
FREQ_MINUTES = 15


def build_asset_metadata():
    rows = []
    asset_counter = 1
    site_building_assets = {}  # (site, building) -> list of asset_ids by type

    for site in SITES:
        for b in range(1, BUILDINGS_PER_SITE + 1):
            building_id = f"{site}_B{b}"
            site_building_assets[building_id] = {}

            for atype, count in ASSET_COUNT_PER_BUILDING.items():
                ids_this_type = []
                for i in range(count):
                    asset_id = f"AST_{asset_counter:04d}"
                    asset_counter += 1

                    install_date = pd.Timestamp("2018-01-01") + pd.Timedelta(
                        days=int(RNG.integers(0, 2200))
                    )
                    capacity = {
                        "Chiller": RNG.uniform(200, 800),   # tons
                        "AHU": RNG.uniform(5000, 20000),    # CFM
                        "Pump": RNG.uniform(20, 100),       # HP
                        "EnergyMeter": np.nan,
                        "EnvSensor": np.nan,
                    }[atype]

                    rows.append({
                        "asset_id": asset_id,
                        "site_id": site,
                        "building_id": building_id,
                        "asset_name": f"{atype}-{building_id}-{i+1}",
                        "asset_type": atype,
                        "manufacturer": RNG.choice(MANUFACTURERS),
                        "installation_date": install_date.date().isoformat(),
                        "capacity": round(capacity, 1) if not np.isnan(capacity) else np.nan,
                        "parent_asset_id": None,  # filled in below once hierarchy is known
                    })
                    ids_this_type.append(asset_id)
                site_building_assets[building_id][atype] = ids_this_type

    df = pd.DataFrame(rows)

    # wire up a simple parent hierarchy: Chiller is parent of AHUs and Pumps,
    # AHUs are parent of a couple of env sensors, pumps parent the energy meter.
    for building_id, by_type in site_building_assets.items():
        chiller = by_type["Chiller"][0] if by_type["Chiller"] else None
        for ahu in by_type.get("AHU", []):
            df.loc[df.asset_id == ahu, "parent_asset_id"] = chiller
        for pump in by_type.get("Pump", []):
            df.loc[df.asset_id == pump, "parent_asset_id"] = chiller
        for meter in by_type.get("EnergyMeter", []):
            parent = by_type["Pump"][0] if by_type.get("Pump") else chiller
            df.loc[df.asset_id == meter, "parent_asset_id"] = parent
        sensors = by_type.get("EnvSensor", [])
        ahus = by_type.get("AHU", [])
        for i, sensor in enumerate(sensors):
            parent = ahus[i % len(ahus)] if ahus else chiller
            df.loc[df.asset_id == sensor, "parent_asset_id"] = parent

    # deliberately leave a couple of orphan / bad records in, real data is never clean
    if len(df) > 5:
        df.loc[df.sample(2, random_state=1).index, "parent_asset_id"] = "AST_9999"  # invalid parent
        df.loc[df.sample(1, random_state=2).index, "manufacturer"] = None

    return df, site_building_assets


def build_connectivity(asset_meta):
    rows = []
    for _, row in asset_meta.iterrows():
        if pd.isna(row.parent_asset_id) or row.parent_asset_id is None:
            continue
        conn_type = {
            "Chiller": "Supplies",
            "AHU": "Monitors",
            "Pump": "Supplies",
            "EnergyMeter": "Monitors",
            "EnvSensor": "Monitors",
        }[row.asset_type]
        rows.append({
            "source_asset_id": row.parent_asset_id,
            "target_asset_id": row.asset_id,
            "connection_type": conn_type,
            "relationship_strength": round(RNG.uniform(0.5, 1.0), 2),
        })

    df = pd.DataFrame(rows)
    # inject a duplicate edge and drop a couple of valid ones -> data quality gaps to detect later
    if len(df) > 3:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def simulate_telemetry(asset_meta):
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("h"),
        periods=int(DAYS_OF_HISTORY * 24 * 60 / FREQ_MINUTES),
        freq=f"{FREQ_MINUTES}min",
    )

    all_rows = []

    for _, asset in asset_meta.iterrows():
        atype = asset.asset_type
        base_temp = {"Chiller": 7, "AHU": 18, "Pump": 25, "EnergyMeter": 22, "EnvSensor": 23}[atype]
        base_power = {"Chiller": 120, "AHU": 15, "Pump": 8, "EnergyMeter": 0, "EnvSensor": 0.1}[atype]
        base_vibr = {"Chiller": 2.5, "AHU": 1.8, "Pump": 3.0, "EnergyMeter": 0.1, "EnvSensor": 0.05}[atype]

        # each asset gets its own slow degradation trend, and a handful get a
        # simulated failure event somewhere in the window
        degrade_rate = RNG.uniform(0, 0.0004)
        will_fail = RNG.random() < 0.18
        fail_idx = RNG.integers(int(len(timestamps) * 0.3), len(timestamps)) if will_fail else None

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            dow = ts.dayofweek
            is_business_hours = 8 <= hour <= 19 and dow < 5

            occ = RNG.poisson(15) if is_business_hours else RNG.poisson(1)
            load_factor = 1.0 + 0.35 * np.sin((hour - 8) / 12 * np.pi) if is_business_hours else 0.35

            deg = degrade_rate * i  # slow drift upward as bearings/filters wear

            approaching_fail = fail_idx is not None and 0 <= (fail_idx - i) <= 96  # last 24h before failure
            fault_flag = 0
            spike = 0.0
            if approaching_fail:
                closeness = 1 - (fail_idx - i) / 96
                spike = closeness * RNG.uniform(1.5, 3.0)
            if fail_idx is not None and i == fail_idx:
                fault_flag = 1

            temp = base_temp + deg * 5 + spike * 2 + RNG.normal(0, 0.6)
            humidity = np.clip(45 + 10 * np.sin(i / 500) + RNG.normal(0, 3), 10, 95)
            pressure = 101.3 + RNG.normal(0, 0.4) - deg * 2
            vibration = max(0, base_vibr + deg * 8 + spike * 1.5 + RNG.normal(0, 0.15))
            power = max(0, base_power * load_factor * (1 + deg * 3) + spike * base_power * 0.4 + RNG.normal(0, base_power * 0.05 + 0.01))

            mode = "Cooling" if is_business_hours else ("Idle" if RNG.random() < 0.6 else "Heating")

            # sprinkle in missing values, sensors drop out sometimes
            row = {
                "timestamp": ts,
                "site_id": asset.site_id,
                "building_id": asset.building_id,
                "asset_id": asset.asset_id,
                "temperature": round(temp, 2),
                "humidity": round(humidity, 2),
                "pressure": round(pressure, 2),
                "vibration": round(vibration, 3),
                "power_consumption": round(power, 3),
                "occupancy_count": int(occ),
                "operating_mode": mode,
                "fault_flag": fault_flag,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # missingness (MCAR-ish, ~1.5% per numeric sensor column) to force real cleaning work
    for col in ["temperature", "humidity", "pressure", "vibration", "power_consumption"]:
        mask = RNG.random(len(df)) < 0.015
        df.loc[mask, col] = np.nan

    # a handful of clearly bad sensor readings (stuck/spiky) for anomaly detection to catch
    n_bad = int(len(df) * 0.002)
    bad_idx = RNG.choice(df.index, size=n_bad, replace=False)
    df.loc[bad_idx, "power_consumption"] = df.loc[bad_idx, "power_consumption"] * RNG.uniform(4, 8)
    df.loc[bad_idx, "vibration"] = df.loc[bad_idx, "vibration"] * RNG.uniform(3, 6)

    return df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)


def main():
    print("Building asset metadata...")
    asset_meta, _ = build_asset_metadata()
    asset_meta.to_csv(RAW_DIR / "asset_metadata.csv", index=False)

    print("Building connectivity table...")
    connectivity = build_connectivity(asset_meta)
    connectivity.to_csv(RAW_DIR / "asset_connectivity.csv", index=False)

    print("Simulating telemetry (this is the slow part)...")
    telemetry = simulate_telemetry(asset_meta)
    telemetry.to_csv(RAW_DIR / "sensor_telemetry.csv", index=False)

    print(f"assets: {len(asset_meta)}, connections: {len(connectivity)}, telemetry rows: {len(telemetry)}")
    print(f"fault events in telemetry: {telemetry.fault_flag.sum()}")
    print("done, files written to", RAW_DIR)


if __name__ == "__main__":
    main()
