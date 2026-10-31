"""
Generate Realistic Hydrological and Meteorological Dataset for Flood Prediction.
Covers Punjab, Haryana, Madhya Pradesh, and Bihar (2018 - 2024) across 17 CWC stations.
Incorporates physical runoff physics, soil saturation dynamics, upstream routing, and
historical flood anomalies (e.g. July 2023 North India Floods, Aug 2020 Narmada Floods,
2019/2021 Bihar Kosi/Ganga Floods).
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path so we can import utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from utils.stations import get_all_stations_flat


def generate_realistic_dataset(output_path=None, seed=42):
    """
    Generate synthetic hydrometeorological time-series dataset with physically grounded dynamics.
    """
    np.random.seed(seed)
    stations = get_all_stations_flat()
    
    # 7-year daily date range: 2018-01-01 to 2024-12-31
    date_range = pd.date_range(start="2018-01-01", end="2024-12-31", freq="D")
    records = []

    print(f"Generating realistic dataset for {len(stations)} stations over {len(date_range)} days...")

    for st in stations:
        state = st["state"]
        district = st["district"]
        station_name = st["station"]
        river = st["river"]
        wl = st["warning_level_m"]
        dl = st["danger_level_m"]
        hfl = st["high_flood_level_m"]
        safe_q = st["safe_discharge_cumec"]
        elev = st["catchment_elevation_m"]
        slope = st["catchment_slope_deg"]
        dist = st["distance_to_river_km"]
        hist_floods = st["historical_flood_count_5yr"]

        # Base dry-season river stage (usually 2-5m below warning level)
        dry_season_stage = wl - np.random.uniform(2.5, 4.5)
        # Base dry-season discharge (10-25% of safe capacity)
        dry_season_q = safe_q * np.random.uniform(0.10, 0.22)

        # State-level continuous variables tracking
        current_stage = dry_season_stage
        current_q = dry_season_q
        soil_moisture = 0.20  # initial dry season moisture (0.0 to 1.0)

        # Temporary series to hold sequential daily data for this station
        station_days = []

        for current_date in date_range:
            month = current_date.month
            day_of_year = current_date.dayofyear
            is_monsoon = 1 if month in [6, 7, 8, 9] else 0

            # ---------------- Rainfall Simulation ----------------
            # Monsoon has higher rain probability and intensity
            if is_monsoon:
                # Heavy monsoon spell chance
                is_storm = np.random.rand() < 0.28
                if is_storm:
                    rain_current = np.random.exponential(scale=35.0) + np.random.uniform(10.0, 45.0)
                else:
                    rain_current = np.random.exponential(scale=6.0) if np.random.rand() < 0.45 else 0.0
            else:
                # Non-monsoon (winter rain or dry spells)
                if month in [1, 2]:  # Western disturbances in North India
                    rain_current = np.random.exponential(scale=8.0) if np.random.rand() < 0.12 else 0.0
                else:
                    rain_current = np.random.exponential(scale=2.0) if np.random.rand() < 0.05 else 0.0

            # Historical Extreme Events Injection:
            # 1. July 8-15, 2023: Massive Punjab and Haryana floods (Yamuna, Sutlej, Ghaggar)
            if state in ["Punjab", "Haryana"] and current_date.year == 2023 and month == 7 and 8 <= current_date.day <= 14:
                rain_current += np.random.uniform(85.0, 180.0)

            # 2. August 28 - September 3, 2020: Massive Narmada & Chambal flood in MP
            if state == "Madhya Pradesh" and current_date.year == 2020 and month == 8 and 27 <= current_date.day <= 31:
                rain_current += np.random.uniform(90.0, 210.0)
            if state == "Madhya Pradesh" and current_date.year == 2022 and month == 8 and 18 <= current_date.day <= 25:
                rain_current += np.random.uniform(75.0, 160.0)

            # 3. Bihar Kosi & Ganga floods (July 2019, August 2021)
            if state == "Bihar" and current_date.year == 2019 and month == 7 and 12 <= current_date.day <= 20:
                rain_current += np.random.uniform(95.0, 220.0)
            if state == "Bihar" and current_date.year == 2021 and month == 8 and 10 <= current_date.day <= 18:
                rain_current += np.random.uniform(80.0, 175.0)

            rain_current = round(float(rain_current), 1)

            # ---------------- Soil Moisture & Runoff ----------------
            # Soil moisture increases with rain and decays through evapotranspiration/percolation
            evaporation = 0.015 if is_monsoon else 0.025
            soil_moisture = soil_moisture * (1.0 - evaporation) + (rain_current / 250.0)
            soil_moisture = float(np.clip(soil_moisture, 0.05, 0.98))
            soil_sat_pct = round(soil_moisture * 100.0, 1)

            # Non-linear runoff generation
            # When soil is dry (<40%), very little runoff. When >80%, runoff coefficient reaches 0.7-0.9
            runoff_coeff = 0.05 + 0.85 * (soil_moisture ** 3.0)
            direct_runoff_mm = rain_current * runoff_coeff

            # ---------------- Hydrological Dynamics & Dam Releases ----------------
            # Upstream surge & Dam releases
            upstream_pulse = 0.0
            dam_release = 0.0

            if is_monsoon and direct_runoff_mm > 15.0:
                upstream_pulse = np.random.uniform(0.15, 0.45) * (direct_runoff_mm / 15.0)
                if st["dam_reservoir"] != "None":
                    # Barrages and dams open gates during heavy inflow
                    dam_release = safe_q * np.random.uniform(0.3, 0.85)

            # Stage relaxation towards baseflow vs surge response
            stage_relaxation = 0.15  # how fast flood water drains
            stage_surge = (direct_runoff_mm / 35.0) * (1.0 + (slope / 2.0)) + (upstream_pulse * 0.8)

            # Historical extreme event surges
            if state in ["Punjab", "Haryana"] and current_date.year == 2023 and month == 7 and 9 <= current_date.day <= 14:
                stage_surge += np.random.uniform(1.8, 3.5)
                dam_release = safe_q * 1.25  # Hathnikund / Bhakra release spike
            if state == "Madhya Pradesh" and current_date.year == 2020 and month == 8 and 28 <= current_date.day <= 31:
                stage_surge += np.random.uniform(2.5, 4.2)
                dam_release = safe_q * 1.35  # Bargi/Tawa massive gate opening
            if state == "Bihar" and current_date.year in [2019, 2021] and month in [7, 8] and rain_current > 60:
                stage_surge += np.random.uniform(1.5, 3.2)
                dam_release = safe_q * 0.95

            # Update stage and discharge
            target_stage = dry_season_stage + (stage_surge * 1.5)
            stage_diff = target_stage - current_stage
            new_stage = current_stage + stage_diff * 0.45
            # Natural boundaries
            min_possible_stage = wl - 5.0
            max_possible_stage = hfl + 1.5
            new_stage = np.clip(new_stage, min_possible_stage, max_possible_stage)

            # Discharge calculated from stage rating curve approximation: Q = a * (H - H0)^b
            stage_above_min = max(0.2, new_stage - min_possible_stage)
            norm_depth = stage_above_min / (dl - min_possible_stage)
            new_q = safe_q * (norm_depth ** 1.85) + dam_release * 0.6
            new_q = round(float(np.clip(new_q, safe_q * 0.05, safe_q * 2.2)), 1)

            # Upstream normalized stage
            upstream_ratio = round(float(np.clip(
                (new_stage / dl) + np.random.normal(0, 0.03) + (upstream_pulse * 0.1),
                0.4, 1.45
            )), 3)
            upstream_q = round(float(np.clip(new_q * np.random.uniform(0.85, 1.2), safe_q * 0.05, safe_q * 2.5)), 1)

            # Forecast rainfall for the next 24 hours (realistic forecast skill: correlated with actual next day)
            # We'll calculate this accurately after all days are generated
            station_days.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "state": state,
                "district": district,
                "station": station_name,
                "river": river,
                "catchment_elevation_m": elev,
                "catchment_slope_deg": slope,
                "distance_to_river_km": dist,
                "historical_flood_count_5yr": hist_floods,
                "warning_level_m": wl,
                "danger_level_m": dl,
                "high_flood_level_m": hfl,
                "safe_discharge_cumec": safe_q,
                "rainfall_intensity_current_mm": rain_current,
                "soil_moisture_index": round(soil_moisture, 3),
                "soil_saturation_pct": soil_sat_pct,
                "water_level_m": round(float(new_stage), 2),
                "discharge_cumec": new_q,
                "upstream_water_level_ratio": upstream_ratio,
                "upstream_discharge_cumec": upstream_q,
                "dam_release_cumec": round(float(dam_release), 1),
                "month": month,
                "monsoon_season": is_monsoon
            })

            current_stage = new_stage
            current_q = new_q

        # Convert station days to DataFrame to compute time-lags and next-24h targets
        st_df = pd.DataFrame(station_days)

        # Rate of changes (24h)
        st_df["water_level_change_24h"] = (st_df["water_level_m"] - st_df["water_level_m"].shift(1)).fillna(0.0).round(2)
        st_df["discharge_change_24h"] = (st_df["discharge_cumec"] - st_df["discharge_cumec"].shift(1)).fillna(0.0).round(1)

        # Cumulative rainfall rolling sums (24h, 48h, 72h)
        st_df["rainfall_cum_24h_mm"] = st_df["rainfall_intensity_current_mm"].round(1)
        st_df["rainfall_cum_48h_mm"] = st_df["rainfall_intensity_current_mm"].rolling(window=2, min_periods=1).sum().round(1)
        st_df["rainfall_cum_72h_mm"] = st_df["rainfall_intensity_current_mm"].rolling(window=3, min_periods=1).sum().round(1)

        # Forecast rainfall next 24h = actual next day rain with realistic meteorological forecast noise
        next_day_rain = st_df["rainfall_intensity_current_mm"].shift(-1).fillna(0.0)
        forecast_noise = np.random.normal(0, 4.0, size=len(st_df))
        st_df["forecast_rainfall_next24h_mm"] = np.clip(next_day_rain + forecast_noise, 0.0, 300.0).round(1)

        # Target label: Did water level breach or exceed danger level in the next 24 hours?
        next_day_stage = st_df["water_level_m"].shift(-1).fillna(st_df["water_level_m"])
        st_df["next_24h_water_level_m"] = next_day_stage.round(2)
        
        # Binary target: 1 if next stage >= danger level DL, or if water is within 10cm of DL with fast rising trend
        st_df["flood_next_24h"] = (
            (next_day_stage >= dl) |
            ((next_day_stage >= (dl - 0.10)) & (st_df["water_level_change_24h"] > 0.40))
        ).astype(int)

        records.append(st_df)

    final_df = pd.concat(records, ignore_index=True)

    # Reorder columns cleanly
    cols_order = [
        "date", "state", "district", "station", "river",
        "warning_level_m", "danger_level_m", "high_flood_level_m", "safe_discharge_cumec",
        "catchment_elevation_m", "catchment_slope_deg", "distance_to_river_km", "historical_flood_count_5yr",
        "month", "monsoon_season",
        "rainfall_intensity_current_mm", "rainfall_cum_24h_mm", "rainfall_cum_48h_mm", "rainfall_cum_72h_mm",
        "forecast_rainfall_next24h_mm", "soil_moisture_index", "soil_saturation_pct",
        "water_level_m", "water_level_change_24h", "discharge_cumec", "discharge_change_24h",
        "upstream_water_level_ratio", "upstream_discharge_cumec", "dam_release_cumec",
        "next_24h_water_level_m", "flood_next_24h"
    ]
    final_df = final_df[cols_order]

    if output_path is None:
        output_path = os.path.join(CURRENT_DIR, "data", "historical_flood_dataset.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    total_samples = len(final_df)
    flood_events = final_df["flood_next_24h"].sum()
    flood_pct = (flood_events / total_samples) * 100.0

    print(f"Dataset successfully created and saved to: {output_path}")
    print(f"Total Records: {total_samples:,}")
    print(f"Total Flood Events: {flood_events:,} ({flood_pct:.2f}% event rate)")
    print(f"State breakdown of flood events:")
    print(final_df.groupby("state")["flood_next_24h"].agg(["count", "sum", "mean"]))
    return final_df


if __name__ == "__main__":
    generate_realistic_dataset()
