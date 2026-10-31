"""
Feature Engineering Pipeline for Flood Prediction System.
Generates hydrological interaction indices, antecedent precipitation index (API),
soil saturation runoff multipliers, and upstream hydraulic surge pressure metrics.
"""

import numpy as np
import pandas as pd
from data_preprocessing import normalize_hydrological_features


# Canonical ordered list of features used in ML models
FEATURE_COLUMNS = [
    "water_level_ratio_danger",
    "water_level_danger_margin_m",
    "is_above_warning_level",
    "water_level_change_24h",
    "discharge_ratio_danger",
    "discharge_change_24h",
    "rainfall_intensity_current_mm",
    "rainfall_cum_24h_mm",
    "rainfall_cum_48h_mm",
    "rainfall_cum_72h_mm",
    "forecast_rainfall_next24h_mm",
    "soil_moisture_index",
    "soil_saturation_pct",
    "runoff_potential_index",
    "upstream_water_level_ratio",
    "upstream_hydraulic_pressure",
    "dam_release_ratio",
    "rapid_rise_indicator",
    "terrain_drainage_speed",
    "catchment_elevation_m",
    "catchment_slope_deg",
    "distance_to_river_km",
    "historical_flood_count_5yr",
    "monsoon_season",
    "month"
]


def extract_engineered_features(df):
    """
    Compute domain-specific hydrological and meteorological indicators.
    Works for both batch DataFrames and single live prediction dictionaries.
    """
    df = df.copy()

    # Ensure baseline station-normalized features are present
    df = normalize_hydrological_features(df)

    # 1. Runoff Potential Index (RPI):
    # Interaction between incoming/recent rain and soil saturation.
    # When soil is 90% saturated, 100mm rain produces drastically more runoff than in dry soil.
    forecast_rain = df.get("forecast_rainfall_next24h_mm", 0.0)
    cum_24h_rain = df.get("rainfall_cum_24h_mm", 0.0)
    saturation = df.get("soil_saturation_pct", 50.0)
    df["runoff_potential_index"] = ((forecast_rain + cum_24h_rain) * (saturation / 100.0)).round(2)

    # 2. Upstream Hydraulic Pressure:
    # Interaction between upstream stage ratio and incoming upstream discharge relative to capacity
    upstream_stage = df.get("upstream_water_level_ratio", 0.8)
    upstream_q = df.get("upstream_discharge_cumec", 1000.0)
    safe_q = df.get("safe_discharge_cumec", 5000.0).replace(0, 5000.0)
    df["upstream_hydraulic_pressure"] = (upstream_stage * (upstream_q / safe_q)).round(3)

    # 3. Dam Release Ratio:
    # Controlled barrage or dam release relative to safe carrying capacity
    dam_q = df.get("dam_release_cumec", 0.0)
    df["dam_release_ratio"] = (dam_q / safe_q).round(3)

    # 4. Rapid Rise Indicator:
    # Flash surge detection: stage rising faster than 0.35m in 24h
    stage_rate = df.get("water_level_change_24h", 0.0)
    q_rate = df.get("discharge_change_24h", 0.0)
    df["rapid_rise_indicator"] = (
        (stage_rate > 0.35) | (q_rate > (0.12 * safe_q))
    ).astype(int)

    # 5. Terrain Drainage Speed:
    # Steeper slopes closer to the river channel drain water into riverbanks much faster
    slope = df.get("catchment_slope_deg", 1.0)
    dist = df.get("distance_to_river_km", 0.5).clip(lower=0.05)
    df["terrain_drainage_speed"] = (slope / dist).round(2)

    # 6. Default fallbacks if seasonal features are missing
    if "month" not in df.columns:
        df["month"] = 7
    if "monsoon_season" not in df.columns:
        df["monsoon_season"] = df["month"].apply(lambda m: 1 if m in [6, 7, 8, 9] else 0)

    # Ensure all required features are populated
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURE_COLUMNS]


def get_feature_names():
    """Return ordered list of feature column names."""
    return list(FEATURE_COLUMNS)
