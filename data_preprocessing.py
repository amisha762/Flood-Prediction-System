"""
Data Preprocessing and Validation Pipeline for Flood Prediction System.
Handles data cleaning, station-specific hydrological normalization, timestamp parsing,
outlier clipping, and time-based / location-aware train-test splits.
"""

import numpy as np
import pandas as pd


# Expected core raw columns and physical sanity bounds
PHYSICAL_BOUNDS = {
    "water_level_m": (0.0, 600.0),
    "discharge_cumec": (0.0, 150000.0),
    "rainfall_intensity_current_mm": (0.0, 600.0),
    "rainfall_cum_24h_mm": (0.0, 600.0),
    "rainfall_cum_48h_mm": (0.0, 900.0),
    "rainfall_cum_72h_mm": (0.0, 1200.0),
    "forecast_rainfall_next24h_mm": (0.0, 600.0),
    "soil_moisture_index": (0.0, 1.0),
    "soil_saturation_pct": (0.0, 100.0),
    "upstream_water_level_ratio": (0.1, 2.5),
    "upstream_discharge_cumec": (0.0, 200000.0),
    "dam_release_cumec": (0.0, 100000.0),
    "catchment_elevation_m": (5.0, 1500.0),
    "catchment_slope_deg": (0.05, 30.0),
    "distance_to_river_km": (0.01, 50.0),
}


def validate_and_clean_data(df, verbose=False):
    """
    Validate data types, remove duplicates, handle missing values, and check physical bounds.
    """
    initial_rows = len(df)
    clean_df = df.copy()

    # 1. Parse dates if present
    if "date" in clean_df.columns:
        clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
        clean_df = clean_df.dropna(subset=["date"])

    # 2. Remove identical duplicate rows
    clean_df = clean_df.drop_duplicates()
    duplicates_removed = initial_rows - len(clean_df)

    # 3. Handle missing values
    # For numerical columns, fill missing using forward-fill within station, then median
    num_cols = clean_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if clean_df[col].isnull().sum() > 0:
            if "station" in clean_df.columns:
                clean_df[col] = clean_df.groupby("station")[col].transform(lambda s: s.ffill().bfill())
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    # 4. Outlier verification and soft clipping within physical feasibility bounds
    for col, (min_val, max_val) in PHYSICAL_BOUNDS.items():
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].clip(lower=min_val, upper=max_val)

    if verbose:
        print(f"Validation complete: {len(clean_df)} records retained. Duplicates dropped: {duplicates_removed}.")

    return clean_df


def normalize_hydrological_features(df):
    """
    Crucial Domain Step:
    Normalize water level and discharge relative to station-specific Danger and Warning Levels.
    Without this, an algorithm cannot generalize across different rivers (e.g. Ganga at 50m vs Sutlej at 235m).
    """
    df = df.copy()

    # Stage relative to Danger Level (ratio > 1.0 means above danger level)
    if "water_level_m" in df.columns and "danger_level_m" in df.columns:
        df["water_level_ratio_danger"] = (df["water_level_m"] / df["danger_level_m"]).round(4)
        df["water_level_danger_margin_m"] = (df["danger_level_m"] - df["water_level_m"]).round(3)

    # Stage relative to Warning Level
    if "water_level_m" in df.columns and "warning_level_m" in df.columns:
        df["water_level_ratio_warning"] = (df["water_level_m"] / df["warning_level_m"]).round(4)
        df["is_above_warning_level"] = (df["water_level_m"] >= df["warning_level_m"]).astype(int)

    # Stage buffer relative to High Flood Level (HFL)
    if "water_level_m" in df.columns and "high_flood_level_m" in df.columns:
        df["hfl_margin_m"] = (df["high_flood_level_m"] - df["water_level_m"]).round(3)

    # Discharge relative to safe channel discharge capacity
    if "discharge_cumec" in df.columns and "safe_discharge_cumec" in df.columns:
        safe_q = df["safe_discharge_cumec"].replace(0, 1000.0)
        df["discharge_ratio_danger"] = (df["discharge_cumec"] / safe_q).round(4)

    return df


def split_data_by_time(df, split_date="2023-01-01"):
    """
    Strict Time-Based Split:
    Train on historical period (e.g. 2018-01-01 to 2022-12-31),
    Evaluate on prospective future period (2023-01-01 to 2024-12-31).
    This strictly prevents look-ahead data leakage in time-series forecasting.
    """
    df["date"] = pd.to_datetime(df["date"])
    split_ts = pd.to_datetime(split_date)

    train_df = df[df["date"] < split_ts].copy()
    test_df = df[df["date"] >= split_ts].copy()

    return train_df, test_df
