"""
Prediction Engine for AI-Based Flood Detection and Probability Prediction.
Takes raw station telemetry, performs validation, feature engineering,
calibrated probability estimation, confidence intervals, and XAI attribution.
Safely detects 'Insufficient Data' to avoid misleading predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd

from data_preprocessing import validate_and_clean_data
from feature_engineering import extract_engineered_features, get_feature_names
from utils.advisories import get_risk_assessment
from utils.stations import get_station_profile
from explainability import FloodExplainer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "flood_model.joblib")


# Critical features without which a physical prediction cannot be made
CRITICAL_HYDROLOGICAL_FIELDS = [
    "water_level_m",
    "danger_level_m",
    "safe_discharge_cumec",
    "rainfall_cum_24h_mm"
]


class FloodPredictor:
    """
    Production-ready inference engine for flood risk estimation.
    """
    def __init__(self, pipeline_path=PIPELINE_PATH):
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Model artifact not found at {pipeline_path}. Run train_model.py first.")
        
        self.pkg = joblib.load(pipeline_path)
        self.calibrated_model = self.pkg["calibrated_model"]
        self.base_model = self.pkg["base_model"]
        self.best_model_name = self.pkg["best_model_name"]
        self.rf_model = self.pkg.get("rf_model")
        self.hgb_model = self.pkg.get("hgb_model")
        self.lr_model = self.pkg.get("lr_model")
        self.scaler = self.pkg["scaler"]
        self.use_scaled = self.pkg["use_scaled"]
        self.feature_names = self.pkg["feature_names"]
        
        # Initialize explainer
        self.explainer = FloodExplainer(pipeline_path)

    def predict_single_station(self, observation_dict):
        """
        Predict flood probability for a single station observation dictionary.
        Returns complete diagnostic report, risk category, and SOP recommendations.
        """
        # 1. Data Freshness and Sufficiency Checks
        missing_fields = []
        for field in CRITICAL_HYDROLOGICAL_FIELDS:
            val = observation_dict.get(field)
            if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
                missing_fields.append(field)

        if missing_fields:
            return {
                "status": "INSUFFICIENT_DATA",
                "is_valid": False,
                "message": f"Critical hydrological data missing: {', '.join(missing_fields)}. Cannot reliably forecast flood risk.",
                "missing_fields": missing_fields,
                "probability_pct": None,
                "risk_category": "Insufficient Data"
            }

        # 2. Convert to DataFrame
        obs_df = pd.DataFrame([observation_dict])

        # Fill station metadata if station and state are provided but thresholds omitted
        if "danger_level_m" not in observation_dict and "station" in observation_dict:
            prof = get_station_profile(
                observation_dict.get("state", "Punjab"),
                observation_dict.get("district", "Jalandhar"),
                observation_dict.get("station", "Phillaur")
            )
            if prof:
                for k, v in prof.items():
                    if k not in obs_df.columns:
                        obs_df[k] = v

        # 3. Clean and feature engineer
        clean_df = validate_and_clean_data(obs_df)
        features_df = extract_engineered_features(clean_df)

        # 4. Prepare model input
        if self.use_scaled:
            X_input = self.scaler.transform(features_df[self.feature_names])
        else:
            X_input = features_df[self.feature_names]

        # 5. Predict Calibrated Probability
        probs = self.calibrated_model.predict_proba(X_input)
        calibrated_prob = float(probs[0, 1]) * 100.0
        calibrated_prob = max(0.1, min(99.9, calibrated_prob))

        # 6. Ensemble Uncertainty Estimation (spread across model candidates)
        # Using predictions from LR, RF, and HGB to estimate model epistemic uncertainty
        try:
            p_lr = float(self.lr_model.predict_proba(self.scaler.transform(features_df[self.feature_names]))[0, 1]) * 100.0
            p_rf = float(self.rf_model.predict_proba(features_df[self.feature_names])[0, 1]) * 100.0
            p_hgb = float(self.hgb_model.predict_proba(features_df[self.feature_names])[0, 1]) * 100.0
            model_spread = [p_lr, p_rf, p_hgb]
            uncertainty_std = float(np.std(model_spread))
            lower_bound = max(0.0, calibrated_prob - 1.96 * uncertainty_std)
            upper_bound = min(100.0, calibrated_prob + 1.96 * uncertainty_std)
        except Exception:
            lower_bound = max(0.0, calibrated_prob - 5.0)
            upper_bound = min(100.0, calibrated_prob + 5.0)
            uncertainty_std = 2.5

        # 7. Map to Risk Assessment and Actionable SOP
        risk_info = get_risk_assessment(calibrated_prob)

        # 8. Local Explainability
        explanation = self.explainer.explain_instance(features_df)

        # Hydrological margins
        current_stage = float(observation_dict.get("water_level_m", 0.0))
        danger_stage = float(observation_dict.get("danger_level_m", 1.0))
        warning_stage = float(observation_dict.get("warning_level_m", danger_stage - 1.0))
        hfl = float(observation_dict.get("high_flood_level_m", danger_stage + 1.5))
        safe_q = float(observation_dict.get("safe_discharge_cumec", 5000.0))
        current_q = float(observation_dict.get("discharge_cumec", 1000.0))

        stage_margin_m = danger_stage - current_stage
        freeboard_pct = (current_stage / danger_stage) * 100.0

        return {
            "status": "SUCCESS",
            "is_valid": True,
            "station": observation_dict.get("station", "Unknown Station"),
            "district": observation_dict.get("district", "Unknown District"),
            "state": observation_dict.get("state", "Unknown State"),
            "river": observation_dict.get("river", "Unknown River"),
            "timestamp": observation_dict.get("timestamp", "Latest"),
            "data_source": observation_dict.get("weather_source", "Sensory Telemetry"),
            # Core Prediction Outputs
            "calibrated_probability_pct": round(calibrated_prob, 1),
            "confidence_interval": {
                "lower_bound_pct": round(lower_bound, 1),
                "upper_bound_pct": round(upper_bound, 1),
                "std_uncertainty": round(uncertainty_std, 1)
            },
            "risk_category": risk_info["risk_category"],
            "risk_badge": risk_info["badge"],
            "risk_color": risk_info["color"],
            "severity_level": risk_info["severity"],
            "advisory": risk_info["advisory"],
            "actionable_sop": risk_info["actions"],
            # Hydrological telemetry summary
            "hydrology_summary": {
                "current_water_level_m": round(current_stage, 2),
                "warning_level_m": round(warning_stage, 2),
                "danger_level_m": round(danger_stage, 2),
                "high_flood_level_m": round(hfl, 2),
                "margin_to_danger_m": round(stage_margin_m, 2),
                "danger_level_capacity_pct": round(freeboard_pct, 1),
                "discharge_cumec": round(current_q, 1),
                "safe_discharge_capacity_cumec": round(safe_q, 1),
                "rainfall_cum_24h_mm": round(float(observation_dict.get("rainfall_cum_24h_mm", 0.0)), 1),
                "forecast_rainfall_next24h_mm": round(float(observation_dict.get("forecast_rainfall_next24h_mm", 0.0)), 1),
                "soil_saturation_pct": round(float(observation_dict.get("soil_saturation_pct", 50.0)), 1)
            },
            # Explainability
            "explanation": explanation
        }

    def predict_batch_dataframe(self, df):
        """
        Process a batch dataframe (e.g. from user CSV upload) and return enriched results.
        """
        results = []
        for _, row in df.iterrows():
            res = self.predict_single_station(row.to_dict())
            results.append({
                "date": row.get("date", "N/A"),
                "station": row.get("station", "N/A"),
                "river": row.get("river", "N/A"),
                "water_level_m": row.get("water_level_m", np.nan),
                "danger_level_m": row.get("danger_level_m", np.nan),
                "flood_probability_pct": res.get("calibrated_probability_pct"),
                "risk_category": res.get("risk_category"),
                "status": res.get("status")
            })
        return pd.DataFrame(results)


# Convenience singleton
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FloodPredictor()
    return _predictor_instance
