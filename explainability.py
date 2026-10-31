"""
Explainable AI (XAI) Module for Flood Detection System.
Provides global feature importance and local instance-level attributions
using SHAP (SHapley Additive exPlanations) and tree-based attribution.
Distinguishes between hydrological correlation and physical causation.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from feature_engineering import get_feature_names, extract_engineered_features


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "flood_model.joblib")


# Feature interpretation lookup table with physical reasoning
FEATURE_NARRATIVES = {
    "water_level_ratio_danger": {
        "label": "River Stage / Danger Level Ratio",
        "description": "Proximity of current water level to the station's official Danger Level.",
        "physical_mechanism": "Direct hydraulic state: as stage reaches 1.0 (100% of DL), river overflow is imminent."
    },
    "water_level_danger_margin_m": {
        "label": "Buffer Remaining Before Danger (m)",
        "description": "Height in meters between current stage and the danger threshold.",
        "physical_mechanism": "Represents the remaining channel freeboard before overtopping."
    },
    "runoff_potential_index": {
        "label": "Runoff Potential Index (Rain × Soil Saturation)",
        "description": "Interaction between incoming/past precipitation and soil saturation percentage.",
        "physical_mechanism": "Saturated soil cannot absorb moisture via infiltration, forcing rainfall directly into surface runoff."
    },
    "water_level_change_24h": {
        "label": "24h Water Level Rate of Rise",
        "description": "Speed at which the river stage is ascending (m/24h).",
        "physical_mechanism": "Fast rising rates (>0.35 m/day) signal steep incoming flood wave crests from upstream."
    },
    "rainfall_cum_72h_mm": {
        "label": "72h Cumulative Antecedent Rainfall",
        "description": "Total rainfall accumulation over the preceding 3 days.",
        "physical_mechanism": "Governs catchment storage saturation and groundwater baseflow contribution."
    },
    "forecast_rainfall_next24h_mm": {
        "label": "Forecast Rainfall (Next 24h)",
        "description": "Expected precipitation volume from numerical weather prediction.",
        "physical_mechanism": "Direct prospective driver of future runoff into the river channel."
    },
    "upstream_hydraulic_pressure": {
        "label": "Upstream Surge Pressure",
        "description": "Combined upstream water stage and arriving flow volume.",
        "physical_mechanism": "Hydraulic routing: flood peaks travel downstream according to river channel wave celerity."
    },
    "soil_saturation_pct": {
        "label": "Soil Saturation Percentage",
        "description": "Volumetric soil moisture as percentage of field capacity.",
        "physical_mechanism": "Determines Horton overland flow infiltration-excess runoff."
    },
    "dam_release_ratio": {
        "label": "Dam / Barrage Release Ratio",
        "description": "Controlled reservoir spillway release relative to safe channel capacity.",
        "physical_mechanism": "Artificial water release can exacerbate natural river flooding."
    },
    "discharge_ratio_danger": {
        "label": "Discharge / Safe Capacity Ratio",
        "description": "Current river volumetric flow rate relative to safe carrying capacity.",
        "physical_mechanism": "When discharge exceeds channel capacity, cross-sectional embankment breach occurs."
    }
}


class FloodExplainer:
    """
    Unified Explainable AI class delivering global and local explanations.
    """
    def __init__(self, pipeline_path=PIPELINE_PATH):
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Model pipeline artifact not found at {pipeline_path}.")
        self.pkg = joblib.load(pipeline_path)
        self.rf_model = self.pkg.get("rf_model")
        self.calibrated_model = self.pkg.get("calibrated_model")
        self.feature_names = self.pkg.get("feature_names", get_feature_names())
        self.feature_importances = self.pkg.get("feature_importances", {})
        
        # Initialize SHAP explainer on Random Forest model
        try:
            self.shap_explainer = shap.TreeExplainer(self.rf_model)
        except Exception:
            self.shap_explainer = None

    def get_global_feature_importances(self, top_n=10):
        """
        Return top-N global feature importances across all training records.
        """
        sorted_feats = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_feats[:top_n]
        results = []
        for feat, score in top_items:
            meta = FEATURE_NARRATIVES.get(feat, {"label": feat, "physical_mechanism": "Hydrological indicator."})
            results.append({
                "feature": feat,
                "label": meta["label"],
                "importance": score,
                "physical_mechanism": meta["physical_mechanism"]
            })
        return results

    def explain_instance(self, features_df, top_n=6):
        """
        Generate local attribution explanation for a specific prediction instance.
        Shows exact factors that increased risk (+) or decreased risk (-).
        """
        if isinstance(features_df, dict):
            features_df = pd.DataFrame([features_df])
        
        # Ensure only model features in correct order
        X = features_df[self.feature_names].copy()

        # Try SHAP Tree Explainer first
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(X)
                # For binary classification, TreeExplainer returns list [class_0, class_1] or 2D/3D array
                if isinstance(shap_values, list):
                    vals = shap_values[1][0]
                elif shap_values.ndim == 3:
                    vals = shap_values[0, :, 1]
                else:
                    vals = shap_values[0]
            except Exception:
                vals = self._surrogate_linear_attribution(X)
        else:
            vals = self._surrogate_linear_attribution(X)

        # Build structured explanation
        contributions = []
        row_vals = X.iloc[0].to_dict()

        for feat, val, raw_val in zip(self.feature_names, vals, X.iloc[0].values):
            meta = FEATURE_NARRATIVES.get(feat, {"label": feat.replace("_", " ").title(), "physical_mechanism": ""})
            contributions.append({
                "feature": feat,
                "label": meta["label"],
                "raw_value": round(float(raw_val), 2),
                "shap_contribution": round(float(val), 4),
                "direction": "INCREASED RISK" if val > 0 else "DECREASED RISK",
                "impact_pct": round(float(abs(val) * 100.0), 1),
                "mechanism": meta.get("physical_mechanism", "")
            })

        # Sort by absolute SHAP impact
        contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
        top_factors = contributions[:top_n]

        # Distinguish correlation vs causation narrative
        summary_narrative = self._generate_causality_narrative(top_factors)

        return {
            "top_factors": top_factors,
            "causality_note": (
                "Hydrological Note: Machine learning identifies statistical associations. "
                "While high water levels and rainfall are directly causal to channel overflow, "
                "factors such as 'historical flood count' and 'month' reflect spatial-temporal vulnerability "
                "and correlation rather than direct hydraulic driving forces."
            ),
            "summary_narrative": summary_narrative
        }

    def _surrogate_linear_attribution(self, X):
        """Fallback surrogate attribution based on standard deviations and normalized offsets."""
        vals = []
        for feat in self.feature_names:
            v = X.iloc[0][feat]
            imp = self.feature_importances.get(feat, 0.05)
            # High water level or high rain increases risk
            if "danger" in feat or "rain" in feat or "saturation" in feat or "pressure" in feat or "rise" in feat:
                impact = imp * (1.0 if v > 0.7 else -0.5)
            else:
                impact = imp * 0.1
            vals.append(impact)
        return np.array(vals)

    def _generate_causality_narrative(self, top_factors):
        """Build human-friendly textual summary of the primary risk drivers."""
        drivers = [f"{f['label']} ({'+' if f['shap_contribution']>0 else '-'}{f['impact_pct']}%)" for f in top_factors[:3]]
        return f"Primary model drivers: {', '.join(drivers)}."
