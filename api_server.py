"""
FLOODGUARD 3D - High-Performance Backend API Server (Flask + CORS)
Serves Live Open-Meteo Weather API, CWC Station Telemetry,
Trained Calibrated Machine Learning Pipeline, and Real-Time Risk Analytics.
"""

import os
import sys
import json
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from utils.stations import (
    STATIONS_DATA, get_all_states, get_districts, get_stations,
    get_station_profile, get_all_stations_flat
)
from utils.advisories import RISK_LEVEL_CONFIG, get_risk_assessment, OFFICIAL_PORTALS
from data_collection import get_live_station_hydrology, fetch_live_weather
from predict import get_predictor


app = Flask(__name__, static_folder=os.path.join(CURRENT_DIR, "frontend", "dist"))
CORS(app)

# Load global ML prediction engine
try:
    predictor = get_predictor()
    print("[OK] ML Prediction Engine & Calibrated Pipeline loaded successfully.")
except Exception as e:
    print(f"[WARN] Could not initialize predictor: {e}")
    predictor = None


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint confirming API status and model readiness."""
    return jsonify({
        "status": "online",
        "system": "FLOODGUARD 3D Intelligence Engine",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "model_loaded": predictor is not None,
        "states_covered": ["Punjab", "Haryana", "Madhya Pradesh", "Bihar"],
        "total_monitoring_stations": len(get_all_stations_flat())
    })


@app.route("/api/stations", methods=["GET"])
def get_stations_catalog():
    """Returns catalog of all 17 CWC monitoring stations with coordinates, danger levels, and river metadata."""
    stations = get_all_stations_flat()
    return jsonify({
        "count": len(stations),
        "states": get_all_states(),
        "stations": stations
    })


@app.route("/api/stations/by-state", methods=["GET"])
def get_stations_grouped():
    """Returns stations grouped hierarchically by state and district."""
    return jsonify(STATIONS_DATA)


@app.route("/api/station/<state>/<district>/<station>/live", methods=["GET"])
def get_station_live_telemetry(state, district, station):
    """
    Fetches real-time weather from Open-Meteo, calculates hydraulic response,
    and runs calibrated ML inference.
    """
    profile = get_station_profile(state, district, station)
    if not profile:
        return jsonify({"error": f"Station '{station}' not found in {district}, {state}."}), 404

    # Fetch live hydrology and weather
    obs = get_live_station_hydrology(state, district, station)
    
    # Run ML Prediction
    if predictor:
        prediction = predictor.predict_single_station(obs)
    else:
        prediction = {"status": "MODEL_UNAVAILABLE", "calibrated_probability_pct": 5.0, "risk_category": "Low"}

    return jsonify({
        "station_profile": profile,
        "telemetry": obs,
        "prediction": prediction,
        "advisory": prediction.get("advisory"),
        "actionable_sop": prediction.get("actionable_sop", []),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    })


@app.route("/api/all-stations-live", methods=["GET"])
def get_all_stations_live():
    """
    Queries live telemetry and calculates real-time risk scores for all 17 stations simultaneously.
    Powers the 3D Earth markers and 2D Leaflet interactive map.
    """
    all_stations = get_all_stations_flat()
    results = []

    for st in all_stations:
        try:
            obs = get_live_station_hydrology(st["state"], st["district"], st["station"])
            if predictor:
                pred = predictor.predict_single_station(obs)
                prob = pred.get("calibrated_probability_pct", 0.0)
                risk = pred.get("risk_category", "Low")
                risk_color = pred.get("risk_color", "#28a745")
                severity = pred.get("severity_level", "NORMAL")
            else:
                prob = 5.0
                risk = "Low"
                risk_color = "#28a745"
                severity = "NORMAL"

            results.append({
                "id": f"{st['state']}_{st['district']}_{st['station']}".replace(" ", "_").lower(),
                "station": st["station"],
                "district": st["district"],
                "state": st["state"],
                "river": st["river"],
                "basin": st["basin"],
                "latitude": st["latitude"],
                "longitude": st["longitude"],
                "warning_level_m": st["warning_level_m"],
                "danger_level_m": st["danger_level_m"],
                "high_flood_level_m": st["high_flood_level_m"],
                "safe_discharge_cumec": st["safe_discharge_cumec"],
                # Live parameters
                "current_water_level_m": obs["water_level_m"],
                "current_discharge_cumec": obs["discharge_cumec"],
                "margin_to_danger_m": round(st["danger_level_m"] - obs["water_level_m"], 2),
                "rainfall_current_mm": obs["rainfall_intensity_current_mm"],
                "rainfall_24h_mm": obs["rainfall_cum_24h_mm"],
                "rainfall_72h_mm": obs["rainfall_cum_72h_mm"],
                "forecast_rainfall_24h_mm": obs["forecast_rainfall_next24h_mm"],
                "soil_saturation_pct": obs["soil_saturation_pct"],
                "weather_source": obs["weather_source"],
                "weather_status": obs["weather_status"],
                # Prediction
                "flood_probability_pct": prob,
                "risk_category": risk,
                "risk_color": risk_color,
                "severity_level": severity,
                "dam_reservoir": st.get("dam_reservoir", "None")
            })
        except Exception as err:
            results.append({
                "id": f"{st['state']}_{st['district']}_{st['station']}".replace(" ", "_").lower(),
                "station": st["station"],
                "district": st["district"],
                "state": st["state"],
                "river": st["river"],
                "latitude": st["latitude"],
                "longitude": st["longitude"],
                "warning_level_m": st["warning_level_m"],
                "danger_level_m": st["danger_level_m"],
                "error": str(err),
                "flood_probability_pct": 0.0,
                "risk_category": "Insufficient Data",
                "risk_color": "#6c757d"
            })

    total = len(results)
    high_or_severe = sum(1 for r in results if r.get("risk_category") in ["High", "Severe"])
    moderate = sum(1 for r in results if r.get("risk_category") == "Moderate")
    low = sum(1 for r in results if r.get("risk_category") == "Low")

    return jsonify({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "total_stations": total,
        "risk_breakdown": {
            "low": low,
            "moderate": moderate,
            "high_or_severe": high_or_severe
        },
        "stations": results
    })


@app.route("/api/predict", methods=["POST"])
def predict_flood_risk():
    """Execute AI inference on arbitrary sensor observations or user-specified inputs."""
    if not predictor:
        return jsonify({"error": "Prediction model is currently unavailable."}), 503

    payload = request.get_json(force=True) or {}
    state = payload.get("state", "Punjab")
    district = payload.get("district", "Jalandhar")
    station = payload.get("station", "Phillaur")

    profile = get_station_profile(state, district, station)
    if profile:
        for k, v in profile.items():
            if k not in payload or payload[k] is None:
                payload[k] = v

    if "water_level_m" not in payload or payload["water_level_m"] is None:
        payload["water_level_m"] = profile["warning_level_m"] - 1.5 if profile else 50.0
    if "rainfall_cum_24h_mm" not in payload or payload["rainfall_cum_24h_mm"] is None:
        payload["rainfall_cum_24h_mm"] = 15.0

    prediction = predictor.predict_single_station(payload)
    return jsonify(prediction)


@app.route("/api/simulate", methods=["POST"])
def simulate_scenario():
    """Run 'What-If' hydraulic simulation with custom sliders."""
    if not predictor:
        return jsonify({"error": "Prediction model is currently unavailable."}), 503

    payload = request.get_json(force=True) or {}
    state = payload.get("state", "Punjab")
    district = payload.get("district", "Jalandhar")
    station = payload.get("station", "Phillaur")

    profile = get_station_profile(state, district, station)
    if not profile:
        return jsonify({"error": f"Station '{station}' not found."}), 404

    wl = profile["warning_level_m"]
    dl = profile["danger_level_m"]
    safe_q = profile["safe_discharge_cumec"]
    water_level = float(payload.get("water_level_m", wl - 0.5))
    water_rise = float(payload.get("water_level_change_24h", 0.3))
    rain_24h = float(payload.get("rainfall_cum_24h_mm", 65.0))
    rain_72h = float(payload.get("rainfall_cum_72h_mm", 120.0))
    forecast_rain = float(payload.get("forecast_rainfall_next24h_mm", 80.0))
    soil_sat = float(payload.get("soil_saturation_pct", 82.0))
    dam_release = float(payload.get("dam_release_cumec", 0.0))
    upstream_ratio = float(payload.get("upstream_water_level_ratio", 0.95))
    is_monsoon = bool(payload.get("is_monsoon", True))

    min_possible = wl - 5.0
    norm_depth = max(0.1, (water_level - min_possible) / (dl - min_possible))
    sim_q = round(float(safe_q * (norm_depth ** 1.85)), 1)

    obs = {
        "state": state,
        "district": district,
        "station": station,
        "river": profile["river"],
        "basin": profile["basin"],
        "warning_level_m": wl,
        "danger_level_m": dl,
        "high_flood_level_m": profile["high_flood_level_m"],
        "safe_discharge_cumec": safe_q,
        "catchment_elevation_m": profile["catchment_elevation_m"],
        "catchment_slope_deg": profile["catchment_slope_deg"],
        "distance_to_river_km": profile["distance_to_river_km"],
        "historical_flood_count_5yr": profile["historical_flood_count_5yr"],
        "water_level_m": water_level,
        "water_level_change_24h": water_rise,
        "discharge_cumec": sim_q,
        "discharge_change_24h": water_rise * (safe_q / 5.0),
        "rainfall_intensity_current_mm": rain_24h / 4.0,
        "rainfall_cum_24h_mm": rain_24h,
        "rainfall_cum_48h_mm": rain_24h + 20.0,
        "rainfall_cum_72h_mm": rain_72h,
        "forecast_rainfall_next24h_mm": forecast_rain,
        "soil_moisture_index": soil_sat / 100.0,
        "soil_saturation_pct": soil_sat,
        "upstream_water_level_ratio": upstream_ratio,
        "upstream_discharge_cumec": sim_q * 1.05,
        "dam_release_cumec": dam_release,
        "monsoon_season": 1 if is_monsoon else 0,
        "month": 7 if is_monsoon else 1,
        "weather_source": "Hydraulic What-If Simulator"
    }

    prediction = predictor.predict_single_station(obs)
    return jsonify({
        "simulation_inputs": payload,
        "prediction": prediction
    })


@app.route("/api/model-metrics", methods=["GET"])
def get_model_metrics():
    """Returns verified empirical test evaluation metrics from models/model_metrics.json."""
    metrics_path = os.path.join(CURRENT_DIR, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Metrics file not found. Run train_model.py first."}), 404


@app.route("/api/official-portals", methods=["GET"])
def get_official_portals():
    """Returns official links to CWC, IMD, NDMA, and State SDMAs."""
    return jsonify(OFFICIAL_PORTALS)


# -----------------------------------------------------------------------------
# Static Frontend Serving
# -----------------------------------------------------------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(CURRENT_DIR, "frontend")
    if path != "" and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    if os.path.exists(os.path.join(dist_dir, "index.html")):
        return send_from_directory(dist_dir, "index.html")
    return jsonify({
        "status": "online",
        "message": "FLOODGUARD 3D API Server is running."
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[*] Starting FLOODGUARD 3D API Server on http://127.0.0.1:{port} ...")
    app.run(host="127.0.0.1", port=port, debug=False)
