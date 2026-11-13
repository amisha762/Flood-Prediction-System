"""
AI-Based Flood Detection and Probability Prediction System
Interactive Web Dashboard built with Streamlit, Plotly, Pandas, and Scikit-learn.
Geographical Coverage: Punjab, Haryana, Madhya Pradesh, and Bihar.
"""

import os
import sys

# -----------------------------------------------------------------------------
# Smart Auto-Launcher & Python Environment Auto-Fix
# If run with a bare `python app.py` or with Python 3.14,
# automatically relaunch with Python 3.10 and Streamlit CLI!
# -----------------------------------------------------------------------------
TARGET_PYTHON = r"C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe"

if os.path.exists(TARGET_PYTHON) and sys.executable.lower() != TARGET_PYTHON.lower():
    import subprocess
    cmd = [TARGET_PYTHON, "-m", "streamlit", "run", os.path.abspath(__file__)]
    subprocess.run(cmd)
    sys.exit(0)

try:
    import streamlit.runtime as st_rt
    if not st_rt.exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
except Exception:
    pass

import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Setup module path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from utils.stations import (
    STATIONS_DATA, get_all_states, get_districts, get_stations,
    get_station_profile, get_all_stations_flat
)
from utils.advisories import RISK_LEVEL_CONFIG, OFFICIAL_PORTALS
from data_collection import get_live_station_hydrology, get_official_sources_catalog
from predict import get_predictor


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Flood Early Warning System | PB-HR-MP-BR",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0d3b66 0%, #001e3d 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #b0c4de;
        margin-top: 6px;
    }
    .badge-bar {
        margin-top: 12px;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(255,255,255,0.15);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.25);
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #888;
        margin-top: 2px;
    }
    .alert-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Initialize System & Cached Predictor
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system_predictor():
    return get_predictor()

try:
    predictor = load_system_predictor()
except Exception as e:
    st.error(f"Error loading trained ML model: {e}. Please ensure 'train_model.py' has been executed.")
    st.stop()


# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🌊 AI-Based Flood Detection & Probability Prediction System</div>
    <div class="sub-title">Hydrological & Meteorological Early Warning Platform for Punjab, Haryana, Madhya Pradesh, and Bihar</div>
    <div class="badge-bar">
        <span class="badge-pill">📍 4 Target States</span>
        <span class="badge-pill">📡 17 CWC Monitoring Stations</span>
        <span class="badge-pill">⚡ Calibrated Supervised ML Pipeline</span>
        <span class="badge-pill">🌐 Live Open-Meteo API Integration</span>
        <span class="badge-pill">🔍 Explainable AI (SHAP Attributions)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Navigation & Location Selection
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/water.png", width=64)
st.sidebar.title("Operational Control")

# Mode Selection
app_mode = st.sidebar.radio(
    "Select Operating Mode:",
    [
        "🟢 Live Telemetry & Forecast",
        "🎛️ Interactive 'What-If' Simulation",
        "🗺️ State-Wise Flood Risk Map",
        "📂 Custom CSV Batch Inference",
        "📚 Official Portals & Diagnostics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Monitoring Station Selection")

all_states = get_all_states()
selected_state = st.sidebar.selectbox("Select State:", all_states, index=0)

available_districts = get_districts(selected_state)
selected_district = st.sidebar.selectbox("Select District:", available_districts, index=0)

available_stations = get_stations(selected_state, selected_district)
selected_station = st.sidebar.selectbox("Select CWC Station:", available_stations, index=0)

station_profile = get_station_profile(selected_state, selected_district, selected_station)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Station Metadata:**
- **River:** {station_profile['river']} ({station_profile['basin']})
- **Warning Level:** `{station_profile['warning_level_m']} m`
- **Danger Level:** `{station_profile['danger_level_m']} m`
- **High Flood Level (HFL):** `{station_profile['high_flood_level_m']} m`
- **Upstream:** {station_profile['upstream_station']}
""")


# =============================================================================
# Helper: Gauge Chart for Calibrated Flood Probability
# =============================================================================
def create_probability_gauge(prob_pct, risk_cat, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"size": 48, "color": color, "weight": "bold"}},
        title={"text": f"<b>Next 24h Flood Probability</b><br><span style='font-size:14px;color:#666'>Calibrated Empirical Risk</span>", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333", "tickvals": [0, 25, 55, 80, 100]},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e0e0e0",
            "steps": [
                {"range": [0, 25], "color": "rgba(40, 167, 69, 0.15)"},
                {"range": [25, 55], "color": "rgba(255, 193, 7, 0.18)"},
                {"range": [55, 80], "color": "rgba(253, 126, 20, 0.22)"},
                {"range": [80, 100], "color": "rgba(220, 53, 69, 0.25)"}
            ],
            "threshold": {
                "line": {"color": "#dc3545", "width": 4},
                "thickness": 0.75,
                "value": 80.0
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# =============================================================================
# MODE 1: LIVE TELEMETRY & FORECAST
# =============================================================================
if app_mode == "🟢 Live Telemetry & Forecast":
    st.subheader(f"📡 Real-Time Telemetry: {selected_station} ({station_profile['river']}, {selected_district}, {selected_state})")

    # Fetch live data
    with st.spinner("Fetching real-time weather and station telemetry..."):
        obs = get_live_station_hydrology(selected_state, selected_district, selected_station)
        pred_res = predictor.predict_single_station(obs)

    # Status Banner
    badge_type = "success" if obs["weather_status"] == "LIVE_SUCCESS" else "warning"
    st.info(f"**Data Stream:** `{obs['weather_status']}` | **Source:** `{obs['weather_source']}` | **Observed at:** `{obs['timestamp']}`")

    # Top Metric Columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">River Water Level</div>
            <div class="metric-value">{obs['water_level_m']} m</div>
            <div class="metric-sub">Danger Level: {obs['danger_level_m']} m</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Freeboard Margin</div>
            <div class="metric-value">{round(obs['danger_level_m'] - obs['water_level_m'], 2)} m</div>
            <div class="metric-sub">Buffer to Danger Level</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">River Discharge</div>
            <div class="metric-value">{obs['discharge_cumec']:,} m³/s</div>
            <div class="metric-sub">Safe: {obs['safe_discharge_cumec']:,} m³/s</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Rain / Forecast</div>
            <div class="metric-value">{obs['rainfall_cum_24h_mm']} / {obs['forecast_rainfall_next24h_mm']} mm</div>
            <div class="metric-sub">Preceding / Expected</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Soil Saturation</div>
            <div class="metric-value">{obs['soil_saturation_pct']}%</div>
            <div class="metric-sub">Runoff readiness</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Main Risk Gauge and Advisory
    left_col, right_col = st.columns([1, 1.4])

    with left_col:
        gauge_fig = create_probability_gauge(
            pred_res["calibrated_probability_pct"],
            pred_res["risk_category"],
            pred_res["risk_color"]
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown(f"""
        <div style="background-color: {pred_res['risk_color']}; color: white; padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.15rem;">
            {pred_res['risk_badge']} — {pred_res['severity_level']}
        </div>
        <div style="font-size: 0.85rem; color: #555; text-align: center; margin-top: 6px;">
            95% Confidence Interval: [{pred_res['confidence_interval']['lower_bound_pct']}% - {pred_res['confidence_interval']['upper_bound_pct']}%]
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.subheader("📋 Actionable Early-Warning Advisory")
        st.markdown(f"""
        **SOP Summary:** {pred_res['advisory']}
        
        **Standard Operational Procedures (SOP):**
        """)
        for action in pred_res["actionable_sop"]:
            st.markdown(f"- ✅ {action}")

        # Emergency contacts trigger
        if pred_res["risk_category"] in ["High", "Severe"]:
            st.error("⚠️ HIGH ALERT: Emergency flood mitigation procedures are triggered for this station.")

    st.markdown("---")

    # Hydrological Level Bar & Explanations
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌊 Water Level vs Critical CWC Thresholds")
        # Visual stage gauge
        stage_df = pd.DataFrame({
            "Threshold": ["Current Level", "Warning Level", "Danger Level", "High Flood Level (HFL)"],
            "Meters": [obs['water_level_m'], obs['warning_level_m'], obs['danger_level_m'], obs['high_flood_level_m']],
            "Color": [pred_res['risk_color'], "#ffc107", "#dc3545", "#6f42c1"]
        })
        bar_fig = px.bar(
            stage_df, x="Threshold", y="Meters", color="Threshold",
            color_discrete_map={
                "Current Level": pred_res['risk_color'],
                "Warning Level": "#ffc107",
                "Danger Level": "#dc3545",
                "High Flood Level (HFL)": "#6f42c1"
            },
            text_auto=".2f"
        )
        bar_fig.update_layout(height=320, showlegend=False, template="plotly_white")
        st.plotly_chart(bar_fig, use_container_width=True)

    with c2:
        st.subheader("🔍 Explainable AI: Top Contributing Factors")
        top_factors = pred_res["explanation"]["top_factors"]
        exp_df = pd.DataFrame(top_factors)

        fig_exp = px.bar(
            exp_df,
            x="shap_contribution",
            y="label",
            orientation="h",
            color="direction",
            color_discrete_map={"INCREASED RISK": "#dc3545", "DECREASED RISK": "#28a745"},
            labels={"shap_contribution": "SHAP Risk Contribution", "label": "Hydrological Factor"},
            title="Feature Impact on Predicted Flood Probability"
        )
        fig_exp.update_layout(height=320, template="plotly_white", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_exp, use_container_width=True)

    # Download Report Button
    st.markdown("---")
    report_text = f"""================================================================================
AI FLOOD DETECTION & EARLY WARNING SITUATION REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}
Station: {selected_station} | River: {station_profile['river']}
District: {selected_district} | State: {selected_state}

1. PREDICTION RESULTS
- Calibrated Flood Probability (Next 24h): {pred_res['calibrated_probability_pct']}%
- Risk Classification: {pred_res['risk_category']} ({pred_res['severity_level']})
- 95% Confidence Bounds: [{pred_res['confidence_interval']['lower_bound_pct']}% - {pred_res['confidence_interval']['upper_bound_pct']}%]

2. HYDROLOGICAL TELEMETRY
- Current Water Level: {obs['water_level_m']} m
- Warning Level: {obs['warning_level_m']} m
- Danger Level: {obs['danger_level_m']} m
- High Flood Level (HFL): {obs['high_flood_level_m']} m
- Discharge: {obs['discharge_cumec']} cumec (Safe capacity: {obs['safe_discharge_cumec']} cumec)
- 24h Preceding Rain: {obs['rainfall_cum_24h_mm']} mm
- Next 24h Forecast Rain: {obs['forecast_rainfall_next24h_mm']} mm
- Soil Saturation: {obs['soil_saturation_pct']}%

3. ACTIONABLE SOP ADVISORY
{pred_res['advisory']}

Standard Actions:
{chr(10).join(['- ' + a for a in pred_res['actionable_sop']])}

================================================================================
Note: System calibrated on Central Water Commission thresholds and Open-Meteo telemetry.
"""
    st.download_button(
        label="📥 Download Flood Situation Report (.txt)",
        data=report_text,
        file_name=f"flood_report_{selected_station.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )


# =============================================================================
# MODE 2: INTERACTIVE "WHAT-IF" SIMULATION
# =============================================================================
elif app_mode == "🎛️ Interactive 'What-If' Simulation":
    st.subheader(f"🎛️ Hydraulic 'What-If' Simulation: {selected_station} ({station_profile['river']})")
    st.info("Test how sudden cloudbursts, reservoir gate openings, or saturated soils impact flood probability in real time.")

    wl = station_profile["warning_level_m"]
    dl = station_profile["danger_level_m"]
    hfl = station_profile["high_flood_level_m"]
    safe_q = station_profile["safe_discharge_cumec"]

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.markdown("**🌊 Hydrological Sliders**")
        sim_stage = st.slider(
            "River Water Level (m):",
            min_value=float(round(wl - 6.0, 1)),
            max_value=float(round(hfl + 1.5, 1)),
            value=float(round(wl - 0.5, 1)),
            step=0.1,
            help=f"Warning Level: {wl}m | Danger Level: {dl}m"
        )
        sim_stage_change = st.slider("24h Stage Rate of Rise (m/day):", min_value=-1.0, max_value=3.5, value=0.4, step=0.1)
        sim_q = st.slider("Discharge Flow Rate (m³/s):", min_value=100, max_value=int(safe_q * 2.2), value=int(safe_q * 0.85), step=100)
        sim_upstream_ratio = st.slider("Upstream River Stage Ratio (Stage/DL):", min_value=0.5, max_value=1.5, value=0.95, step=0.05)
        sim_dam_release = st.slider("Upstream Dam/Barrage Release (m³/s):", min_value=0, max_value=int(safe_q * 1.5), value=int(safe_q * 0.2), step=200)

    with col_sim2:
        st.markdown("**🌧️ Meteorological & Catchment Sliders**")
        sim_rain_24h = st.slider("Cumulative 24h Rainfall (mm):", min_value=0.0, max_value=350.0, value=65.0, step=5.0)
        sim_rain_72h = st.slider("Cumulative 72h Rainfall (mm):", min_value=0.0, max_value=500.0, value=120.0, step=10.0)
        sim_forecast_rain = st.slider("Forecast Next 24h Rainfall (mm):", min_value=0.0, max_value=300.0, value=80.0, step=5.0)
        sim_soil_sat = st.slider("Catchment Soil Saturation (%):", min_value=5.0, max_value=100.0, value=82.0, step=1.0)
        sim_monsoon = st.selectbox("Season:", ["Monsoon (June-Sep)", "Non-Monsoon"], index=0)

    # Build observation payload
    sim_obs = {
        "state": selected_state,
        "district": selected_district,
        "station": selected_station,
        "river": station_profile["river"],
        "basin": station_profile["basin"],
        "warning_level_m": wl,
        "danger_level_m": dl,
        "high_flood_level_m": hfl,
        "safe_discharge_cumec": safe_q,
        "catchment_elevation_m": station_profile["catchment_elevation_m"],
        "catchment_slope_deg": station_profile["catchment_slope_deg"],
        "distance_to_river_km": station_profile["distance_to_river_km"],
        "historical_flood_count_5yr": station_profile["historical_flood_count_5yr"],
        # Sliders
        "water_level_m": sim_stage,
        "water_level_change_24h": sim_stage_change,
        "discharge_cumec": sim_q,
        "discharge_change_24h": sim_stage_change * (safe_q / 5.0),
        "upstream_water_level_ratio": sim_upstream_ratio,
        "upstream_discharge_cumec": sim_q * 1.05,
        "dam_release_cumec": sim_dam_release,
        "rainfall_intensity_current_mm": sim_rain_24h / 4.0,
        "rainfall_cum_24h_mm": sim_rain_24h,
        "rainfall_cum_48h_mm": sim_rain_24h + 20.0,
        "rainfall_cum_72h_mm": sim_rain_72h,
        "forecast_rainfall_next24h_mm": sim_forecast_rain,
        "soil_moisture_index": sim_soil_sat / 100.0,
        "soil_saturation_pct": sim_soil_sat,
        "monsoon_season": 1 if "Monsoon" in sim_monsoon else 0,
        "month": 7 if "Monsoon" in sim_monsoon else 1,
        "weather_source": "User Scenario Simulation"
    }

    sim_res = predictor.predict_single_station(sim_obs)

    st.markdown("---")
    st.subheader("📊 Simulation Prediction Outcome")

    res_col1, res_col2 = st.columns([1, 1.3])
    with res_col1:
        sim_gauge = create_probability_gauge(
            sim_res["calibrated_probability_pct"],
            sim_res["risk_category"],
            sim_res["risk_color"]
        )
        st.plotly_chart(sim_gauge, use_container_width=True)
        st.markdown(f"""
        <div style="background-color: {sim_res['risk_color']}; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.15rem;">
            {sim_res['risk_badge']}
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"### Advisory: {sim_res['risk_category']} Risk")
        st.write(sim_res["advisory"])
        st.markdown("**Recommended Operations:**")
        for act in sim_res["actionable_sop"]:
            st.markdown(f"- {act}")

        # Top factor explanation
        st.markdown("**Primary Drivers in this Simulation:**")
        for factor in sim_res["explanation"]["top_factors"][:3]:
            st.markdown(f"- **{factor['label']}**: {factor['direction']} (Impact: {factor['impact_pct']}%)")


# =============================================================================
# MODE 3: STATE-WISE FLOOD RISK MAP
# =============================================================================
elif app_mode == "🗺️ State-Wise Flood Risk Map":
    st.subheader("🗺️ Multi-Station Geospatial Flood Risk Map (Punjab, Haryana, MP, Bihar)")
    st.markdown("Live spatial overview of all 17 Central Water Commission monitoring stations.")

    stations_all = get_all_stations_flat()
    map_records = []

    with st.spinner("Analyzing telemetry across all 17 stations..."):
        for st_item in stations_all:
            # Get hydrology and prediction
            h_data = get_live_station_hydrology(st_item["state"], st_item["district"], st_item["station"])
            p_data = predictor.predict_single_station(h_data)

            map_records.append({
                "Station": st_item["station"],
                "District": st_item["district"],
                "State": st_item["state"],
                "River": st_item["river"],
                "Latitude": st_item["latitude"],
                "Longitude": st_item["longitude"],
                "Probability": p_data["calibrated_probability_pct"],
                "Risk": p_data["risk_category"],
                "Color": p_data["risk_color"],
                "Water_Level_m": h_data["water_level_m"],
                "Danger_Level_m": h_data["danger_level_m"],
                "Rain_24h_mm": h_data["rainfall_cum_24h_mm"]
            })

    map_df = pd.DataFrame(map_records)

    
    # Plotly MapLibre Scatter Map
    fig_map = px.scatter_map(
        map_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Station",
        hover_data={
            "River": True,
            "State": True,
            "District": True,
            "Water_Level_m": True,
            "Danger_Level_m": True,
            "Probability": ":.1f%",
            "Risk": True,
            "Latitude": False,
            "Longitude": False
        },
        color="Risk",
        color_discrete_map={
            "Low": "#28a745",
            "Moderate": "#ffc107",
            "High": "#fd7e14",
            "Severe": "#dc3545"
        },
        size=[18] * len(map_df),
        zoom=4.7,
        center={"lat": 26.5, "lon": 80.5},
        map_style="carto-positron",
        title="<b>CWC Monitoring Stations by Predicted Flood Risk Category</b>"
    )

    fig_map.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig_map, use_container_width=True)


# =============================================================================
# MODE 4: CUSTOM CSV BATCH INFERENCE
# =============================================================================
elif app_mode == "📂 Custom CSV Batch Inference":
    st.subheader("📂 Custom CSV Batch Prediction & Testing")
    st.markdown("Upload a custom sensor log or historical observation CSV to execute batch flood probability predictions.")

    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows. Sample view:")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🚀 Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    processed_results = predictor.predict_batch_dataframe(batch_df)
                
                st.success(f"Batch prediction completed for {len(processed_results)} observations!")
                st.dataframe(processed_results, use_container_width=True)

                # Download results CSV
                csv_bytes = processed_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predicted Results CSV",
                    data=csv_bytes,
                    file_name="batch_flood_predictions.csv",
                    mime="text/csv"
                )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")
    else:
        st.info("💡 You can test this using the pre-generated development dataset located at `data/historical_flood_dataset.csv`.")


# =============================================================================
# MODE 5: OFFICIAL PORTALS & DIAGNOSTICS
# =============================================================================
elif app_mode == "📚 Official Portals & Diagnostics":
    st.subheader("📚 Official Hydrological Portals & Model Diagnostics")

    tab1, tab2, tab3 = st.tabs(["🏛️ Official Government Portals", "📈 Model Evaluation Curves", "ℹ️ System Limitations & Viva Points"])

    with tab1:
        st.markdown("### Official Indian River & Weather Monitoring Authorities")
        portals = get_official_sources_catalog()
        for p in portals:
            st.markdown(f"""
            #### 🌐 [{p['agency']} - {p['portal']}]({p['url']})
            - **URL:** `{p['url']}`
            - **Variables:** {p['variables']}
            - **Resolution:** {p['resolution']}
            - **Geographical Scope:** {p['coverage']}
            - **Access / Licensing:** {p['access']}
            ---
            """)

    
    with tab2:
        st.markdown("### 📈 Model Evaluation Curves")

        report_html_path = os.path.join(
            CURRENT_DIR,
            "reports",
            "model_evaluation_report.html"
        )

        if not os.path.exists(report_html_path):
            with st.spinner(
                "Generating ROC, Precision-Recall and Calibration Curves..."
            ):
                try:
                    from evaluation import generate_evaluation_diagnostics

                    generate_evaluation_diagnostics(save_plots=True)

                except Exception as err:
                    st.error(
                        f"Could not generate evaluation curves: {err}"
                    )

        if os.path.exists(report_html_path):
            with open(report_html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            st.components.v1.html(
                html_content,
                height=950,
                scrolling=True
            )
        else:
            st.warning(
                "Evaluation report is unavailable. "
                "Please check the model and dataset files."
            )

    with tab3:
        st.markdown("""
        ### 🎓 Academic Project Notes & Viva Defence Points

        #### 1. Why Station-Specific Normalization is Critical:
        In hydrological systems, absolute water level in meters cannot be compared directly across rivers. 
        A river stage of 50.0m is dangerously overflowing in Patna (Ganga), whereas 50.0m in Phillaur (Sutlej, elevation 235m) 
        would be physically impossible or underground. 
        Our pipeline standardizes inputs using:
        $$\\text{Stage Ratio} = \\frac{\\text{Water Level (m)}}{\\text{Station Danger Level (m)}}$$

        #### 2. Prevention of Future-Data Leakage:
        - Evaluated with a **strict temporal split**: Trained on historical seasons (2018–2022), evaluated on prospective future years (2023–2024).
        - No future rainfall or future river stages are ever used to construct current-day features.

        #### 3. Why Probability Calibration is Essential:
        Standard tree ensembles and neural networks often produce overconfident uncalibrated probabilities (e.g. predicting 99% when the empirical risk is only 60%). 
        We apply **Platt Scaling (Sigmoid CalibratedClassifierCV)** to ensure that when our system displays a 75% flood probability, 
        approximately 75 out of 100 historical instances under identical conditions resulted in actual flood events.

        #### 4. Project Limitations:
        - Open-Meteo provides atmospheric and soil data; live telemetry for all remote CWC gauge stations requires authorized access credentials for production real-time feeds.
        - Riverbed siltation and embankment breaches alter local stage-discharge rating curves dynamically over decades.
        """)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85rem;">
    <b>AI-Based Flood Detection and Probability Prediction System</b><br>
    Final Year B.Tech Engineering Major Project | Aligned with Central Water Commission (CWC) & NDMA Standards.<br>
    <i>Disclaimer: For academic, research, and technical demonstration purposes. Emergency public advisories must be verified with official District Disaster Management Authorities.</i>
</div>
""", unsafe_allow_html=True)
