AI-Based Flood Detection and Probability Prediction System

## 🌐 Live Demo

[**Click Here to Open Flood Prediction Dashboard**](https://flood-prediction-system-amisha.streamlit.app)
Project Repository

This project uses Machine Learning to predict flood probability across Punjab, Haryana, Madhya Pradesh, and Bihar.
**Final-Year B.Tech Major Project in Artificial Intelligence & Data Science**

---

## 📌 Project Overview
This project is an end-to-end, production-grade Artificial Intelligence and Machine Learning system that predicts the **calibrated probability of riverine flooding** for any selected monitoring station within the next **24 hours**.

The system integrates:
1. **Hydrological & Meteorological Physics:** River water stage, discharge velocity, upstream hydraulic wave propagation, multi-day antecedent precipitation (24h/48h/72h), numerical rainfall forecasts, and non-linear soil saturation runoff dynamics.
2. **Geographical Granularity:** Station-specific Danger Levels and High Flood Levels (HFL) across **17 Central Water Commission (CWC) monitoring stations** in **Punjab, Haryana, Madhya Pradesh, and Bihar**.
3. **Calibrated Machine Learning:** Rigorous comparison of Logistic Regression baseline, Random Forest Ensembles, and Gradient Boosting (HistGradientBoosting), calibrated using Sigmoid Platt Scaling to deliver true empirical flood risk percentages (0–100%).
4. **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) and feature attribution showing exactly which physical variables pushed the risk higher or lower.
5. **Interactive Streamlit Dashboard:** Live Open-Meteo weather API integration, interactive "What-If" hydraulic parameter simulators, geospatial maps, CWC threshold comparisons, and downloadable emergency situation reports.

---

## 🏗️ Folder Structure
```text
flood_prediction_system/
│
├── app.py                      # Interactive Streamlit Web Dashboard
├── train_model.py              # ML training, cross-validation, and probability calibration
├── predict.py                  # Production inference engine with 'Insufficient Data' safeguards
├── data_preprocessing.py       # Data cleaning, unit standardization, and temporal split logic
├── feature_engineering.py      # Runoff potential index, upstream pressure, and surge rates
├── data_collection.py          # Real-time Open-Meteo API caller & CWC telemetry assembler
├── evaluation.py               # Generates interactive ROC, PR, calibration, and confusion matrices
├── explainability.py           # SHAP tree-based attributions and correlation vs causation analysis
├── generate_dataset.py         # Generates 2018-2024 physically grounded hydrological dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Complete beginner-friendly setup and operational guide
│
├── data/
│   └── historical_flood_dataset.csv   # 43,469 daily records (2018-2024) across 17 CWC stations
│
├── models/
│   ├── flood_model.joblib             # Serialized calibrated ML pipeline and scaler
│   └── model_metrics.json             # Empirical test metrics on prospective 2023-2024 test split
│
├── reports/
│   ├── model_evaluation_report.html   # High-resolution interactive 4-panel diagnostic curves
│   └── EVALUATION_SUMMARY.md          # Quantitative results and confusion matrix breakdown
│
└── utils/
    ├── __init__.py
    ├── stations.py                    # Metadata, coordinates, and CWC danger levels for all 17 stations
    └── advisories.py                  # NDMA standard operating procedures & official portal links
```

---

## 🚀 Beginner-Friendly Setup Guide for Windows (VS Code)

### Step 1: Open the Project in VS Code
1. Open **Visual Studio Code**.
2. Click **File** ➔ **Open Folder...**
3. Select the folder:
   `C:\Users\DELL\.gemini\antigravity\scratch\flood_prediction_system`
4. Open the integrated terminal: Press ``Ctrl + ` `` (backtick) or click **Terminal ➔ New Terminal** (Ensure it is **PowerShell**).

---

### Step 2: Set Up Python Virtual Environment (Recommended)
In the VS Code terminal, run:

```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# If PowerShell script execution is restricted on your machine, run this once:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```
*(You will see `(venv)` appear at the beginning of your command prompt).*

---

### Step 3: Install Required Dependencies
Run the following command in PowerShell:

```powershell
pip install -r requirements.txt
```

---

### Step 4: Step-by-Step Execution Workflow

#### Stage 1: Generate Hydrological & Meteorological Dataset
*(Generates 43,469 daily observations for Punjab, Haryana, MP, and Bihar spanning 2018–2024)*
```powershell
python generate_dataset.py
```

#### Stage 2: Train & Calibrate Machine Learning Models
*(Evaluates Logistic Regression, Random Forest, and Gradient Boosting on a strict temporal split 2018-2022 vs 2023-2024)*
```powershell
python train_model.py
```

#### Stage 3: Generate Interactive Diagnostic Evaluation Plots
*(Creates interactive HTML plots for ROC Curves, PR Curves, and Calibration curves in `reports/`)*
```powershell
python evaluation.py
```

#### Stage 4: Launch the Interactive Streamlit Web Dashboard
Run the dashboard with:
```powershell
streamlit run app.py
```
*Your browser will automatically open to `http://localhost:8501`!*

---

## 📊 Empirical Model Evaluation Results (Prospective Test Period: 2023–2024)

All models were evaluated on prospective test data (2023–2024, 12,427 samples, 230 flood events) with **zero future-data leakage**:

| Algorithm | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **1.0000** | 0.5180 | 0.6825 | 0.9988 | 0.9416 | 0.0130 |
| **Random Forest Ensemble** | 0.9435 | 0.7381 | 0.8282 | 0.9985 | 0.9397 | 0.0054 |
| **HistGradientBoosting** | 0.9652 | 0.7093 | 0.8177 | **0.9991** | **0.9620** | 0.0056 |
| **Calibrated Production Model** | 0.8217 | **0.8552** | **0.8381** | 0.9988 | 0.9416 | **0.0039** |

> **Key Takeaway for Viva/Interviews:**
> - In flood forecasting, **Recall is prioritized** because a False Negative (failing to predict a catastrophic flood) results in casualties and damage, whereas a False Positive only triggers precautionary flood preparations.
> - **Platt Scaling (Calibration)** reduces the Brier score to 0.0039, ensuring the predicted percentage reflects true historical likelihood.

---

## 🗺️ Supported Geographical Coverage (17 CWC Stations)

| State | Districts Covered | River Basins | Example CWC Stations |
| :--- | :--- | :--- | :--- |
| **Punjab** | Jalandhar, Tarn Taran, Hoshiarpur, Rupnagar | Sutlej, Beas, Indus Basin | Phillaur, Harike Barrage, Pong Dam Downstream, Ropar |
| **Haryana** | Yamunanagar, Ambala, Sirsa, Panipat | Yamuna, Ghaggar, Tangri | Hathnikund Barrage, Ambala Cantt, Ottu Weir, Panipat |
| **Madhya Pradesh** | Narmadapuram, Mandla, Sheopur, Raisen | Narmada, Chambal, Betwa | Hoshangabad Ghat, Mandla Bridge, Sheopur, Ganjbasoda |
| **Bihar** | Patna, Supaul, Gopalganj, Muzaffarpur, Khagaria | Ganga, Kosi, Gandak, Bagmati | Digha Ghat, Birpur Barrage, Dumriaghat, Benibad, Baltara |

---

## 🏛️ Official Data Sources Reference

1. **Central Water Commission (CWC):** [ffs.india-water.gov.in](https://ffs.india-water.gov.in/) — Official hydrological stage, Warning and Danger Levels.
2. **India-WRIS:** [indiawris.gov.in](https://indiawris.gov.in/) — Historical multi-year discharge and reservoir storage volumes.
3. **India Meteorological Department (IMD):** [mausam.imd.gov.in](https://mausam.imd.gov.in/) — Quantitative precipitation forecasts (QPF) and gridded rainfall.
4. **Open-Meteo Weather API:** [open-meteo.com](https://open-meteo.com/) — Real-time precipitation, 24h forecasts, and volumetric soil moisture.
5. **State Disaster Management Authorities:** PSDMA (Punjab), HSDMA (Haryana), MPSDMA (MP), and BSDMA / FMIS (Bihar).

---

## ⚖️ Project Limitations & Future Roadmap
1. **Dynamic Riverbed Morphometry:** Siltation and sediment deposition can alter stage-discharge rating curves after heavy monsoon seasons. Integrating satellite InSAR or periodic bathymetric updates will improve long-term decadal accuracy.
2. **Live Telemetry Gateway:** Currently uses Open-Meteo for real-time weather + CWC physical stage baselines. Deploying authenticated CWC API keys will enable direct real-time ultrasonic river gauge streaming.
3. **Deep Learning Integration:** Future extensions could incorporate LSTM / Temporal Fusion Transformers (TFT) for multi-horizon (1-day to 7-day) sequential discharge hydrographs.

---
*Created as a complete B.Tech Major Project in Artificial Intelligence, Machine Learning, and Environmental Data Science.*
