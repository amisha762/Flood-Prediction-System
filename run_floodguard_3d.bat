@echo off
title FLOODGUARD 3D — AI Flood Prediction & Early Warning Command Center
color 0B
echo ==============================================================================
echo       FLOODGUARD 3D // AI FLOOD INTELLIGENCE COMMAND CENTER
echo       Target States: Punjab, Haryana, Madhya Pradesh, Bihar
echo ==============================================================================
echo [*] Initializing AI Prediction Engine (Scikit-Learn Calibrated Pipeline)...
echo [*] Connecting Live Open-Meteo Weather API Telemetry...
echo [*] Starting High-Performance Backend Server on http://127.0.0.1:8000 ...
echo ==============================================================================

cd /d "%~dp0"

:: Start browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: Run API Server via Python 3.10
"C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe" api_server.py

if errorlevel 1 (
    echo.
    echo [ERROR] Server encountered an error. Press any key to exit.
    pause
)
