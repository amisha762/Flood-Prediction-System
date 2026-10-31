@echo off
title AI Flood Detection & Early Warning System
echo ======================================================================
echo Launching AI Flood Detection Dashboard...
echo ======================================================================
cd /d "%~dp0"
"C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Something went wrong. Press any key to exit.
    pause
)
