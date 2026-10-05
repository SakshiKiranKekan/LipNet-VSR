@echo off
title LipNet: Visual Speech Recognition from Lip Movements Using Deep Learning
echo ===============================================================================
echo   LipNet: Visual Speech Recognition from Lip Movements Using Deep Learning
echo ===============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.9+.
    pause
    exit /b
)

echo.
echo [2/3] Checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo [3/3] Launching Gradio Web Interface...
echo Open your browser at http://127.0.0.1:7860
echo.
python app.py

pause
