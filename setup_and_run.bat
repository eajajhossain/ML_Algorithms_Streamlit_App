@echo off
echo ============================================
echo   ML Algorithms - Streamlit App Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/3] Installing dependencies...
pip install --upgrade pip >nul
pip install -r requirements.txt

echo [3/3] Launching app...
echo.
echo ============================================
echo   App running at: http://localhost:8501
echo   Press Ctrl+C to stop
echo ============================================
echo.
streamlit run app.py

pause
