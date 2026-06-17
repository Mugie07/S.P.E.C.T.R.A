@echo off
REM Setup script for 3D Reconstruction project
REM Run this to automatically set up the Streamlit dashboard

echo.
echo ================================
echo 3D Reconstruction Setup
echo ================================
echo.

echo [1/3] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo [1/3] DONE
echo.

echo [2/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing Python dependencies
    pause
    exit /b 1
)
echo [2/3] DONE
echo.

echo [3/3] Creating necessary directories...
if not exist "data\uploads" mkdir data\uploads
if not exist "data\results" mkdir data\results
echo [3/3] DONE
echo.

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo Next steps:
echo.
echo 1. Start the Streamlit dashboard:
echo    - Run: streamlit run spectra_dashboard\main.py
echo    OR use start_dashboard.bat
echo.
echo 2. Open the Streamlit URL shown in the terminal
echo.
echo See FULLSTACK_SETUP.md for detailed instructions
echo.
pause
