@echo off
REM Start Streamlit dashboard

echo Starting S.P.E.C.T.R.A. Streamlit dashboard...
echo.
echo Backend live logs: data\logs\live_pipeline.log
echo Streamlit logs: data\logs\streamlit_dashboard.log
echo.

if not exist data\logs mkdir data\logs
echo %date% %time% ^| INFO ^| Dashboard launcher started.>> data\logs\launcher.log

set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set PYTHON_EXE=C:\Project_anaconda\envs\reconstruction\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python
echo %date% %time% ^| INFO ^| Dashboard Python: %PYTHON_EXE%>> data\logs\launcher.log
"%PYTHON_EXE%" -m streamlit run spectra_dashboard\main.py --server.port 8501 1>> data\logs\streamlit_dashboard.log 2>> data\logs\streamlit_dashboard.err.log

pause
