#!/bin/bash
# Start Streamlit dashboard

echo "Starting S.P.E.C.T.R.A. Streamlit dashboard..."
echo "Backend live logs: data/logs/live_pipeline.log"
echo "Streamlit logs: data/logs/streamlit_dashboard.log"

mkdir -p data/logs
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | Dashboard launcher started." >> data/logs/launcher.log

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python -m streamlit run spectra_dashboard/main.py --server.port 8501 >> data/logs/streamlit_dashboard.log 2>> data/logs/streamlit_dashboard.err.log
