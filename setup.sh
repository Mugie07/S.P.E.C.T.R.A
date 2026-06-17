#!/bin/bash
# Setup script for 3D Reconstruction project
# Run this to automatically set up the Streamlit dashboard

echo ""
echo "================================"
echo "3D Reconstruction Setup"
echo "================================"
echo ""

echo "[1/3] Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate
echo "[1/3] DONE"
echo ""

echo "[2/3] Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing Python dependencies"
    exit 1
fi
echo "[2/3] DONE"
echo ""

echo "[3/3] Creating necessary directories..."
mkdir -p data/uploads
mkdir -p data/results
echo "[3/3] DONE"
echo ""

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the Streamlit dashboard:"
echo "   - Run: streamlit run spectra_dashboard/main.py"
echo "   OR use start_dashboard.sh"
echo ""
echo "2. Open the Streamlit URL shown in the terminal"
echo ""
echo "See FULLSTACK_SETUP.md for detailed instructions"
echo ""
