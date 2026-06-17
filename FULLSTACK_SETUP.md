# S.P.E.C.T.R.A. Streamlit Layout

## Project Structure

```text
recon3d/
|-- spectra_dashboard/
|   `-- main.py
|-- src/
|   |-- utils/
|   |   |-- smart_frame_selector.py
|   |   |-- cleanup.py
|   |   `-- evaluate_reconstruction.py
|   |-- depth/
|   |   `-- midas_depth.py
|   |-- sfm/
|   |   `-- sparse_recon_live.py
|   |-- fusion/
|   |   `-- depth_fusion.py
|   |-- mesh/
|   |   `-- mesh_surface.py
|   `-- export/
|       `-- gaussian_export.py
|-- data/
|-- setup.bat
|-- setup.sh
|-- start_dashboard.bat
`-- start_dashboard.sh
```

## Start

Windows:
```bat
setup.bat
start_dashboard.bat
```

Linux/Mac:
```bash
bash setup.sh
bash start_dashboard.sh
```

This project is now Streamlit-only. There is no Flask API or React frontend in the active layout.
