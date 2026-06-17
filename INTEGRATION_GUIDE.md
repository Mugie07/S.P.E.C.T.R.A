# Integration Guide

The app is now Streamlit-only.

## Dashboard

Add UI controls and pages under:

```text
spectra_dashboard/
`-- main.py
```

## Pipeline Modules

Use this active layout:

```text
src/utils/smart_frame_selector.py
src/depth/midas_depth.py
src/sfm/sparse_recon_live.py
src/fusion/depth_fusion.py
src/utils/cleanup.py
src/mesh/mesh_surface.py
src/utils/evaluate_reconstruction.py
src/export/gaussian_export.py
```

Import these modules directly from Streamlit when wiring dashboard actions.
