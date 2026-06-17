# S.P.E.C.T.R.A

S.P.E.C.T.R.A is a Streamlit dashboard for image-based 3D reconstruction. It stages uploaded images, runs the reconstruction pipeline, and exposes pipeline status, depth previews, point cloud outputs, metrics, reports, and export readiness from one interface.

## Run locally

```bash
python -m streamlit run spectra_dashboard/main.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```

On Windows, you can also use:

```bat
start_dashboard.bat
```

## Streamlit Community Cloud

Use these deployment settings:

```text
Main file path: spectra_dashboard/main.py
Python version: 3.10
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full deployment checklist.

## Notes

- The dashboard accepts JPG, JPEG, and PNG image uploads.
- Large local datasets and generated reconstruction outputs under `data/` are intentionally excluded from Git.
- Free cloud hosting may be slower for heavy backend reconstruction stages than the local workstation.
