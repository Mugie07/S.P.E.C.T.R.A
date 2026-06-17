# S.P.E.C.T.R.A. Streamlit Setup

S.P.E.C.T.R.A. is now a Streamlit dashboard. The UI home is:

```text
spectra_dashboard/main.py
```

## Start

```bash
python -m streamlit run spectra_dashboard/main.py --server.port 8501
```

The dashboard currently includes:

- Image upload controls
- Full reconstruction pipeline controls
- Live backend status and logs
- Result artifact metrics
- Output browser for files in `data/results`
- Report preview/download links

There is no Flask API in the active layout. Add dashboard panels under `spectra_dashboard/` as the app grows.
