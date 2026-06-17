# S.P.E.C.T.R.A Streamlit Cloud Deployment

This project deploys as a Streamlit Community Cloud app.

## Local access remains unchanged

Cloud deployment does not replace the local project. You can still run:

```bash
python -m streamlit run spectra_dashboard/main.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Streamlit Cloud settings

Use these values when creating the app:

```text
Repository: your GitHub repository for this project
Branch: main
Main file path: spectra_dashboard/main.py
Python version: 3.10
```

Python 3.10 is recommended because this project pins `open3d==0.17.0`, which is safer on older supported Python versions than on Streamlit Cloud's current default.

## Important deployment notes

- `requirements.txt` declares the Python packages.
- `packages.txt` declares Linux system packages used by OpenCV.
- The large local `data/` directory is intentionally ignored by Git.
- On Streamlit Cloud, the app will start with empty runtime data folders and can still show the dashboard shell, reports, pipeline controls, and any generated files produced during that cloud session.
- Heavy reconstruction stages may be slower or resource-limited on the free tier.

## Deploy steps

1. Push this project to a clean GitHub repository.
2. Open `https://share.streamlit.io`.
3. Click **Create app**.
4. Choose **Yup, I have an app**.
5. Select the repository and branch.
6. Set the main file path to `spectra_dashboard/main.py`.
7. Open **Advanced settings** and select Python `3.10`.
8. Click **Deploy**.

After deployment, Streamlit gives you a public `streamlit.app` link that you can share.
