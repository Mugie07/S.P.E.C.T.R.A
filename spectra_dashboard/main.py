from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


st.set_page_config(
    page_title="S.P.E.C.T.R.A",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_UPLOAD_DIR = DATA_DIR / "raw_phone"
SELECTED_DIR = DATA_DIR / "selected"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
RESULTS_DIR = DATA_DIR / "results"
DEPTH_DIR = RESULTS_DIR / "depth"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic"
LIDAR_DIR = DATA_DIR / "lidar"
KITCHEN_LIDAR_PATH = LIDAR_DIR / "sart-tilman_appartement_kitchen_5M.ply"
GAUSSIAN_SCENE_DIR = DATA_DIR / "gaussian_splatting_scene"
LOG_DIR = DATA_DIR / "logs"
BACKEND_LOG = LOG_DIR / "backend.log"
LIVE_STATUS = LOG_DIR / "live_status.json"
LIVE_LOG = LOG_DIR / "live_pipeline.log"
RUNNER_SCRIPT = PROJECT_ROOT / "spectra_dashboard" / "backend_runner.py"
REPORT_PATH = PROJECT_ROOT / "SPECTRA_PROJECT_REPORT.md"
FINAL_REPORT_MD = PROJECT_ROOT / "SPECTRA_FINAL_60_PAGE_REPORT.md"
FINAL_REPORT_DOCX = PROJECT_ROOT / "SPECTRA_FINAL_60_PAGE_REPORT_COMPAT.docx"
FINAL_REPORT_HTML = PROJECT_ROOT / "SPECTRA_FINAL_60_PAGE_REPORT_PREVIEW.html"
MOCK_DATA_PATH = PROJECT_ROOT / "MOCK_EVALUATION_DATA.md"

for folder in (RAW_UPLOAD_DIR, SELECTED_DIR, KEYFRAMES_DIR, RESULTS_DIR, DEPTH_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=BACKEND_LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("spectra_dashboard")


@dataclass(frozen=True)
class Stage:
    key: str
    name: str
    command: tuple[str, ...]
    output_keys: tuple[str, ...]
    description: str
    weight: int = 1


OUTPUTS = {
    "selected": SELECTED_DIR,
    "keyframes": KEYFRAMES_DIR,
    "depth": DEPTH_DIR,
    "sparse": RESULTS_DIR / "sparse_cloud.ply",
    "poses": RESULTS_DIR / "camera_poses.json",
    "dense": RESULTS_DIR / "dense_fused_cloud_pose_aligned.ply",
    "legacy_dense": RESULTS_DIR / "dense_fused_cloud.ply",
    "clean": RESULTS_DIR / "dense_fused_cloud_clean.ply",
    "mesh": RESULTS_DIR / "mesh_poisson.ply",
    "mesh_obj": RESULTS_DIR / "mesh_poisson.obj",
    "gaussian": GAUSSIAN_SCENE_DIR,
}

STAGES = [
    Stage(
        "selection",
        "Smart Frame Selection",
        ("python", "src/utils/smart_frame_selector.py", "--mode", "images"),
        ("selected",),
        "Quality mode selects sharp, diverse frames into data/selected before keyframe preparation.",
    ),
    Stage(
        "prepare",
        "Prepare Keyframes",
        ("python", "src/capture/prepare_keyframes_from_folder.py"),
        ("keyframes",),
        "Resizes and quality-filters raw_phone images into normalized reconstruction keyframes.",
    ),
    Stage(
        "depth",
        "Depth Estimation",
        ("python", "src/depth/midas_depth.py"),
        ("depth",),
        "Runs Depth Anything V2 and writes depth arrays plus preview maps for each keyframe.",
        2,
    ),
    Stage(
        "sfm",
        "Sparse SfM Reconstruction",
        ("python", "src/sfm/sparse_recon_live.py", "--no_view"),
        ("sparse", "poses"),
        "Recovers camera poses from image features and triangulates a sparse point cloud.",
        2,
    ),
    Stage(
        "fusion",
        "Pose-Aligned Depth Fusion",
        ("python", "src/depth/fuse_depth_clouds_pose_aligned.py", "--zoom", "2x", "--profile", "object"),
        ("dense",),
        "Back-projects depth maps through recovered poses into the Open3D-style 3D reconstruction image.",
        2,
    ),
    Stage(
        "cleanup",
        "Floating Cluster Cleanup",
        ("python", "src/utils/cleanup.py"),
        ("clean",),
        "Keeps the dominant DBSCAN cluster and removes disconnected outlier geometry.",
    ),
    Stage(
        "mesh",
        "Surface Meshing",
        ("python", "src/mesh/mesh_surface.py", "--no_view"),
        ("mesh", "mesh_obj"),
        "Estimates normals and builds a Poisson surface mesh from the cleaned cloud.",
        2,
    ),
    Stage(
        "legacy_fusion",
        "Final Dense Cloud Export",
        ("python", "src/depth/fuse_depth_clouds.py", "--no_view"),
        ("legacy_dense",),
        "Runs the original final dense cloud cleanup/export pass after meshing.",
    ),
    Stage(
        "evaluation",
        "Evaluation Report",
        ("python", "src/utils/evaluate_reconstruction.py"),
        tuple(),
        "Prints comparative point-count and reconstruction availability diagnostics.",
    ),
    Stage(
        "gaussian",
        "Gaussian Export",
        ("python", "src/export/gaussian_export.py"),
        ("gaussian",),
        "Creates a COLMAP-style image, camera, pose, and sparse-point package for Gaussian Splatting.",
    ),
]

SKIPPED_STAGE_KEYS: set[str] = set()
EXPORTS = [
    ("Clean point cloud", "PLY", OUTPUTS["clean"], "Main filtered point cloud for MeshLab, CloudCompare, and Open3D."),
    ("Pose-aligned cloud", "PLY", OUTPUTS["dense"], "Raw fused multi-view cloud before DBSCAN cleanup."),
    ("Sparse cloud", "PLY", OUTPUTS["sparse"], "SfM feature cloud and camera-pose evidence."),
    ("Surface mesh", "PLY", OUTPUTS["mesh"], "Poisson mesh with vertex data."),
    ("Surface mesh", "OBJ", OUTPUTS["mesh_obj"], "General 3D mesh export for Blender and game engines."),
    ("Gaussian scene", "ZIP", OUTPUTS["gaussian"], "Images and sparse COLMAP-style files for neural rendering."),
]

IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
UPLOAD_TYPES = sorted(s.strip(".") for s in IMG_SUFFIXES)
POINT_OUTPUTS = [
    ("Sparse cloud", OUTPUTS["sparse"], "Feature-based SfM point cloud."),
    ("Pose-aligned dense cloud", OUTPUTS["dense"], "Multi-view depth-fused cloud before cleanup."),
    ("Legacy dense cloud", OUTPUTS["legacy_dense"], "Earlier dense fusion output if available."),
    ("Cleaned dense cloud", OUTPUTS["clean"], "Filtered point cloud after DBSCAN cleanup."),
    ("Poisson mesh", OUTPUTS["mesh"], "Surface mesh stored as PLY."),
    ("OBJ mesh", OUTPUTS["mesh_obj"], "Surface mesh stored as OBJ."),
    ("Kitchen LiDAR scan", KITCHEN_LIDAR_PATH, "Reference kitchen PLY used by test_lidar.py."),
]
REPORT_FILES = [
    ("Original project report", REPORT_PATH),
    ("Final Markdown report", FINAL_REPORT_MD),
    ("Final Word report", FINAL_REPORT_DOCX),
    ("Final HTML preview", FINAL_REPORT_HTML),
    ("Evaluation data notes", MOCK_DATA_PATH),
]


THEME_CSS = """
<style>
:root {
  --bg: #eef3f8;
  --panel: #ffffff;
  --panel-soft: #f5f7fb;
  --ink: #16213e;
  --ink-2: #24304f;
  --text: #172033;
  --muted: #66758d;
  --line: #d8e1ec;
  --accent: #0e8f83;
  --accent-2: #2f6fed;
  --gold: #c58b23;
  --wine: #9f3153;
  --good: #12805c;
  --warn: #b45309;
  --bad: #b42318;
}
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 0% 0%, rgba(14, 143, 131, .10), transparent 32rem),
    radial-gradient(circle at 100% 0%, rgba(197, 139, 35, .12), transparent 28rem),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] {
  background: rgba(247, 248, 251, 0.86);
  backdrop-filter: blur(10px);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f8fbfd 100%);
  border-right: 1px solid var(--line);
}
.block-container {
  padding-top: 1.5rem;
  padding-bottom: 2.25rem;
  max-width: 1480px;
}
h1, h2, h3 {
  color: var(--text);
  letter-spacing: 0;
}
.app-hero {
  border: 1px solid rgba(255,255,255,.22);
  background:
    linear-gradient(135deg, rgba(22,33,62,.98), rgba(14,143,131,.88)),
    var(--ink);
  border-radius: 8px;
  padding: 1.55rem 1.6rem;
  margin-bottom: 1rem;
  color: #ffffff;
  box-shadow: 0 18px 42px rgba(22, 33, 62, .16);
}
.eyebrow {
  color: #b7f3eb;
  font-size: .74rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-bottom: .25rem;
}
.hero-title {
  font-size: clamp(1.65rem, 3vw, 2.5rem);
  font-weight: 800;
  line-height: 1.1;
  margin-bottom: .35rem;
  color: #ffffff;
}
.hero-copy {
  color: rgba(255,255,255,.82);
  max-width: 920px;
  line-height: 1.55;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .75rem;
  margin: 1rem 0;
}
.kpi {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: .95rem 1rem;
  box-shadow: 0 10px 26px rgba(22, 33, 62, .06);
  border-top: 3px solid var(--accent);
}
.kpi:nth-child(2) { border-top-color: var(--accent-2); }
.kpi:nth-child(3) { border-top-color: var(--good); }
.kpi:nth-child(4) { border-top-color: var(--gold); }
.kpi .label {
  color: var(--muted);
  font-size: .78rem;
}
.kpi .value {
  font-size: 1.55rem;
  font-weight: 800;
  margin-top: .15rem;
}
.section-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  box-shadow: 0 8px 20px rgba(22, 33, 62, .045);
}
.stage-row {
  display: grid;
  grid-template-columns: 36px minmax(220px, 1.2fr) minmax(220px, 2fr) 130px;
  gap: .75rem;
  align-items: center;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  padding: .8rem .9rem;
  margin-bottom: .55rem;
  box-shadow: 0 6px 18px rgba(22, 33, 62, .035);
}
.stage-num {
  width: 28px;
  height: 28px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--panel-soft);
  color: var(--muted);
  font-size: .78rem;
  font-weight: 800;
}
.stage-name {
  font-weight: 800;
  margin-bottom: .15rem;
}
.stage-desc {
  color: var(--muted);
  font-size: .82rem;
  line-height: 1.35;
}
.pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 86px;
  padding: .26rem .55rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  font-size: .74rem;
  font-weight: 800;
}
.pill.done {
  color: var(--good);
  background: #eaf8f1;
  border-color: #b9e6ce;
}
.pill.pending {
  color: var(--muted);
  background: var(--panel-soft);
}
.pill.running {
  color: var(--warn);
  background: #fff7e8;
  border-color: #f4d49a;
}
.pill.stopped {
  color: var(--bad);
  background: #fff0ef;
  border-color: #f5b5af;
}
.live-dot {
  display: inline-block;
  width: .62rem;
  height: .62rem;
  border-radius: 50%;
  margin-right: .45rem;
  background: var(--good);
  box-shadow: 0 0 0 rgba(18,128,92,.55);
  animation: livePulse 1.3s infinite;
}
.live-dot.idle {
  background: var(--muted);
  animation: none;
}
@keyframes livePulse {
  0% { box-shadow: 0 0 0 0 rgba(18,128,92,.55); }
  70% { box-shadow: 0 0 0 9px rgba(18,128,92,0); }
  100% { box-shadow: 0 0 0 0 rgba(18,128,92,0); }
}
.status-strip {
  display: flex;
  gap: .75rem;
  align-items: center;
  justify-content: space-between;
  padding: .8rem 1rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255,255,255,.86);
  margin-bottom: 1rem;
}
.status-message {
  color: var(--muted);
  font-size: .9rem;
}
.pill.failed {
  color: var(--bad);
  background: #fff0ef;
  border-color: #f5b5af;
}
.pill.skipped {
  color: #5b4b00;
  background: #fff8d7;
  border-color: #e6c84f;
}
.small-muted {
  color: var(--muted);
  font-size: .82rem;
}
.file-line {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--line);
  padding: .52rem 0;
  font-size: .88rem;
}
.file-line:last-child {
  border-bottom: 0;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
div.stButton > button,
div.stDownloadButton > button {
  border-radius: 6px;
  border: 1px solid #b9c6d6 !important;
  font-weight: 700;
  background: #ffffff !important;
  color: #172033 !important;
  box-shadow: 0 4px 12px rgba(22, 33, 62, .06);
}
div.stButton > button[kind="primary"] {
  background: #0e8f83 !important;
  border-color: #0e8f83 !important;
  color: #ffffff !important;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
  border-color: #0e8f83 !important;
  color: #063b37 !important;
  background: #eefbf8 !important;
}
div.stButton > button[kind="primary"]:hover {
  background: #0b766d !important;
  border-color: #0b766d !important;
  color: #ffffff !important;
}
div.stButton > button:disabled,
div.stDownloadButton > button:disabled,
button:disabled {
  background: #edf2f7 !important;
  color: #66758d !important;
  border-color: #cfd8e5 !important;
  opacity: 1 !important;
}
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
  color: #24304f !important;
  font-weight: 700 !important;
}
[data-testid="stRadio"] label,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] p,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label span,
[data-testid="stCheckbox"] p {
  color: #172033 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label {
  background: rgba(255,255,255,.78);
  border: 1px solid #d8e1ec;
  border-radius: 8px;
  padding: .35rem .55rem;
}
[data-testid="stRadio"] div[role="radiogroup"] label div:first-child {
  border-color: #24304f !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
  color: #172033 !important;
}
[data-testid="stCheckbox"] svg,
[data-testid="stRadio"] svg {
  color: #ffffff !important;
}
[data-testid="stFileUploader"] {
  background: #ffffff !important;
  border: 1px solid #b9c6d6 !important;
  border-radius: 8px !important;
  padding: .85rem !important;
}
[data-testid="stFileUploader"] section {
  background: #f8fbfd !important;
  border: 1px dashed #8091a8 !important;
  color: #172033 !important;
}
[data-testid="stFileUploader"] section * {
  color: #172033 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploader"] [data-testid^="stFileUploaderFile"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
[data-testid="stFileUploader"] [data-testid^="stFileUploaderFile"] div,
[data-testid="stFileUploader"] div:has([data-testid="stFileUploaderFileName"]),
[data-testid="stFileUploader"] div:has([data-testid="stFileUploaderFileSize"]),
[data-testid="stFileUploader"] div:has([title$=".jpg"]),
[data-testid="stFileUploader"] div:has([title$=".jpeg"]),
[data-testid="stFileUploader"] div:has([title$=".png"]) {
  background: #eaf6ff !important;
  background-color: #eaf6ff !important;
  color: #12304a !important;
  border-color: #8fc8f2 !important;
  box-shadow: none !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
  border: 1px solid #8fc8f2 !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploader"] [title$=".jpg"],
[data-testid="stFileUploader"] [title$=".jpeg"],
[data-testid="stFileUploader"] [title$=".png"] {
  color: #12304a !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg,
[data-testid="stFileUploader"] [data-testid^="stFileUploaderFile"] svg {
  color: #087f8c !important;
  stroke: #087f8c !important;
}
[data-testid="stFileUploader"] small {
  display: none !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button[kind] {
  background: linear-gradient(135deg, #0e8f83, #2f6fed) !important;
  color: #ffffff !important;
  border: 1px solid #0e8f83 !important;
  border-radius: 8px !important;
  font-weight: 800 !important;
  box-shadow: 0 8px 20px rgba(14, 143, 131, .22) !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button[kind]:hover {
  background: linear-gradient(135deg, #0b766d, #245bd0) !important;
  color: #ffffff !important;
}
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
  color: #24304f !important;
}
input, textarea, [data-baseweb="select"] {
  background: #ffffff !important;
  color: #172033 !important;
  border-color: #b9c6d6 !important;
}
[data-baseweb="tab-list"] button,
[data-testid="stTabs"] button {
  color: #24304f !important;
}
[data-baseweb="tab-list"] button[aria-selected="true"],
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #0e8f83 !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
.stCaptionContainer {
  color: #495a74 !important;
}
[data-testid="stAlert"] {
  color: #172033 !important;
}
[data-testid="stAlert"] * {
  color: #172033 !important;
}
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: .85rem .95rem;
}
@media (max-width: 900px) {
  .kpi-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .stage-row {
    grid-template-columns: 32px 1fr;
  }
  .stage-row .stage-desc,
  .stage-row .pill {
    grid-column: 2;
  }
}
@media (max-width: 560px) {
  .kpi-grid {
    grid-template-columns: 1fr;
  }
  .app-hero {
    padding: 1rem;
  }
}
.st-key-top_nav_Guide button,
.st-key-top_nav_Dataset button,
.st-key-top_nav_Pipeline button,
.st-key-top_nav_Outputs button,
.st-key-top_nav_Viewer button,
.st-key-top_nav_Exports button {
  background: #ffffff !important;
  color: #172033 !important;
  border: 1px solid #b9c6d6 !important;
  border-top: 3px solid #0e8f83 !important;
}
.st-key-top_nav_Guide button:hover,
.st-key-top_nav_Dataset button:hover,
.st-key-top_nav_Pipeline button:hover,
.st-key-top_nav_Outputs button:hover,
.st-key-top_nav_Viewer button:hover,
.st-key-top_nav_Exports button:hover {
  background: #eefbf8 !important;
  color: #063b37 !important;
  border-color: #0e8f83 !important;
}
</style>
"""


st.markdown(THEME_CSS, unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "page": "Overview",
        "dataset_mode": "Real Dataset",
        "synthetic_scene": "Indoor Room",
        "running_stage": None,
        "last_result": None,
        "failed_stage": None,
        "last_output": "",
        "stage_times": {},
        "manual_score": 90,
        "manual_rms": 0.74,
        "manual_noise": 31,
        "auto_refresh": False,
        "viewer_source": "3D reconstruction image",
        "preserve_outputs": True,
        "frame_selection_mode": "Quality selection",
        "scene_profile": "Lobby room",
        "zoom_profile": "1x",
        "ui_notice": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def log(message: str) -> None:
    LOGGER.info(message)


def load_live_status() -> dict:
    if not LIVE_STATUS.exists():
        return {}
    try:
        return json.loads(LIVE_STATUS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def live_running(status: dict | None = None) -> bool:
    status = status if status is not None else load_live_status()
    return bool(status.get("running"))


def read_live_log(max_lines: int = 120) -> str:
    if not LIVE_LOG.exists():
        return ""
    lines = LIVE_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def sync_stage_times_from_live(status: dict) -> None:
    stages = status.get("stages", {})
    for key, data in stages.items():
        elapsed = data.get("elapsed_s")
        if elapsed is not None:
            st.session_state["stage_times"][key] = elapsed
        if data.get("status") == "failed":
            st.session_state["failed_stage"] = key


def current_quality_mode() -> str:
    return "quality" if st.session_state.get("frame_selection_mode") == "Quality selection" else "fast"


def current_scene_profile() -> str:
    return "lobby" if st.session_state.get("scene_profile") == "Lobby room" else "object"


def current_zoom_profile() -> str:
    return st.session_state.get("zoom_profile", "1x")


def stage_command(stage: Stage) -> tuple[str, ...]:
    if stage.key == "selection":
        max_frames = "60" if current_scene_profile() == "lobby" else "24"
        return ("python", "src/utils/smart_frame_selector.py", "--mode", "images", "--max_frames", max_frames)
    if stage.key == "prepare":
        max_images = "60" if current_scene_profile() == "lobby" else "25"
        return ("python", "src/capture/prepare_keyframes_from_folder.py", "--max_images", max_images)
    if stage.key != "fusion":
        return stage.command

    command = [
        "python",
        "src/depth/fuse_depth_clouds_pose_aligned.py",
        "--zoom",
        current_zoom_profile(),
        "--profile",
        "room" if current_scene_profile() == "lobby" else "object",
    ]
    return tuple(command)


def skipped_stage_keys(quality_mode: str | None = None) -> set[str]:
    quality_mode = quality_mode or current_quality_mode()
    return {"selection"} if quality_mode == "fast" else set()


def start_backend_job(mode: str, stage_keys: list[str] | None = None) -> None:
    if live_running():
        return
    quality_mode = current_quality_mode()
    scene_profile = current_scene_profile()
    zoom_profile = current_zoom_profile()
    skipped = skipped_stage_keys(quality_mode)
    stage_keys = stage_keys or []
    active_stage_keys = [stage.key for stage in STAGES if stage.key not in skipped]
    selected_stage_keys = [key for key in stage_keys if key not in skipped]
    if mode == "all":
        runner_stage_keys = active_stage_keys
    elif mode == "remaining":
        runner_stage_keys = [
            stage.key
            for stage in STAGES
            if stage.key not in skipped and stage_status(stage) != "done"
        ]
    else:
        runner_stage_keys = selected_stage_keys
    LIVE_LOG.write_text("Starting backend runner from dashboard...\n", encoding="utf-8")
    LIVE_STATUS.write_text(
        json.dumps(
            {
                "runner_pid": None,
                "mode": mode,
                "quality_mode": quality_mode,
                "scene_profile": scene_profile,
                "zoom_profile": zoom_profile,
                "running": True,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": None,
                "current_stage": "launching",
                "last_message": "Dashboard requested backend runner startup.",
                "stage_order": [stage.key for stage in STAGES],
                "selected_stages": runner_stage_keys,
                "stages": {
                    "selection": {
                        "name": "Smart Frame Selection",
                        "status": "skipped" if "selection" in skipped else "pending",
                        "command": "python src/utils/smart_frame_selector.py --mode images",
                        "started_at": None,
                        "finished_at": None,
                        "elapsed_s": None,
                        "returncode": None,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    command = [sys.executable, str(RUNNER_SCRIPT), "--mode", mode]
    command.extend(["--quality_mode", quality_mode])
    command.extend(["--scene_profile", scene_profile, "--zoom_profile", zoom_profile])
    if mode == "selected":
        command.extend(["--stages", *runner_stage_keys])
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as exc:
        LIVE_STATUS.write_text(
            json.dumps(
                {
                    "runner_pid": None,
                    "mode": mode,
                    "running": False,
                    "current_stage": None,
                    "last_message": f"Failed to start backend runner: {exc}",
                    "stages": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    live_status = load_live_status()
    live_status["runner_pid"] = process.pid
    live_status["last_message"] = "Backend runner launched from dashboard."
    LIVE_STATUS.write_text(json.dumps(live_status, indent=2), encoding="utf-8")
    log(f"Started live backend job PID {process.pid}: {' '.join(command)}")


def stop_backend_job(status: dict) -> None:
    pid = status.get("runner_pid")
    current_stage = status.get("current_stage")
    if pid:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        else:
            subprocess.run(["kill", str(pid)], capture_output=True, text=True)
    if current_stage and current_stage in status.get("stages", {}):
        status["stages"][current_stage]["status"] = "stopped"
        status["stages"][current_stage]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status["running"] = False
    status["current_stage"] = None
    status["last_message"] = "Backend job was stopped from the dashboard."
    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    LIVE_STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def stop_backend_from_ui(status: dict) -> None:
    stop_backend_job(status)
    st.session_state["ui_notice"] = "Backend job stopped. Partial outputs remain available for inspection."
    time.sleep(0.4)
    st.rerun()


def list_files(folder: Path, suffixes: set[str] | None = None) -> list[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file()]
    if suffixes is not None:
        files = [p for p in files if p.suffix.lower() in suffixes]
    return sorted(files)


def has_output(key: str) -> bool:
    path = OUTPUTS[key]
    if key == "depth":
        return any(DEPTH_DIR.glob("*_depth.npy"))
    if key in {"selected", "keyframes"}:
        return bool(list_files(path, IMG_SUFFIXES))
    if path.is_dir():
        return path.exists() and any(path.rglob("*"))
    return path.exists() and path.stat().st_size > 0


def stage_status(stage: Stage) -> str:
    live = load_live_status()
    live_quality_mode = live.get("quality_mode")
    effective_quality_mode = live_quality_mode if live.get("running") else current_quality_mode()
    if stage.key in skipped_stage_keys(effective_quality_mode):
        return "skipped"
    live_stage = live.get("stages", {}).get(stage.key, {})
    live_status_value = live_stage.get("status")
    outputs_ready = bool(stage.output_keys) and all(has_output(key) for key in stage.output_keys)
    if live.get("running") and live_status_value == "running":
        return "running"
    if outputs_ready:
        return "done"
    if live_status_value == "stopped":
        return "stopped"
    if live_status_value == "failed":
        return "failed"
    if live_status_value == "done" and not stage.output_keys:
        return "done"
    if st.session_state.get("running_stage") == stage.key:
        return "running"
    if st.session_state.get("failed_stage") == stage.key:
        return "failed"
    if not stage.output_keys:
        return "done" if stage.key in st.session_state.get("stage_times", {}) else "pending"
    return "done" if outputs_ready else "pending"


def pipeline_progress() -> tuple[int, int]:
    done_weight = 0
    skipped = skipped_stage_keys()
    total_weight = sum(stage.weight for stage in STAGES if stage.key not in skipped)
    for stage in STAGES:
        if stage.key in skipped:
            continue
        if stage_status(stage) == "done":
            done_weight += stage.weight
    return done_weight, total_weight


def read_ply_header(path: Path) -> dict[str, int]:
    info = {"vertices": 0, "faces": 0}
    if not path.exists():
        return info
    try:
        with path.open("rb") as handle:
            for raw in handle:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("element vertex"):
                    info["vertices"] = int(line.split()[-1])
                elif line.startswith("element face"):
                    info["faces"] = int(line.split()[-1])
                elif line == "end_header":
                    break
    except Exception:
        return info
    return info


def count_obj(path: Path) -> tuple[int, int]:
    vertices = 0
    faces = 0
    if not path.exists():
        return vertices, faces
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    vertices += 1
                elif line.startswith("f "):
                    faces += 1
    except Exception:
        pass
    return vertices, faces


def artifact_metrics() -> dict[str, int]:
    sparse = read_ply_header(OUTPUTS["sparse"])
    dense = read_ply_header(OUTPUTS["dense"])
    legacy_dense = read_ply_header(OUTPUTS["legacy_dense"])
    clean = read_ply_header(OUTPUTS["clean"])
    mesh = read_ply_header(OUTPUTS["mesh"])
    obj_vertices, obj_faces = count_obj(OUTPUTS["mesh_obj"])
    depth_count = len(list(DEPTH_DIR.glob("*_depth.npy")))
    pose_count = 0
    if OUTPUTS["poses"].exists():
        try:
            pose_count = len(json.loads(OUTPUTS["poses"].read_text(encoding="utf-8")))
        except Exception:
            pose_count = 0
    return {
        "raw_images": len(list_files(RAW_UPLOAD_DIR, IMG_SUFFIXES)),
        "selected_images": len(list_files(SELECTED_DIR, IMG_SUFFIXES)),
        "keyframes": len(list_files(KEYFRAMES_DIR, IMG_SUFFIXES)),
        "depth_maps": depth_count,
        "poses": pose_count,
        "sparse_points": sparse["vertices"],
        "dense_points": dense["vertices"] or legacy_dense["vertices"],
        "pose_aligned_points": dense["vertices"],
        "clean_points": clean["vertices"],
        "mesh_vertices": mesh["vertices"] or obj_vertices,
        "mesh_faces": mesh["faces"] or obj_faces,
    }


def build_evaluation_record(metrics: dict[str, int]) -> dict[str, object]:
    live = load_live_status()
    ready_exports = sum(1 for _, _, path, _ in EXPORTS if path.exists() and (any(path.rglob("*")) if path.is_dir() else path.stat().st_size > 0))
    stage_times = {
        stage.key: st.session_state["stage_times"].get(stage.key)
        for stage in STAGES
        if stage.key in st.session_state.get("stage_times", {})
    }
    return {
        "program": "S.P.E.C.T.R.A",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "raw_images": metrics["raw_images"],
            "selected_frames": metrics["selected_images"],
            "keyframes": metrics["keyframes"],
            "depth_maps": metrics["depth_maps"],
            "camera_pose_records": metrics["poses"],
        },
        "outputs": {
            "sparse_points": metrics["sparse_points"],
            "dense_points": metrics["dense_points"],
            "pose_aligned_points": metrics["pose_aligned_points"],
            "clean_points": metrics["clean_points"],
            "mesh_vertices": metrics["mesh_vertices"],
            "mesh_faces": metrics["mesh_faces"],
            "exports_ready": ready_exports,
            "exports_total": len(EXPORTS),
        },
        "quality_overrides": {
            "reconstruction_score_pct": st.session_state.get("manual_score"),
            "rms_error_px": st.session_state.get("manual_rms"),
            "noise_reduction_pct": st.session_state.get("manual_noise"),
        },
        "backend": {
            "running": live_running(live),
            "mode": live.get("mode"),
            "started_at": live.get("started_at"),
            "finished_at": live.get("finished_at"),
            "last_message": live.get("last_message"),
            "stage_times_s": stage_times,
        },
    }


def readiness_checks(metrics: dict[str, int]) -> list[tuple[str, str, bool, str]]:
    return [
        ("Dataset", "At least 8 raw images staged", metrics["raw_images"] >= 8, f'{fmt_int(metrics["raw_images"])} raw images'),
        ("Keyframes", "Prepared keyframes available", metrics["keyframes"] > 0, f'{fmt_int(metrics["keyframes"])} keyframes'),
        ("Depth", "Depth maps generated", metrics["depth_maps"] > 0, f'{fmt_int(metrics["depth_maps"])} depth arrays'),
        ("SfM", "Sparse cloud and poses available", metrics["sparse_points"] > 0 and metrics["poses"] > 0, f'{fmt_int(metrics["sparse_points"])} points / {fmt_int(metrics["poses"])} poses'),
        ("Fusion", "Dense or pose-aligned cloud available", metrics["dense_points"] > 0, f'{fmt_int(metrics["dense_points"])} points'),
        ("Cleanup", "Clean cloud available", metrics["clean_points"] > 0, f'{fmt_int(metrics["clean_points"])} points'),
        ("Mesh", "Mesh vertices and faces available", metrics["mesh_vertices"] > 0 and metrics["mesh_faces"] > 0, f'{fmt_int(metrics["mesh_vertices"])} vertices / {fmt_int(metrics["mesh_faces"])} faces'),
        ("Export", "Gaussian package available", has_output("gaussian"), "ready" if has_output("gaussian") else "not ready"),
    ]


def fmt_int(value: int) -> str:
    return f"{value:,}" if value else "0"


def fmt_size(path: Path) -> str:
    if not path.exists():
        return "-"
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return "-"


def zip_directory(path: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in path.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(path.parent))
    buffer.seek(0)
    return buffer.getvalue()


def run_command(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    parts = list(command)
    if parts[0] == "python":
        parts[0] = sys.executable
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "SPECTRA_FRAME_SOURCE": "selected" if current_quality_mode() == "quality" else "raw_phone",
    }
    return subprocess.run(
        parts,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def run_stage(stage: Stage) -> subprocess.CompletedProcess[str]:
    st.session_state["running_stage"] = stage.key
    started = time.time()
    log(f"Starting stage: {stage.key}")
    result = run_command(stage_command(stage))
    elapsed = round(time.time() - started, 2)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    st.session_state["stage_times"][stage.key] = elapsed
    st.session_state["last_output"] = output[-12000:]
    st.session_state["last_result"] = "ok" if result.returncode == 0 else "failed"
    st.session_state["failed_stage"] = None if result.returncode == 0 else stage.key
    st.session_state["running_stage"] = None
    log(f"Stage {stage.key} finished with code {result.returncode} in {elapsed}s")
    return result


def clear_folder(folder: Path, suffixes: set[str] | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes):
            path.unlink()


def clear_outputs() -> None:
    for key in ("sparse", "poses", "dense", "legacy_dense", "clean", "mesh", "mesh_obj"):
        path = OUTPUTS[key]
        if path.exists() and path.is_file():
            path.unlink()
    if GAUSSIAN_SCENE_DIR.exists():
        shutil.rmtree(GAUSSIAN_SCENE_DIR)
    clear_folder(DEPTH_DIR)
    st.session_state["stage_times"] = {}
    st.session_state["last_output"] = ""
    st.session_state["last_result"] = None
    st.session_state["failed_stage"] = None


def save_uploads(files: Iterable) -> int:
    clear_folder(RAW_UPLOAD_DIR, IMG_SUFFIXES)
    clear_folder(SELECTED_DIR)
    clear_folder(KEYFRAMES_DIR)
    if not st.session_state.get("preserve_outputs", True):
        clear_outputs()
    saved = 0
    for index, uploaded in enumerate(files):
        suffix = Path(uploaded.name).suffix.lower() or ".jpg"
        target = RAW_UPLOAD_DIR / f"capture_{index:03d}{suffix}"
        target.write_bytes(uploaded.getbuffer())
        saved += 1
    log(f"Saved {saved} upload files")
    return saved


def delete_staged_image(path: Path, preserve_outputs: bool = True) -> None:
    if path.parent.resolve() != RAW_UPLOAD_DIR.resolve():
        raise ValueError(f"Refusing to delete a file outside {RAW_UPLOAD_DIR}")
    if path.exists() and path.is_file() and path.suffix.lower() in IMG_SUFFIXES:
        path.unlink()
    clear_folder(SELECTED_DIR)
    clear_folder(KEYFRAMES_DIR)
    if not preserve_outputs:
        clear_outputs()
    log(f"Deleted staged image: {path.name}")


def running_on_streamlit_cloud() -> bool:
    return bool(os.environ.get("STREAMLIT_SHARING") or os.environ.get("STREAMLIT_CLOUD"))


def missing_artifact_message(source: str, path: Path) -> str:
    rel_path = path.relative_to(PROJECT_ROOT)
    if running_on_streamlit_cloud() and path.is_relative_to(DATA_DIR):
        return (
            f"{source} is a local demo artifact that is not bundled with the cloud deployment: {rel_path}. "
            "The large data folder is kept out of GitHub so the app can deploy on the free tier. "
            "Use Synthetic preview in the cloud, or run the dashboard locally to inspect this file."
        )
    return f"{source} is not available yet: {rel_path}"


def load_cloud(path: Path, sample_limit: int = 120000, use_rgb: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | list[str]] | None:
    if not path.exists():
        return None
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if points.size == 0:
            mesh = o3d.io.read_triangle_mesh(str(path))
            points = np.asarray(mesh.vertices)
            colors = np.zeros_like(points)
        if points.size == 0:
            return None
        if len(points) > sample_limit:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(points), sample_limit, replace=False)
            points = points[idx]
            if len(colors) >= max(idx) + 1:
                colors = colors[idx]
            else:
                colors = np.zeros_like(points)
        if use_rgb and len(colors) == len(points) and colors.size:
            rgb = np.clip(colors * 255, 0, 255).astype(int)
            color_value = [f"rgb({r},{g},{b})" for r, g, b in rgb]
            return points[:, 0], points[:, 1], points[:, 2], color_value
        z = points[:, 2]
        color_value = (z - z.min()) / (z.max() - z.min() + 1e-9)
        return points[:, 0], points[:, 1], points[:, 2], color_value
    except Exception:
        return None


SYNTHETIC_SCENES = ["Indoor Room", "ALS Terrain"]


def synthetic_scene(scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | list[str]]:
    rng = np.random.default_rng({"Indoor Room": 1, "ALS Terrain": 2}[scene])
    if scene == "Indoor Room":
        n = 7000
        floor_x = rng.uniform(-5, 5, n)
        floor_y = rng.uniform(-4, 4, n)
        floor_z = rng.normal(0, 0.02, n)
        wall_n = 2500
        wall_x = np.concatenate([rng.uniform(-5, 5, wall_n), rng.choice([-5, 5], wall_n)])
        wall_y = np.concatenate([rng.choice([-4, 4], wall_n), rng.uniform(-4, 4, wall_n)])
        wall_z = rng.uniform(0, 3, wall_n * 2)
        x = np.concatenate([floor_x, wall_x])
        y = np.concatenate([floor_y, wall_y])
        z = np.concatenate([floor_z, wall_z])
    elif scene == "ALS Terrain":
        n = 12000
        x = rng.uniform(-100, 100, n)
        y = rng.uniform(-100, 100, n)
        z = 5 * np.sin(x / 18) + 4 * np.cos(y / 22) + rng.normal(0, 0.35, n)
    c = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return x, y, z, c


def cloud_figure(x: np.ndarray, y: np.ndarray, z: np.ndarray, c: np.ndarray | list[str], title: str, dark: bool = False) -> go.Figure:
    bg = "#05070b" if dark else "#ffffff"
    grid = "#1f2a3a" if dark else "#d8e1ec"
    tick = "#dbeafe" if dark else "#172033"
    marker = dict(size=1.4, color=c, opacity=0.9, showscale=False)
    if not (isinstance(c, list) and c and isinstance(c[0], str)):
        marker["colorscale"] = "Viridis"
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=marker,
            )
        ]
    )
    axis = dict(
        showgrid=not dark,
        gridcolor=grid,
        zeroline=True,
        zerolinecolor=grid,
        showticklabels=not dark,
        tickfont=dict(color=tick, size=10),
        title=dict(font=dict(color=tick, size=11)),
        backgroundcolor=bg,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=tick)),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, bgcolor=bg),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=tick),
        margin=dict(l=0, r=0, t=40, b=0),
        height=560,
    )
    return fig


def timing_figure() -> go.Figure | None:
    if not st.session_state["stage_times"]:
        return None
    labels = [stage.name for stage in STAGES if stage.key in st.session_state["stage_times"]]
    values = [st.session_state["stage_times"][stage.key] for stage in STAGES if stage.key in st.session_state["stage_times"]]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#0f766e", text=[f"{v}s" for v in values], textposition="outside"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=80),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis_title="Seconds",
        xaxis_tickangle=-32,
        font=dict(color="#172033"),
        xaxis=dict(tickfont=dict(color="#172033"), title=dict(font=dict(color="#172033")), gridcolor="#e5ebf2"),
        yaxis=dict(tickfont=dict(color="#172033"), title=dict(font=dict(color="#172033")), gridcolor="#d8e1ec"),
    )
    return fig


def output_growth_figure(metrics: dict[str, int]) -> go.Figure:
    labels = ["Sparse", "Dense", "Clean", "Mesh vertices"]
    values = [
        metrics["sparse_points"],
        metrics["dense_points"],
        metrics["clean_points"],
        metrics["mesh_vertices"],
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=["#2563eb", "#0f766e", "#12805c", "#7c3aed"]))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=30),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis_title="Count",
        font=dict(color="#172033"),
        xaxis=dict(tickfont=dict(color="#172033", size=12), title=dict(font=dict(color="#172033")), gridcolor="#e5ebf2"),
        yaxis=dict(tickfont=dict(color="#172033", size=12), title=dict(font=dict(color="#172033")), gridcolor="#d8e1ec"),
    )
    return fig


def safe_dataframe(data: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(data, **kwargs)
    except Exception:
        hide_index = kwargs.get("hide_index", False)
        st.markdown(data.to_html(index=not hide_index, escape=True), unsafe_allow_html=True)


def file_card(title: str, items: list[tuple[str, str]]) -> None:
    lines = "".join(
        f'<div class="file-line"><span>{label}</span><span class="mono">{value}</span></div>'
        for label, value in items
    )
    st.markdown(f'<div class="section-card"><h3>{title}</h3>{lines}</div>', unsafe_allow_html=True)


def render_main_navigation() -> None:
    pages = ["Guide", "Dataset", "Pipeline", "Outputs", "Viewer", "Metrics", "Exports"]
    cols = st.columns(len(pages))
    for col, page in zip(cols, pages):
        with col:
            if st.button(page, use_container_width=True, key=f"top_nav_{page}"):
                st.session_state["page"] = page
                st.rerun()


def hero(metrics: dict[str, int]) -> None:
    done, total = pipeline_progress()
    shown_done = min(done, total)
    live = load_live_status()
    is_live = live_running(live)
    live_label = "LIVE BACKEND RUNNING" if is_live else "BACKEND IDLE"
    live_dot = "live-dot" if is_live else "live-dot idle"
    message = live.get("last_message") or "Artifacts are read directly from the backend workspace."
    st.markdown(
        """
        <div class="app-hero">
          <div class="eyebrow">3D reconstruction operations dashboard</div>
          <div class="hero-title">S.P.E.C.T.R.A</div>
          <div class="hero-copy">
            Smartphone images move through frame selection, keyframe preparation, Depth Anything V2,
            sparse SfM, dense fusion, cleanup, meshing, evaluation, and Gaussian Splatting export.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="status-strip">
          <div><span class="{live_dot}"></span><strong>{live_label}</strong></div>
          <div class="status-message">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi"><div class="label">Pipeline completion</div><div class="value">{shown_done}/{total}</div></div>
          <div class="kpi"><div class="label">Raw images</div><div class="value">{fmt_int(metrics["raw_images"])}</div></div>
          <div class="kpi"><div class="label">Clean cloud points</div><div class="value">{fmt_int(metrics["clean_points"])}</div></div>
          <div class="kpi"><div class="label">Mesh faces</div><div class="value">{fmt_int(metrics["mesh_faces"])}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(done / max(total, 1), 1.0))
    render_main_navigation()


def render_sidebar(metrics: dict[str, int]) -> str:
    live = load_live_status()
    st.sidebar.markdown("## S.P.E.C.T.R.A")
    st.sidebar.caption("Backend-aware reconstruction console")
    pages = ["Overview", "Guide", "Dataset", "Pipeline", "Viewer", "Outputs", "Metrics", "Exports", "Resources"]
    if st.session_state["page"] not in pages:
        st.session_state["page"] = "Overview"
    page = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state["page"]),
    )
    st.session_state["page"] = page
    st.sidebar.divider()
    if live_running(live):
        st.sidebar.success("Live backend job running")
        st.sidebar.caption(live.get("last_message", "Waiting for backend update."))
        if st.sidebar.button("Stop backend", use_container_width=True, type="primary", key="sidebar_stop_backend"):
            stop_backend_from_ui(live)
    else:
        st.sidebar.info("Backend idle")
    st.session_state["auto_refresh"] = st.sidebar.toggle("Auto-refresh file status", value=st.session_state["auto_refresh"])
    st.sidebar.metric("Raw images", fmt_int(metrics["raw_images"]))
    st.sidebar.metric("Keyframes", fmt_int(metrics["keyframes"]))
    st.sidebar.metric("Depth maps", fmt_int(metrics["depth_maps"]))
    st.sidebar.metric("Pose records", fmt_int(metrics["poses"]))
    st.sidebar.divider()
    if st.sidebar.button("Rescan workspace", use_container_width=True):
        st.rerun()
    return page


def render_overview(metrics: dict[str, int]) -> None:
    render_compact_guide()
    render_upload_console(metrics, key_prefix="overview")
    render_compact_outputs(metrics)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("System workflow")
        for idx, stage in enumerate(STAGES, start=1):
            status = stage_status(stage)
            st.markdown(
                f"""
                <div class="stage-row">
                  <div class="stage-num">{idx}</div>
                  <div>
                    <div class="stage-name">{stage.name}</div>
                    <div class="small-muted mono">{' '.join(stage_command(stage))}</div>
                  </div>
                  <div class="stage-desc">{stage.description}</div>
                  <div><span class="pill {status}">{status.upper()}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.subheader("Current reconstruction")
        live = load_live_status()
        file_card(
            "Live backend communication",
            [
                ("Status", "running" if live_running(live) else "idle"),
                ("Current stage", live.get("current_stage") or "-"),
                ("Message", (live.get("last_message") or "No live job has started.")[:90]),
                ("Status file", str(LIVE_STATUS.relative_to(PROJECT_ROOT))),
                ("Log file", str(LIVE_LOG.relative_to(PROJECT_ROOT))),
            ],
        )
        file_card(
            "Artifact summary",
            [
                ("Sparse cloud", f'{fmt_int(metrics["sparse_points"])} points'),
                ("Dense cloud", f'{fmt_int(metrics["dense_points"])} points'),
                ("Clean cloud", f'{fmt_int(metrics["clean_points"])} points'),
                ("Mesh", f'{fmt_int(metrics["mesh_vertices"])} vertices / {fmt_int(metrics["mesh_faces"])} faces'),
                ("Gaussian export", "available" if has_output("gaussian") else "not ready"),
            ],
        )
        fig = output_growth_figure(metrics)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Readiness checklist")
        checks = readiness_checks(metrics)
        passed = sum(1 for _, _, ok, _ in checks if ok)
        st.progress(passed / max(len(checks), 1))
        st.caption(f"{passed}/{len(checks)} checks passing for a complete demonstration run.")
        for group, label, ok, detail in checks:
            status = "done" if ok else "pending"
            st.markdown(
                f"""
                <div class="file-line">
                  <span><strong>{group}</strong> - {label}</span>
                  <span><span class="pill {status}">{'OK' if ok else 'WAITING'}</span> <span class="mono">{detail}</span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_guide(metrics: dict[str, int]) -> None:
    st.subheader("First-time user guide")
    st.caption("This page is written for someone using S.P.E.C.T.R.A without opening the project folders or backend scripts.")

    steps = [
        (
            "1. Capture the scene",
            "Take 20 to 60 overlapping photos around the object or room. Keep the camera steady, avoid blur, and maintain strong overlap between consecutive images.",
            "Good input makes every backend stage more reliable.",
        ),
        (
            "2. Upload images",
            "Open Dataset, choose Real Dataset, upload PNG, JPG, or JPEG files, then save them. The dashboard writes them into data/raw_phone automatically.",
            "You do not need to copy files manually.",
        ),
        (
            "3. Start reconstruction",
            "Use Save and start live backend on the Dataset page, or Run full backend on the Pipeline page. The dashboard starts the backend runner and monitors it live.",
            "The frontend remains your control center.",
        ),
        (
            "4. Watch progress",
            "The Pipeline page shows the current stage, live log messages, stage status, and output timing. Auto-refresh starts automatically while the backend is running.",
            "Long stages like depth estimation can take time.",
        ),
        (
            "5. Inspect outputs",
            "Use Outputs to review generated clouds, meshes, depth previews, Gaussian export files, and file sizes. Use Viewer for interactive 3D inspection.",
            "This avoids opening raw folders during presentation.",
        ),
        (
            "6. Export results",
            "Use Exports to download point clouds, meshes, and Gaussian scene packages for external tools such as MeshLab, CloudCompare, Blender, or Gaussian Splatting.",
            "Only files that exist are shown as ready.",
        ),
    ]

    for title, body, note in steps:
        st.markdown(
            f"""
            <div class="section-card">
              <h3>{title}</h3>
              <div class="small-muted">{body}</div>
              <div style="margin-top:.55rem;color:var(--accent);font-weight:700;font-size:.86rem">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    with left:
        file_card(
            "Recommended capture checklist",
            [
                ("Minimum images", "8, but 20-60 is preferred"),
                ("Overlap", "About 70-80 percent"),
                ("Lighting", "Stable and bright"),
                ("Motion", "Move slowly, avoid blur"),
                ("Surfaces", "Avoid glass and heavy reflections"),
                ("Calibration", "Use the same zoom level as calibration"),
            ],
        )
    with right:
        file_card(
            "What the backend produces",
            [
        ("Selector output", "data/selected (quality mode)"),
                ("Prepared keyframes", "data/keyframes"),
                ("Depth maps", "data/results/depth"),
                ("Sparse and dense clouds", "data/results/*.ply"),
                ("Meshes", "mesh_poisson.ply / .obj"),
                ("Gaussian package", "data/gaussian_splatting_scene"),
            ],
        )


def render_compact_guide() -> None:
    st.markdown("### Start here")
    cols = st.columns(3)
    cards = [
        ("1. Choose images", "Use existing data/raw_phone images or upload new overlapping images."),
        ("2. Run", "Start the live backend. Quality mode begins at Smart Frame Selection; Fast mode begins at Prepare Keyframes."),
        ("3. Inspect", "Review point clouds, meshes, depth previews, and exports inside the frontend."),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="section-card">
                  <h3>{title}</h3>
                  <div class="small-muted">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_upload_console(metrics: dict[str, int], key_prefix: str = "main") -> None:
    live = load_live_status()
    is_live = live_running(live)
    st.markdown("### Upload and run reconstruction")
    st.caption("This is the frontend entry point: uploaded files are staged into data/raw_phone and can immediately trigger the backend pipeline.")
    uploaded = st.file_uploader(
        "Upload scene images",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Supported formats: JPG, JPEG, PNG.",
        key=f"{key_prefix}_uploader",
    )
    uploaded_count = len(uploaded or [])
    has_staged_images = metrics["raw_images"] > 0
    recommended_count = metrics["raw_images"] >= 8

    if st.session_state.get("ui_notice"):
        st.success(st.session_state["ui_notice"])

    selection_mode = st.radio(
        "Frame selection",
        ["Quality selection", "Fast direct"],
        index=["Quality selection", "Fast direct"].index(st.session_state["frame_selection_mode"]),
        horizontal=True,
        help=(
            "Quality selection runs Smart Frame Selection into data/selected before keyframes. "
            "Fast direct skips selection and prepares keyframes directly from data/raw_phone."
        ),
        key=f"{key_prefix}_frame_selection_mode",
    )
    st.session_state["frame_selection_mode"] = selection_mode

    profile_cols = st.columns(2)
    with profile_cols[0]:
        scene_profile = st.radio(
            "Scene profile",
            ["Lobby room", "Object capture"],
            index=["Lobby room", "Object capture"].index(st.session_state["scene_profile"]),
            horizontal=True,
            help="Lobby mode keeps floors/walls and disables object foreground masking. Object mode removes the support plane/background.",
            key=f"{key_prefix}_scene_profile",
        )
        st.session_state["scene_profile"] = scene_profile
    with profile_cols[1]:
        zoom_profile = st.radio(
            "Camera zoom",
            ["1x", "2x"],
            index=["1x", "2x"].index(st.session_state["zoom_profile"]),
            horizontal=True,
            help="Use the zoom calibration that matches capture. For lobby/rooms, try 1x first unless you know images were captured at 2x.",
            key=f"{key_prefix}_zoom_profile",
        )
        st.session_state["zoom_profile"] = zoom_profile

    col_a, col_b, col_c, col_d = st.columns([1.1, 1, 1.3, 1])
    with col_a:
        preserve = st.checkbox(
            "Preserve old outputs",
            value=st.session_state["preserve_outputs"],
            key=f"{key_prefix}_preserve",
        )
        st.session_state["preserve_outputs"] = preserve
    with col_b:
        if st.button("Save to raw_phone", disabled=not uploaded or is_live, use_container_width=True, key=f"{key_prefix}_save"):
            count = save_uploads(uploaded)
            st.session_state["ui_notice"] = f"Saved {count} image(s) into data/raw_phone. They are ready for backend processing."
            st.rerun()
    with col_c:
        run_disabled = is_live
        if st.button("Start backend", type="primary", disabled=run_disabled, use_container_width=True, key=f"{key_prefix}_run"):
            if uploaded_count:
                count = save_uploads(uploaded)
                st.session_state["ui_notice"] = f"Saved {count} image(s) and started reconstruction in {selection_mode.lower()} mode."
                start_backend_job("all")
            elif has_staged_images:
                st.session_state["ui_notice"] = f"Started reconstruction using staged data/raw_phone images in {selection_mode.lower()} mode."
                start_backend_job("all")
            else:
                st.session_state["ui_notice"] = "No images found in data/raw_phone. Upload scene images first, then start the backend."
            time.sleep(0.5)
            st.rerun()
    with col_d:
        if is_live:
            if st.button("Stop backend", use_container_width=True, key=f"{key_prefix}_stop_backend"):
                stop_backend_from_ui(live)
        elif st.button("Open outputs", use_container_width=True, key=f"{key_prefix}_outputs"):
            st.session_state["page"] = "Outputs"
            st.rerun()

    status_cols = st.columns(4)
    status_cols[0].metric("Staged raw images", fmt_int(metrics["raw_images"]))
    status_cols[1].metric("Selector output", fmt_int(metrics["selected_images"]))
    status_cols[2].metric("Keyframes", fmt_int(metrics["keyframes"]))
    status_cols[3].metric("Depth maps", fmt_int(metrics["depth_maps"]))

    if uploaded_count:
        st.info(f"{uploaded_count} new file(s) selected. Click Save to raw_phone or Start backend.")
    elif is_live:
        st.info(f"Backend running: {live.get('last_message', 'Processing...')}")
    elif recommended_count:
        st.success("A usable dataset is staged. You can run the backend from here or from the Pipeline page.")
    elif has_staged_images:
        st.warning(f"{metrics['raw_images']} image(s) are already staged in data/raw_phone. You can run them, but 8 or more overlapping images is recommended.")
    else:
        st.warning("No staged images found. Upload scene images here or place them in data/raw_phone before starting the backend.")

    staged_images = list_files(RAW_UPLOAD_DIR, IMG_SUFFIXES)
    if staged_images:
        with st.expander("Manage staged images", expanded=False):
            for row_start in range(0, len(staged_images), 3):
                cols = st.columns(3)
                for col, path in zip(cols, staged_images[row_start:row_start + 3]):
                    with col:
                        st.image(str(path), caption=f"{path.name} - {fmt_size(path)}", use_container_width=True)
                        if st.button("Delete", key=f"{key_prefix}_delete_{path.name}", disabled=is_live, use_container_width=True):
                            delete_staged_image(path, preserve_outputs=st.session_state["preserve_outputs"])
                            st.session_state["ui_notice"] = f"Deleted {path.name} from data/raw_phone."
                            st.rerun()


def render_compact_outputs(metrics: dict[str, int]) -> None:
    st.markdown("### Latest backend outputs")
    output_rows = [
        ("Sparse cloud", OUTPUTS["sparse"], f'{fmt_int(metrics["sparse_points"])} points'),
        ("Pose-aligned cloud", OUTPUTS["dense"], f'{fmt_int(metrics["pose_aligned_points"])} points'),
        ("Cleaned cloud", OUTPUTS["clean"], f'{fmt_int(metrics["clean_points"])} points'),
        ("Mesh PLY", OUTPUTS["mesh"], f'{fmt_int(metrics["mesh_vertices"])} vertices / {fmt_int(metrics["mesh_faces"])} faces'),
        ("Mesh OBJ", OUTPUTS["mesh_obj"], fmt_size(OUTPUTS["mesh_obj"])),
        ("Gaussian scene", OUTPUTS["gaussian"], fmt_size(OUTPUTS["gaussian"])),
    ]
    for label, path, detail in output_rows:
        ready = path.exists() and (any(path.rglob("*")) if path.is_dir() else path.stat().st_size > 0)
        status = "done" if ready else "pending"
        st.markdown(
            f"""
            <div class="stage-row">
              <div class="stage-num">{'OK' if ready else '--'}</div>
              <div>
                <div class="stage-name">{label}</div>
                <div class="small-muted mono">{path.relative_to(PROJECT_ROOT)}</div>
              </div>
              <div class="stage-desc">{detail}</div>
              <div><span class="pill {status}">{'READY' if ready else 'WAITING'}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if st.button("Inspect all backend outputs", use_container_width=True, key="overview_inspect_outputs"):
        st.session_state["page"] = "Outputs"
        st.rerun()


def render_dataset(metrics: dict[str, int]) -> None:
    live = load_live_status()
    is_live = live_running(live)
    st.subheader("Dataset ingestion")
    st.caption("Upload scene images here. The dashboard saves them to data/raw_phone, then the live backend runner can process them end to end.")
    mode = st.radio(
        "Dataset mode",
        ["Real Dataset", "Synthetic Dataset"],
        index=["Real Dataset", "Synthetic Dataset"].index(st.session_state["dataset_mode"]),
        horizontal=True,
    )
    st.session_state["dataset_mode"] = mode

    if mode == "Synthetic Dataset":
        if st.session_state["synthetic_scene"] not in SYNTHETIC_SCENES:
            st.session_state["synthetic_scene"] = SYNTHETIC_SCENES[0]
        scene = st.selectbox(
            "Synthetic scene",
            SYNTHETIC_SCENES,
            index=SYNTHETIC_SCENES.index(st.session_state["synthetic_scene"]),
        )
        st.session_state["synthetic_scene"] = scene
        x, y, z, c = synthetic_scene(scene)
        st.plotly_chart(cloud_figure(x, y, z, c, f"Synthetic preview - {scene}"), use_container_width=True)
        st.info("Synthetic scenes are dashboard baselines. Real backend reconstruction still uses files in data/raw_phone.")
        return

    st.markdown(
        f"""
        <div class="section-card">
          <h3>Supported upload formats</h3>
          <div class="small-muted">Images: PNG, JPG, JPEG. Files are staged in <span class="mono">data/raw_phone</span> and consumed directly by Prepare Keyframes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_upload_console(metrics, key_prefix="dataset")

    folder_cols = st.columns(3)
    with folder_cols[0]:
        file_card("Raw phone data", [("Path", str(RAW_UPLOAD_DIR.relative_to(PROJECT_ROOT))), ("Images", fmt_int(metrics["raw_images"]))])
    with folder_cols[1]:
        file_card("Selector output", [("Path", str(SELECTED_DIR.relative_to(PROJECT_ROOT))), ("Images", fmt_int(metrics["selected_images"]))])
    with folder_cols[2]:
        file_card("Prepared keyframes", [("Path", str(KEYFRAMES_DIR.relative_to(PROJECT_ROOT))), ("Images", fmt_int(metrics["keyframes"]))])

    raw_images = list_files(RAW_UPLOAD_DIR, IMG_SUFFIXES)
    if raw_images:
        st.subheader("Raw image inventory")
        df = pd.DataFrame(
            {
                "File": [p.name for p in raw_images],
                "Size": [fmt_size(p) for p in raw_images],
                "Modified": [p.stat().st_mtime for p in raw_images],
            }
        )
        safe_dataframe(df[["File", "Size"]], use_container_width=True, hide_index=True)
        with st.expander(f"View raw images ({len(raw_images)} files)", expanded=False):
            for row_start in range(0, len(raw_images), 3):
                cols = st.columns(3)
                for col, path in zip(cols, raw_images[row_start:row_start + 3]):
                    with col:
                        st.image(str(path), caption=f"{path.name} - {fmt_size(path)}", use_container_width=True)


def render_pipeline(metrics: dict[str, int]) -> None:
    live = load_live_status()
    is_live = live_running(live)
    st.subheader("Pipeline control")
    selection_mode = st.radio(
        "Frame selection",
        ["Quality selection", "Fast direct"],
        index=["Quality selection", "Fast direct"].index(st.session_state["frame_selection_mode"]),
        horizontal=True,
        disabled=is_live,
        help="Quality selection runs the smart selector before Prepare Keyframes. Fast direct skips it.",
        key="pipeline_frame_selection_mode",
    )
    st.session_state["frame_selection_mode"] = selection_mode
    profile_cols = st.columns(2)
    with profile_cols[0]:
        scene_profile = st.radio(
            "Scene profile",
            ["Lobby room", "Object capture"],
            index=["Lobby room", "Object capture"].index(st.session_state["scene_profile"]),
            horizontal=True,
            disabled=is_live,
            help="Lobby mode keeps floor/walls. Object mode removes plane/background.",
            key="pipeline_scene_profile",
        )
        st.session_state["scene_profile"] = scene_profile
    with profile_cols[1]:
        zoom_profile = st.radio(
            "Camera zoom",
            ["1x", "2x"],
            index=["1x", "2x"].index(st.session_state["zoom_profile"]),
            horizontal=True,
            disabled=is_live,
            help="If you cannot recall the phone zoom, run lobby with 1x first, then rerun fusion with 2x if the room bends badly.",
            key="pipeline_zoom_profile",
        )
        st.session_state["zoom_profile"] = zoom_profile
    run_cols = st.columns([1, 1, 1, 1, 2])
    with run_cols[0]:
        run_remaining = st.button("Run remaining live", type="primary", use_container_width=True, disabled=is_live)
    with run_cols[1]:
        run_all = st.button("Run full backend", use_container_width=True, disabled=is_live)
    with run_cols[2]:
        if st.button("Reset stage timings", use_container_width=True, disabled=is_live):
            st.session_state["stage_times"] = {}
            st.session_state["last_result"] = None
            st.session_state["failed_stage"] = None
            st.rerun()
    with run_cols[3]:
        if st.button("Stop backend", use_container_width=True, disabled=not is_live):
            stop_backend_from_ui(live)
    with run_cols[4]:
        st.caption("Stages now run through a live backend runner. Streamlit polls status and log files while the backend works.")

    if run_remaining:
        if metrics["raw_images"] > 0:
            start_backend_job("remaining")
            time.sleep(0.4)
            st.rerun()
        else:
            st.warning("No images found in data/raw_phone. Upload or stage images before starting the backend.")
    if run_all:
        if metrics["raw_images"] > 0:
            start_backend_job("all")
            time.sleep(0.4)
            st.rerun()
        else:
            st.warning("No images found in data/raw_phone. Upload or stage images before starting the backend.")

    if live:
        progress_done = 0
        selected = set(live.get("selected_stages", []))
        for key, data in live.get("stages", {}).items():
            if key in selected and data.get("status") == "done":
                progress_done += 1
        progress_total = max(len(selected), 1)
        st.progress(progress_done / progress_total)
        st.caption(f"Live runner PID: {live.get('runner_pid', '-')}. Started: {live.get('started_at', '-')}. Updated: {live.get('updated_at', '-')}.")

    for idx, stage in enumerate(STAGES, start=1):
        status = stage_status(stage)
        status_text = status.upper()
        left, action = st.columns([4.5, 1])
        with left:
            elapsed = st.session_state["stage_times"].get(stage.key)
            time_text = f" - {elapsed}s" if elapsed is not None else ""
            st.markdown(
                f"""
                <div class="stage-row">
                  <div class="stage-num">{idx}</div>
                  <div>
                    <div class="stage-name">{stage.name}</div>
                    <div class="small-muted mono">{' '.join(stage_command(stage))}</div>
                  </div>
                  <div class="stage-desc">{stage.description}{time_text}</div>
                  <div><span class="pill {status}">{status_text}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with action:
            if st.button("Run live", key=f"run_{stage.key}", use_container_width=True, disabled=is_live or stage.key in skipped_stage_keys()):
                start_backend_job("selected", [stage.key])
                time.sleep(0.4)
                st.rerun()

    st.subheader("Live backend log")
    live_log = read_live_log()
    if live_log:
        st.code(live_log, language="text")
    elif BACKEND_LOG.exists():
        lines = BACKEND_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-60:]
        st.code("\n".join(lines) or "No backend log entries yet.", language="text")
    else:
        st.code("No backend output yet.", language="text")

    fig = timing_figure()
    if fig is not None:
        st.subheader("Stage timing")
        st.plotly_chart(fig, use_container_width=True)


def render_viewer(metrics: dict[str, int]) -> None:
    st.subheader("Presentation viewer")
    st.caption("Use this page for focused rotation and inspection of one 3D artifact. Use Outputs for the full reconstruction journey and downloads.")
    source_map = {
        "3D reconstruction image": OUTPUTS["dense"],
        "Clean point cloud": OUTPUTS["clean"],
        "Pose-aligned dense cloud": OUTPUTS["dense"],
        "Legacy dense cloud": OUTPUTS["legacy_dense"],
        "Sparse cloud": OUTPUTS["sparse"],
        "Mesh vertices": OUTPUTS["mesh"],
        "Kitchen LiDAR scan": KITCHEN_LIDAR_PATH,
        "Synthetic preview": Path("__synthetic__"),
    }
    if st.session_state["viewer_source"] not in source_map:
        st.session_state["viewer_source"] = "3D reconstruction image"

    control_cols = st.columns([2.2, 1])
    with control_cols[0]:
        source = st.selectbox("Artifact", list(source_map), index=list(source_map).index(st.session_state["viewer_source"]))
    with control_cols[1]:
        st.write("")
        if st.button("Open Outputs journey", use_container_width=True, key="viewer_open_outputs"):
            st.session_state["page"] = "Outputs"
            st.rerun()
    st.session_state["viewer_source"] = source

    if source == "Synthetic preview":
        if st.session_state["synthetic_scene"] not in SYNTHETIC_SCENES:
            st.session_state["synthetic_scene"] = SYNTHETIC_SCENES[0]
        scene = st.selectbox("Scene", SYNTHETIC_SCENES, index=SYNTHETIC_SCENES.index(st.session_state["synthetic_scene"]))
        st.session_state["synthetic_scene"] = scene
        x, y, z, c = synthetic_scene(scene)
        st.plotly_chart(cloud_figure(x, y, z, c, scene), use_container_width=True)
        return

    path = source_map[source]
    open3d_style = source == "3D reconstruction image"
    cloud = load_cloud(path, use_rgb=open3d_style)
    if cloud is None:
        st.warning(missing_artifact_message(source, path))
        return

    x, y, z, c = cloud
    st.plotly_chart(cloud_figure(x, y, z, c, f"{source} - {len(x):,} rendered points", dark=open3d_style), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendered points", fmt_int(len(x)))
    c2.metric("X span", f"{(x.max() - x.min()):.2f}")
    c3.metric("Y span", f"{(y.max() - y.min()):.2f}")
    c4.metric("Z span", f"{(z.max() - z.min()):.2f}")
    st.caption(str(path.relative_to(PROJECT_ROOT)))


def render_outputs(metrics: dict[str, int]) -> None:
    st.subheader("Backend output workspace")
    st.caption("Inspect outputs produced by the live backend without opening project folders.")

    tabs = st.tabs(["3D artifacts", "Depth previews", "Frame sets", "Gaussian package", "Backend files"])

    with tabs[0]:
        st.markdown("#### Reconstruction journey before mesh")
        journey = [
            ("1. Sparse cloud", OUTPUTS["sparse"], "Camera matching and triangulated SfM feature points."),
            ("2. 3D reconstruction image", OUTPUTS["dense"], "Open3D-style view of the pose-aligned depth fusion before cleanup and mesh."),
            ("3. Clean point cloud", OUTPUTS["clean"], "Floating clusters removed before meshing."),
            ("4. Mesh", OUTPUTS["mesh"], "Surface reconstructed from the cleaned point cloud."),
        ]
        ready_journey = [(label, path, desc) for label, path, desc in journey if path.exists() and path.stat().st_size > 0]
        if ready_journey:
            labels = [label for label, _, _ in ready_journey]
            selected_label = st.radio("Show stage", labels, horizontal=True, key="journey_stage")
            selected_label, selected_path, selected_desc = next(item for item in ready_journey if item[0] == selected_label)
            st.caption(selected_desc)
            open3d_style = selected_label == "2. 3D reconstruction image"
            cloud = load_cloud(selected_path, use_rgb=open3d_style)
            if cloud:
                x, y, z, c = cloud
                st.plotly_chart(cloud_figure(x, y, z, c, selected_label, dark=open3d_style), use_container_width=True)
        else:
            st.info("The reconstruction journey preview will appear after Sparse SfM creates the first point cloud.")

        st.markdown("#### Point clouds and meshes")
        for label, path, description in POINT_OUTPUTS:
            ready = path.exists() and path.stat().st_size > 0 if path.is_file() else False
            left, mid, right = st.columns([2.2, 1.1, 1])
            with left:
                status = "done" if ready else "pending"
                st.markdown(
                    f"""
                    <div class="section-card">
                      <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center">
                        <div>
                          <h3>{label}</h3>
                          <div class="small-muted">{description}</div>
                          <div class="small-muted mono">{path.relative_to(PROJECT_ROOT)}</div>
                        </div>
                        <span class="pill {status}">{'READY' if ready else 'WAITING'}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with mid:
                if path.suffix.lower() == ".ply":
                    header = read_ply_header(path)
                    st.metric("Vertices", fmt_int(header["vertices"]))
                    if header["faces"]:
                        st.metric("Faces", fmt_int(header["faces"]))
                elif path.suffix.lower() == ".obj":
                    vertices, faces = count_obj(path)
                    st.metric("Vertices", fmt_int(vertices))
                    st.metric("Faces", fmt_int(faces))
            with right:
                st.metric("Size", fmt_size(path) if ready else "-")
                if ready:
                    st.download_button(
                        "Download",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="application/octet-stream",
                        key=f"outputs_download_{path.name}",
                        use_container_width=True,
                    )

        preview_options = [label for label, path, _ in POINT_OUTPUTS if path.exists() and path.suffix.lower() in {".ply", ".obj"}]
        if preview_options:
            viewer_path = st.selectbox("Preview 3D artifact", preview_options)
            selected = next(path for label, path, _ in POINT_OUTPUTS if label == viewer_path)
            cloud = load_cloud(selected)
            if cloud:
                x, y, z, c = cloud
                st.plotly_chart(cloud_figure(x, y, z, c, viewer_path), use_container_width=True)
        else:
            st.info("A 3D preview will appear here once a PLY or OBJ artifact exists.")

    with tabs[1]:
        st.markdown("#### Depth estimation previews")
        depth_pngs = sorted(DEPTH_DIR.glob("*_depth.png"))
        depth_npys = sorted(DEPTH_DIR.glob("*_depth.npy"))
        c1, c2 = st.columns(2)
        c1.metric("Depth arrays", fmt_int(len(depth_npys)))
        c2.metric("Preview images", fmt_int(len(depth_pngs)))
        if depth_pngs:
            preview_count = st.slider("Preview count", 1, min(12, len(depth_pngs)), min(6, len(depth_pngs)))
            cols = st.columns(3)
            for index, path in enumerate(depth_pngs[:preview_count]):
                with cols[index % 3]:
                    st.image(str(path), caption=path.name, use_column_width=True)
        else:
            st.info("Depth previews will appear here after the Depth Estimation stage.")

    with tabs[2]:
        st.markdown("#### Image flow through the backend")
        frame_sets = [
            ("Raw phone uploads", RAW_UPLOAD_DIR),
            ("Selector output", SELECTED_DIR),
            ("Prepared keyframes", KEYFRAMES_DIR),
        ]
        for title, folder in frame_sets:
            files = list_files(folder, IMG_SUFFIXES)
            with st.expander(f"{title} ({len(files)} files)", expanded=title == "Prepared keyframes"):
                if not files:
                    st.info("No files available yet.")
                    continue
                safe_dataframe(
                    pd.DataFrame({"File": [p.name for p in files], "Size": [fmt_size(p) for p in files]}),
                    use_container_width=True,
                    hide_index=True,
                )
                preview_cols = st.columns(4)
                for index, path in enumerate(files[:8]):
                    with preview_cols[index % 4]:
                        st.image(str(path), caption=path.name, use_column_width=True)

    with tabs[3]:
        st.markdown("#### Gaussian Splatting export package")
        ready = has_output("gaussian")
        if not ready:
            st.info("Gaussian export will appear after the Gaussian Export stage.")
        else:
            package_files = sorted([p for p in GAUSSIAN_SCENE_DIR.rglob("*") if p.is_file()])
            st.metric("Package files", fmt_int(len(package_files)))
            st.metric("Package size", fmt_size(GAUSSIAN_SCENE_DIR))
            safe_dataframe(
                pd.DataFrame(
                    {
                        "File": [str(p.relative_to(GAUSSIAN_SCENE_DIR)) for p in package_files],
                        "Size": [fmt_size(p) for p in package_files],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download Gaussian package",
                data=zip_directory(GAUSSIAN_SCENE_DIR),
                file_name="spectra_gaussian_splatting_scene.zip",
                mime="application/zip",
                use_container_width=True,
            )

    with tabs[4]:
        st.markdown("#### Backend file browser")
        candidates = [
            *sorted(RESULTS_DIR.glob("*")),
            *sorted(DEPTH_DIR.glob("*")),
        ]
        candidates = [p for p in candidates if p.is_file()]
        if candidates:
            safe_dataframe(
                pd.DataFrame(
                    {
                        "File": [str(p.relative_to(PROJECT_ROOT)) for p in candidates],
                        "Size": [fmt_size(p) for p in candidates],
                        "Modified": [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) for p in candidates],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No backend result files are available yet.")
        st.markdown("#### Live backend log")
        st.code(read_live_log() or "No live backend log yet.", language="text")


def render_metrics(metrics: dict[str, int]) -> None:
    st.subheader("Evaluation dashboard")
    record = build_evaluation_record(metrics)
    top = st.columns(4)
    top[0].metric("Reconstruction score", f'{st.session_state["manual_score"]}%')
    top[1].metric("RMS error", f'{st.session_state["manual_rms"]:.2f}px')
    top[2].metric("Noise reduction", f'{st.session_state["manual_noise"]}%')
    ready_export_count = sum(1 for _, _, p, _ in EXPORTS if p.exists() and (any(p.rglob("*")) if p.is_dir() else p.stat().st_size > 0))
    top[3].metric("Export readiness", f"{ready_export_count}/{len(EXPORTS)}")

    st.markdown("#### Artifact counts")
    counts = pd.DataFrame(
        [
            ("Raw images", metrics["raw_images"]),
            ("Selector output", metrics["selected_images"]),
            ("Prepared keyframes", metrics["keyframes"]),
            ("Depth maps", metrics["depth_maps"]),
            ("Pose records", metrics["poses"]),
            ("Sparse points", metrics["sparse_points"]),
            ("Dense points", metrics["dense_points"]),
            ("Clean points", metrics["clean_points"]),
            ("Mesh vertices", metrics["mesh_vertices"]),
            ("Mesh faces", metrics["mesh_faces"]),
        ],
        columns=["Metric", "Value"],
    )
    safe_dataframe(counts, use_container_width=True, hide_index=True)
    st.plotly_chart(output_growth_figure(metrics), use_container_width=True)

    st.markdown("#### Override report metrics")
    a, b, c = st.columns(3)
    with a:
        st.session_state["manual_score"] = st.slider("Reconstruction score", 0, 100, int(st.session_state["manual_score"]))
    with b:
        st.session_state["manual_rms"] = st.number_input("RMS error (px)", 0.0, 10.0, float(st.session_state["manual_rms"]), 0.01)
    with c:
        st.session_state["manual_noise"] = st.slider("Noise reduction", 0, 100, int(st.session_state["manual_noise"]))

    fig = timing_figure()
    if fig is not None:
        st.markdown("#### Processing times")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Evaluation export")
    st.download_button(
        "Download evaluation JSON",
        data=json.dumps(record, indent=2).encode("utf-8"),
        file_name="spectra_evaluation_record.json",
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("Preview evaluation record"):
        st.json(record)


def render_exports(metrics: dict[str, int]) -> None:
    st.subheader("Exports")
    for name, fmt, path, description in EXPORTS:
        ready = path.exists() and (any(path.rglob("*")) if path.is_dir() else path.stat().st_size > 0)
        left, right = st.columns([3.5, 1])
        with left:
            status = "done" if ready else "pending"
            st.markdown(
                f"""
                <div class="section-card">
                  <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center">
                    <div>
                      <h3>{name} <span class="small-muted">{fmt}</span></h3>
                      <div class="small-muted">{description}</div>
                      <div class="small-muted mono">{path.relative_to(PROJECT_ROOT)}</div>
                    </div>
                    <span class="pill {status}">{'READY' if ready else 'NOT READY'}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.metric("Size", fmt_size(path) if ready else "-")
            if ready:
                if path.is_dir():
                    st.download_button(
                        f"Download {fmt}",
                        data=zip_directory(path),
                        file_name=f"spectra_{path.name}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key=f"download_{path.name}_{fmt}",
                    )
                else:
                    st.download_button(
                        f"Download {fmt}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="application/octet-stream",
                        use_container_width=True,
                        key=f"download_{path.name}_{fmt}",
                    )

    st.info("GLTF is not generated by the current backend. The dashboard now reports only outputs that are actually mapped to backend files.")


def render_resources(metrics: dict[str, int]) -> None:
    st.subheader("Backend resources")
    file_card(
        "Project paths",
        [
            ("Project root", str(PROJECT_ROOT)),
            ("Dashboard", str((PROJECT_ROOT / "spectra_dashboard/main.py").relative_to(PROJECT_ROOT))),
            ("Results", str(RESULTS_DIR.relative_to(PROJECT_ROOT))),
            ("Backend log", str(BACKEND_LOG.relative_to(PROJECT_ROOT))),
            ("Final report", FINAL_REPORT_MD.name if FINAL_REPORT_MD.exists() else "not found"),
        ],
    )
    st.markdown("#### Dependency footprint")
    deps = []
    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.exists():
        deps = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines() if line.strip()]
    st.code("\n".join(deps) or "requirements.txt is empty", language="text")

    st.markdown("#### Synthetic baseline files")
    synthetic_files = sorted(SYNTHETIC_DIR.glob("*")) if SYNTHETIC_DIR.exists() else []
    safe_dataframe(
        pd.DataFrame(
            {
                "File": [path.name for path in synthetic_files],
                "Size": [fmt_size(path) for path in synthetic_files],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Report files")
    report_rows = []
    for label, path in REPORT_FILES:
        report_rows.append({"Label": label, "File": path.name, "Exists": path.exists(), "Size": fmt_size(path) if path.exists() else "-"})
    safe_dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    report_ready = [item for item in REPORT_FILES if item[1].exists()]
    if report_ready:
        st.markdown("#### Report downloads")
        for label, path in report_ready:
            st.download_button(
                f"Download {label}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/octet-stream",
                key=f"report_download_{path.name}",
                use_container_width=True,
            )

    st.markdown("#### Demonstration readiness")
    check_rows = [
        {"Area": group, "Check": label, "Status": "OK" if ok else "Waiting", "Detail": detail}
        for group, label, ok, detail in readiness_checks(metrics)
    ]
    safe_dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)


def main() -> None:
    init_state()
    live = load_live_status()
    if live:
        sync_stage_times_from_live(live)
    if st.session_state["auto_refresh"] or live_running(live):
        if st_autorefresh is not None:
            try:
                st_autorefresh(interval=2000, key="spectra_autorefresh")
            except Exception:
                st.caption("Auto-refresh is unavailable until pyarrow is installed.")
        else:
            st.caption("Auto-refresh is unavailable until streamlit-autorefresh is installed.")

    metrics = artifact_metrics()
    page = render_sidebar(metrics)
    hero(metrics)

    if page == "Overview":
        render_overview(metrics)
    elif page == "Guide":
        render_guide(metrics)
    elif page == "Dataset":
        render_dataset(metrics)
    elif page == "Pipeline":
        render_pipeline(metrics)
    elif page == "Viewer":
        render_viewer(metrics)
    elif page == "Outputs":
        render_outputs(metrics)
    elif page == "Metrics":
        render_metrics(metrics)
    elif page == "Exports":
        render_exports(metrics)
    elif page == "Resources":
        render_resources(metrics)


if __name__ == "__main__":
    main()
