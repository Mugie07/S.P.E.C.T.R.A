from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "SPECTRA_FINAL_60_PAGE_REPORT.md"
OUT = ROOT / "print_support" / "SPECTRA_FINALZ_REDUCED_PRINT.md"


def replace_section(text, heading, replacement, next_heading_level="##"):
    pattern = re.compile(
        rf"(^## {re.escape(heading)}\n)(.*?)(?=^## |\Z)",
        flags=re.M | re.S,
    )
    return pattern.sub(rf"\1\n{replacement.strip()}\n\n", text)


def remove_code_fences(text):
    return re.sub(r"\n```(?:text|bash|json|python|mermaid)?\n.*?\n```\n", "\n", text, flags=re.S)


def compact_appendices(text):
    appendix_start = text.find("# Appendices")
    if appendix_start == -1:
        return text
    main = text[:appendix_start].rstrip()
    appendices = """# Appendices

## Appendix A: Summary of Supporting Project Files

The printed report keeps the appendix concise. Full source code, detailed JSON records, environment files, and Mermaid diagram definitions remain available in the `recon3d` project workspace. The most relevant supporting files are summarized below.

| Evidence Area | File or Folder | Purpose |
|---|---|---|
| Dashboard | `spectra_dashboard/main.py` | Streamlit interface and page navigation |
| Backend runner | `spectra_dashboard/backend_runner.py` | Executes pipeline stages from the dashboard |
| Frame selection | `src/utils/smart_frame_selector.py` | Selects useful frames from raw captures |
| Depth estimation | `src/depth/midas_depth.py` | Runs Depth Anything V2 and stores depth previews |
| Sparse reconstruction | `src/sfm/sparse_recon_live.py` | Estimates camera motion and sparse points |
| Dense fusion | `src/fusion/depth_fusion.py` | Fuses depth maps into a dense point cloud |
| Cleanup | `src/utils/cleanup.py` | Applies DBSCAN-based outlier removal |
| Meshing | `src/mesh/mesh_surface.py` | Produces Poisson mesh outputs |
| Evaluation | `src/utils/evaluate_reconstruction.py` | Checks output availability and metrics |
| Export | `src/export/gaussian_export.py` | Prepares Gaussian Splatting dataset files |

## Appendix B: Backend Command Summary

The full pipeline can be reproduced by running the frame selection, keyframe preparation, depth estimation, sparse reconstruction, dense fusion, cleanup, meshing, evaluation, and Gaussian export scripts in order. The printed appendix avoids listing long command and code blocks because these files are already included in the submitted project folder.

## Appendix C: Evaluation Checklist

| Check | Evidence Required |
|---|---|
| Raw dataset available | Image count in `data/raw_phone` |
| Selected frames available | Image count in `data/selected` |
| Keyframes available | Image count in `data/keyframes` |
| Depth maps available | `.npy` and `.png` outputs in `data/results/depth` |
| Camera poses available | `camera_poses.json` exists |
| Sparse cloud valid | `sparse_cloud.ply` exists and contains points |
| Dense cloud valid | Dense point cloud exists |
| Cleaned cloud valid | Cleaned output exists after DBSCAN filtering |
| Mesh valid | PLY and OBJ mesh outputs exist |
| Gaussian export valid | Export folder contains images and sparse files |

## Appendix D: Final Submission Notes

Before final submission, the printed copy should be checked for page numbering, figure numbering, table numbering, signatures, and any university-specific declaration fields. Long JSON schemas, full environment files, and conversion commands should remain in the digital project submission instead of the printed report.
"""
    return main + "\n\n" + appendices + "\n"


def main():
    text = SRC.read_text(encoding="utf-8")

    # Remove explicit page markers from the source version so pagination is controlled by Word.
    text = re.sub(r"\n\*\*Page \d+\*\*\n\n\\pagebreak\n", "\n\n", text)

    replacements = {
        "2.2 Foundations of 3D Reconstruction": """
3D reconstruction is the process of estimating the shape and structure of a real or synthetic scene in three dimensions. Outputs may appear as sparse point clouds, dense point clouds, meshes, textured surfaces, or neural rendering datasets. For this project, the most relevant forms are sparse point clouds for camera geometry, dense point clouds for visible scene coverage, meshes for surface representation, and Gaussian Splatting export files for later neural rendering workflows.

S.P.E.C.T.R.A follows a passive image-based approach because it uses ordinary monocular images rather than LiDAR, structured light, or stereo hardware. This choice supports the aim of making reconstruction more accessible using devices such as smartphones and webcams, although it also makes the final quality dependent on image overlap, texture, calibration, pose stability, and depth estimation.
""",
        "2.3 Photogrammetry and Smartphone-Based Capture": """
Photogrammetry estimates geometry from overlapping photographs. In S.P.E.C.T.R.A, this idea is adapted for smartphone capture, where the user records multiple views of a scene and the system selects useful frames for reconstruction. Smartphones are suitable for a student prototype because they are affordable, portable, and widely available, but they also introduce practical limitations such as blur, rolling-shutter effects, changing exposure, and imperfect calibration.
""",
        "2.4 Structure from Motion": """
Structure from Motion estimates camera movement and sparse 3D structure from multiple overlapping images. It detects shared image features, matches them across views, estimates relative camera pose, and triangulates sparse 3D points. In S.P.E.C.T.R.A, the SfM stage uses ORB features, descriptor matching, essential matrix estimation, RANSAC, pose recovery, and triangulation to produce `sparse_cloud.ply` and `camera_poses.json`.
""",
        "2.5 ORB Feature Detection and Matching": """
ORB is used because it is fast, available in OpenCV, and suitable for a lightweight prototype. S.P.E.C.T.R.A uses ORB during frame selection to reject weak images and during sparse reconstruction to recover feature correspondences for pose estimation. Its limitation is that it performs poorly on blurred, textureless, reflective, or repetitive surfaces, which directly affects camera stability and sparse reconstruction quality.
""",
        "2.6 Camera Calibration and Intrinsic Parameters": """
Camera calibration provides intrinsic parameters such as focal length and principal point, which are required for back-projecting image pixels into 3D space. S.P.E.C.T.R.A stores calibration resources and camera intrinsics in the `data` folder, with fallback estimates when exact calibration is unavailable. This is practical for demonstration, but accurate reconstruction should use calibration matching the exact camera, resolution, zoom level, and capture mode.
""",
        "2.8 Point Cloud Generation and Fusion": """
Point cloud generation converts image evidence into 3D points. S.P.E.C.T.R.A first produces a sparse point cloud from feature triangulation, then uses depth maps, keyframe colors, camera poses, and intrinsics to create denser scene geometry. The dense fusion stage back-projects depth pixels into camera coordinates, transforms them into a shared world coordinate system, attaches color values, and writes the resulting cloud for cleanup and meshing.
""",
        "2.10 Mesh Reconstruction and Surface Generation": """
A mesh converts point-based reconstruction into a surface representation that can be viewed, edited, or exported to 3D tools. S.P.E.C.T.R.A uses Poisson surface reconstruction through Open3D to generate PLY and OBJ mesh outputs. This improves visual inspection, but mesh quality still depends on the density, noise level, and completeness of the fused point cloud.
""",
        "2.16 Conceptual Framework": """
The conceptual framework links four layers of the project: image capture, reconstruction processing, dashboard interaction, and evaluation/export. Raw smartphone images are converted into selected keyframes, depth maps, camera poses, point clouds, meshes, metrics, and Gaussian Splatting export files. The dashboard connects these stages so that the workflow can be demonstrated and evaluated from one interface.
""",
        "3.16 Visualization Method": """
Visualization is handled through dashboard pages that load available reconstruction outputs from `data/results`. The viewer supports inspection of point cloud, mesh, and comparison outputs so that users can understand the effect of each stage without switching between many separate tools.
""",
        "3.17 Metrics Method": """
The metrics page summarizes stage completion, timing, point counts, mesh availability, quality indicators, and export readiness. In the current report, some values are mock evaluation values used to demonstrate the reporting structure, while synthetic baseline metrics are read from generated JSON files.
""",
        "3.18 Export Method": """
The export page checks whether reconstruction outputs are ready for downstream use. Supported outputs include point clouds, mesh files, and a Gaussian Splatting-ready scene folder. GLTF remains a future enhancement because it is listed as expected but not yet mapped to a produced backend file.
""",
        "4.12 Dependency Discussion": """
The implementation uses Streamlit, NumPy, Open3D, OpenCV, PyTorch, Transformers, Plotly, Pandas, and image-processing helpers. The current dependency list should therefore be expanded before final packaging so that another user can reproduce the full dashboard and backend pipeline more reliably.
""",
        "5.4.2 Dependency Improvements": """
The dependency file should be updated to include all libraries imported by the dashboard and backend scripts, including OpenCV, PyTorch, Transformers, Pandas, Plotly, Pillow, and Open3D. This would make installation clearer and reduce setup errors during demonstration or future development.
""",
    }
    for heading, replacement in replacements.items():
        text = replace_section(text, heading, replacement)

    # Insert real project visuals in Chapter 4. These replace some explanation rather than adding generic decoration.
    text = text.replace(
        "## 4.2 Pipeline Timing Results\n",
        "## 4.2 Pipeline Timing Results\n\n![Nine-Stage Pipeline Timing Profile](figure_pipeline_timing_profile.png)\n\n",
    )
    text = text.replace(
        "## 4.3 Real Reconstruction Output Snapshot\n",
        "## 4.3 Real Reconstruction Output Snapshot\n\n![Real S.P.E.C.T.R.A Pipeline Evidence from Capture to Reconstruction Preview](figure_real_pipeline_evidence.png)\n\n",
    )
    text = text.replace(
        "## 4.5 Sparse vs Dense Output\n",
        "## 4.5 Sparse vs Dense Output\n\n![Top-View Projection of Real Reconstruction Outputs](figure_real_geometry_projection.png)\n\n",
    )
    text = text.replace(
        "## 4.8 Synthetic Baseline Results\n",
        "## 4.8 Synthetic Baseline Results\n\n![Synthetic Baseline Reconstruction Comparison](figure_synthetic_baseline_comparison.png)\n\n",
    )

    text = compact_appendices(text)

    # Keep the main report readable by removing remaining fenced code blocks after summaries are in place.
    text = remove_code_fences(text)

    OUT.write_text(text, encoding="utf-8")
    print(OUT)
    print("words", len(re.findall(r"\b[\w'-]+\b", text)))


if __name__ == "__main__":
    main()
