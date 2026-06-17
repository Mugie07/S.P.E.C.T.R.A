# S.P.E.C.T.R.A Mock Evaluation Data Document

Generated: 25 May 2026  
Program: S.P.E.C.T.R.A 3D Reconstruction Dashboard  
Entry point: `spectra_dashboard/main.py`

## 1. Purpose

This document provides mock evaluation data for demonstrating, testing, and reporting the S.P.E.C.T.R.A reconstruction workflow. It is designed to match the dashboard pages and backend stages used by the program:

- Upload and dataset ingestion
- Reconstruction pipeline execution
- 3D point cloud and mesh inspection
- Reconstruction comparison
- Metrics dashboard
- Export readiness

The values below are suitable for demo reports, supervisor review, UI testing, and sample analytics. Values marked as "mock" are representative and should be replaced with measured values during final evaluation.

## 2. Evaluation Dataset Summary

| Dataset ID | Dataset Type | Scene | Source | Input Frames | Selected Keyframes | Capture Notes |
|---|---|---|---|---:|---:|---|
| EVAL-REAL-001 | Real Dataset | Indoor kitchen | Smartphone images | 35 | 25 | Handheld overlapping image sequence |
| EVAL-SYN-001 | Synthetic Dataset | Indoor Room | Procedural synthetic scene | N/A | N/A | Clean controlled baseline |
| EVAL-SYN-002 | Synthetic Dataset | ALS Terrain | Procedural synthetic scene | N/A | N/A | Large outdoor terrain baseline |
| EVAL-SYN-003 | Synthetic Dataset | Urban Scene | Procedural synthetic scene | N/A | N/A | Building and street-like geometry |

## 3. Pipeline Stage Evaluation Data

The dashboard defines 9 reconstruction stages. The table below gives mock stage timing and status data for a complete real-dataset run.

| Stage No. | Stage Key | Stage Name | Expected Output | Mock Status | Mock Time (s) |
|---:|---|---|---|---|---:|
| 1 | keyframes | Keyframe Extraction | `data/selected/*.jpg` | Complete | 4.82 |
| 2 | prepare | Prepare Keyframes | `data/keyframes/*.jpg` | Complete | 3.17 |
| 3 | depth | Depth Estimation | `data/results/depth/*_depth.npy` and PNG previews | Complete | 96.44 |
| 4 | sfm | SfM Sparse Reconstruction | `data/results/sparse_cloud.ply` | Complete | 28.76 |
| 5 | fusion | Depth Fusion to Point Cloud | `data/results/dense_fused_cloud.ply` | Complete | 42.11 |
| 6 | cleanup | Noise Cleanup | `data/results/dense_fused_cloud_clean.ply` | Complete | 8.93 |
| 7 | mesh | Surface Meshing | `data/results/mesh_poisson.ply` and `.obj` | Complete | 34.58 |
| 8 | evaluation | Evaluation | Console evaluation report | Complete | 2.05 |
| 9 | gaussian | Gaussian Splatting Export | `data/gaussian_splatting_scene/` | Complete | 15.90 |

Total mock processing time: 236.76 seconds

## 4. Real Reconstruction Output Snapshot

These values are aligned with the current result files in `data/results`.

| Output File | Artifact Type | Vertex Count | Face Count | Mock Evaluation Status |
|---|---|---:|---:|---|
| `sparse_cloud.ply` | Sparse point cloud | 1,224 | N/A | Available |
| `dense_fused_cloud.ply` | Dense fused point cloud | 20,987 | N/A | Available |
| `dense_fused_cloud_clean.ply` | Cleaned dense point cloud | 14,683 | N/A | Available |
| `dense_fused_cloud_pose_aligned.ply` | Pose-aligned dense point cloud | 88,259 | N/A | Available |
| `mesh_poisson.ply` | Poisson surface mesh | 35,196 | 70,077 | Available |
| `mesh_poisson.obj` | OBJ surface mesh | Mock: 35,196 | Mock: 70,077 | Available |

## 5. Mock Quality Metrics

| Metric | Value | Unit | Interpretation |
|---|---:|---|---|
| Reconstruction score | 90 | % | Good reconstruction confidence for a smartphone-based pipeline |
| RMS reprojection error | 0.74 | pixels | Acceptable image-space alignment |
| Noise reduction | 31 | % | Moderate cleanup while preserving scene structure |
| Coverage | 92 | % | Strong multi-view coverage from selected keyframes |
| Overlap quality | 86 | % | Sufficient cross-view overlap for SfM and fusion |
| Density score | 88 | % | Dense output is suitable for visual inspection and meshing |
| Camera pose recovery | 25 / 25 | frames | All selected keyframes estimated successfully |
| Export readiness | 4 / 5 | formats | PLY, OBJ, mesh PLY, and Gaussian scene available; GLTF is listed but not mapped in current outputs |

## 6. Synthetic Baseline Metrics

The synthetic metrics below come from `data/results/synthetic/*_metrics.json` and represent clean controlled baselines.

| Scene | Reconstruction Time (s) | Points | Point Density | Bounding Box (m) | Mesh Vertices | Mesh Triangles | Normal Consistency | Mesh Completeness |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Indoor Room | 10.85 | 54,520 | 220.9476 per m3 | 10.070 x 8.069 x 3.037 | 88,040 | 173,014 | 69.8% | Open surface |
| ALS Terrain | 12.47 | 67,419 | 0.0754 per m3 | 199.990 x 199.997 x 22.347 | 93,679 | 183,445 | 91.5% | Open surface |
| Urban Scene | 12.52 | 61,161 | 0.0172 per m3 | 300.975 x 299.908 x 39.305 | 149,734 | 295,225 | 85.0% | Open surface |

## 7. Comparative Evaluation Data

| Comparison Mode | Left Output | Right Output | Mock Finding |
|---|---|---|---|
| Sparse vs Dense | `sparse_cloud.ply` | `dense_fused_cloud.ply` | Dense fusion increases point count from 1,224 to 20,987 points |
| Raw vs Cleaned | `dense_fused_cloud_pose_aligned.ply` | `dense_fused_cloud_clean.ply` | Cleanup removes outliers and reduces noisy points from 88,259 to 14,683 |
| Mesh vs Gaussian | `mesh_poisson.ply` | `data/gaussian_splatting_scene/` | Mesh supports geometry editing; Gaussian scene supports neural-view rendering |
| Synthetic vs Real | Synthetic baselines | Smartphone reconstruction | Synthetic data gives upper-bound performance under clean conditions |

## 8. Mock Evaluation Record

```json
{
  "evaluation_id": "SPECTRA-EVAL-2026-05-25-001",
  "program": "S.P.E.C.T.R.A",
  "dataset": {
    "id": "EVAL-REAL-001",
    "type": "Real Dataset",
    "scene": "Indoor kitchen",
    "input_frames": 35,
    "selected_keyframes": 25
  },
  "pipeline": {
    "stages_total": 9,
    "stages_complete": 9,
    "total_time_s": 236.76,
    "status": "Complete"
  },
  "outputs": {
    "sparse_cloud_points": 1224,
    "dense_fused_points": 20987,
    "clean_cloud_points": 14683,
    "pose_aligned_points": 88259,
    "mesh_vertices": 35196,
    "mesh_faces": 70077
  },
  "quality": {
    "reconstruction_score_pct": 90,
    "rms_error_px": 0.74,
    "noise_reduction_pct": 31,
    "coverage_pct": 92,
    "overlap_pct": 86,
    "density_pct": 88
  },
  "exports": {
    "ply_point_cloud": true,
    "obj_mesh": true,
    "ply_mesh": true,
    "gltf_web_3d": false,
    "gaussian_scene": true
  }
}
```

## 9. Acceptance Criteria

| Criterion | Target | Mock Result | Pass/Fail |
|---|---:|---:|---|
| Minimum input images | 8 images | 35 images | Pass |
| Minimum selected keyframes | 8 frames | 25 frames | Pass |
| Sparse cloud generated | File exists | Available | Pass |
| Dense cloud generated | File exists | Available | Pass |
| Cleaned cloud generated | File exists | Available | Pass |
| Mesh generated | File exists | Available | Pass |
| Reconstruction score | >= 80% | 90% | Pass |
| RMS error | <= 1.00 px | 0.74 px | Pass |
| Noise reduction | >= 20% | 31% | Pass |
| Export package readiness | >= 3 formats | 4 formats | Pass |

## 10. Notes for Final Evaluation

- Replace mock timing values with `st.session_state["stage_times"]` from an actual dashboard run.
- Replace mock quality values with measured values from the evaluation script or a more detailed benchmark tool.
- The current dashboard lists GLTF export as an expected format, but `BACKEND_OUTPUTS` does not currently map a GLTF file path.
- The synthetic scenes are useful as upper-bound baselines because they avoid motion blur, exposure changes, and real camera calibration errors.
- The real-dataset values are suitable for a demonstration but should be validated with visual inspection in the 3D Viewer and external tools such as MeshLab or CloudCompare.
