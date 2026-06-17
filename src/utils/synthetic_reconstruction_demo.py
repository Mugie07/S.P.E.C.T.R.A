"""
synthetic_reconstruction_demo.py
---------------------------------
Demonstrates the Recon3D pipeline on synthetic point cloud data.
Runs three scenes through the same meshing and evaluation logic used
for real captured data, producing side-by-side comparable results.

Scenes:
  1. Indoor room   — floor, walls, ceiling, furniture
  2. ALS terrain   — ground, vegetation, buildings
  3. Urban scene   — roads, vehicles, trees, buildings

Usage:
    python src/utils/synthetic_reconstruction_demo.py

Outputs (saved to data/results/synthetic/):
  - <scene>_cloud.ply       raw point cloud
  - <scene>_mesh.ply        Poisson surface mesh
  - <scene>_mesh.obj        exportable mesh
  - <scene>_metrics.json    evaluation metrics
  - synthetic_report.json   combined report across all scenes
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Open3D is required: pip install open3d")

# Import the synthetic data generators from Dr. Florent Poux's module.
# Place synthetic_data_generator.py in src/utils/ alongside this script.
from synthetic_data_generator import (
    generate_als_terrain,
    generate_urban_scene,
    generate_indoor_scene,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUT_DIR = Path("data/results/synthetic")

# ---------------------------------------------------------------------------
# Meshing parameters — same logic as mesh_surface.py
# ---------------------------------------------------------------------------

# Poisson depth per scene type — indoor needs less detail than terrain
POISSON_DEPTH = {
    "indoor":  8,
    "terrain": 9,
    "urban":   9,
}

# Density trim quantile — removes low-confidence Poisson surface
DENSITY_TRIM_QUANTILE = 0.05

# Normal estimation parameters
NORMAL_RADIUS = {
    "indoor":  0.3,
    "terrain": 3.0,
    "urban":   5.0,
}
NORMAL_MAX_NN = 30

# Voxel downsample before meshing — larger for bigger scenes
PRE_MESH_VOXEL = {
    "indoor":  0.05,
    "terrain": 0.5,
    "urban":   0.8,
}


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def build_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Convert numpy arrays to Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        np.clip(colors.astype(np.float64), 0.0, 1.0)
    )
    return pcd


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float,
) -> o3d.geometry.PointCloud:
    """Estimate and orient surface normals for Poisson reconstruction."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=NORMAL_MAX_NN,
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


def run_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Run Poisson surface reconstruction, return mesh and density values."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=0,
        scale=1.1,
        linear_fit=False,
    )
    return mesh, np.asarray(densities)


def trim_and_clean(
    mesh: o3d.geometry.TriangleMesh,
    densities: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Remove low-density vertices and clean degenerate geometry."""
    threshold = np.quantile(densities, DENSITY_TRIM_QUANTILE)
    mesh.remove_vertices_by_mask(densities < threshold)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pcd: o3d.geometry.PointCloud,
    mesh: o3d.geometry.TriangleMesh,
    scene_name: str,
    elapsed: float,
) -> dict:
    """
    Compute reconstruction quality metrics comparable to evaluate_reconstruction.py.
    These same metrics apply to both synthetic and real captured data.
    """
    points = np.asarray(pcd.points)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Point cloud metrics
    n_points = len(points)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_extent = np.asarray(bbox.get_extent())
    scene_volume = float(np.prod(bbox_extent)) if np.all(bbox_extent > 0) else 1.0
    point_density = n_points / scene_volume

    # Mesh metrics
    n_vertices = len(vertices)
    n_triangles = len(triangles)

    # Surface area
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    surface_area = float(mesh_copy.get_surface_area())

    # Normal consistency — measures how smooth the surface is.
    # Higher = more consistent normals = smoother surface.
    normals = np.asarray(mesh.vertex_normals)
    if len(normals) > 1:
        # Sample pairs of adjacent normals and measure dot product consistency
        sample_size = min(1000, len(normals) - 1)
        idx = np.random.choice(len(normals) - 1, sample_size, replace=False)
        dot_products = np.abs(
            np.sum(normals[idx] * normals[idx + 1], axis=1)
        )
        normal_consistency = float(np.mean(dot_products))
    else:
        normal_consistency = 0.0

    # Watertight check
    is_watertight = mesh.is_watertight()

    metrics = {
        "scene": scene_name,
        "reconstruction_time_s": round(elapsed, 2),
        "point_cloud": {
            "n_points": int(n_points),
            "point_density_per_m3": round(point_density, 4),
            "bounding_box_m": [round(float(e), 3) for e in bbox_extent],
        },
        "mesh": {
            "n_vertices": int(n_vertices),
            "n_triangles": int(n_triangles),
            "surface_area_m2": round(surface_area, 4),
            "normal_consistency": round(normal_consistency, 4),
            "is_watertight": bool(is_watertight),
        },
        "quality_summary": {
            "normal_consistency_pct": round(normal_consistency * 100, 1),
            "mesh_completeness": "watertight" if is_watertight else "open surface",
        }
    }
    return metrics


# ---------------------------------------------------------------------------
# Per-scene reconstruction pipeline
# ---------------------------------------------------------------------------

def reconstruct_scene(
    name: str,
    points: np.ndarray,
    colors: np.ndarray,
    out_dir: Path,
) -> dict:
    """
    Run the full reconstruction pipeline on one synthetic scene.
    Mirrors the real pipeline: point cloud → normals → Poisson → clean → evaluate.
    """
    print(f"\n{'='*55}")
    print(f"  Scene: {name.upper()}")
    print(f"{'='*55}")

    t_start = time.time()

    # --- Build point cloud ---
    print(f"  Building point cloud ({len(points):,} points)...")
    pcd = build_point_cloud(points, colors)

    # --- Voxel downsample ---
    voxel = PRE_MESH_VOXEL[name]
    print(f"  Voxel downsampling (voxel={voxel})...")
    pcd = pcd.voxel_down_sample(voxel)
    print(f"  After downsample: {len(pcd.points):,} points")

    # --- Statistical outlier removal ---
    print("  Removing outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After outlier removal: {len(pcd.points):,} points")

    # --- Normal estimation ---
    print("  Estimating normals...")
    pcd = estimate_normals(pcd, radius=NORMAL_RADIUS[name])

    # --- Save point cloud ---
    cloud_path = out_dir / f"{name}_cloud.ply"
    o3d.io.write_point_cloud(str(cloud_path), pcd)
    print(f"  Saved cloud -> {cloud_path}")

    # --- Poisson reconstruction ---
    depth = POISSON_DEPTH[name]
    print(f"  Running Poisson reconstruction (depth={depth})...")
    mesh, densities = run_poisson(pcd, depth)
    print(f"  Raw mesh: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    # --- Trim and clean ---
    print("  Trimming low-density surface...")
    mesh = trim_and_clean(mesh, densities)
    print(f"  Final mesh: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")

    t_elapsed = time.time() - t_start

    # --- Save mesh ---
    ply_path = out_dir / f"{name}_mesh.ply"
    obj_path = out_dir / f"{name}_mesh.obj"
    o3d.io.write_triangle_mesh(str(ply_path), mesh)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)
    print(f"  Saved mesh -> {ply_path}")
    print(f"  Saved mesh -> {obj_path}")

    # --- Evaluate ---
    print("  Computing metrics...")
    metrics = compute_metrics(pcd, mesh, name, t_elapsed)

    metrics_path = out_dir / f"{name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics -> {metrics_path}")

    # --- Print summary ---
    print(f"\n  --- {name.upper()} RESULTS ---")
    print(f"  Reconstruction time : {t_elapsed:.1f}s")
    print(f"  Points              : {metrics['point_cloud']['n_points']:,}")
    print(f"  Vertices            : {metrics['mesh']['n_vertices']:,}")
    print(f"  Triangles           : {metrics['mesh']['n_triangles']:,}")
    print(f"  Surface area        : {metrics['mesh']['surface_area_m2']:.2f} m²")
    print(f"  Normal consistency  : {metrics['quality_summary']['normal_consistency_pct']:.1f}%")
    print(f"  Mesh completeness   : {metrics['quality_summary']['mesh_completeness']}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nRecon3D — Synthetic Reconstruction Demo")
    print("Mugwanya Osbert | ISBAT University | Faculty of ICT")
    print("Supervisor: Mr. Umesh Kumar")
    print("="*55)

    all_metrics = []

    # --- Scene 1: Indoor Room ---
    print("\nGenerating indoor scene...")
    points, labels, colors = generate_indoor_scene(
        n_points=80000,
        room_dims=(10, 8, 3),
        seed=42,
    )
    metrics = reconstruct_scene("indoor", points, colors, OUT_DIR)
    all_metrics.append(metrics)

    # --- Scene 2: ALS Terrain ---
    print("\nGenerating ALS terrain scene...")
    points, labels, colors = generate_als_terrain(
        n_ground=50000,
        n_vegetation=20000,
        n_buildings=10000,
        extent=(200, 200),
        seed=42,
    )
    metrics = reconstruct_scene("terrain", points, colors, OUT_DIR)
    all_metrics.append(metrics)

    # --- Scene 3: Urban Scene ---
    print("\nGenerating urban scene...")
    points, labels, colors = generate_urban_scene(
        n_points=100000,
        extent=(300, 300),
        n_buildings=15,
        seed=42,
    )
    metrics = reconstruct_scene("urban", points, colors, OUT_DIR)
    all_metrics.append(metrics)

    # --- Combined report ---
    report = {
        "project": "Recon3D — Synthetic Validation",
        "author": "Mugwanya Osbert",
        "institution": "ISBAT University, Faculty of ICT",
        "supervisor": "Mr. Umesh Kumar",
        "description": (
            "Synthetic reconstruction results demonstrating pipeline correctness "
            "on clean ground-truth data. These results complement real-world "
            "captured reconstructions and establish an upper-bound quality baseline."
        ),
        "scenes": all_metrics,
    }

    report_path = OUT_DIR / "synthetic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print("SYNTHETIC DEMO COMPLETE")
    print(f"{'='*55}")
    print(f"All outputs saved to: {OUT_DIR}")
    print(f"Combined report    : {report_path}")
    print("\nFiles produced:")
    for scene in ["indoor", "terrain", "urban"]:
        print(f"  {scene}_cloud.ply")
        print(f"  {scene}_mesh.ply")
        print(f"  {scene}_mesh.obj")
        print(f"  {scene}_metrics.json")
    print("  synthetic_report.json")
    print("\nNext: open .ply files in Open3D viewer or MeshLab for visualisation.")


if __name__ == "__main__":
    main()