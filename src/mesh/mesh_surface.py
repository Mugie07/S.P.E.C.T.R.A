"""
mesh_surface.py
---------------
Converts the cleaned dense point cloud into a textured 3D surface mesh
using Poisson surface reconstruction.

Run after cleanup_cloud.py:
    python src/depth/mesh_surface.py

Input  : data/results/dense_fused_cloud_clean.ply
Output : data/results/mesh_poisson.ply   (mesh with vertex colours)
         data/results/mesh_poisson.obj   (exportable mesh for other tools)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Open3D is required. Install with: pip install open3d")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Input: cleaned point cloud from cleanup_cloud.py
# If clean cloud doesn't exist yet, falls back to the raw fused cloud
INPUT_PLY   = Path("data/results/dense_fused_cloud_clean.ply")
FALLBACK_PLY = Path("data/results/dense_fused_cloud_pose_aligned.ply")

OUTPUT_PLY  = Path("data/results/mesh_poisson.ply")
OUTPUT_OBJ  = Path("data/results/mesh_poisson.obj")

# ---------------------------------------------------------------------------
# Poisson reconstruction parameters
# ---------------------------------------------------------------------------

# Depth of the octree used for reconstruction.
# Higher = finer detail but slower and needs more points.
# 8  = good for small objects like a tin with ~50k points
# 9  = better detail, needs ~100k+ points
# 10 = high detail, needs dense cloud, slow
POISSON_DEPTH = 8

# After Poisson reconstruction, low-density vertices are trimmed.
# This removes the "watery" surface that Poisson adds around sparse areas.
# Range 0.0-1.0. Higher = more aggressive trimming.
# 0.01 = very conservative (keeps almost everything)
# 0.05 = balanced — recommended starting point
# 0.10 = aggressive (may remove thin parts of the object)
DENSITY_TRIM_QUANTILE = 0.05

# Normal estimation search radius — how many neighbours to use.
# Larger radius = smoother normals but slower.
NORMAL_RADIUS = 0.1
NORMAL_MAX_NN = 30

# Voxel size for optional pre-meshing downsample (set to None to skip)
# Downsampling before meshing speeds up Poisson and reduces noise in the mesh.
PRE_MESH_VOXEL = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the cleaned dense point cloud into a Poisson mesh."
    )
    parser.add_argument(
        "--no_view",
        action="store_true",
        help="Save mesh outputs without launching the Open3D viewer.",
    )
    return parser.parse_args()


def load_point_cloud() -> o3d.geometry.PointCloud:
    """Load the best available point cloud."""
    if INPUT_PLY.exists():
        print(f"Loading cleaned cloud: {INPUT_PLY}")
        pcd = o3d.io.read_point_cloud(str(INPUT_PLY))
    elif FALLBACK_PLY.exists():
        print(f"Clean cloud not found, using fallback: {FALLBACK_PLY}")
        pcd = o3d.io.read_point_cloud(str(FALLBACK_PLY))
    else:
        raise FileNotFoundError(
            f"No input point cloud found.\n"
            f"Expected: {INPUT_PLY} or {FALLBACK_PLY}\n"
            "Run fuse_depth_clouds_pose_aligned.py and cleanup_cloud.py first."
        )
    print(f"Loaded {len(pcd.points):,} points")
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Estimate and orient surface normals — required by Poisson reconstruction.
    Normals define which way each surface faces; without them Poisson cannot
    build a watertight mesh.
    """
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN,
        )
    )

    # Orient normals consistently so they all point outward from the surface.
    # Without this step normals may randomly point inward or outward, producing
    # a badly inverted mesh.
    print("Orienting normals (this may take a moment)...")
    pcd.orient_normals_consistent_tangent_plane(k=15)

    return pcd


def run_poisson(
    pcd: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Run Poisson surface reconstruction.
    Returns the raw mesh and per-vertex density values.
    Density values indicate how well each mesh vertex is supported by the
    input point cloud — low density = poorly supported, likely noise.
    """
    print(f"Running Poisson reconstruction (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=POISSON_DEPTH,
        width=0,
        scale=1.1,
        linear_fit=False,
    )
    print(f"Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    return mesh, np.asarray(densities)


def trim_low_density(
    mesh: o3d.geometry.TriangleMesh,
    densities: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """
    Remove mesh vertices that are poorly supported by the point cloud.
    Poisson reconstruction fills in gaps which creates a smooth closed surface,
    but the filled-in regions have low density scores. Trimming them removes
    the spurious surface added around sparse or empty areas.
    """
    threshold = np.quantile(densities, DENSITY_TRIM_QUANTILE)
    print(f"Trimming low-density vertices (threshold={threshold:.4f}, "
          f"quantile={DENSITY_TRIM_QUANTILE})...")

    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"After trimming: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")
    return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Standard mesh cleanup — removes degenerate triangles, duplicate vertices,
    and unreferenced geometry that can cause issues in downstream tools.
    """
    print("Cleaning mesh...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def main() -> None:
    args = parse_args()

    # --- Load ---
    pcd = load_point_cloud()

    # --- Optional pre-mesh downsample ---
    if PRE_MESH_VOXEL is not None:
        print(f"Pre-mesh voxel downsample (voxel={PRE_MESH_VOXEL})...")
        pcd = pcd.voxel_down_sample(PRE_MESH_VOXEL)
        print(f"After downsample: {len(pcd.points):,} pts")

    # --- Normals ---
    pcd = estimate_normals(pcd)

    # --- Poisson reconstruction ---
    mesh, densities = run_poisson(pcd)

    # --- Trim spurious low-density surface ---
    mesh = trim_low_density(mesh, densities)

    # --- Mesh cleanup ---
    mesh = clean_mesh(mesh)

    # --- Compute vertex normals for shading in viewer ---
    mesh.compute_vertex_normals()

    # --- Save outputs ---
    OUTPUT_PLY.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving PLY mesh -> {OUTPUT_PLY}")
    o3d.io.write_triangle_mesh(str(OUTPUT_PLY), mesh)

    print(f"Saving OBJ mesh -> {OUTPUT_OBJ}")
    o3d.io.write_triangle_mesh(str(OUTPUT_OBJ), mesh)

    print(f"\nFinal mesh: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles")
    print("Done.")

    if args.no_view:
        return

    # --- Visualise ---
    print("Launching viewer (close window to exit)...")
    vis = o3d.visualization.Visualizer()
    if vis.create_window(window_name="Poisson Mesh", width=900, height=700):
        vis.add_geometry(mesh)
        render = vis.get_render_option()
        render.mesh_show_back_face = True
        render.background_color = np.array([0.1, 0.1, 0.1])
        vis.run()
        vis.destroy_window()
    else:
        print("Warning: could not open viewer window.")


if __name__ == "__main__":
    main()
