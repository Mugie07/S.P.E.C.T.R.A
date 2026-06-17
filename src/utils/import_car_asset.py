from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "results" / "synthetic"
OUT_CLOUD = OUT_DIR / "car_dataset_cloud.ply"
OUT_META = OUT_DIR / "car_dataset_source.json"


def fallback_car_colors(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    p = (points - mins) / span
    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    colors = np.full((len(points), 3), [0.12, 0.36, 0.68], dtype=np.float64)
    tire_mask = z < 0.24
    glass_mask = (z > 0.58) & (np.abs(y - 0.5) > 0.22) & (x > 0.18) & (x < 0.78)
    light_mask = (x > 0.93) & (z > 0.30) & (z < 0.55)
    tail_mask = (x < 0.07) & (z > 0.30) & (z < 0.55)

    colors[tire_mask] = [0.03, 0.03, 0.035]
    colors[glass_mask] = [0.28, 0.56, 0.78]
    colors[light_mask] = [0.95, 0.90, 0.65]
    colors[tail_mask] = [0.75, 0.08, 0.08]

    shade = 0.72 + 0.28 * z
    colors *= shade[:, None]
    return np.clip(colors, 0.0, 1.0)


def normalize(points: np.ndarray) -> np.ndarray:
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    points = points - center
    scale = np.max(points.max(axis=0) - points.min(axis=0))
    if scale > 0:
        points = points / scale * 5.8
    points[:, 2] -= points[:, 2].min()
    return points


def load_as_point_cloud(path: Path, n_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) and len(mesh.triangles):
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if len(colors) != len(points) or not np.any(colors):
            colors = None
        return points, colors

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError(f"No vertices or points could be read from {path}")
    colors = np.asarray(pcd.colors)
    if len(points) > n_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), n_points, replace=False)
        points = points[idx]
        colors = colors[idx] if len(colors) >= idx.max() + 1 else None
    elif len(colors) != len(points) or not np.any(colors):
        colors = None
    return points, colors


def write_cloud(points: np.ndarray, colors: np.ndarray, output: Path) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
    o3d.io.write_point_cloud(str(output), pcd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an academic/downloaded car mesh or point cloud into the dashboard synthetic car PLY."
    )
    parser.add_argument("input", type=Path, help="Input car asset: .ply, .obj, .off, .stl, or another Open3D-readable file.")
    parser.add_argument("--source-name", default="External car dataset", help="Dataset/source name for metadata.")
    parser.add_argument("--source-url", default="", help="Dataset/source URL for metadata.")
    parser.add_argument("--points", type=int, default=140000, help="Number of points to sample when input is a mesh.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points, colors = load_as_point_cloud(input_path, args.points)
    points = normalize(points)
    if colors is None:
        colors = fallback_car_colors(points)
    write_cloud(points, colors, OUT_CLOUD)

    metadata = {
        "source_name": args.source_name,
        "source_url": args.source_url,
        "input_path": str(input_path),
        "output_path": str(OUT_CLOUD),
        "points": int(len(points)),
        "note": "Converted into a normalized RGB point cloud for the synthetic car dashboard preview.",
    }
    OUT_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_CLOUD.relative_to(PROJECT_ROOT)} ({len(points):,} points)")
    print(f"Wrote {OUT_META.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
