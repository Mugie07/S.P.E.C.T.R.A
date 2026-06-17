from __future__ import annotations

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "results" / "synthetic"


def indoor_room_colors(points: np.ndarray, saturation: float = 1.25) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    p = (points - mins) / span
    x, y, z = p[:, 0], p[:, 1], p[:, 2]

    colors = np.full((len(points), 3), [0.86, 0.82, 0.74], dtype=np.float64)

    floor = z < 0.08
    ceiling = z > 0.92
    wall = ((x < 0.045) | (x > 0.955) | (y < 0.045) | (y > 0.955)) & ~floor & ~ceiling
    furniture = ~(floor | ceiling | wall)

    plank = 0.5 + 0.5 * np.sin((x * 22.0 + y * 4.0) * np.pi)
    colors[floor] = np.column_stack(
        [
            0.58 + 0.12 * plank[floor],
            0.43 + 0.08 * plank[floor],
            0.28 + 0.05 * plank[floor],
        ]
    )

    wall_tint = 0.5 + 0.5 * np.sin((x * 3.0 + y * 2.0) * np.pi)
    colors[wall] = np.column_stack(
        [
            0.86 + 0.05 * wall_tint[wall],
            0.84 + 0.04 * wall_tint[wall],
            0.78 + 0.04 * wall_tint[wall],
        ]
    )

    colors[ceiling] = [0.94, 0.93, 0.89]

    furniture_group = np.floor((x * 4.0) + (y * 3.0)).astype(int) % 4
    furniture_palette = np.array(
        [
            [0.48, 0.30, 0.18],
            [0.18, 0.34, 0.46],
            [0.56, 0.52, 0.45],
            [0.36, 0.24, 0.17],
        ]
    )
    colors[furniture] = furniture_palette[furniture_group[furniture]]

    light = 0.72 + 0.28 * (0.55 * z + 0.45 * (1.0 - y))
    colors *= light[:, None]

    gray = colors.mean(axis=1, keepdims=True)
    colors = gray + saturation * (colors - gray)
    return np.clip(colors, 0.0, 1.0)


def colorize_geometry(input_path: Path, output_path: Path) -> None:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if len(mesh.vertices):
        points = np.asarray(mesh.vertices)
        colors = indoor_room_colors(points)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_colors=True)
        return

    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError(f"No points or vertices found in {input_path}")
    pcd.colors = o3d.utility.Vector3dVector(indoor_room_colors(points))
    o3d.io.write_point_cloud(str(output_path), pcd)


def render_preview(input_path: Path, output_path: Path, sample_limit: int = 45000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0:
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
    if len(points) == 0:
        raise ValueError(f"No renderable points found in {input_path}")

    if len(points) > sample_limit:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(points), sample_limit, replace=False)
        points = points[idx]
        colors = colors[idx] if len(colors) >= len(idx) else indoor_room_colors(points)

    fig = plt.figure(figsize=(13, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=np.clip(colors, 0, 1), s=1.1, linewidths=0)
    ax.view_init(elev=22, azim=-52)
    ax.set_axis_off()
    ax.set_facecolor("#f4f2ed")
    fig.patch.set_facecolor("#f4f2ed")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(mins[2], mins[2] + 2 * radius)
    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def create_dense_point_cloud(mesh_path: Path, output_path: Path, n_points: int = 220000) -> None:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"No mesh triangles found in {mesh_path}")
    if len(mesh.vertex_colors) != len(mesh.vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(indoor_room_colors(np.asarray(mesh.vertices)))
    dense = mesh.sample_points_uniformly(number_of_points=n_points)
    dense.colors = o3d.utility.Vector3dVector(indoor_room_colors(np.asarray(dense.points), saturation=1.55))
    o3d.io.write_point_cloud(str(output_path), dense)


def main() -> None:
    inputs = [
        SYNTHETIC_DIR / "indoor_cloud.ply",
        SYNTHETIC_DIR / "indoor_mesh.ply",
    ]
    for input_path in inputs:
        if not input_path.exists():
            print(f"Missing: {input_path}")
            continue
        output_path = input_path.with_name(f"{input_path.stem}_realistic_color{input_path.suffix}")
        colorize_geometry(input_path, output_path)
        print(f"Wrote: {output_path.relative_to(PROJECT_ROOT)}")

    preview_source = SYNTHETIC_DIR / "indoor_cloud_realistic_color.ply"
    if preview_source.exists():
        preview_path = SYNTHETIC_DIR / "indoor_room_realistic_preview.png"
        render_preview(preview_source, preview_path)
        print(f"Wrote: {preview_path.relative_to(PROJECT_ROOT)}")

    mesh_source = SYNTHETIC_DIR / "indoor_mesh_realistic_color.ply"
    if mesh_source.exists():
        dense_path = SYNTHETIC_DIR / "indoor_room_dense_saturated.ply"
        create_dense_point_cloud(mesh_source, dense_path)
        print(f"Wrote: {dense_path.relative_to(PROJECT_ROOT)}")

        dense_preview_path = SYNTHETIC_DIR / "indoor_room_dense_saturated_preview.png"
        render_preview(dense_path, dense_preview_path, sample_limit=90000)
        print(f"Wrote: {dense_preview_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
