from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.depth.depth_utils import (
    CameraCalibration,
    compute_sequence_distance_range,
    load_camera_calibration,
    load_midas_distance_map,
    write_point_cloud_ply,
)


def depth_to_points(
    depth: np.ndarray,
    color: np.ndarray,
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pixel_points = np.stack((xs.reshape(-1), ys.reshape(-1)), axis=-1).astype(np.float32).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(
        pixel_points,
        calibration.K,
        calibration.dist_coeffs,
    ).reshape(-1, 2).astype(np.float32)

    z = depth.astype(np.float32)
    x = rays[:, 0].reshape(h, w) * z
    y = rays[:, 1].reshape(h, w) * z

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3) / 255.0

    mask = np.isfinite(points).all(axis=1)
    mask &= (z.reshape(-1) > np.percentile(z[z > 0], 5)) if np.any(z > 0) else False
    mask &= (z.reshape(-1) < np.percentile(z[z > 0], 95)) if np.any(z > 0) else False

    points = points[mask]
    colors = colors[mask]

    return points, colors


def find_first_image_depth_pair(
    image_dir: Path,
    depth_dir: Path,
) -> tuple[Path, Path]:
    image_paths = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpeg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    for image_path in image_paths:
        depth_path = depth_dir / f"{image_path.stem}_depth.npy"
        if depth_path.exists():
            return image_path, depth_path

    raise FileNotFoundError(
        f"No matching depth maps found in {depth_dir} for images in {image_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the first matching keyframe/depth pair to a dense point cloud."
    )
    parser.add_argument(
        "--no_view",
        action="store_true",
        help="Save outputs without launching the Open3D viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_path, depth_path = find_first_image_depth_pair(
        Path("data/keyframes"),
        Path("data/results/depth"),
    )
    out_path = Path("data/results") / f"{img_path.stem}_dense_cloud.ply"

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("Could not read image.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_h, image_w = img_rgb.shape[:2]
    calibration = load_camera_calibration(image_w, image_h)
    print(f"Using camera calibration: {calibration.source}")
    distance_range = compute_sequence_distance_range([depth_path])
    depth = load_midas_distance_map(depth_path, distance_range=distance_range)
    if depth is None:
        raise RuntimeError("Could not load depth map.")

    points, colors = depth_to_points(depth, img_rgb, calibration)

    if o3d is None:
        print("Open3D not installed; saving raw dense cloud without visualization.")
        write_point_cloud_ply(out_path, points, colors)
        print(f"Saved dense point cloud to: {out_path}")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved dense point cloud to: {out_path}")

    if not args.no_view:
        o3d.visualization.draw_geometries([pcd], window_name="Dense Point Cloud from Depth")


if __name__ == "__main__":
    main()
