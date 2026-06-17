from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.depth.depth_utils import load_camera_calibration


def load_sorted_image_paths(img_dir: Path) -> list[Path]:
    image_paths = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return image_paths


def load_pose_lookup(poses_path: Path) -> dict[int, dict[str, np.ndarray]]:
    with open(poses_path, "r", encoding="utf-8") as f:
        poses_data = json.load(f)

    pose_lookup = {
        int(item["frame"]): {
            "R": np.array(item["R"], dtype=np.float64),
            "t": np.array(item["t"], dtype=np.float64).reshape(3, 1),
        }
        for item in poses_data
    }
    if not pose_lookup:
        raise RuntimeError(f"No poses found in {poses_path}")
    return pose_lookup


def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Return COLMAP-style quaternion [qw, qx, qy, qz]."""
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

    q /= np.linalg.norm(q) + 1e-12
    if q[0] < 0:
        q = -q
    return q


def load_sparse_points(ply_path: Path) -> np.ndarray:
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing sparse cloud: {ply_path}")
    if o3d is None:
        raise RuntimeError("Open3D is required to export sparse points for Gaussian Splatting.")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) == 0:
        raise RuntimeError(f"Sparse cloud is empty: {ply_path}")
    return points


def undistort_and_copy_images(
    image_paths: list[Path],
    pose_lookup: dict[int, dict[str, np.ndarray]],
    out_images_dir: Path,
) -> tuple[np.ndarray, int, int, list[tuple[int, str]]]:
    sample = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f"Could not read image: {image_paths[0]}")
    h, w = sample.shape[:2]

    calibration = load_camera_calibration(w, h)
    print(f"Using camera calibration: {calibration.source}")
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        calibration.K,
        calibration.dist_coeffs,
        (w, h),
        alpha=0.0,
        newImgSize=(w, h),
    )

    exported = []
    out_images_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, img_path in enumerate(image_paths):
        if frame_idx not in pose_lookup:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        print(f"  [{frame_idx}] Processing frame (calibration source: {calibration.source})")

        if calibration.dist_coeffs.size > 0 and not np.allclose(calibration.dist_coeffs, 0.0):
            img_out = cv2.undistort(img, calibration.K, calibration.dist_coeffs, None, new_K)
        else:
            img_out = img

        out_name = img_path.name
        cv2.imwrite(str(out_images_dir / out_name), img_out)
        exported.append((frame_idx, out_name))

    if not exported:
        raise RuntimeError("No images with poses were exported.")

    return new_K, w, h, exported


def write_cameras_txt(cameras_path: Path, K: np.ndarray, width: int, height: int) -> None:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    with open(cameras_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}\n")


def write_images_txt(
    images_path: Path,
    exported_images: list[tuple[int, str]],
    pose_lookup: dict[int, dict[str, np.ndarray]],
) -> None:
    with open(images_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(exported_images)}\n")

        for image_id, (frame_idx, image_name) in enumerate(exported_images, start=1):
            R_wc = pose_lookup[frame_idx]["R"]
            t_wc = pose_lookup[frame_idx]["t"]

            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            qvec = quaternion_from_rotation_matrix(R_cw)
            tx, ty, tz = t_cw.reshape(-1)

            f.write(
                f"{image_id} "
                f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
                f"{tx:.12f} {ty:.12f} {tz:.12f} 1 {image_name}\n"
            )
            f.write("\n")


def write_points3d_txt(points_path: Path, points: np.ndarray) -> None:
    with open(points_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(points)}\n")
        for point_id, point in enumerate(points, start=1):
            x, y, z = point.tolist()
            f.write(f"{point_id} {x:.12f} {y:.12f} {z:.12f} 255 255 255 1.0\n")


def main() -> None:
    base_dir = Path("data/gaussian_splatting_scene")
    images_dir = base_dir / "images"
    sparse_dir = base_dir / "sparse" / "0"

    img_dir = Path("data/keyframes")
    poses_path = Path("data/results/camera_poses.json")
    sparse_cloud_path = Path("data/results/sparse_cloud.ply")

    if base_dir.exists():
        shutil.rmtree(base_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_sorted_image_paths(img_dir)
    pose_lookup = load_pose_lookup(poses_path)
    sparse_points = load_sparse_points(sparse_cloud_path)
    K, width, height, exported_images = undistort_and_copy_images(image_paths, pose_lookup, images_dir)

    write_cameras_txt(sparse_dir / "cameras.txt", K, width, height)
    write_images_txt(sparse_dir / "images.txt", exported_images, pose_lookup)
    write_points3d_txt(sparse_dir / "points3D.txt", sparse_points)

    print(f"Exported Gaussian Splatting dataset to: {base_dir}")
    print(f"Images exported: {len(exported_images)}")
    print(f"Sparse points exported: {len(sparse_points)}")
    print("Dataset layout:")
    print(f"  {images_dir}")
    print(f"  {sparse_dir / 'cameras.txt'}")
    print(f"  {sparse_dir / 'images.txt'}")
    print(f"  {sparse_dir / 'points3D.txt'}")


if __name__ == "__main__":
    main()
