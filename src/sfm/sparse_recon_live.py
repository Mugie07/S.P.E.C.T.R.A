from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import sys
import argparse

import json
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

MIN_DENSE_FUSION_POSE_INLIERS = 25
MIN_DENSE_FUSION_SCALE_SUPPORT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sparse camera poses and a sparse point cloud from keyframes."
    )
    parser.add_argument(
        "--no_view",
        action="store_true",
        help="Save outputs without launching the Open3D live viewer.",
    )
    return parser.parse_args()


def load_image_paths(folder: str = "data/keyframes") -> List[Path]:
    p = Path(folder)
    paths = sorted(
        list(p.glob("*.jpg")) +
        list(p.glob("*.png")) +
        list(p.glob("*.jpeg"))
    )

    print(f"DEBUG: Found {len(paths)} image files")

    if not paths:
        raise FileNotFoundError(f"No images found in {folder}.")

    return paths


def load_images(paths: List[Path]) -> tuple[List[Path], List[np.ndarray]]:
    valid_paths = []
    imgs = []
    for fp in paths:
        print(f"Loading: {fp.name}")
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Failed to read {fp.name}")
            continue
        valid_paths.append(fp)
        imgs.append(img)

    if not imgs:
        raise RuntimeError("Images exist but could not be read.")

    return valid_paths, imgs


def orb_features(img: np.ndarray, nfeatures: int = 6000) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10,
    )
    kps, des = orb.detectAndCompute(gray, None)
    return kps, des


def match_descriptors(des1: np.ndarray, des2: np.ndarray, ratio: float = 0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    good = sorted(good, key=lambda m: m.distance)
    return good


def to_points(kps1, kps2, matches) -> Tuple[np.ndarray, np.ndarray]:
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def undistort_points(
    pts: np.ndarray,
    calibration: CameraCalibration,
) -> np.ndarray:
    if calibration.dist_coeffs.size == 0 or np.allclose(calibration.dist_coeffs, 0.0):
        return pts

    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, calibration.K, calibration.dist_coeffs, P=calibration.K)
    return undistorted.reshape(-1, 2)


def triangulate_points(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / (pts4d[3] + 1e-12)).T
    return pts3d


def filter_points_basic(pts3d: np.ndarray, max_abs: float = 200.0) -> np.ndarray:
    m = np.isfinite(pts3d).all(axis=1)
    pts = pts3d[m]
    m2 = (np.abs(pts) < max_abs).all(axis=1)
    return pts[m2]


def estimate_translation_scale(
    pts3d: np.ndarray,
    depth_map: np.ndarray | None,
    calibration: CameraCalibration,
) -> tuple[float, int]:
    """
    Estimate translation scale by matching triangulated depth to the shared
    pseudo-metric depth map used later during dense fusion.
    """
    if depth_map is None or len(pts3d) < 10:
        return 1.0, 0

    h, w = depth_map.shape
    valid = pts3d[:, 2] > 0
    pts3d = pts3d[valid]
    if len(pts3d) < 5:
        return 1.0, 0

    image_points, _ = cv2.projectPoints(
        pts3d.astype(np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        calibration.K,
        calibration.dist_coeffs,
    )
    image_points = image_points.reshape(-1, 2)

    u = np.rint(image_points[:, 0]).astype(np.int32)
    v = np.rint(image_points[:, 1]).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        return 1.0, 0

    z_depth = depth_map[v[inside], u[inside]]
    z_tri = pts3d[inside, 2]
    valid_depth = z_depth > 0
    if not np.any(valid_depth):
        return 1.0, 0

    scales = z_depth[valid_depth] / z_tri[valid_depth]
    scales = scales[np.isfinite(scales) & (scales > 1e-4) & (scales < 100.0)]

    if len(scales) < 5:
        return 1.0, int(len(scales))

    lo = float(np.percentile(scales, 20))
    hi = float(np.percentile(scales, 80))
    inlier_scales = scales[(scales >= lo) & (scales <= hi)]
    if len(inlier_scales) < 5:
        inlier_scales = scales

    return float(np.median(inlier_scales)), int(len(inlier_scales))


def stabilize_scale(
    raw_scale: float,
    support_count: int,
    scale_history: list[float],
    min_support: int = 12,
    max_step_ratio: float = 1.8,
    ema_alpha: float = 0.35,
) -> float:
    """Keep depth-derived translation scale from jumping wildly between frames."""
    if support_count < min_support or not np.isfinite(raw_scale) or raw_scale <= 1e-4:
        return scale_history[-1] if scale_history else 1.0

    if not scale_history:
        return raw_scale

    reference = float(np.median(scale_history[-5:]))
    lower = reference / max_step_ratio
    upper = reference * max_step_ratio
    clamped = float(np.clip(raw_scale, lower, upper))
    return (1.0 - ema_alpha) * scale_history[-1] + ema_alpha * clamped


def main() -> None:
    args = parse_args()
    image_paths = load_image_paths("data/keyframes")
    image_paths, images = load_images(image_paths)
    depth_dir = Path("data/results/depth")
    depth_paths = [depth_dir / f"{path.stem}_depth.npy" for path in image_paths]
    distance_range = compute_sequence_distance_range(depth_paths)
    image_h, image_w = images[0].shape[:2]
    calibration = load_camera_calibration(image_w, image_h)
    K = calibration.K

    print(f"Loaded {len(images)} images. Size: {image_w}x{image_h}")
    print(f"Using camera calibration: {calibration.source}")
    if "calibration mismatch" in calibration.source:
        print("Warning: calibration file did not match the reconstruction images, so fallback intrinsics are being used.")
    print(f"Using shared sequence depth range: [{distance_range[0]:.4f}, {distance_range[1]:.4f}]")
    print("Building sparse reconstruction...")

    R_w_c = np.eye(3, dtype=np.float64)
    t_w_c = np.zeros((3, 1), dtype=np.float64)

    all_points_world: List[np.ndarray] = []
    poses = [{
        "frame": 0,
        "R": R_w_c.tolist(),
        "t": t_w_c.reshape(-1).tolist(),
    }]

    if o3d is None or args.no_view:
        print("Continuing without live visualization.")
        vis = None
        pcd = None
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Sparse Reconstruction (Live Mode)", width=1100, height=750)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.point_size = 2.0

    prev_img = images[0]
    prev_kp, prev_des = orb_features(prev_img)
    prev_depth = load_midas_distance_map(
        depth_dir / f"{image_paths[0].stem}_depth.npy",
        distance_range=distance_range,
    )
    scale_history = [1.0]

    for i in range(1, len(images)):
        curr_img = images[i]
        curr_depth = load_midas_distance_map(
            depth_dir / f"{image_paths[i].stem}_depth.npy",
            distance_range=distance_range,
        )
        curr_kp, curr_des = orb_features(curr_img)
        print(f"[{i}] Processing frame (calibration source: {calibration.source})")

        if prev_des is None or curr_des is None or len(prev_kp) < 50 or len(curr_kp) < 50:
            print(f"[{i}] Not enough features. Skipping frame.")
            prev_img, prev_kp, prev_des, prev_depth = curr_img, curr_kp, curr_des, curr_depth
            continue

        matches = match_descriptors(prev_des, curr_des, ratio=0.75)
        if len(matches) < 25:
            print(f"[{i}] Too few matches ({len(matches)}). Skipping frame.")
            prev_img, prev_kp, prev_des, prev_depth = curr_img, curr_kp, curr_des, curr_depth
            continue

        pts1, pts2 = to_points(prev_kp, curr_kp, matches)
        pts1 = undistort_points(pts1, calibration)
        pts2 = undistort_points(pts2, calibration)

        E, inliers = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None or inliers is None:
            print(f"[{i}] Essential matrix failed. Skipping frame.")
            prev_img, prev_kp, prev_des, prev_depth = curr_img, curr_kp, curr_des, curr_depth
            continue

        inliers = inliers.ravel().astype(bool)
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        if len(pts1_in) < 20:
            print(f"[{i}] Too few inliers ({len(pts1_in)}). Skipping frame.")
            prev_img, prev_kp, prev_des, prev_depth = curr_img, curr_kp, curr_des, curr_depth
            continue

        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)
        mask_pose = mask_pose.ravel().astype(bool)

        pts1_pose = pts1_in[mask_pose]
        pts2_pose = pts2_in[mask_pose]

        if len(pts1_pose) < 15:
            print(f"[{i}] Pose inliers too few ({len(pts1_pose)}). Skipping frame.")
            prev_img, prev_kp, prev_des, prev_depth = curr_img, curr_kp, curr_des, curr_depth
            continue

        pts3d_prev = triangulate_points(K, R_rel, t_rel, pts1_pose, pts2_pose)
        pts3d_prev = filter_points_basic(pts3d_prev)
        raw_scale, scale_support = estimate_translation_scale(pts3d_prev, prev_depth, calibration)
        scale = stabilize_scale(raw_scale, scale_support, scale_history)
        scale_history.append(scale)
        t_rel_scaled = t_rel * scale
        use_for_dense_fusion = (
            len(pts1_pose) >= MIN_DENSE_FUSION_POSE_INLIERS and
            scale_support >= MIN_DENSE_FUSION_SCALE_SUPPORT
        )
        dense_fusion_score = float(len(pts1_pose) * max(scale_support, 1))

        pts_world = (R_w_c @ pts3d_prev.T + t_w_c).T
        all_points_world.append(pts_world)

        R_prev = R_w_c.copy()
        t_prev = t_w_c.copy()

        R_w_c = R_prev @ R_rel
        t_w_c = t_prev + (R_prev @ t_rel_scaled)

        poses.append({
            "frame": i,
            "R": R_w_c.tolist(),
            "t": t_w_c.reshape(-1).tolist(),
            "matches": len(matches),
            "inliers": len(pts1_in),
            "pose_inliers": len(pts1_pose),
            "raw_scale": float(raw_scale),
            "scale": float(scale),
            "scale_support": int(scale_support),
            "dense_fusion_score": dense_fusion_score,
            "use_for_dense_fusion": bool(use_for_dense_fusion),
        })

        pts_concat = np.vstack(all_points_world) if all_points_world else np.zeros((0, 3))
        if o3d is not None and pcd is not None and vis is not None:
            pcd.points = o3d.utility.Vector3dVector(pts_concat)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

        print(
            f"[{i}] matches={len(matches)} inliers={len(pts1_in)} "
            f"points_added={len(pts_world)} total={len(pts_concat)} "
            f"scale={scale:.3f} raw_scale={raw_scale:.3f} support={scale_support} "
            f"dense_ok={use_for_dense_fusion}"
        )

        prev_img, prev_kp, prev_des, prev_depth = curr_img, curr_kp, curr_des, curr_depth

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "sparse_cloud.ply"
    final_pts = np.vstack(all_points_world) if all_points_world else np.zeros((0, 3))
    if o3d is not None and pcd is not None:
        o3d.io.write_point_cloud(str(out_ply), pcd)
    else:
        write_point_cloud_ply(out_ply, final_pts)

    poses_path = out_dir / "camera_poses.json"
    with open(poses_path, "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)

    print(f"Saved camera poses to: {poses_path}")
    print(f"Saved sparse point cloud to: {out_ply}")

    if o3d is not None and vis is not None:
        print("Press Q in the Open3D window (or close it) to finish.")
        for _ in range(300):
            vis.poll_events()
            vis.update_renderer()

        vis.destroy_window()


if __name__ == "__main__":
    main()
