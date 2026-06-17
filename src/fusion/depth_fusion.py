from __future__ import annotations

import json
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

# CHANGED: was 5 — increased to use all available frames instead of a narrow
# top-5 window, which was skipping high-scoring frames like 013 and 015
MAX_FUSION_FRAMES = 10

# CHANGED: was 30000 — recaptured frames yield only 4k-20k points so the old
# threshold was rejecting every single selected frame and crashing the pipeline
MIN_POINTS_PER_FRAME = 4000

# CHANGED: was 3.5 — outdoor captures have a wider disparity range (up to 9.07)
# so the old ceiling was clipping too much valid far depth
MAX_DEPTH_METERS = 5.0

# CHANGED: was 5.0 — lowered to keep more near-object depth that was being
# clipped by the percentile cut, reducing point count per frame
DEPTH_NEAR_PERCENTILE = 2.0

# CHANGED: was 65.0 — raised to retain more of the scene depth instead of
# aggressively cutting the far half of every depth map
DEPTH_FAR_PERCENTILE = 80.0

# CHANGED: was 0.42 for both — widened the elliptical crop window so more
# valid pixels near the frame edges are included in back-projection
CENTER_RADIUS_X = 0.55
CENTER_RADIUS_Y = 0.55

# Unchanged ICP parameters
ICP_VOXEL_SIZE = 0.03
ICP_MAX_CORRESPONDENCE = 0.12
ICP_MIN_FITNESS = 0.28
ICP_ACCEPT_FITNESS = 0.40

MIN_POSE_INLIERS = 120
MIN_SCALE_SUPPORT = 70
MIN_DENSE_FUSION_SCORE = 8000.0

# Presentation cleanup runs inside the fusion stage so the first 3D view is
# already audience-ready instead of waiting for the later cleanup stage.
POST_VOXEL_SIZE = 0.018
POST_STAT_NEIGHBORS = 24
POST_STAT_STD_RATIO = 1.35
POST_RADIUS = 0.08
POST_RADIUS_MIN_POINTS = 8
POST_CLUSTER_EPS = 0.08
POST_CLUSTER_MIN_POINTS = 15
POST_RADIUS_PERCENTILE = 97.5


def depth_to_points(
    depth: np.ndarray,
    color: np.ndarray,
    calibration: CameraCalibration,
    foreground_mask: np.ndarray | None = None,
    center_radius_x: float = CENTER_RADIUS_X,
    center_radius_y: float = CENTER_RADIUS_Y,
    near_percentile: float = DEPTH_NEAR_PERCENTILE,
    far_percentile: float = DEPTH_FAR_PERCENTILE,
    max_depth_meters: float = MAX_DEPTH_METERS,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    ys_full, xs_full = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    nx = (xs_full - (w / 2.0)) / max(w * center_radius_x, 1.0)
    ny = (ys_full - (h / 2.0)) / max(h * center_radius_y, 1.0)
    center_mask = (nx * nx + ny * ny) <= 1.0

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid_depth):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    near_clip = float(np.percentile(depth[valid_depth], near_percentile))
    far_clip = float(np.percentile(depth[valid_depth], far_percentile))
    far_clip = min(far_clip, max_depth_meters)
    if far_clip <= max(0.3, near_clip):
        far_clip = min(max_depth_meters, max(0.31, near_clip + 0.5))

    mask = (
        center_mask &
        np.isfinite(depth) &
        (depth > max(0.3, near_clip)) &
        (depth < far_clip)
    )
    if foreground_mask is not None:
        if foreground_mask.shape != depth.shape:
            foreground_mask = cv2.resize(
                foreground_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        mask &= foreground_mask

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    pixel_points = np.stack((xs, ys), axis=-1).astype(np.float32).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(
        pixel_points,
        calibration.K,
        calibration.dist_coeffs,
    ).reshape(-1, 2).astype(np.float32)

    z = depth[ys, xs].astype(np.float32)
    x = rays[:, 0] * z
    y = rays[:, 1] * z
    points = np.stack((x, y, z), axis=-1).astype(np.float32)
    colors = color[ys, xs].astype(np.float32) / 255.0

    max_points = 100000
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T + t.reshape(3, 1)).T


def build_o3d_point_cloud(points: np.ndarray, colors: np.ndarray) -> "o3d.geometry.PointCloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def foreground_mask_from_image(color_rgb: np.ndarray) -> np.ndarray:
    h, w = color_rgb.shape[:2]
    rect_x = int(w * 0.05)
    rect_y = int(h * 0.06)
    rect_w = int(w * 0.90)
    rect_h = int(h * 0.78)
    rect = (rect_x, rect_y, rect_w, rect_h)

    grabcut_mask = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            color_rgb,
            grabcut_mask,
            rect,
            bg_model,
            fg_model,
            4,
            cv2.GC_INIT_WITH_RECT,
        )
        mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    except cv2.error:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255

    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    keep = mask > 0
    if keep.sum() < int(0.10 * h * w):
        fallback = np.zeros((h, w), dtype=bool)
        fallback[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = True
        return fallback
    return keep


def select_point_indices(pcd: "o3d.geometry.PointCloud", keep_mask: np.ndarray) -> "o3d.geometry.PointCloud":
    if len(keep_mask) != len(pcd.points):
        return pcd
    keep_indices = np.flatnonzero(keep_mask)
    if len(keep_indices) < max(500, int(0.35 * len(pcd.points))):
        print("Radius trim skipped because it would remove too much geometry.")
        return pcd
    return pcd.select_by_index(keep_indices.tolist())


def trim_far_outliers(pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
    points = np.asarray(pcd.points)
    if len(points) < 1000:
        return pcd

    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    robust_sigma = max(1.4826 * mad, 1e-6)
    percentile_limit = float(np.percentile(distances, POST_RADIUS_PERCENTILE))
    mad_limit = median_distance + (4.0 * robust_sigma)
    limit = min(percentile_limit, mad_limit)
    keep_mask = distances <= limit
    before = len(points)
    pcd = select_point_indices(pcd, keep_mask)
    print(f"After radius trim: {len(pcd.points):,} pts (from {before:,})")
    return pcd


def keep_main_cluster(pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
    if len(pcd.points) < 1000:
        return pcd

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=POST_CLUSTER_EPS,
            min_points=POST_CLUSTER_MIN_POINTS,
            print_progress=False,
        )
    )
    cluster_labels = labels[labels >= 0]
    if len(cluster_labels) == 0:
        print("DBSCAN cleanup skipped because no stable cluster was found.")
        return pcd

    counts = np.bincount(cluster_labels)
    largest_label = int(np.argmax(counts))
    keep_indices = np.flatnonzero(labels == largest_label)
    if len(keep_indices) < max(500, int(0.30 * len(pcd.points))):
        print("DBSCAN cleanup skipped because the largest cluster was too small.")
        return pcd

    before = len(pcd.points)
    pcd = pcd.select_by_index(keep_indices.tolist())
    print(f"After main-cluster cleanup: {len(pcd.points):,} pts (from {before:,})")
    return pcd


def refine_points_with_icp(
    points_world: np.ndarray,
    colors: np.ndarray,
    reference_points: np.ndarray,
    reference_colors: np.ndarray,
) -> tuple[np.ndarray, dict[str, float] | None, bool]:
    if o3d is None or len(reference_points) < 1000 or len(points_world) < 1000:
        return points_world, None, True

    source = build_o3d_point_cloud(points_world, colors)
    target = build_o3d_point_cloud(reference_points, reference_colors)

    source_ds = source.voxel_down_sample(ICP_VOXEL_SIZE)
    target_ds = target.voxel_down_sample(ICP_VOXEL_SIZE)
    if len(source_ds.points) < 200 or len(target_ds.points) < 200:
        return points_world, None, True

    reg = o3d.pipelines.registration.registration_icp(
        source_ds,
        target_ds,
        ICP_MAX_CORRESPONDENCE,
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    info = {
        "fitness": float(reg.fitness),
        "rmse": float(reg.inlier_rmse),
    }
    if reg.fitness < ICP_MIN_FITNESS:
        return points_world, info, False

    transformed = np.asarray(
        source.transform(reg.transformation).points,
        dtype=np.float32,
    )
    accepted = reg.fitness >= ICP_ACCEPT_FITNESS
    return transformed, info, accepted


def select_frames_for_fusion(
    pose_lookup: dict[int, dict[str, object]],
    max_frames: int = MAX_FUSION_FRAMES,
    min_pose_inliers: int = MIN_POSE_INLIERS,
    min_scale_support: int = MIN_SCALE_SUPPORT,
    min_dense_fusion_score: float = MIN_DENSE_FUSION_SCORE,
) -> set[int]:
    eligible = []
    for frame_idx, pose in pose_lookup.items():
        if not bool(pose.get("use_for_dense_fusion", True)):
            continue
        if int(pose.get("pose_inliers", 0)) < min_pose_inliers:
            continue
        if int(pose.get("scale_support", 0)) < min_scale_support:
            continue
        score = float(pose.get("dense_fusion_score", 0.0))
        if score < min_dense_fusion_score:
            continue
        eligible.append((score, frame_idx))

    if not eligible:
        print("Warning: no frames passed strict pose-quality gates; falling back to score-ranked frames.")
        fallback = []
        for frame_idx, pose in pose_lookup.items():
            if not bool(pose.get("use_for_dense_fusion", True)):
                continue
            fallback.append((float(pose.get("dense_fusion_score", 0.0)), frame_idx))
        fallback.sort(reverse=True)
        return {idx for _, idx in fallback[:max_frames]}

    eligible.sort(key=lambda item: item[0], reverse=True)
    return {frame_idx for _, frame_idx in eligible[:max_frames]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse depth maps into a pose-aligned dense point cloud."
    )
    parser.add_argument(
        "--no_view",
        action="store_true",
        help="Save outputs without launching the Open3D viewer.",
    )
    # CHANGED: new flag — plane removal is now optional rather than always on.
    # Use --remove_plane for isolated object capture (tin, mug, etc.) to strip
    # the table/floor. Leave it off for room-scale reconstruction where the
    # floor and walls are part of the scene you want to keep.
    parser.add_argument(
        "--remove_plane",
        action="store_true",
        help=(
            "Remove the dominant ground/background plane after outlier filtering. "
            "Use for isolated object capture. "
            "Do NOT use for room-scale reconstruction — the floor is part of the scene."
        ),
    )
    parser.add_argument(
        "--zoom",
        choices=("1x", "2x"),
        default="2x",
        help=(
            "Camera zoom level used during capture. "
            "'2x' loads data/camera_intrinsics_2x.json for object captures. "
            "'1x' loads data/camera_intrinsics_1x.json for room/wide captures."
        ),
    )
    parser.add_argument(
        "--foreground_mask",
        action="store_true",
        help="Use GrabCut foreground masking to suppress background points before fusion.",
    )
    parser.add_argument(
        "--profile",
        choices=("object", "room", "strict"),
        default="object",
        help=(
            "Fusion tuning profile. 'object' keeps thin/planar objects more aggressively, "
            "'room' preserves wider scene geometry, and 'strict' uses conservative cleanup."
        ),
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Override the maximum number of pose-ranked frames used for dense fusion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile == "room":
        fusion_defaults = {
            "max_frames": 40,
            "min_pose_inliers": 90,
            "min_scale_support": 45,
            "min_score": 4500.0,
            "center_radius_x": 0.72,
            "center_radius_y": 0.72,
            "near_pct": 1.0,
            "far_pct": 92.0,
            "max_depth": 6.0,
            "voxel": 0.016,
            "stat_neighbors": 20,
            "stat_std": 1.8,
            "radius": 0.10,
            "radius_min_points": 5,
            "keep_cluster": True,
        }
    elif args.profile == "strict":
        fusion_defaults = {
            "max_frames": MAX_FUSION_FRAMES,
            "min_pose_inliers": MIN_POSE_INLIERS,
            "min_scale_support": MIN_SCALE_SUPPORT,
            "min_score": MIN_DENSE_FUSION_SCORE,
            "center_radius_x": CENTER_RADIUS_X,
            "center_radius_y": CENTER_RADIUS_Y,
            "near_pct": DEPTH_NEAR_PERCENTILE,
            "far_pct": DEPTH_FAR_PERCENTILE,
            "max_depth": MAX_DEPTH_METERS,
            "voxel": POST_VOXEL_SIZE,
            "stat_neighbors": POST_STAT_NEIGHBORS,
            "stat_std": POST_STAT_STD_RATIO,
            "radius": POST_RADIUS,
            "radius_min_points": POST_RADIUS_MIN_POINTS,
            "keep_cluster": True,
        }
    else:
        fusion_defaults = {
            "max_frames": 30,
            "min_pose_inliers": 55,
            "min_scale_support": 25,
            "min_score": 1800.0,
            "center_radius_x": 0.78,
            "center_radius_y": 0.78,
            "near_pct": 0.5,
            "far_pct": 95.0,
            "max_depth": 5.5,
            "voxel": 0.012,
            "stat_neighbors": 16,
            "stat_std": 2.2,
            "radius": 0.075,
            "radius_min_points": 3,
            "keep_cluster": False,
        }
    max_fusion_frames = max(1, args.max_frames or int(fusion_defaults["max_frames"]))
    img_dir = Path("data/keyframes")
    depth_dir = Path("data/results/depth")
    poses_path = Path("data/results/camera_poses.json")
    out_path = Path("data/results/dense_fused_cloud_pose_aligned.ply")

    if not poses_path.exists():
        raise FileNotFoundError(
            f"Missing poses file: {poses_path}. Run sparse reconstruction first."
        )

    with open(poses_path, "r", encoding="utf-8") as f:
        poses_data = json.load(f)

    pose_lookup = {
        int(item["frame"]): {
            "R": np.array(item["R"], dtype=np.float32),
            "t": np.array(item["t"], dtype=np.float32).reshape(3, 1),
            "use_for_dense_fusion": bool(item.get("use_for_dense_fusion", True)),
            "scale_support": int(item.get("scale_support", 0)),
            "pose_inliers": int(item.get("pose_inliers", 0)),
            "dense_fusion_score": float(item.get("dense_fusion_score", 0.0)),
        }
        for item in poses_data
    }

    image_paths = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )

    print(f"DEBUG: Found {len(image_paths)} images in {img_dir}")

    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    depth_paths = [depth_dir / f"{img_path.stem}_depth.npy" for img_path in image_paths]
    distance_range = compute_sequence_distance_range(depth_paths)
    image_h, image_w = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR).shape[:2]
    calibration = load_camera_calibration(image_w, image_h, zoom=args.zoom)

    all_points, all_colors = [], []
    selected_frames = select_frames_for_fusion(
        pose_lookup,
        max_frames=max_fusion_frames,
        min_pose_inliers=int(fusion_defaults["min_pose_inliers"]),
        min_scale_support=int(fusion_defaults["min_scale_support"]),
        min_dense_fusion_score=float(fusion_defaults["min_score"]),
    )

    print(f"Found {len(image_paths)} images.")
    print(f"Loaded {len(pose_lookup)} camera poses.")
    print(f"Using camera calibration: {calibration.source}")
    print(f"Using zoom profile: {args.zoom}")
    print(f"Using fusion profile: {args.profile}")
    if "calibration mismatch" in calibration.source:
        print("Warning: calibration file did not match the reconstruction images, so fallback intrinsics are being used.")
    print(f"Using shared sequence depth range: [{distance_range[0]:.4f}, {distance_range[1]:.4f}]")
    print(f"Using top {len(selected_frames)} pose-ranked frames for dense fusion.")
    print("Fusing depth clouds with pose alignment...")

    for idx, img_path in enumerate(image_paths):
        if idx not in pose_lookup:
            print(f"  [{idx}] No pose - skipping {img_path.name}")
            continue
        if selected_frames and idx not in selected_frames:
            print(
                f"  [{idx}] Lower-ranked frame for dense fusion "
                f"(score={pose_lookup[idx]['dense_fusion_score']:.1f}) - skipping {img_path.name}"
            )
            continue
        if not pose_lookup[idx]["use_for_dense_fusion"]:
            print(
                f"  [{idx}] Weak pose for dense fusion "
                f"(pose_inliers={pose_lookup[idx]['pose_inliers']} "
                f"scale_support={pose_lookup[idx]['scale_support']}) - skipping {img_path.name}"
            )
            continue

        depth = load_midas_distance_map(
            depth_dir / f"{img_path.stem}_depth.npy",
            distance_range=distance_range,
        )
        if depth is None:
            print(f"  [{idx}] No depth map - skipping {img_path.name}")
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [{idx}] Unreadable image - skipping {img_path.name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"  [{idx}] Processing frame (calibration source: {calibration.source})")

        scale = 0.5
        img_rgb = cv2.resize(img_rgb, None, fx=scale, fy=scale)
        depth = cv2.resize(depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        foreground_mask = foreground_mask_from_image(img_rgb) if args.foreground_mask else None
        calib_scaled = CameraCalibration(
            K=calibration.K.copy(),
            dist_coeffs=calibration.dist_coeffs.copy(),
            width=depth.shape[1],
            height=depth.shape[0],
            source=calibration.source,
        )
        calib_scaled.K[0, 0] *= scale
        calib_scaled.K[1, 1] *= scale
        calib_scaled.K[0, 2] *= scale
        calib_scaled.K[1, 2] *= scale

        points, colors = depth_to_points(
            depth,
            img_rgb,
            calib_scaled,
            foreground_mask,
            center_radius_x=float(fusion_defaults["center_radius_x"]),
            center_radius_y=float(fusion_defaults["center_radius_y"]),
            near_percentile=float(fusion_defaults["near_pct"]),
            far_percentile=float(fusion_defaults["far_pct"]),
            max_depth_meters=float(fusion_defaults["max_depth"]),
        )
        if len(points) < MIN_POINTS_PER_FRAME:
            print(f"  [{idx}] Too few dense points ({len(points)}) - skipping {img_path.name}")
            continue

        R = pose_lookup[idx]["R"]
        t = pose_lookup[idx]["t"]

        points_world = transform_points(points, R, t)
        icp_info = None
        icp_accept = True
        if all_points:
            reference_points = np.vstack(all_points)
            reference_colors = np.vstack(all_colors)
            points_world, icp_info, icp_accept = refine_points_with_icp(
                points_world,
                colors,
                reference_points,
                reference_colors,
            )
            if not icp_accept:
                print(
                    f"  [{idx}] ICP rejected frame "
                    f"(fitness={icp_info['fitness']:.3f} rmse={icp_info['rmse']:.4f}) - skipping {img_path.name}"
                )
                continue

        all_points.append(points_world)
        all_colors.append(colors)

        if icp_info is None:
            print(f"  [{idx + 1}/{len(image_paths)}] {len(points_world):,} pts - {img_path.name}")
        else:
            print(
                f"  [{idx + 1}/{len(image_paths)}] {len(points_world):,} pts - {img_path.name} "
                f"| icp_fitness={icp_info['fitness']:.3f} rmse={icp_info['rmse']:.4f}"
            )

    if not all_points:
        raise RuntimeError("No valid aligned point clouds were created.")

    points_concat = np.vstack(all_points)
    colors_concat = np.vstack(all_colors)

    print(f"\nTotal raw fused points: {len(points_concat):,}")

    if o3d is None:
        print("Open3D not installed; saving raw fused cloud without Open3D filtering/visualization.")
        center = points_concat.mean(axis=0, keepdims=True)
        points_concat = points_concat - center
        print(f"Final point count: {len(points_concat):,}")
        write_point_cloud_ply(out_path, points_concat, colors_concat)
        print(f"\nSaved -> {out_path}")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_concat)
    pcd.colors = o3d.utility.Vector3dVector(colors_concat)

    print("Voxel downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=float(fusion_defaults["voxel"]))

    print("Removing statistical outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=int(fusion_defaults["stat_neighbors"]),
        std_ratio=float(fusion_defaults["stat_std"]),
    )

    print("Removing isolated radius outliers...")
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=int(fusion_defaults["radius_min_points"]),
        radius=float(fusion_defaults["radius"]),
    )

    print("Trimming distant floating points...")
    pcd = trim_far_outliers(pcd)

    # CHANGED: plane removal is now gated behind --remove_plane flag.
    # Old behaviour: plane was always removed (which would destroy floor/wall
    # data if you ever scale up to room reconstruction).
    # New behaviour: pass --remove_plane for isolated objects (tin, mug, etc.),
    # omit it for room-scale scenes where floor and walls must be preserved.
    if args.remove_plane:
        print("Removing ground plane (--remove_plane flag is set)...")
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000,
        )
        pcd = pcd.select_by_index(inliers, invert=True)
        print(f"After plane removal: {len(pcd.points):,} pts")
    else:
        print("Skipping plane removal (use --remove_plane to strip floor/background).")

    if bool(fusion_defaults["keep_cluster"]):
        print("Keeping the dominant reconstruction cluster...")
        pcd = keep_main_cluster(pcd)
    else:
        print("Skipping dominant-cluster cleanup for permissive object profile.")

    pcd.translate(-pcd.get_center())

    print(f"Final point count: {len(pcd.points):,}")

    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"\nSaved -> {out_path}")

    if args.no_view:
        return

    print("Launching visualizer...")

    vis = o3d.visualization.Visualizer()

    if not vis.create_window(
        window_name="Pose-Aligned Dense Fused Cloud",
        width=900,
        height=700,
    ):
        print("Warning: Failed to create Open3D window. Skipping visualization.")
    else:
        vis.add_geometry(pcd)

        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0, 0, 0])

        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()
