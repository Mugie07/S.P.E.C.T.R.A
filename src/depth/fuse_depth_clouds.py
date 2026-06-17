from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the pose-aligned fused cloud and save the final dense cloud."
    )
    parser.add_argument(
        "--no_view",
        action="store_true",
        help="Save outputs without launching the Open3D viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Read the POSE-ALIGNED cloud from step 3
    in_path  = Path("data/results/dense_fused_cloud_pose_aligned.ply")
    out_path = Path("data/results/dense_fused_cloud.ply")

    if not in_path.exists():
        raise FileNotFoundError(
            f"Pose-aligned cloud not found: {in_path}\n"
            "Run fuse_depth_clouds_pose_aligned.py first."
        )

    print(f"Loading pose-aligned cloud from: {in_path}")
    pcd = o3d.io.read_point_cloud(str(in_path))
    print(f"Loaded {len(pcd.points):,} points")

    print("Voxel downsampling...")
    pcd = pcd.voxel_down_sample(voxel_size=0.02)

    print("Removing statistical outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Second pass with tighter cleanup.
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    print(f"Final point count: {len(pcd.points):,}")
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved final cloud -> {out_path}")

    if not args.no_view:
        o3d.visualization.draw_geometries(
            [pcd], window_name="Final Dense Point Cloud")


if __name__ == "__main__":
    main()
