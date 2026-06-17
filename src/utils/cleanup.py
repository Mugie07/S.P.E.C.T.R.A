import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("data/results/dense_fused_cloud_pose_aligned.ply")

print(f"Before cleanup: {len(pcd.points):,} pts")

labels = np.array(pcd.cluster_dbscan(eps=0.08, min_points=8))
valid_labels = labels[labels >= 0]

if len(valid_labels) == 0:
    print("Cleanup skipped: no stable DBSCAN cluster was found.")
else:
    largest = np.bincount(valid_labels).argmax()
    keep_indices = np.where(labels == largest)[0]
    min_keep = max(500, int(0.25 * len(pcd.points)))
    if len(keep_indices) < min_keep:
        print(
            "Cleanup skipped: dominant cluster would remove too much geometry "
            f"({len(keep_indices):,}/{len(pcd.points):,} pts)."
        )
    else:
        pcd = pcd.select_by_index(keep_indices)

print(f"After cleanup: {len(pcd.points):,} pts")

o3d.io.write_point_cloud("data/results/dense_fused_cloud_clean.ply", pcd)
print("Saved -> data/results/dense_fused_cloud_clean.ply")
