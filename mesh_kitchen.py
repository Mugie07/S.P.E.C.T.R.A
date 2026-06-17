import open3d as o3d
import numpy as np
from pathlib import Path

print("Loading LiDAR point cloud...")
pcd = o3d.io.read_point_cloud(r'C:\Users\Hp\recon3d\data\lidar\sart-tilman_appartement_kitchen_5M.ply')
print(f'Points loaded: {len(pcd.points):,}')

# Estimate normals (required for Poisson)
print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)

# Poisson surface reconstruction
print("Running Poisson reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)

# Remove low-density vertices (clean up outer artifacts)
densities = np.asarray(densities)
keep = densities > np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(~keep)
mesh.compute_vertex_normals()

# Save
out_path = r'C:\Users\Hp\recon3d\data\results\kitchen_mesh.ply'
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
o3d.io.write_triangle_mesh(out_path, mesh)
print(f'Mesh saved: {out_path}')
print(f'Triangles: {len(mesh.triangles):,}')

# Visualise
print("Opening viewer...")
o3d.visualization.draw_geometries([mesh], window_name='Kitchen Mesh', width=1200, height=800)
