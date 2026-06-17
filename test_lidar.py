import open3d as o3d
pcd = o3d.io.read_point_cloud(r'C:\Users\Hp\recon3d\data\lidar\sart-tilman_appartement_kitchen_5M.ply')
print('Points:', len(pcd.points))
o3d.visualization.draw_geometries([pcd], window_name='Kitchen Scan')
