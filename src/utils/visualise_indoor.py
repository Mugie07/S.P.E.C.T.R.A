import open3d as o3d
import numpy as np
from pathlib import Path
import sys
sys.path.append('src/utils')
from synthetic_data_generator import generate_indoor_scene

points, labels, colors = generate_indoor_scene(n_points=80000, room_dims=(10, 8, 3), seed=42)

# Boost color saturation
colors = np.clip(colors.astype(np.float64), 0.0, 1.0)
# Increase contrast: push colors away from grey
grey = colors.mean(axis=1, keepdims=True)
colors = np.clip(grey + 2.5 * (colors - grey), 0.0, 1.0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
pcd.colors = o3d.utility.Vector3dVector(colors)

# Render with larger points
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Synthetic Indoor Scene', width=1200, height=800)
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.point_size = 3.0
opt.background_color = np.array([0.05, 0.05, 0.05])  # dark background makes colours pop
vis.run()
vis.destroy_window()