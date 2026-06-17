"""
3D Geodata Academy - Synthetic Data Generator
Copyright (c) 3D Geodata Academy
Made by Dr. Florent Poux
https://learngeodata.eu
License: NC (Commercial use for 3D Geodata Academy members only)

Realistic synthetic data generation for point cloud tutorials.
- ALS/LiDAR terrain simulation
- Urban scene generation
- Indoor scene generation
- Multi-class labeled datasets
"""

import numpy as np


def generate_als_terrain(n_ground=50000, n_vegetation=20000, n_buildings=10000,
                         extent=(200, 200), seed=42):
    """
    Generate realistic ALS LiDAR terrain with ground, vegetation, and buildings.

    Args:
        n_ground: Number of ground points
        n_vegetation: Number of vegetation points
        n_buildings: Number of building points
        extent: (width, height) of the terrain in meters
        seed: Random seed for reproducibility

    Returns:
        points: Nx3 array of coordinates
        labels: N array of class labels (0=ground, 1=vegetation, 2=building)
        colors: Nx3 array of RGB colors (0-1)
    """
    np.random.seed(seed)
    w, h = extent

    # Ground with terrain undulation (realistic terrain)
    gx = np.random.uniform(0, w, n_ground)
    gy = np.random.uniform(0, h, n_ground)
    gz = (2 * np.sin(gx / 30) + 1.5 * np.cos(gy / 25) +
          0.5 * np.sin((gx + gy) / 20) +
          np.random.normal(0, 0.05, n_ground))
    ground = np.column_stack([gx, gy, gz])
    ground_colors = np.column_stack([
        np.random.uniform(0.35, 0.45, n_ground),  # Brown-ish
        np.random.uniform(0.30, 0.40, n_ground),
        np.random.uniform(0.15, 0.25, n_ground)
    ])

    # Vegetation above terrain (trees and shrubs)
    vx = np.random.uniform(0, w, n_vegetation)
    vy = np.random.uniform(0, h, n_vegetation)
    terrain_z = 2 * np.sin(vx / 30) + 1.5 * np.cos(vy / 25) + 0.5 * np.sin((vx + vy) / 20)
    vz = terrain_z + np.random.exponential(5, n_vegetation) + 1.5
    vegetation = np.column_stack([vx, vy, vz])
    vegetation_colors = np.column_stack([
        np.random.uniform(0.1, 0.3, n_vegetation),
        np.random.uniform(0.4, 0.7, n_vegetation),  # Green
        np.random.uniform(0.1, 0.2, n_vegetation)
    ])

    # Buildings (flat roof boxes)
    buildings_list = []
    building_colors_list = []
    n_buildings_count = 5
    points_per_building = n_buildings // n_buildings_count

    for _ in range(n_buildings_count):
        cx, cy = np.random.uniform(20, w - 30, 2)
        bw, bh = np.random.uniform(15, 30, 2)
        roof_h = np.random.uniform(8, 15)

        bx = np.random.uniform(cx, cx + bw, points_per_building)
        by = np.random.uniform(cy, cy + bh, points_per_building)
        base_z = 2 * np.sin(cx / 30) + 1.5 * np.cos(cy / 25)
        bz = np.full(points_per_building, base_z + roof_h) + np.random.normal(0, 0.02, points_per_building)
        buildings_list.append(np.column_stack([bx, by, bz]))

        # Gray rooftops
        gray = np.random.uniform(0.5, 0.7, points_per_building)
        building_colors_list.append(np.column_stack([gray, gray, gray]))

    buildings = np.vstack(buildings_list)
    building_colors = np.vstack(building_colors_list)

    # Combine all
    all_points = np.vstack([ground, vegetation, buildings])
    all_colors = np.vstack([ground_colors, vegetation_colors, building_colors])
    labels = np.concatenate([
        np.zeros(n_ground, dtype=np.int32),
        np.ones(n_vegetation, dtype=np.int32),
        np.full(len(buildings), 2, dtype=np.int32)
    ])

    return all_points, labels, all_colors


def generate_urban_scene(n_points=100000, extent=(300, 300), n_buildings=15, seed=42):
    """
    Generate urban scene with roads, buildings, vegetation, and vehicles.

    Args:
        n_points: Total approximate number of points
        extent: Scene dimensions in meters
        n_buildings: Number of buildings
        seed: Random seed

    Returns:
        points: Nx3 array
        labels: N array (0=ground, 1=road, 2=building, 3=vegetation, 4=vehicle)
        colors: Nx3 array
    """
    np.random.seed(seed)
    w, h = extent

    points_list = []
    colors_list = []
    labels_list = []

    # Ground / grass areas
    n_ground = n_points // 3
    gx = np.random.uniform(0, w, n_ground)
    gy = np.random.uniform(0, h, n_ground)
    gz = np.random.normal(0, 0.03, n_ground)
    points_list.append(np.column_stack([gx, gy, gz]))
    colors_list.append(np.column_stack([
        np.random.uniform(0.2, 0.35, n_ground),
        np.random.uniform(0.45, 0.6, n_ground),
        np.random.uniform(0.1, 0.2, n_ground)
    ]))
    labels_list.append(np.zeros(n_ground, dtype=np.int32))

    # Roads (grid pattern)
    n_road = n_points // 5
    road_width = 8

    # Horizontal roads
    for y_pos in [50, 150, 250]:
        n_r = n_road // 6
        rx = np.random.uniform(0, w, n_r)
        ry = np.random.uniform(y_pos - road_width/2, y_pos + road_width/2, n_r)
        rz = np.random.normal(0.02, 0.01, n_r)
        points_list.append(np.column_stack([rx, ry, rz]))
        gray = np.random.uniform(0.3, 0.4, n_r)
        colors_list.append(np.column_stack([gray, gray, gray]))
        labels_list.append(np.ones(n_r, dtype=np.int32))

    # Vertical roads
    for x_pos in [100, 200]:
        n_r = n_road // 6
        rx = np.random.uniform(x_pos - road_width/2, x_pos + road_width/2, n_r)
        ry = np.random.uniform(0, h, n_r)
        rz = np.random.normal(0.02, 0.01, n_r)
        points_list.append(np.column_stack([rx, ry, rz]))
        gray = np.random.uniform(0.3, 0.4, n_r)
        colors_list.append(np.column_stack([gray, gray, gray]))
        labels_list.append(np.ones(n_r, dtype=np.int32))

    # Buildings
    n_building_pts = n_points // 4
    pts_per_building = n_building_pts // n_buildings

    for i in range(n_buildings):
        cx = np.random.uniform(20, w - 40)
        cy = np.random.uniform(20, h - 40)
        bw = np.random.uniform(15, 35)
        bd = np.random.uniform(15, 35)
        bh = np.random.uniform(10, 40)

        bx = np.random.uniform(cx, cx + bw, pts_per_building)
        by = np.random.uniform(cy, cy + bd, pts_per_building)
        bz = np.full(pts_per_building, bh) + np.random.normal(0, 0.05, pts_per_building)
        points_list.append(np.column_stack([bx, by, bz]))

        # Building colors (various)
        if i % 3 == 0:
            colors_list.append(np.column_stack([
                np.random.uniform(0.6, 0.75, pts_per_building),
                np.random.uniform(0.5, 0.65, pts_per_building),
                np.random.uniform(0.4, 0.55, pts_per_building)
            ]))
        else:
            gray = np.random.uniform(0.5, 0.7, pts_per_building)
            colors_list.append(np.column_stack([gray, gray * 0.95, gray * 0.9]))
        labels_list.append(np.full(pts_per_building, 2, dtype=np.int32))

    # Vegetation (trees)
    n_veg = n_points // 8
    n_trees = 30
    pts_per_tree = n_veg // n_trees

    for _ in range(n_trees):
        tx = np.random.uniform(0, w)
        ty = np.random.uniform(0, h)
        tree_h = np.random.uniform(5, 15)

        vx = np.random.normal(tx, 2, pts_per_tree)
        vy = np.random.normal(ty, 2, pts_per_tree)
        vz = np.random.exponential(tree_h / 3, pts_per_tree) + 1
        points_list.append(np.column_stack([vx, vy, vz]))
        colors_list.append(np.column_stack([
            np.random.uniform(0.1, 0.25, pts_per_tree),
            np.random.uniform(0.4, 0.65, pts_per_tree),
            np.random.uniform(0.1, 0.2, pts_per_tree)
        ]))
        labels_list.append(np.full(pts_per_tree, 3, dtype=np.int32))

    # Vehicles (simple boxes)
    n_vehicles = 20
    pts_per_vehicle = 200

    for _ in range(n_vehicles):
        vx_pos = np.random.uniform(20, w - 20)
        vy_pos = np.random.uniform(20, h - 20)

        vx = np.random.uniform(vx_pos, vx_pos + 4.5, pts_per_vehicle)
        vy = np.random.uniform(vy_pos, vy_pos + 2, pts_per_vehicle)
        vz = np.random.uniform(0.1, 1.5, pts_per_vehicle)
        points_list.append(np.column_stack([vx, vy, vz]))

        # Random vehicle colors
        base_color = np.random.uniform(0.2, 0.9, 3)
        colors_list.append(np.tile(base_color + np.random.normal(0, 0.05, (pts_per_vehicle, 3)), 1).clip(0, 1))
        labels_list.append(np.full(pts_per_vehicle, 4, dtype=np.int32))

    return np.vstack(points_list), np.concatenate(labels_list), np.vstack(colors_list)


def generate_indoor_scene(n_points=80000, room_dims=(10, 8, 3), seed=42):
    """
    Generate indoor room scan with floor, walls, ceiling, and furniture.

    Args:
        n_points: Approximate total points
        room_dims: (length, width, height) in meters
        seed: Random seed

    Returns:
        points: Nx3 array
        labels: N array (0=floor, 1=wall, 2=ceiling, 3=furniture)
        colors: Nx3 array
    """
    np.random.seed(seed)
    L, W, H = room_dims

    points_list = []
    colors_list = []
    labels_list = []

    # Floor
    n_floor = n_points // 6
    fx = np.random.uniform(0, L, n_floor)
    fy = np.random.uniform(0, W, n_floor)
    fz = np.random.normal(0, 0.005, n_floor)
    points_list.append(np.column_stack([fx, fy, fz]))
    colors_list.append(np.column_stack([
        np.random.uniform(0.55, 0.65, n_floor),
        np.random.uniform(0.45, 0.55, n_floor),
        np.random.uniform(0.35, 0.45, n_floor)
    ]))
    labels_list.append(np.zeros(n_floor, dtype=np.int32))

    # Walls
    n_wall = n_points // 4
    n_per_wall = n_wall // 4

    # Wall at y=0
    wx = np.random.uniform(0, L, n_per_wall)
    wy = np.random.normal(0, 0.01, n_per_wall)
    wz = np.random.uniform(0, H, n_per_wall)
    points_list.append(np.column_stack([wx, wy, wz]))

    # Wall at y=W
    wx = np.random.uniform(0, L, n_per_wall)
    wy = np.random.normal(W, 0.01, n_per_wall)
    wz = np.random.uniform(0, H, n_per_wall)
    points_list.append(np.column_stack([wx, wy, wz]))

    # Wall at x=0
    wx = np.random.normal(0, 0.01, n_per_wall)
    wy = np.random.uniform(0, W, n_per_wall)
    wz = np.random.uniform(0, H, n_per_wall)
    points_list.append(np.column_stack([wx, wy, wz]))

    # Wall at x=L
    wx = np.random.normal(L, 0.01, n_per_wall)
    wy = np.random.uniform(0, W, n_per_wall)
    wz = np.random.uniform(0, H, n_per_wall)
    points_list.append(np.column_stack([wx, wy, wz]))

    # Wall colors (beige/white)
    for _ in range(4):
        colors_list.append(np.column_stack([
            np.random.uniform(0.85, 0.95, n_per_wall),
            np.random.uniform(0.82, 0.92, n_per_wall),
            np.random.uniform(0.75, 0.85, n_per_wall)
        ]))
        labels_list.append(np.ones(n_per_wall, dtype=np.int32))

    # Ceiling
    n_ceiling = n_points // 6
    cx = np.random.uniform(0, L, n_ceiling)
    cy = np.random.uniform(0, W, n_ceiling)
    cz = np.random.normal(H, 0.005, n_ceiling)
    points_list.append(np.column_stack([cx, cy, cz]))
    colors_list.append(np.full((n_ceiling, 3), 0.95))
    labels_list.append(np.full(n_ceiling, 2, dtype=np.int32))

    # Furniture (simple boxes)
    n_furniture = n_points // 4
    furniture_items = [
        {'pos': (1, 1), 'size': (2, 1, 0.75), 'name': 'desk'},
        {'pos': (5, 3), 'size': (2, 2.5, 0.4), 'name': 'table'},
        {'pos': (8, 1), 'size': (1.5, 0.8, 1.8), 'name': 'cabinet'},
        {'pos': (7, 5), 'size': (0.8, 0.8, 0.45), 'name': 'chair'},
    ]

    pts_per_item = n_furniture // len(furniture_items)
    for item in furniture_items:
        px, py = item['pos']
        sw, sd, sh = item['size']

        fx = np.random.uniform(px, px + sw, pts_per_item)
        fy = np.random.uniform(py, py + sd, pts_per_item)
        fz = np.random.uniform(0.01, sh, pts_per_item)
        points_list.append(np.column_stack([fx, fy, fz]))

        # Wood-ish colors
        colors_list.append(np.column_stack([
            np.random.uniform(0.5, 0.65, pts_per_item),
            np.random.uniform(0.35, 0.45, pts_per_item),
            np.random.uniform(0.2, 0.3, pts_per_item)
        ]))
        labels_list.append(np.full(pts_per_item, 3, dtype=np.int32))

    return np.vstack(points_list), np.concatenate(labels_list), np.vstack(colors_list)


def generate_multi_return_lidar(n_points=100000, extent=(200, 200), max_returns=4, seed=42):
    """
    Generate multi-return LiDAR data simulating canopy penetration.

    Args:
        n_points: Total points (sum of all returns)
        extent: Scene dimensions
        max_returns: Maximum returns per pulse
        seed: Random seed

    Returns:
        points: Nx3 array
        return_number: N array of return numbers (1 to max_returns)
        num_returns: N array of total returns for each pulse
        intensity: N array of simulated intensity values
    """
    np.random.seed(seed)
    w, h = extent

    # Generate pulse origins (fewer than total points)
    n_pulses = n_points // 2
    pulse_x = np.random.uniform(0, w, n_pulses)
    pulse_y = np.random.uniform(0, h, n_pulses)

    # Terrain height
    terrain_z = 2 * np.sin(pulse_x / 30) + 1.5 * np.cos(pulse_y / 25)

    points_list = []
    return_list = []
    num_returns_list = []
    intensity_list = []

    for i in range(n_pulses):
        x, y, base_z = pulse_x[i], pulse_y[i], terrain_z[i]

        # Determine number of returns based on "vegetation density"
        veg_density = 0.3 + 0.4 * np.sin(x / 50) * np.cos(y / 50)
        n_returns = np.random.binomial(max_returns - 1, veg_density) + 1

        # Generate returns from top to bottom
        z_heights = np.sort(np.random.uniform(base_z, base_z + 15, n_returns))[::-1]
        z_heights[-1] = base_z + np.random.normal(0, 0.03)  # Last return = ground

        for r in range(n_returns):
            points_list.append([x + np.random.normal(0, 0.02),
                               y + np.random.normal(0, 0.02),
                               z_heights[r]])
            return_list.append(r + 1)
            num_returns_list.append(n_returns)
            # Intensity decreases with return number
            base_intensity = 200 if r == n_returns - 1 else 120  # Ground stronger
            intensity_list.append(base_intensity + np.random.normal(0, 20))

    return (np.array(points_list),
            np.array(return_list, dtype=np.int32),
            np.array(num_returns_list, dtype=np.int32),
            np.clip(np.array(intensity_list), 0, 255).astype(np.uint8))


def generate_powerline_corridor(length=500, width=50, n_points=200000, seed=42):
    """
    Generate power line corridor data with towers, wires, vegetation.

    Args:
        length: Corridor length in meters
        width: Corridor width in meters
        n_points: Total points
        seed: Random seed

    Returns:
        points: Nx3 array
        labels: N array (0=ground, 1=vegetation, 2=tower, 3=wire)
    """
    np.random.seed(seed)

    points_list = []
    labels_list = []

    # Ground
    n_ground = n_points // 2
    gx = np.random.uniform(0, length, n_ground)
    gy = np.random.uniform(-width/2, width/2, n_ground)
    gz = 0.5 * np.sin(gx / 100) + np.random.normal(0, 0.1, n_ground)
    points_list.append(np.column_stack([gx, gy, gz]))
    labels_list.append(np.zeros(n_ground, dtype=np.int32))

    # Vegetation (avoiding center corridor)
    n_veg = n_points // 4
    vx = np.random.uniform(0, length, n_veg)
    vy = np.random.choice([-1, 1], n_veg) * np.random.uniform(width/4, width/2, n_veg)
    base_z = 0.5 * np.sin(vx / 100)
    vz = base_z + np.random.exponential(3, n_veg) + 0.5
    points_list.append(np.column_stack([vx, vy, vz]))
    labels_list.append(np.ones(n_veg, dtype=np.int32))

    # Towers (every 100m)
    n_towers = int(length / 100)
    tower_pts = 2000
    tower_height = 25

    for i in range(n_towers):
        tx = i * 100 + 50
        # Tower legs
        for leg_y in [-2, 2]:
            lx = np.random.normal(tx, 0.1, tower_pts // 4)
            ly = np.random.normal(leg_y, 0.1, tower_pts // 4)
            lz = np.random.uniform(0, tower_height, tower_pts // 4)
            points_list.append(np.column_stack([lx, ly, lz]))
            labels_list.append(np.full(tower_pts // 4, 2, dtype=np.int32))

        # Cross arms
        cx = np.random.normal(tx, 0.1, tower_pts // 2)
        cy = np.random.uniform(-8, 8, tower_pts // 2)
        cz = np.random.uniform(tower_height - 3, tower_height, tower_pts // 2)
        points_list.append(np.column_stack([cx, cy, cz]))
        labels_list.append(np.full(tower_pts // 2, 2, dtype=np.int32))

    # Wires
    n_wire = n_points // 10
    wire_offsets = [-6, -3, 0, 3, 6]  # 5 wires
    pts_per_wire = n_wire // len(wire_offsets)

    for offset in wire_offsets:
        wx = np.random.uniform(0, length, pts_per_wire)
        wy = np.full(pts_per_wire, offset) + np.random.normal(0, 0.05, pts_per_wire)
        # Catenary sag
        tower_spacing = 100
        sag = 2 * np.sin(np.pi * (wx % tower_spacing) / tower_spacing)
        wz = tower_height - 2 - sag + np.random.normal(0, 0.02, pts_per_wire)
        points_list.append(np.column_stack([wx, wy, wz]))
        labels_list.append(np.full(pts_per_wire, 3, dtype=np.int32))

    return np.vstack(points_list), np.concatenate(labels_list)