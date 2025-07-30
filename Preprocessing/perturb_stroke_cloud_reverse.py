import random
import numpy as np

import random

def stroke_node_features_to_polyline(stroke_node_features, is_feature_line):
    """
    Convert stroke_node_features into a list of stroke objects (polyline format).

    Each stroke object has:
    - type: "feature_line" or "normal_line"
    - feature_id: 0 (for now)
    - geometry: list of 3D points
    - id: unique id per stroke
    - opacity: random in [0.6, 0.8] for feature lines, [0.1, 0.3] otherwise
    """
    stroke_cloud = build_stroke_cloud_from_node_features(stroke_node_features)

    stroke_objects = []

    for idx, pts in enumerate(stroke_cloud):
        if is_feature_line[idx][0] == 1:
            opacity = random.uniform(0.6, 0.8)
            stroke_type = "feature_line"
        else:
            opacity = random.uniform(0.1, 0.3)
            stroke_type = "normal_line"

        stroke_obj = {
            "type": stroke_type,
            "feature_id": 0,
            "geometry": pts.tolist(),
            "id": idx,
            "opacity": opacity
        }
        stroke_objects.append(stroke_obj)

    return stroke_objects


def build_stroke_cloud_from_node_features(stroke_node_features, 
                                           num_points_straight=5, 
                                           num_points_arc=10, 
                                           num_points_circle=20):
    """
    Build a perturbed stroke cloud from clean stroke_node_features.

    Returns:
    - List of (M, 3) arrays, one array per perturbed stroke
    """
    stroke_cloud = []

    for stroke in stroke_node_features:
        if all(v == 0 for v in stroke):
            continue  # skip padded strokes

        stroke_type = stroke[-1]

        if stroke_type == 1:  # Straight line
            start = stroke[0:3]
            end = stroke[3:6]
            pts = np.linspace(start, end, num_points_straight)

            stroke_cloud.append(pts)

        elif stroke_type == 3:  # Arc
            pts = reconstruct_arc_points(stroke, num_points=num_points_arc)
            if pts.size == 0:
                return []
            stroke_cloud.append(pts)

        elif stroke_type == 2:  # Circle
            pts = reconstruct_circle_points(stroke, num_points=num_points_circle)
            stroke_cloud.append(pts)

    return stroke_cloud


import numpy as np

import numpy as np

def reconstruct_arc_points(stroke, num_points=10):
    """
    Reconstruct a 1/4 circle arc from start, end, and center.
    Only returns the arc if it passes geometric checks (no 3/4 circles).
    """
    start = np.array(stroke[0:3])
    end = np.array(stroke[3:6])
    center = np.array(stroke[7:10])

    # Detect axis-aligned plane
    tol = 1e-4
    if abs(start[0] - center[0]) < tol and abs(end[0] - center[0]) < tol:
        const_axis = 0
        var_axes = [1, 2]
    elif abs(start[1] - center[1]) < tol and abs(end[1] - center[1]) < tol:
        const_axis = 1
        var_axes = [0, 2]
    elif abs(start[2] - center[2]) < tol and abs(end[2] - center[2]) < tol:
        const_axis = 2
        var_axes = [0, 1]
    else:
        return np.empty((0, 3))

    # Project to 2D
    start_2d = start[var_axes] - center[var_axes]
    end_2d = end[var_axes] - center[var_axes]
    angle_start = np.arctan2(start_2d[1], start_2d[0])
    angle_end = np.arctan2(end_2d[1], end_2d[0])
    radius = np.linalg.norm(start_2d)
    max_valid_dist = radius + 1e-3  # small tolerance

    def sample_arc(angle0, angle1):
        angles = np.linspace(angle0, angle1, num_points)
        arc_2d = np.stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ], axis=1)
        arc_3d = np.zeros((num_points, 3))
        arc_3d[:, var_axes[0]] = arc_2d[:, 0] + center[var_axes[0]]
        arc_3d[:, var_axes[1]] = arc_2d[:, 1] + center[var_axes[1]]
        arc_3d[:, const_axis] = center[const_axis]
        return arc_3d

    # Try CCW direction
    arc_ccw_3d = sample_arc(angle_start, angle_end)
    if all(
        min(np.linalg.norm(pt - start), np.linalg.norm(pt - end)) <= max_valid_dist
        for pt in arc_ccw_3d
    ):
        return arc_ccw_3d

    # Try CW direction
    arc_cw_3d = sample_arc(angle_end, angle_start)
    if all(
        min(np.linalg.norm(pt - start), np.linalg.norm(pt - end)) <= max_valid_dist
        for pt in arc_cw_3d
    ):
        return arc_cw_3d

    # If neither direction is valid, reject
    return np.empty((0, 3))


def reconstruct_circle_points(stroke, num_points=20):
    center = np.array(stroke[0:3])
    normal = np.array(stroke[3:6])
    radius = stroke[7]

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Find two orthogonal vectors in the circle's plane
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    t_vals = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    pts = []

    for t in t_vals:
        point = center + radius * (np.cos(t) * u + np.sin(t) * v)
        pts.append(point)

    return np.array(pts)
