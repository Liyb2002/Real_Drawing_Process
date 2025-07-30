import os
import re
import open3d as o3d
import numpy as np
import shutil

def find_largest_step_index(stl_files):
    step_indices = []
    for f in stl_files:
        match = re.match(r"step_(\d+)\.stl", f)
        if match:
            step_indices.append(int(match.group(1)))
    return max(step_indices) if step_indices else None

def find_largest_step_index_gt(stl_files):
    step_indices = []
    for f in stl_files:
        match = re.match(r"shape_(\d+)\.stl", f)
        if match:
            step_indices.append(int(match.group(1)))
    return max(step_indices) if step_indices else None

def normalize_mesh(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    diagonal = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    if diagonal == 0:
        return mesh  # skip if degenerate
    mesh.translate(-center)  # center at origin
    mesh.scale(1.0 / diagonal, center=(0, 0, 0))  # scale to unit size
    return mesh

def mesh2mesh_distance(mesh1, mesh2):
    pcd = mesh1.sample_points_uniformly(1000)
    points = np.asarray(pcd.points, dtype='float32')

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh2))
    return scene.compute_distance(points).numpy().sum()

def compare_meshes(mesh1, mesh2):
    mesh1 = normalize_mesh(mesh1)
    mesh2 = normalize_mesh(mesh2)
    return mesh2mesh_distance(mesh1, mesh2) + mesh2mesh_distance(mesh2, mesh1)

# Base directories
base_dir = os.path.join(os.getcwd(), 'dataset', 'cad2sketch_cleaned')
gt_base_dir = os.path.join(os.getcwd(), 'dataset', 'cad2sketch')
success = 0


# Iterate over subfolders
for folder in os.listdir(base_dir):
    if not folder.isdigit():
        continue

    predicted_dir = os.path.join(base_dir, folder, 'canvas')
    gt_dir = os.path.join(gt_base_dir, folder)

    predicted_files = os.listdir(predicted_dir) if os.path.exists(predicted_dir) else []
    gt_files = os.listdir(gt_dir) if os.path.exists(gt_dir) else []

    x = find_largest_step_index(predicted_files)
    y = find_largest_step_index_gt(gt_files)

    if x is None or y is None:
        continue

    pred_mesh_path = os.path.join(predicted_dir, f'step_{x}.stl')
    gt_mesh_path = os.path.join(gt_dir, f'shape_{y}.stl')

    predicted_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)

    result = compare_meshes(predicted_mesh, gt_mesh)


    if result > 0.1:
        shutil.rmtree(os.path.join(base_dir, folder), ignore_errors=True)
    else:
        success += 1
        print(f"Folder {folder} — Distance: {result:.6f}")
    
print("total success", success)
