import numpy as np
import json
from scipy.spatial import cKDTree
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add


def read_step(filepath):
    try:
        step_reader = STEPControl_Reader()
        step_reader.ReadFile(str(filepath))
        step_reader.TransferRoot(1)
        shape = step_reader.Shape()
        return shape
    except Exception as e:
        print(f"Error reading STEP file: {filepath}, {e}")
        return None


def sample_points_from_shape(shape, sample_density, tolerance=1e-5):
    if shape is None:
        print("Shape is None, skipping sampling.")
        return np.array([])
    try:
        BRepMesh_IncrementalMesh(shape, tolerance)
        points = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods.Face(explorer.Current())
            adaptor = BRepAdaptor_Surface(face)
            umin, umax = adaptor.FirstUParameter(), adaptor.LastUParameter()
            vmin, vmax = adaptor.FirstVParameter(), adaptor.LastVParameter()
            u_step = (umax - umin) / sample_density
            v_step = (vmax - vmin) / sample_density
            u = umin
            while u <= umax:
                v = vmin
                while v <= vmax:
                    point = adaptor.Value(u, v)
                    points.append((point.X(), point.Y(), point.Z()))
                    v += v_step
                u += u_step
            explorer.Next()
        return np.array(points)
    except Exception as e:
        print(f"Error sampling points from shape: {e}")
        return np.array([])

def chamfer_distance(points1, points2, bbox_scale):
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        print("Empty point cloud detected!")
        return float('inf')
    try:
        tree = cKDTree(points2)
        dist, _ = tree.query(points1)

        # Cap each distance at bbox_scale
        # dist = np.minimum(dist, bbox_scale)

        # Sum of clipped distances
        return np.sum(dist)
    except Exception as e:
        print(f"Error computing Chamfer distance: {e}")
        return float('inf')


def apply_transform_from_json(shape, matrix_path):
    """Apply 4x4 transform matrix (from JSON file) to the shape."""
    try:
        with open(matrix_path, 'r') as f:
            matrix = json.load(f)

        r00, r01, r02, tx = matrix[0]
        r10, r11, r12, ty = matrix[1]
        r20, r21, r22, tz = matrix[2]

        trsf = gp_Trsf()
        trsf.SetValues(
            r00, r01, r02, tx,
            r10, r11, r12, ty,
            r20, r21, r22, tz
        )
        transform = BRepBuilderAPI_Transform(shape, trsf, True)
        return transform.Shape()
    except Exception as e:
        print(f"Error applying transform: {e}")
        return shape


from OCC.Core.BRepBndLib import brepbndlib

def compute_bbox_scale(shape):
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin
    max_range = max(x_range, y_range, z_range)
    return max_range


def compute_fidelity_score(gt_brep_path, output_brep_path, matrix_path, strict = False, sample_density=250, tolerance=1e-5):
    """
    Computes the fidelity score based on Chamfer distances between two BREP files.
    """
    try:
        # Read shapes
        if not Path(gt_brep_path).exists():
            return 0
        
        gt_shape = read_step(gt_brep_path)
        output_shape = read_step(output_brep_path)

        # Apply transformation to GT shape
        gt_shape = apply_transform_from_json(gt_shape, matrix_path)

        if gt_shape is None or output_shape is None:
            print("Invalid shape detected, skipping fidelity computation.")
            return 0

        # Sample points
        gt_points = sample_points_from_shape(gt_shape, tolerance, sample_density)
        output_points = sample_points_from_shape(output_shape, tolerance, sample_density)

        if gt_points.shape[0] == 0 or output_points.shape[0] == 0:
            print("Insufficient points sampled, skipping fidelity computation.")
            return 0

        bbox_scale = compute_bbox_scale(gt_shape)

        # Compute directional Chamfer distances
        gt_to_output = chamfer_distance(gt_points, output_points, bbox_scale)
        output_to_gt = chamfer_distance(output_points, gt_points, bbox_scale)

        # print("gt_to_output", gt_to_output)
        # print("output_to_gt", output_to_gt)

        # Normalize distances using bounding box scale
        norm_gt_to_output = gt_to_output / bbox_scale
        norm_output_to_gt = output_to_gt / bbox_scale

        # print(f"bbox_scale = {bbox_scale}")
        # print(f"normalized gt_to_output = {norm_gt_to_output}")
        # print(f"normalized output_to_gt = {norm_output_to_gt}")

        fidelity_score = min(1, 3 / (1 + norm_gt_to_output + norm_output_to_gt))

        if strict:
            fidelity_score = 1 / (1 + gt_to_output * 1000 + output_to_gt * 1000)

        return fidelity_score
    except Exception as e:
        print(f"Error computing fidelity score: {e}")
        return 0



def compute_fidelity_direct(gt_brep_path, output_brep_path, strict = False, sample_density=250, tolerance=1e-5):
    """
    Computes the fidelity score based on Chamfer distances between two BREP files.
    """
    try:
        # Read shapes
        if not Path(gt_brep_path).exists():
            return 0
        
        gt_shape = read_step(gt_brep_path)
        output_shape = read_step(output_brep_path)


        if gt_shape is None or output_shape is None:
            print("Invalid shape detected, skipping fidelity computation.")
            return 0

        # Sample points
        gt_points = sample_points_from_shape(gt_shape, tolerance, sample_density)
        output_points = sample_points_from_shape(output_shape, tolerance, sample_density)

        if gt_points.shape[0] == 0 or output_points.shape[0] == 0:
            print("Insufficient points sampled, skipping fidelity computation.")
            return 0

        bbox_scale = compute_bbox_scale(gt_shape)

        # Compute directional Chamfer distances
        gt_to_output = chamfer_distance(gt_points, output_points, bbox_scale)
        output_to_gt = chamfer_distance(output_points, gt_points, bbox_scale)

        # print("gt_to_output", gt_to_output)
        # print("output_to_gt", output_to_gt)

        # Normalize distances using bounding box scale
        norm_gt_to_output = gt_to_output / bbox_scale
        norm_output_to_gt = output_to_gt / bbox_scale

        # print(f"bbox_scale = {bbox_scale}")
        # print(f"normalized gt_to_output = {norm_gt_to_output}")
        # print(f"normalized output_to_gt = {norm_output_to_gt}")

        fidelity_score = min(1, 3 / (1 + norm_gt_to_output + norm_output_to_gt))

        if strict:
            fidelity_score = 1 / (1 + gt_to_output * 1000 + output_to_gt * 1000)

        return fidelity_score
    except Exception as e:
        print(f"Error computing fidelity score: {e}")
        return 0
