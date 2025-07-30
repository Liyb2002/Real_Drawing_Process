import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from math import inf
from numpy.linalg import norm
from scipy.interpolate import CubicSpline

import os
import pathlib

import Preprocessing.cad2sketch_stroke_features

def get_kth_operation(op_to_index_matrix, k):
    squeezed_matrix = op_to_index_matrix.squeeze(0)
    try:
        kth_operation = squeezed_matrix[:, k].unsqueeze(1)
        return kth_operation
    except IndexError:
        return None



def get_extruded_face(kth_operation, extrude_strokes, stroke_cloud_loops):
    # kth_operation and extrude_strokes are assumed to be numpy arrays with shape (num_strokes, 1)

    extruded_face_mask = (kth_operation == 1) & (extrude_strokes < 0.5)
    
    chosen_strokes = (extruded_face_mask == 1).nonzero(as_tuple=True)[0]  # Indices of chosen stroke
    loop_chosen_mask = []
    for loop in stroke_cloud_loops:
        if all(stroke in chosen_strokes for stroke in loop):
            loop_chosen_mask.append(1)  # Loop is chosen
        else:
            loop_chosen_mask.append(0)  # Loop is not chosen
    
    
    loop_chosen_mask = torch.tensor(loop_chosen_mask, dtype=torch.float).flatten()
    
    return loop_chosen_mask, chosen_strokes


def get_all_operation_strokes(stroke_operations_order_matrix, program_whole, operation):
    # Find the indices in program_whole where the operation occurs
    ks = [i for i, op in enumerate(program_whole) if op == operation]

    # Check if we found any valid indices
    if len(ks) == 0:
        return None

    # Squeeze the matrix to remove any singleton dimensions at the 0-th axis
    squeezed_matrix = stroke_operations_order_matrix.squeeze(0)

    # Initialize an empty list to collect all columns corresponding to ks
    operation_strokes_list = []

    # Collect the k-th columns and stack them into a new matrix
    for k in ks:
        kth_operation = squeezed_matrix[:, k].unsqueeze(1)  # Extract and unsqueeze the k-th column
        operation_strokes_list.append(kth_operation)

    # Stack all the k-th columns into a matrix of shape (op_to_index.shape[0], n)
    all_operation_strokes = torch.cat(operation_strokes_list, dim=1)

    # Perform a logical OR (any row that has a 1 in any column will have 1 in the result)
    result_strokes = (all_operation_strokes > 0).any(dim=1).float().unsqueeze(1)

    # Return the result as a column vector of shape (op_to_index.shape[0], 1)
    return result_strokes

def get_feature_strokes(gnn_graph):

    features_strokes = gnn_graph['stroke'].x[:, -1]

    # Iterate over each stroke and modify features_strokes based on the condition
    for i, stroke in enumerate(gnn_graph['stroke'].x):
        if stroke[7] > 0:
            features_strokes[i] = 1
        elif features_strokes[i] != 0 and features_strokes[i] != 1:
            features_strokes[i] = 0

    return features_strokes.clone()



def is_perpendicular(line1, line2, tol=5e-5):
    """Check if two lines (each defined by two 3D points) are perpendicular."""
    vec1 = np.array(line1[1]) - np.array(line1[0])
    vec2 = np.array(line2[1]) - np.array(line2[0])
    dot_product = np.dot(vec1, vec2)

    return abs(dot_product) < tol


def min_distance_to_endpoints(p1, p2, endpoints):
    distances = [np.linalg.norm(np.array(p1) - np.array(ep)) for ep in endpoints] + \
                [np.linalg.norm(np.array(p2) - np.array(ep)) for ep in endpoints]
    return min(distances)


def point_matches(p, endpoints, threshold):
    """Return True if point `p` is within `threshold` distance of any point in `endpoints`."""
    for ep in endpoints:
        if np.linalg.norm(np.array(p) - np.array(ep)) < threshold:
            return True
    return False


def is_colinear_with_extension(line1, line2, tolerance=1e-6):
    # Check if line1 is on the extension of line2
    p1, p2 = np.array(line1[0]), np.array(line1[1])
    q1, q2 = np.array(line2[0]), np.array(line2[1])
    v1 = p2 - p1
    v2 = q2 - q1

    # Check if directions are parallel (cross product near zero)
    if np.linalg.norm(np.cross(v1, v2)) > tolerance:
        return False

    # Project p1 onto line q1-q2
    if np.linalg.norm(np.cross(v2, p1 - q1)) > tolerance and \
       np.linalg.norm(np.cross(v2, p2 - q1)) > tolerance:
        return False

    return True


def choose_extrude_strokes(stroke_selection_mask, sketch_selection_mask, stroke_node_features):
    num_strokes = stroke_node_features.shape[0]
    extrude_strokes = np.zeros((num_strokes, 1), dtype=int)

    from_circle = any(
        sketch_selection_mask[i] == 1 and stroke_node_features[i][-1] == 2
        for i in range(num_strokes)
    )

    if from_circle:
        return choose_extrude_strokes_from_circle(stroke_selection_mask, sketch_selection_mask, stroke_node_features)

    sketch_lines = []
    sketch_endpoints = []
    for i in range(num_strokes):
        if sketch_selection_mask[i] == 1:
            p1 = tuple(stroke_node_features[i][:3])
            p2 = tuple(stroke_node_features[i][3:6])
            sketch_lines.append((p1, p2))
            sketch_endpoints.extend([p1, p2])

    if len(sketch_endpoints) == 0:
        return extrude_strokes
    
    for i in range(num_strokes):
        if stroke_selection_mask[i] == 1:
            p1 = tuple(stroke_node_features[i][:3])
            p2 = tuple(stroke_node_features[i][3:6])
            candidate_line = (p1, p2)
            length = np.linalg.norm(np.array(p2) - np.array(p1))
            if all(is_perpendicular(candidate_line, sketch_line) for sketch_line in sketch_lines) and \
               min_distance_to_endpoints(p1, p2, sketch_endpoints) < 0.2 * length:
                extrude_strokes[i] = 1

    for i in range(num_strokes):
        if extrude_strokes[i] == 1:
            p1 = tuple(stroke_node_features[i][:3])
            p2 = tuple(stroke_node_features[i][3:6])
            line = (p1, p2)
            for sketch_line in sketch_lines:
                if is_colinear_with_extension(line, sketch_line):
                    extrude_strokes[i] = 0
                    break
    
    # fallback
    if not np.any(extrude_strokes):
        for i in range(num_strokes):
            if stroke_node_features[i][-1] != 1:
                continue
            p1 = tuple(stroke_node_features[i][:3])
            p2 = tuple(stroke_node_features[i][3:6])
            candidate_line = (p1, p2)
            length = np.linalg.norm(np.array(p2) - np.array(p1))
            threshold = 0.2 * length
            if all(is_perpendicular(candidate_line, sketch_line) for sketch_line in sketch_lines):
                match_count = int(point_matches(p1, sketch_endpoints, threshold)) + \
                              int(point_matches(p2, sketch_endpoints, threshold))
                if match_count == 1:
                    extrude_strokes[i] = 1

    return extrude_strokes


# Circle Feature: 10 values + type 2
# 0-2: center, 3-5:normal, 6:alpha_value, 7:radius, 8-9: 0

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def choose_extrude_strokes_from_circle(stroke_selection_mask, sketch_selection_mask, stroke_node_features):
    num_strokes = stroke_node_features.shape[0]
    extrude_strokes = np.zeros((num_strokes, 1), dtype=int)

    sketch_center = None
    sketch_radius = 0
    cylinder_height = 0
    EPSILON = 5e-5

    # Find one circle in the selected sketch
    for i in range(num_strokes):
        if sketch_selection_mask[i] == 1 and stroke_node_features[i][-1] == 2:
            sketch_center = stroke_node_features[i][0:3]
            sketch_radius = stroke_node_features[i][7]
            break

    if sketch_center is None:
        return extrude_strokes  # No valid circle found

    # Find matching circle with same radius but different center
    for i in range(num_strokes):
        if stroke_node_features[i][-1] == 2:
            center = stroke_node_features[i][0:3]
            radius = stroke_node_features[i][7]


            center_dist = dist(center, sketch_center)
            radius_diff = abs(radius - sketch_radius) 

            radius_threshold = sketch_radius * 0.1
            center_threshold = sketch_radius * 0.1

            if center_dist > center_threshold and radius_diff < radius_threshold:
                cylinder_height = center_dist
                break
    
    # Find lines that connect the two circles (extrude strokes)
    for i in range(num_strokes):
        if stroke_node_features[i][-1] == 1:
            point_1 = stroke_node_features[i][0:3]
            point_2 = stroke_node_features[i][3:6]
            length = dist(point_1, point_2)


            if abs(length - cylinder_height) < 5e-4 and dist(point_1, point_2) > 1e-5:
                d1 = dist(point_1, sketch_center)
                d2 = dist(point_2, sketch_center)

                if abs(d1 - sketch_radius) < 5e-4 or abs(d2 - sketch_radius) < 5e-4:
                    extrude_strokes[i] = 1

    return extrude_strokes



def choose_fillet_strokes(raw_fillet_stroke_idx, stroke_node_features):
    # Filter raw_fillet_stroke_idx based on conditions in stroke_node_features
    filtered_strokes = [
        idx for idx in raw_fillet_stroke_idx
        if stroke_node_features[idx][-1] == 3
    ]
    
    num_strokes = stroke_node_features.shape[0]
    stroke_selection_matrix = torch.zeros((num_strokes, 1), dtype=torch.float32)
    
    # Set the value to 1 for the selected strokes
    for idx in filtered_strokes:
        stroke_selection_matrix[idx] = 1.0  # Use 1.0 to ensure float32 type
    
    return filtered_strokes, stroke_selection_matrix


def dist(point1, point2):
    # Calculate the Euclidean distance between two points
    return norm([p1 - p2 for p1, p2 in zip(point1, point2)])

def choose_chamfer_strokes(raw_chamfer_stroke_idx, stroke_node_features):
    # Filter `raw_chamfer_stroke_idx` based on conditions in `stroke_node_features`
    min_stroke_length = inf
    chamfer_strokes = []

    # Find the minimum stroke length among the specified strokes
    for idx in raw_chamfer_stroke_idx:
        stroke_length = dist(stroke_node_features[idx][0:3], stroke_node_features[idx][3:6])
        if stroke_length < min_stroke_length:
            min_stroke_length = stroke_length

    # Collect strokes with the minimum length found
    for idx in raw_chamfer_stroke_idx:
        if dist(stroke_node_features[idx][0:3], stroke_node_features[idx][3:6]) == min_stroke_length:
            chamfer_strokes.append(idx)  # Fix: `append` is a method and should use parentheses

    # Create a selection matrix with the same number of strokes as in stroke_node_features
    num_strokes = stroke_node_features.shape[0]
    stroke_selection_matrix = torch.zeros((num_strokes, 1), dtype=torch.float32)

    # Set the value to 1 for the selected strokes
    for idx in chamfer_strokes:
        stroke_selection_matrix[idx] = 1.0  # Use 1.0 to ensure float32 type

    return chamfer_strokes, stroke_selection_matrix
#------------------------------------------------------------------------------------------------------#



def stroke_to_face(kth_operation, face_to_stroke):
    num_faces = len(face_to_stroke)
    face_chosen = torch.zeros((num_faces, 1), dtype=torch.float32)
    
    for i, strokes in enumerate(face_to_stroke):
        if all(kth_operation[stroke].item() == 1 for stroke in strokes):
            face_chosen[i] = 1

    return face_chosen



#------------------------------------------------------------------------------------------------------#



def program_mapping(program, device):
    operation_map = {
        'sketch': 1,
        'extrude': 2,
        'fillet': 3,
        'chamfer': 4,
        'start': 9, 
        'terminate': 0,
        'padding': 10
    }
    
    # Map each operation in the program list to its corresponding value
    mapped_program = [operation_map.get(op, -1) for op in program] 
    mapped_program.insert(0,9)
    
    for i in range (20 - len(mapped_program)):
        mapped_program.append(10)
    
    mapped_program_tensor = torch.tensor(mapped_program, dtype=torch.long, device=device)

    return mapped_program_tensor


def program_gt_mapping(program, device):
    operation_map = {
        'sketch': 1,
        'extrude': 2,
        'fillet': 3,
        'chamfer': 4,
        'start': 9, 
        'terminate': 0,
        'padding': 10
    }
    
    # Map each operation in the program list to its corresponding value
    mapped_program = [operation_map.get(op, -1) for op in program]
    
    mapped_program_tensor = torch.tensor(mapped_program, dtype=torch.long, device=device)

    return mapped_program_tensor

#------------------------------------------------------------------------------------------------------#

def find_edge_features_slice(tensor, i):
    """
    Extract edges from the tensor where both nodes are within the range
    [i * 200, (i + 1) * 200), and adjust the values using modulo 200.

    Args:
    tensor (torch.Tensor): Input tensor of shape (2, n), where each column represents an edge between two nodes.
    i (int): The batch index.

    Returns:
    torch.Tensor: Filtered and adjusted tensor of shape (2, k) where both nodes in each edge are within [i * 200, (i + 1) * 200),
                  adjusted to range [0, 199] via modulo operation.
    """
    # Define the start and end of the range based on i
    start = i * 200
    end = (i + 1) * 200
    
    # Get the two rows representing the edges
    edges = tensor
    
    # Create a mask where both nodes in each edge are within the range [start, end)
    mask = (edges[0] >= start) & (edges[0] < end) & (edges[1] >= start) & (edges[1] < end)
    
    # Apply the mask to filter the edges
    filtered_edges = edges[:, mask]
    
    # Adjust the values to range [0, 199] using modulo 200
    adjusted_edges = filtered_edges % 200
    
    return adjusted_edges


#------------------------------------------------------------------------------------------------------#

def face_is_not_in_brep(matrix, face_to_stroke, node_features, edge_features):
    # Find the max index in the matrix
    max_index = torch.argmax(matrix).item()
    
    # Get the strokes associated with the chosen face
    chosen_face_strokes = face_to_stroke[max_index]
    
    # Check if any of the strokes are in edge_features
    for stroke_index in chosen_face_strokes:
        stroke_value = node_features[stroke_index]
        if any(torch.equal(stroke_value, edge) for edge in edge_features):
            return False
    
    return True


def predict_face_coplanar_with_brep(predicted_index, coplanar_matrix, node_features):
    # Step 1: Find all coplanar faces with the predicted_index using coplanar_matrix
    coplanar_faces = torch.where(coplanar_matrix[predicted_index] == 1)[0]
    
    # Step 2: Check if any of the coplanar faces are used, using the last column of node_features
    if torch.any(node_features[coplanar_faces, -1] == 1):
        return True
    
    # Step 3: If no coplanar face is used, return False
    return False




#------------------------------------------------------------------------------------------------------#


def build_intersection_matrix(node_features):
    num_strokes = node_features.shape[0]
    intersection_matrix = torch.zeros((num_strokes, num_strokes), dtype=torch.int32)

    for i in range(num_strokes):
        for j in range(i + 1, num_strokes):
            # Extract points for strokes i and j
            stroke_i_points = node_features[i].view(2, 3)
            stroke_j_points = node_features[j].view(2, 3)
            
            # Check for intersection
            if (stroke_i_points[0] == stroke_j_points[0]).all() or \
               (stroke_i_points[0] == stroke_j_points[1]).all() or \
               (stroke_i_points[1] == stroke_j_points[0]).all() or \
               (stroke_i_points[1] == stroke_j_points[1]).all():
                intersection_matrix[i, j] = 1
                intersection_matrix[j, i] = 1

    return intersection_matrix


#------------------------------------------------------------------------------------------------------#


def clean_face_choice(predicted_index, node_features):
    # Check if the predicted_index itself is not being used
    if node_features[predicted_index, -1] == 0:
        return True
    else:
        return False




#------------------------------------------------------------------------------------------------------#
def vis_left_graph(stroke_node_features):
    """
    Visualizes strokes in 3D space with color coding based on the last feature.
    Strokes are initially plotted in blue with a hand-drawn effect, 
    and then strokes with stroke[-1] == 1 are highlighted in green.

    Parameters:
    - stroke_node_features: A numpy array or list containing the features of each stroke.
      Each stroke should contain its start and end coordinates, with additional
      flags indicating if it's a circle or arc and the color coding based on the last element.
    """
    
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    perturb_factor = 0.002  # Adjusted perturbation factor for hand-drawn effect

    # First pass: Plot all strokes in blue with perturbations
    for stroke in stroke_node_features:
        start, end = stroke[:3], stroke[3:6]
        alpha_value = stroke[6]
        
        # Ignore invalid strokes marked with specific values
        if stroke[-2] == -1 and stroke[-3] == -1 and stroke[-4] == -1:
            continue
        
        # Set color to blue for the initial pass
        color = 'black'

        # Update min and max limits for rescaling (ignoring circles)
        if stroke[-2] == 1:
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        if stroke[-2] == 2:
            # Circle face
            x_values, y_values, z_values = plot_circle(stroke)
            ax.plot(x_values, y_values, z_values, color=color, linewidth=0.5, alpha = alpha_value)
            continue

        if stroke[-2] == 3:
            # Arc
            x_values, y_values, z_values = plot_arc(stroke)
            ax.plot(x_values, y_values, z_values, color=color, linewidth=0.5, alpha = alpha_value)
            continue

        else:
            # Hand-drawn effect for regular stroke line
            x_values = np.array([start[0], end[0]])
            y_values = np.array([start[1], end[1]])
            z_values = np.array([start[2], end[2]])
            
            # Perturb points along the line
            perturbations = np.random.normal(0, perturb_factor, (10, 3))  # 10 intermediate points
            t = np.linspace(0, 1, 10)
            x_interpolated = np.linspace(x_values[0], x_values[1], 10) + perturbations[:, 0]
            y_interpolated = np.linspace(y_values[0], y_values[1], 10) + perturbations[:, 1]
            z_interpolated = np.linspace(z_values[0], z_values[1], 10) + perturbations[:, 2]

            # Smooth with cubic splines
            cs_x = CubicSpline(t, x_interpolated)
            cs_y = CubicSpline(t, y_interpolated)
            cs_z = CubicSpline(t, z_interpolated)
            smooth_t = np.linspace(0, 1, 100)
            smooth_x = cs_x(smooth_t)
            smooth_y = cs_y(smooth_t)
            smooth_z = cs_z(smooth_t)

            # Plot with hand-drawn effect
            ax.plot(smooth_x, smooth_y, smooth_z, color=color, linewidth=0.5, alpha = alpha_value)

    # Second pass: Plot selected strokes in green with hand-drawn effect to overlay the blue ones
    for stroke in stroke_node_features:
        if stroke[-1] == 1:
            start, end = stroke[:3], stroke[3:6]
            
            # Ignore invalid strokes marked with specific values
            if stroke[-2] == -1 and stroke[-3] == -1 and stroke[-4] == -1:
                continue
            
            color = 'blue'
            
            if stroke[-2] == 2:
                # Circle face
                x_values, y_values, z_values = plot_circle(stroke)
                ax.plot(x_values, y_values, z_values, color=color, linewidth=1, alpha = alpha_value)
                continue

            if stroke[-2] == 3:
                # Arc
                x_values, y_values, z_values = plot_arc(stroke)
                ax.plot(x_values, y_values, z_values, color=color, linewidth=1, alpha = alpha_value)
                continue

            else:
                # Hand-drawn effect for selected green stroke line
                x_values = np.array([start[0], end[0]])
                y_values = np.array([start[1], end[1]])
                z_values = np.array([start[2], end[2]])
                
                perturbations = np.random.normal(0, perturb_factor, (10, 3))
                t = np.linspace(0, 1, 10)
                x_interpolated = np.linspace(x_values[0], x_values[1], 10) + perturbations[:, 0]
                y_interpolated = np.linspace(y_values[0], y_values[1], 10) + perturbations[:, 1]
                z_interpolated = np.linspace(z_values[0], z_values[1], 10) + perturbations[:, 2]

                cs_x = CubicSpline(t, x_interpolated)
                cs_y = CubicSpline(t, y_interpolated)
                cs_z = CubicSpline(t, z_interpolated)
                smooth_t = np.linspace(0, 1, 100)
                smooth_x = cs_x(smooth_t)
                smooth_y = cs_y(smooth_t)
                smooth_z = cs_z(smooth_t)

                ax.plot(smooth_x, smooth_y, smooth_z, color=color, linewidth=1, alpha = alpha_value)

    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])


    # Show plot
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

def vis_left_graph_loops(stroke_node_features, loop_node_features, stroke_cloud_loops):
    """
    Visualizes strokes and loops in 3D space.
    
    1. Plots regular strokes in black.
    2. Highlights strokes associated with used loops (loop_node_features[-1] == 1) in red.
    3. Considers stroke types (line, circle, arc) for plotting.
    
    Parameters:
    - stroke_node_features: A numpy array of shape (num_strokes, 12) representing stroke features.
    - loop_node_features: A numpy array of shape (num_loops, 12) representing loop features.
    - stroke_cloud_loops: A list of sublists, where each sublist contains the stroke indices corresponding to a loop.
    """
    stroke_node_features = stroke_node_features[:, :-1]  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    perturb_factor = 0.000002

    # Helper to plot a single stroke (with optional color override)
    def plot_stroke(idx, color='black'):
        stroke = stroke_node_features[idx]
        start, end = stroke[:3], stroke[3:6]

        if stroke[-2] == -1 and stroke[-3] == -1 and stroke[-4] == -1:
            return

        nonlocal x_min, x_max, y_min, y_max, z_min, z_max

        if stroke[-1] == 1:
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        if stroke[-1] == 2:
            x_values, y_values, z_values = plot_circle(stroke)
            ax.plot(x_values, y_values, z_values, color=color, alpha=1.0)
        elif stroke[-1] == 3:
            x_values, y_values, z_values = plot_arc(stroke)
            ax.plot(x_values, y_values, z_values, color=color, alpha=1.0)
        else:
            x_values = np.array([start[0], end[0]])
            y_values = np.array([start[1], end[1]])
            z_values = np.array([start[2], end[2]])

            perturbations = np.random.normal(0, perturb_factor, (10, 3))
            t = np.linspace(0, 1, 10)
            x_interpolated = np.linspace(x_values[0], x_values[1], 10) + perturbations[:, 0]
            y_interpolated = np.linspace(y_values[0], y_values[1], 10) + perturbations[:, 1]
            z_interpolated = np.linspace(z_values[0], z_values[1], 10) + perturbations[:, 2]

            cs_x = CubicSpline(t, x_interpolated)
            cs_y = CubicSpline(t, y_interpolated)
            cs_z = CubicSpline(t, z_interpolated)
            smooth_t = np.linspace(0, 1, 100)
            smooth_x = cs_x(smooth_t)
            smooth_y = cs_y(smooth_t)
            smooth_z = cs_z(smooth_t)

            ax.plot(smooth_x, smooth_y, smooth_z, color=color, alpha=1.0, linewidth=0.5)

    # Plot all strokes initially in black
    for idx in range(len(stroke_node_features)):
        plot_stroke(idx, color='black')

    # Plot strokes from used loops in red
    for i, loop_row in enumerate(loop_node_features):
        if loop_row[-1] == 1:
            stroke_indices = stroke_cloud_loops[i]
            for idx in stroke_indices:
                plot_stroke(idx, color='red')

    # Center and scale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.show()



def vis_brep(brep):
    """
    Visualize the brep strokes and circular/cylindrical faces in 3D space if brep is not empty.
    
    Parameters:
    brep (np.ndarray or torch.Tensor): A matrix with shape (num_strokes, 6) representing strokes.
                       Each row contains two 3D points representing the start and end of a stroke.
                       If brep.shape[0] == 0, the function returns without plotting.
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Check if brep is empty
    if brep.shape[0] == 0:
        plt.title('Empty Plot')
        plt.show()
        return

    # Convert brep to numpy if it's a tensor
    if not isinstance(brep, np.ndarray):
        brep = brep.numpy()

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all brep strokes and circle/cylinder faces in blue

    # Last values
    # Straight Line: 1
    # Circle Feature: 2
    # Cylinder Face Feature: 3
    # Arc Feature: 4

    for stroke in brep:
        
        if stroke[-1] == 3:
            # Cylinder face
            center = stroke[:3]
            normal = stroke[3:6]
            height = stroke[6]
            radius = stroke[7]

            # Generate points for the cylinder's base circle (less dense)
            theta = np.linspace(0, 2 * np.pi, 30)  # Less dense with 30 points
            x_values = radius * np.cos(theta)
            y_values = radius * np.sin(theta)
            z_values = np.zeros_like(theta)

            # Combine the coordinates into a matrix (3, 30)
            base_circle_points = np.array([x_values, y_values, z_values])

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            # Rotation logic using Rodrigues' formula
            z_axis = np.array([0, 0, 1])  # Z-axis is the default normal for the cylinder

            # Rotate the base circle points to align with the normal vector (even if normal is aligned)
            rotation_axis = np.cross(z_axis, normal)
            if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
                rotation_axis /= np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

                # Create the rotation matrix using the rotation axis and angle (Rodrigues' rotation formula)
                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                              [rotation_axis[2], 0, -rotation_axis[0]],
                              [-rotation_axis[1], rotation_axis[0], 0]])

                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

                # Rotate the base circle points
                rotated_base_circle_points = np.dot(R, base_circle_points)
            else:
                rotated_base_circle_points = base_circle_points

            # Translate the base circle to the center point
            x_base = rotated_base_circle_points[0] + center[0]
            y_base = rotated_base_circle_points[1] + center[1]
            z_base = rotated_base_circle_points[2] + center[2]

            # Plot the base circle
            ax.plot(x_base, y_base, z_base, color='blue')

            # Plot vertical lines to create the "cylinder" (but without filling the body)
            x_top = x_base - normal[0] * height
            y_top = y_base - normal[1] * height
            z_top = z_base - normal[2] * height

            # Plot lines connecting the base and top circle with reduced density
            for i in range(0, len(x_base), 3):  # Fewer lines by skipping points
                ax.plot([x_base[i], x_top[i]], [y_base[i], y_top[i]], [z_base[i], z_top[i]], color='blue')

            # Update axis limits for the cylinder points
            x_min, x_max = min(x_min, x_base.min(), x_top.min()), max(x_max, x_base.max(), x_top.max())
            y_min, y_max = min(y_min, y_base.min(), y_top.min()), max(y_max, y_base.max(), y_top.max())
            z_min, z_max = min(z_min, z_base.min(), z_top.min()), max(z_max, z_base.max(), z_top.max())

        elif stroke[-1] == 2:
            # Circle face (same rotation logic as shared)
            x_values, y_values, z_values = plot_circle(stroke)
            ax.plot(x_values, y_values, z_values, color='blue')
        
        elif stroke[-1] == 4:
            # plot arc 
            x_values, y_values, z_values = plot_arc(stroke)
            ax.plot(x_values, y_values, z_values, color='blue')


        else:
            # Plot the stroke
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

            # Update axis limits for the stroke points
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

    # Compute the center of the shape
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Compute the maximum difference across x, y, z directions
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Set the same limits for x, y, and z axes centered around the computed center
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()


def vis_brep_with_indices(brep, indices):
    """
    Visualize the BREP strokes and circular/cylindrical faces in 3D space,
    highlighting the specified edges in red.

    Parameters:
    - brep (np.ndarray or torch.Tensor): A matrix with shape (num_strokes, 12) representing strokes.
        Each row contains two 3D points representing the start and end of a stroke.
    - indices (list): List of indices in the BREP to highlight in red.
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Check if BREP is empty
    if brep.shape[0] == 0:
        plt.title('Empty Plot')
        plt.show()
        return

    # Convert BREP to numpy if it's a tensor
    if not isinstance(brep, np.ndarray):
        brep = brep.numpy()

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot blue strokes first
    for i, stroke in enumerate(brep):
        if i not in indices:  # Plot only non-highlighted strokes
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

            # Update axis limits
            x_min = min(x_min, start[0], end[0])
            x_max = max(x_max, start[0], end[0])
            y_min = min(y_min, start[1], end[1])
            y_max = max(y_max, start[1], end[1])
            z_min = min(z_min, start[2], end[2])
            z_max = max(z_max, start[2], end[2])

    # Plot red strokes last
    for i in indices:
        stroke = brep[i]
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', linewidth=2)

    # Compute the center of the shape
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Compute the maximum difference across x, y, z directions
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Set the same limits for x, y, and z axes centered around the computed center
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()




# Straight Line: 10 values + type 1
# 0-2: point1, 3-5:point2, 6:alpha_value, 7-9: 0

# Circle Feature: 10 values + type 2
# 0-2: center, 3-5:normal, 6:alpha_value, 7:radius, 8-9: 0

# Arc Feature: 10 values + type 3
# 0-2: point1, 3-5:point2, 6:alpha_value, 7-9:center

# Ellipse Feature: 10 values + type 4
# 0-2: center, 3-5:normal, 6:alpha_value, 7: major axis, 8: minor axis, 9: orientation

# Closed Line: 10 values + type 5
# 0-2: point1, 3-5: point2, 6:alpha_value, 7-9: random point in the line

# Curved Line: 10 values + type 6
# 0-2: point1, 3-5: point2, 6:alpha_value, 7-9: random point in the line


def vis_selected_strokes(stroke_node_features, selected_stroke_idx, data_idx, alpha_value=0.7):
    """
    Visualizes selected strokes in 3D space with a hand-drawn effect.

    Parameters:
    - stroke_node_features: A numpy array or list containing the features of each stroke.
    - selected_stroke_idx: A list or array of indices of the strokes that should be highlighted in red.
    - data_idx: string to locate the dataset subfolder
    - alpha_value: Float, default 0.7, controls transparency of red highlighted strokes.
    """
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import random

    # Load the already-processed all_lines
    final_edges_file_path = os.path.join(
        os.getcwd(), 'dataset', 'small', data_idx, 'final_edges.json')
    final_edges_data = read_json(final_edges_file_path)


    all_lines = []

    for key, stroke in final_edges_data.items():
        all_lines.append(stroke)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # === Set Camera View ===
    ax.view_init(elev=-150, azim=57, roll=0)  # Match the provided camera inclination and azimuth
    # ax.dist = 7  # Simulate distance/zoom, optional

    # Clean plot styling
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_axis_off()

    # Bounding box init
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # === First pass: plot all strokes ===
    for i, stroke in enumerate(all_lines):
        geometry = stroke["geometry"]
        if len(geometry) < 2:
            continue
        
        stroke_type = 'feature_line'
        
        if stroke_type == 'feature_line' or stroke_type == 'extrude_line' or stroke_type == 'fillet_line':
            alpha = 0.6
        else:
            alpha = 0.2

        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='black',
                    linewidth=0.6,
                    alpha=alpha)

    # === Rescale view ===
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # === Second pass: highlight selected strokes in red ===
    for idx in selected_stroke_idx:
        if idx < len(all_lines):
            geometry = all_lines[idx]["geometry"]
            if len(geometry) < 2:
                continue
            for j in range(1, len(geometry)):
                start = geometry[j - 1]
                end = geometry[j]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='red',
                        linewidth=2.0,
                        alpha=alpha_value)
        else:
            stroke = stroke_node_features[idx]
            start = stroke[0:3]
            end = stroke[3:6]
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='red',
                    linewidth=2.0,
                    alpha=alpha_value)

    plt.show()



def vis_selected_strokes_synthetic(stroke_node_features, selected_stroke_idx, data_idx, alpha_value=0.7):
    """
    Visualizes selected strokes in 3D space with a hand-drawn effect.

    Parameters:
    - stroke_node_features: A numpy array or list containing the features of each stroke.
    - selected_stroke_idx: A list or array of indices of the strokes that should be highlighted in red.
    - data_idx: string to locate the dataset subfolder
    - alpha_value: Float, default 0.7, controls transparency of red highlighted strokes.
    """
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import random

    # Load the already-processed all_lines
    final_edges_file_path = os.path.join(
        os.getcwd(), 'dataset', 'mid', data_idx, 'perturbed_all_lines.json')
    all_lines = read_json(final_edges_file_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-30, azim=50, roll=100)  # Match the provided camera inclination and azimuth

    # Clean plot styling
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_axis_off()

    # Bounding box init
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # === First pass: plot all strokes ===
    for i, stroke in enumerate(all_lines):
        geometry = stroke["geometry"]
        if len(geometry) < 2:
            continue

        # Assign opacity if not already in the line

        alpha = stroke["opacity"]

        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='black',
                    linewidth=0.6,  # thicker for higher opacity
                    alpha=alpha)

    # === Rescale view ===
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # === Second pass: highlight selected strokes in red ===
    for idx in selected_stroke_idx:
        if idx < len(all_lines):
            geometry = all_lines[idx]["geometry"]
            if len(geometry) < 2:
                continue
            for j in range(1, len(geometry)):
                start = geometry[j - 1]
                end = geometry[j]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='red',
                        linewidth=3.0,
                        alpha=1.0)
        else:
            stroke = stroke_node_features[idx]
            start = stroke[0:3]
            end = stroke[3:6]
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='red',
                    linewidth=3.0,
                    alpha=1.0)

    plt.show()






def vis_used_strokes(stroke_node_features, data_idx, alpha_value=0.7):
    """
    Visualizes selected strokes in 3D space with a hand-drawn effect.

    Parameters:
    - stroke_node_features: A numpy array or list containing the features of each stroke.
      Each stroke should contain its start and end coordinates, and potentially a flag indicating if it's a circle.
    - selected_stroke_idx: A list or array of indices of the strokes that should be highlighted in red.
    - alpha_value: Float, optional. The transparency level of the lines (0.0 is fully transparent, 1.0 is fully opaque).
    """
    
    # Initialize the 3D plot
    final_edges_file_path = os.path.join(
    os.getcwd(), 'dataset', 'cad2sketch_annotated', data_idx, 'final_edges.json')
    final_edges_data = read_json(final_edges_file_path)
    all_lines = extract_all_lines(final_edges_data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove axis labels, ticks, and background
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_axis_off()

    # Initialize bounding box variables
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # First pass: plot all strokes in black and compute bounding box
    for stroke in all_lines:
        geometry = stroke["geometry"]

        if len(geometry) < 2:
            continue

        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            # Update bounding box
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            # Plot the stroke in black
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    color='black', 
                    linewidth=0.5)

    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Second pass: highlight used strokes in red
    for idx, stroke in enumerate(stroke_node_features):

        if stroke[-1] != 0:

            if idx < len(all_lines):
                geometry = all_lines[idx]["geometry"]
                if len(geometry) < 2:
                    continue
                for j in range(1, len(geometry)):
                    start = geometry[j - 1]
                    end = geometry[j]
                    ax.plot([start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]],
                            color='red',
                            linewidth=1.0)
            else:
                stroke = stroke_node_features[idx]
                start = stroke[0:3]
                end = stroke[3:6]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='red',
                        linewidth=1.0)

    plt.show()




def vis_used_strokes_lifted(stroke_node_features, data_idx, lifted_stroke_node_features_bbox, cleaned_stroke_node_features_bbox):
    """
    Visualizes selected strokes in 3D space with a hand-drawn effect.

    Parameters:
    - stroke_node_features: A numpy array or list containing the features of each stroke.
      Each stroke should contain its start and end coordinates, and potentially a flag indicating if it's a circle.
    - selected_stroke_idx: A list or array of indices of the strokes that should be highlighted in red.
    - alpha_value: Float, optional. The transparency level of the lines (0.0 is fully transparent, 1.0 is fully opaque).
    """
    
    # Get the mapping file
    mapping_path = os.path.join(os.getcwd(), 'dataset', 'mapping', 'mapping_train.json')
    with open(mapping_path, 'r') as f:
        mapping_file = json.load(f)

    lifted_path = mapping_file[data_idx]
    final_edges_data = read_json(os.path.join(lifted_path, 'final_edges.json'))
    all_lines = extract_all_lines(final_edges_data)

    stroke_node_features = Preprocessing.cad2sketch_stroke_features.transform_stroke_node_features_reverse(stroke_node_features, lifted_stroke_node_features_bbox, cleaned_stroke_node_features_bbox)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove axis labels, ticks, and background
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_axis_off()

    # Initialize bounding box variables
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # First pass: plot all strokes in black and compute bounding box
    for stroke in all_lines:
        geometry = stroke["geometry"]

        if len(geometry) < 2:
            continue

        for j in range(1, len(geometry)):
            start = geometry[j - 1]
            end = geometry[j]

            # Update bounding box
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])

            # Plot the stroke in black
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    color='black', 
                    linewidth=0.5)

    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Second pass: highlight used strokes in red
    for idx, stroke in enumerate(stroke_node_features):

        if stroke[-1] != 0:

            if idx < len(all_lines):
                geometry = all_lines[idx]["geometry"]
                if len(geometry) < 2:
                    continue
                for j in range(1, len(geometry)):
                    start = geometry[j - 1]
                    end = geometry[j]
                    ax.plot([start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]],
                            color='blue',
                            linewidth=1.0)
            else:
                stroke = stroke_node_features[idx]
                start = stroke[0:3]
                end = stroke[3:6]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='blue',
                        linewidth=1.0)

    plt.show()





def vis_selected_loops(stroke_node_features, edge_features_slice, selected_loop_idx, data_idx):
    """
    Visualizes selected strokes in 3D space with a hand-drawn effect.

    Parameters:
    - stroke_node_features: A numpy array or list containing the features of each stroke.
      Each stroke should contain its start and end coordinates, and potentially a flag indicating if it's a circle.
    - edge_features_slice: A 2D array where each row contains [stroke_index, loop_index].
    - selected_loop_idx: A list (length 1) or array of the loop index to highlight.
    - data_idx: String or identifier for loading the corresponding file.
    """

    # Load final edges
    final_edges_file_path = os.path.join(
        os.getcwd(), 'dataset', 'cad2sketch_annotated', data_idx, 'final_edges.json')
    final_edges_data = read_json(final_edges_file_path)
    all_lines = extract_all_lines(final_edges_data)

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_axis_off()

    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for stroke in all_lines:
        geometry = stroke["geometry"]
        if len(geometry) < 2:
            continue
        for j in range(1, len(geometry)):
            start, end = geometry[j - 1], geometry[j]
            x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
            y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
            z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    color='black', 
                    linewidth=0.5)

    # Normalize view
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    target_loop = selected_loop_idx[0]

    # Step 1: Extract loop indices (second row)
    loop_indices = edge_features_slice[1]  # shape: (n,)

    # Step 2: Create mask where loop index == target
    mask = loop_indices == target_loop  # shape: (n,), bool tensor

    # Step 3: Apply mask to stroke indices (first row)
    stroke_indices = edge_features_slice[0]  # shape: (n,)
    loop_strokes = stroke_indices[mask].tolist()  # final result as list

    for idx in loop_strokes:
        if idx < len(all_lines):
            geometry = all_lines[idx]["geometry"]
            if len(geometry) < 2:
                continue
            for j in range(1, len(geometry)):
                start = geometry[j - 1]
                end = geometry[j]
                ax.plot([start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color='red',
                        linewidth=1.0)
        else:
            stroke = stroke_node_features[idx]
            start = stroke[0:3]
            end = stroke[3:6]
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color='red',
                    linewidth=1.0)

    plt.show()



#------------------------------------------------------------------------------------------------------#

def find_selected_strokes_from_loops(strokes_to_loops, selected_loop_idx):
    selected_stroke_idx = []
    for stroke_idx, loop_idx in zip(strokes_to_loops[0], strokes_to_loops[1]):
        if loop_idx.item() == selected_loop_idx[0]:
            selected_stroke_idx.append(stroke_idx)

    return selected_stroke_idx


#------------------------------------------------------------------------------------------------------#



def plot_circle(stroke):
    center = stroke[:3]
    normal = stroke[3:6]
    radius = stroke[7]

    # Generate circle points in the XY plane
    theta = np.linspace(0, 2 * np.pi, 30)  # Less dense with 30 points
    x_values = radius * np.cos(theta)
    y_values = radius * np.sin(theta)
    z_values = np.zeros_like(theta)

    # Combine the coordinates into a matrix (3, 30)
    circle_points = np.array([x_values, y_values, z_values])

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Rotation logic using Rodrigues' formula
    z_axis = np.array([0, 0, 1])  # Z-axis is the default normal for the circle

    rotation_axis = np.cross(z_axis, normal)
    if np.linalg.norm(rotation_axis) > 0:  # Check if rotation is needed
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

        # Create the rotation matrix using the rotation axis and angle (Rodrigues' rotation formula)
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0]])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        # Rotate the circle points
        rotated_circle_points = np.dot(R, circle_points)
    else:
        rotated_circle_points = circle_points

    # Translate the circle to the center point
    x_values = rotated_circle_points[0] + center[0]
    y_values = rotated_circle_points[1] + center[1]
    z_values = rotated_circle_points[2] + center[2]


    return x_values, y_values, z_values



def plot_arc(stroke):
    import numpy as np

    # Extract start and end points from the stroke
    start_point = np.array(stroke[:3])
    end_point = np.array(stroke[3:6])

    # Generate a straight line with 100 points between start_point and end_point
    t = np.linspace(0, 1, 100)  # Parameter for interpolation
    line_points = (1 - t)[:, None] * start_point + t[:, None] * end_point

    # Return x, y, z coordinates of the line points
    return line_points[:, 0], line_points[:, 1], line_points[:, 2]





#---------------------- Copied Files -----------------------------#   
import json
def read_json( file_path):
    """
    Reads a JSON file and returns its contents.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None



def extract_all_lines(final_edges_data):
    feature_lines = []

    for key, stroke in final_edges_data.items():
        
        feature_lines.append(stroke)

    return feature_lines
