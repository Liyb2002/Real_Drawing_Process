import numpy as np
import random
from shapely.geometry import Polygon, Point
from shapely.geometry.polygon import orient
from shapely import affinity
import pyrr
import json 
import torch
import math

import matplotlib.pyplot as plt
from itertools import permutations, combinations
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

import Preprocessing.proc_CAD.global_thresholding


def compute_normal(face_vertices, other_point):
    if len(face_vertices) < 3:
        raise ValueError("Need at least three points to define a plane")


    p1 = np.array(face_vertices[0].position)
    p2 = np.array(face_vertices[1].position)
    p3 = np.array(face_vertices[2].position)

    # Create vectors from the first three points
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute the cross product to find the normal
    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("The points do not form a valid plane")
    normal_unit = normal / norm

    # Use the other point to check if the normal should be flipped
    reference_vector = other_point.position - p1
    if np.dot(normal_unit, reference_vector) > 0:
        normal_unit = -normal_unit  # Flip the normal if it points towards the other point

    return normal_unit.tolist()


#----------------------------------------------------------------------------------#


def round_position(position, decimals=10):
    return tuple(round(coord, decimals) for coord in position)



#----------------------------------------------------------------------------------#




def find_target_verts(target_vertices, edges) :
    target_pos_1 = round_position(target_vertices[0])
    target_pos_2 = round_position(target_vertices[1])
    target_positions = {target_pos_1, target_pos_2}
    
    for edge in edges:
        verts = edge.vertices()
        if len(verts) ==2 :
            edge_positions = {
                round_position([verts[0].X, verts[0].Y, verts[0].Z]), 
                round_position([verts[1].X, verts[1].Y, verts[1].Z])
                }
        
            if edge_positions == target_positions:
                return edge
        
    return None


#----------------------------------------------------------------------------------#


def get_neighbor_verts(vert, non_app_edge, Edges):
    #get the neighbor of the given vert

    neighbors = []
    for edge in Edges:
        if edge.id == non_app_edge.id:
            continue

        if edge.vertices[0].position == vert.position:
            neighbors.append(edge.vertices[1])
        elif edge.vertices[1].position == vert.position:
            neighbors.append(edge.vertices[0])  

    return neighbors

def find_edge_from_verts(vert_1, vert_2, edges):
    vert_1_id = vert_1.id  # Get the ID of vert_1
    vert_2_id = vert_2.id  # Get the ID of vert_2

    for edge in edges:
        # Get the IDs of the vertices in the current edge
        edge_vertex_ids = [vertex.id for vertex in edge.vertices]

        # Check if both vertex IDs are present in the current edge
        if vert_1_id in edge_vertex_ids and vert_2_id in edge_vertex_ids:
            return edge  # Return the edge that contains both vertices

    return None  # Return None if no edge contains both vertices
    

#----------------------------------------------------------------------------------#
def compute_fillet_new_vert(old_vert, neighbor_verts, amount):
    move_positions = []
    old_position = old_vert.position

    # Compute new positions for the arc endpoints
    for neighbor_vert in neighbor_verts:
        direction_vector = [
            neighbor_vert.position[i] - old_position[i]
            for i in range(len(old_position))
        ]
        norm = math.sqrt(sum(x**2 for x in direction_vector))
        normalized_vector = [x / norm for x in direction_vector]
        move_position = [
            old_position[i] + normalized_vector[i] * amount
            for i in range(len(old_position))
        ]
        move_positions.append(move_position)

    # Extract the moved positions
    start_point = move_positions[0]
    end_point = move_positions[1]

    # Compute the plane normal (cross product of vectors from old_vert to start and end)
    vec1 = [start_point[i] - old_position[i] for i in range(len(old_position))]
    vec2 = [end_point[i] - old_position[i] for i in range(len(old_position))]
    normal = [
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
    ]
    norm = math.sqrt(sum(x**2 for x in normal))
    normal = [x / norm for x in normal]  # Normalize the plane normal

    # Find the midpoint between start_point and end_point
    midpoint = [
        (start_point[i] + end_point[i]) / 2
        for i in range(len(start_point))
    ]

    # Project the midpoint onto the plane to find the arc center
    vec_from_old_to_midpoint = [
        midpoint[i] - old_position[i]
        for i in range(len(old_position))
    ]
    distance_from_plane = sum(
        vec_from_old_to_midpoint[i] * normal[i]
        for i in range(len(normal))
    )
    center = [
        midpoint[i] - distance_from_plane * normal[i]
        for i in range(len(midpoint))
    ]

    return move_positions, center


#----------------------------------------------------------------------------------#

def find_rectangle_on_plane(points, normal):
    """
    Find a new rectangle on the same plane as the given larger rectangle, with a translation.
    
    Args:
        points: List of 4 numpy arrays representing the vertices of the larger rectangle.
    
    Returns:
        list: A list of 4 numpy arrays representing the vertices of the new rectangle.
    """
    # Convert points to numpy array for easy manipulation
    points = np.array(points)
    
    # Extract the coordinates
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]
    
    # Check which coordinate is the same for all points (defining the plane)
    if np.all(x_vals == x_vals[0]):
        fixed_coord = 'x'
        fixed_value = x_vals[0]
    elif np.all(y_vals == y_vals[0]):
        fixed_coord = 'y'
        fixed_value = y_vals[0]
    elif np.all(z_vals == z_vals[0]):
        fixed_coord = 'z'
        fixed_value = z_vals[0]
    
    # Determine the min and max for the other two coordinates
    if fixed_coord == 'x':
        min_y, max_y = np.min(y_vals), np.max(y_vals)
        min_z, max_z = np.min(z_vals), np.max(z_vals)
        new_min_y = min_y + (max_y - min_y) * 0.1
        new_max_y = max_y - (max_y - min_y) * 0.1
        new_min_z = min_z + (max_z - min_z) * 0.1
        new_max_z = max_z - (max_z - min_z) * 0.1
        new_points = [
            np.array([fixed_value, new_min_y, new_min_z]),
            np.array([fixed_value, new_max_y, new_min_z]),
            np.array([fixed_value, new_max_y, new_max_z]),
            np.array([fixed_value, new_min_y, new_max_z])
        ]
    elif fixed_coord == 'y':
        min_x, max_x = np.min(x_vals), np.max(x_vals)
        min_z, max_z = np.min(z_vals), np.max(z_vals)
        new_min_x = min_x + (max_x - min_x) * 0.1
        new_max_x = max_x - (max_x - min_x) * 0.1
        new_min_z = min_z + (max_z - min_z) * 0.1
        new_max_z = max_z - (max_z - min_z) * 0.1
        new_points = [
            np.array([new_min_x, fixed_value, new_min_z]),
            np.array([new_max_x, fixed_value, new_min_z]),
            np.array([new_max_x, fixed_value, new_max_z]),
            np.array([new_min_x, fixed_value, new_max_z])
        ]
    elif fixed_coord == 'z':
        min_x, max_x = np.min(x_vals), np.max(x_vals)
        min_y, max_y = np.min(y_vals), np.max(y_vals)
        new_min_x = min_x + (max_x - min_x) * 0.1
        new_max_x = max_x - (max_x - min_x) * 0.1
        new_min_y = min_y + (max_y - min_y) * 0.1
        new_max_y = max_y - (max_y - min_y) * 0.1
        new_points = [
            np.array([new_min_x, new_min_y, fixed_value]),
            np.array([new_max_x, new_min_y, fixed_value]),
            np.array([new_max_x, new_max_y, fixed_value]),
            np.array([new_min_x, new_max_y, fixed_value])
        ]
    
    return new_points


def find_triangle_on_plane(points, normal):

    four_pts = find_rectangle_on_plane(points, normal)
    idx1, idx2 = 0, 1
    point1 = four_pts[idx1]
    point2 = four_pts[idx2]

    point3 = 0.5 * (four_pts[2] + four_pts[3])

    return [point1, point2, point3]


def find_triangle_to_cut(points, normal):

    points = np.array(points)
    
    # Randomly shuffle the indices to choose three points
    start_index = np.random.randint(0, 4)

    # Determine the indices of the three points
    indices = [(start_index + i) % 4 for i in range(3)]

    
    # Use the second point as the pin point
    pin_index = indices[1]
    pin_point = points[pin_index]
    
    # Interpolate between the pin point and the other two points
    point1 = 0.5 * random.random() * (pin_point + points[indices[0]])
    point2 = 0.5 * random.random() * (pin_point + points[indices[2]])

    return [pin_point, point1, point2]


def random_circle(points, normal):
    """
    Given four vertices of a rectangle, compute a random circle on the same plane.

    Args:
        points: List of 4 numpy arrays representing the vertices of the larger rectangle.
        normal: The normal vector of the plane containing the points.

    Returns:
        center: The randomly selected center point for the circle.
        radius: The randomly sampled radius for the circle.
    """
    
    # Find the rectangle and the center
    four_pts = np.array(find_rectangle_on_plane(points, normal))

    min_x, min_y, min_z = np.min(four_pts, axis=0)
    max_x, max_y, max_z = np.max(four_pts, axis=0)
    
    random_x = random.uniform(min_x, max_x)
    random_y = random.uniform(min_y, max_y)
    random_z = random.uniform(min_z, max_z)

    center = np.array([random_x, random_y, random_z])

    # Compute the distances to the x, y, z axes for each point
    x_distances = [abs(point[0] - center[0]) for point in points]
    y_distances = [abs(point[1] - center[1]) for point in points]
    z_distances = [abs(point[2] - center[2]) for point in points]
    
    # Find the minimum distances for x, y, and z axes
    min_x_dist = min(x_distances)
    min_y_dist = min(y_distances)
    min_z_dist = min(z_distances)
    
    # Ignore the zero distance (because the points are on the same plane)
    distances = [d for d in [min_x_dist, min_y_dist, min_z_dist] if d > 0]
    
    # Find the second smallest distance, which will be the max radius
    max_radius = sorted(distances)[0] if len(distances) >= 2 else distances[0]
    
    # Randomly sample a radius between 0 and max_radius
    radius = random.uniform(max_radius * 0.5, max_radius * 0.8) 
    
    return radius, center




#----------------------------------------------------------------------------------#




def project_points(feature_lines, obj_center, img_dims=[1000, 1000]):

    obj_center = np.array(obj_center)
    cam_pos = obj_center + np.array([5,0,5])
    up_vec = np.array([0,1,0])
    view_mat = pyrr.matrix44.create_look_at(cam_pos,
                                            np.array([0, 0, 0]),
                                            up_vec)
    near = 0.001
    far = 1.0
    total_view_points = []

    for edge_info in feature_lines:
        view_points = []
        vertices = edge_info['vertices']
        if edge_info['is_curve']:
            vertices = edge_info['sampled_points']
        for p in vertices:
            p -= obj_center
            hom_p = np.ones(4)
            hom_p[:3] = p
            proj_p = np.matmul(view_mat.T, hom_p)
            view_points.append(proj_p)
            
            total_view_points.append(proj_p)
        edge_info['projected_edge'].append(np.array(view_points))
    
    # for edge_info in feature_lines:
    #    plt.plot(edge_info['projected_edge'][0][:, 0], edge_info['projected_edge'][0][:, 1], c="black")
    # plt.show()



    total_view_points = np.array(total_view_points)
    max = np.array([np.max(total_view_points[:, 0]), np.max(total_view_points[:, 1]), np.max(total_view_points[:, 2])])
    min = np.array([np.min(total_view_points[:, 0]), np.min(total_view_points[:, 1]), np.min(total_view_points[:, 2])])

    max_dim = np.maximum(np.abs(max[0]-min[0]), np.abs(max[1]-min[1]))
    proj_mat = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=-max_dim/2, right=max_dim/2, bottom=-max_dim/2, top=max_dim/2,
                                                                              near=near, far=far)

    total_projected_points = []
    projected_edges = []

    for edge_info in feature_lines:
        projected_points = []
        for p in edge_info['projected_edge'][0]:
            proj_p = np.matmul(proj_mat, p)
            proj_p[:3] /= proj_p[-1]
            total_projected_points.append(proj_p[:2])
            projected_points.append(proj_p[:2])
        projected_edges.append(np.array(projected_points))

        edge_info['projected_edge'] = projected_edges[-1]
    total_projected_points = np.array(total_projected_points)

    # screen-space
    # scale to take up 80% of the image
    max = np.array([np.max(total_projected_points[:, 0]), np.max(total_projected_points[:, 1])])
    min = np.array([np.min(total_projected_points[:, 0]), np.min(total_projected_points[:, 1])])
    bbox_diag = np.linalg.norm(max - min)
    screen_diag = np.sqrt(img_dims[0]*img_dims[0]+img_dims[1]*img_dims[1])


    for edge_info in feature_lines:
        scaled_points = []
        for p in edge_info['projected_edge']:
            p[1] *= -1
            p *= 0.5*screen_diag/bbox_diag
            p += np.array([img_dims[0]/2, img_dims[1]/2])
            scaled_points.append(p)
        edge_info['projected_edge'] = np.array(scaled_points)

    
    # for edge_info in feature_lines:
    #     f_line = edge_info['projected_edge']
    #     plt.plot(f_line[:, 0], f_line[:, 1], c="black")
    # plt.show()

    return feature_lines


#----------------------------------------------------------------------------------#

def program_to_string(file_path):

    Op_string = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for Op in data:
            Op_string.append(Op['operation'][0])

    return Op_string


def program_to_tensor(program):
    operation_to_index = {'terminate': 0, 'sketch': 1, 'extrude': 2, 'fillet': 3}
    Op_indices = []

    for Op in program:
        Op_indices.append(operation_to_index[Op])

    return torch.tensor(Op_indices, dtype=torch.long)


def sketch_face_selection(file_path):

    boundary_points = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for Op in data:
            if Op['operation'][0] == 'sketch':
                boundary_points.append(Op['operation'][1])
            else:
                boundary_points.append([])

    return boundary_points

#----------------------------------------------------------------------------------#

def expected_extrude_point(point, sketch_face_normal, extrude_amount):
    x, y, z = point
    a, b, c = sketch_face_normal
    x_extruded = x - a * extrude_amount
    y_extruded = y - b * extrude_amount
    z_extruded = z - c * extrude_amount
    return [x_extruded, y_extruded, z_extruded]


def expected_lvl(point, sketch_face_normal, extrude_amount):
    x, y, z = point
    a, b, c = sketch_face_normal
    
    # Check for non-zero component in sketch_face_normal to determine the axis
    if a != 0:
        x_extruded = x - a * extrude_amount
        return 'x', x_extruded
    elif b != 0:
        y_extruded = y - b * extrude_amount
        return 'y', y_extruded
    elif c != 0:
        z_extruded = z - c * extrude_amount
        return 'z', z_extruded
    else:
        raise ValueError("At least one component of the sketch_face_normal should be non-zero.")


def canvas_has_point(canvas, point):
    edges = canvas.edges()   
    point = round_position(point)
    
    for edge in edges:
        verts = edge.vertices()
        if len(verts) ==2 :
            edge_positions = [
                round_position([verts[0].X, verts[0].Y, verts[0].Z]), 
                round_position([verts[1].X, verts[1].Y, verts[1].Z])
                ]
    
            if point == edge_positions[0] or point == edge_positions[1]:
                return True
        
    return False


def canvas_has_lvl(canvas, expected_axis, expected_value):
    edges = canvas.edges()   
    
    for edge in edges:
        verts = edge.vertices()

        for vert in verts:
            # Check if the vertex is on the same level as the expected axis and value
            if expected_axis == 'x' and vert.X == expected_value:
                return True
            elif expected_axis == 'y' and vert.Y == expected_value:
                return True
            elif expected_axis == 'z' and vert.Z == expected_value:
                return True
        
    return False


def print_canvas_points(canvas):
    edges = canvas.edges()    
    
    for edge in edges:
        verts = edge.vertices()
        if len(verts) ==2 :
            edge_positions = [
                round_position([verts[0].X, verts[0].Y, verts[0].Z]), 
                round_position([verts[1].X, verts[1].Y, verts[1].Z])
                ]
        # print("edge_positions", edge_positions)




#----------------------------------------------------------------------------------#



def preprocess_features(features):
    processed_features = [] 
    for _, f in features:
        processed_features.append(f)
    
    return torch.tensor(processed_features)



#----------------------------------------------------------------------------------#


def face_to_stroke(stroke_cloud_faces, stroke_features):
    num_strokes = stroke_features.shape[0]
    stroke_ids_per_face = []
    
    for face_id, face in stroke_cloud_faces.items():
        face_stroke_ids = []
        # Get all combinations of two vertices
        vertex_combinations = list(combinations(face.vertices, 2))
        
        for comb in vertex_combinations:
            vert1_pos = np.array(comb[0].position)
            vert2_pos = np.array(comb[1].position)
            
            for stroke_id in range(num_strokes):
                start_point = stroke_features[stroke_id, :3]
                end_point = stroke_features[stroke_id, 3:]
                
                if (np.allclose(vert1_pos, start_point) and np.allclose(vert2_pos, end_point)) or \
                   (np.allclose(vert1_pos, end_point) and np.allclose(vert2_pos, start_point)):
                    face_stroke_ids.append(stroke_id)
                    break
        
        stroke_ids_per_face.append(face_stroke_ids)
    
    return stroke_ids_per_face



#----------------------------------------------------------------------------------#


def chosen_face_id(boundary_points, edge_features):
    print("edge_features", len(edge_features))
    print("boundary_points", len(boundary_points))


def is_planar(points, tol=0.17):
    if len(points) < 3:
        return True

    points = np.array(points)
    p0, p1, p2 = points[:3]

    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)

    normal = normal / np.linalg.norm(normal)

    for p in points[3:]:
        dist = abs(np.dot(p - p0, normal))
        if dist > tol:
            return False
    
    return True

def no_colinear(group_points, tol=1e-7):
    """
    Return True if no three points in the group are colinear.
    
    Parameters:
        group_points: list of 3D points
        tol: numerical tolerance for detecting colinearity
        
    Returns:
        True if all point triplets are non-colinear, False otherwise
    """
    points = np.array(group_points)
    
    if len(points) < 3:
        return True  # Less than 3 points can't be colinear

    for p1, p2, p3 in combinations(points, 3):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) < tol:
            return False  # These three are colinear

    return True


#----------------------------------------------------------------------------------#
import numpy as np
from itertools import combinations, permutations


def no_colinear(group_points, tol=1e-5):
    """
    Return True if no three points in the group are colinear.
    
    Parameters:
        group_points: list of 3D points
        tol: numerical tolerance for detecting colinearity
        
    Returns:
        True if all point triplets are non-colinear, False otherwise
    """
    points = np.array(group_points)
    
    if len(points) < 3:
        return True  # Less than 3 points can't be colinear

    for p1, p2, p3 in combinations(points, 3):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) < tol:
            return False  # These three are colinear

    return True


def is_closed_loop(group, group_points, threshold):
    unique_points = []
    for pt in group_points:
        if not any(np.linalg.norm(np.array(pt) - np.array(up)) < threshold for up in unique_points):
            unique_points.append(pt)

    print("len(group)", len(group), "len(unique_points)", len(unique_points))
    print("-------")
    return len(group) == len(unique_points)

def face_aggregate_networkx(stroke_matrix):
    """
    Finds all valid planar loops (triangles or quads) from strokes using NetworkX,
    where each stroke is a 3D line segment and points are considered identical if they are 'close enough'
    according to a custom closeness function.

    Parameters:
    stroke_matrix (numpy.ndarray): A matrix of shape (num_strokes, 7). Each row is a stroke with start/end 3D points.

    Returns:
    list: A list of sets of stroke indices that form valid planar loops (triangles or quads).
    """
    
    full_stroke_matrix = np.array(stroke_matrix)          
    stroke_matrix = full_stroke_matrix[:, :6]
    stroke_type = full_stroke_matrix[:, -1]
    G = nx.Graph()
    edge_to_stroke_id = {}

    point_registry = []

    def get_registered_point(point, current_stroke):
        for registered_point, reg_stroke in point_registry:
            if Preprocessing.proc_CAD.global_thresholding.point_close(point, registered_point, current_stroke, reg_stroke):
                return registered_point
        point_registry.append((point, current_stroke))
        return point

    # Build graph with custom closeness-based point matching
    for idx, stroke in enumerate(stroke_matrix):
        if stroke_type[idx] != 1:
            continue

        start_raw = tuple(stroke[:3])
        end_raw = tuple(stroke[3:])

        start_point = get_registered_point(start_raw, stroke)
        end_point = get_registered_point(end_raw, stroke)

        G.add_edge(start_point, end_point)
        edge_to_stroke_id[(start_point, end_point)] = idx
        edge_to_stroke_id[(end_point, start_point)] = idx  # undirected

    valid_groups = []

    def check_valid_edges(edges):
        point_count = {}
        for edge in edges:
            point_count[edge[0]] = point_count.get(edge[0], 0) + 1
            point_count[edge[1]] = point_count.get(edge[1], 0) + 1
        return all(count == 2 for count in point_count.values())

    nodes = list(G.nodes)

    for group_nodes in combinations(nodes, 3):
        subgraph = G.subgraph(group_nodes)
        if nx.is_connected(subgraph):
            for perm_edges in permutations(combinations(group_nodes, 2), 3):
                if check_valid_edges(perm_edges):
                    strokes_in_group = [
                        edge_to_stroke_id.get(edge) or edge_to_stroke_id.get((edge[1], edge[0]))
                        for edge in perm_edges
                    ]
                    if None not in strokes_in_group:
                        valid_groups.append(sorted(strokes_in_group))

    for group_nodes in combinations(nodes, 4):
        subgraph = G.subgraph(group_nodes)
        if nx.is_connected(subgraph):
            for perm_edges in permutations(combinations(group_nodes, 2), 4):
                if check_valid_edges(perm_edges):
                    strokes_in_group = [
                        edge_to_stroke_id.get(edge) or edge_to_stroke_id.get((edge[1], edge[0]))
                        for edge in perm_edges
                    ]
                    if None not in strokes_in_group:
                        valid_groups.append(sorted(strokes_in_group))

    # Remove duplicates
    unique_groups = list(set(frozenset(group) for group in valid_groups))


    final_groups = []
    for group in unique_groups:
        group_points = []
        for edge_id in group:
            stroke = stroke_matrix[edge_id]
            group_points.append(stroke[:3])  # start point
            group_points.append(stroke[3:6]) # end point
            threshold = dist(stroke[:3], stroke[3:6]) * 0.2
        
        if is_planar(group_points) and no_colinear(group_points):
            final_groups.append(group)
        
        

    return final_groups





def face_aggregate_networkx_accumulate(new_stroke_matrix, old_stroke_matrix):
    """
    Aggregates strokes from old and new stroke matrices to find loops,
    considering at least one edge from the new stroke matrix.

    Parameters:
    - new_stroke_matrix (numpy.ndarray): Matrix of new strokes (num_new_strokes, 10).
    - old_stroke_matrix (numpy.ndarray): Matrix of old strokes (num_old_strokes, 10).

    Returns:
    - combined_stroke_matrix: Combined matrix with shape (num_old_strokes + num_new_strokes, 6).
    - num_old_strokes: Number of old strokes.
    - num_new_strokes: Number of new strokes.
    """
    # Ensure old_stroke_matrix and new_stroke_matrix have consistent dimensions
    if old_stroke_matrix.size == 0:
        combined_stroke_matrix = new_stroke_matrix[:, :6]
        num_old_strokes = 0
        num_new_strokes = new_stroke_matrix.shape[0]
    elif new_stroke_matrix.size == 0:
        combined_stroke_matrix = old_stroke_matrix[:, :6]
        num_old_strokes = old_stroke_matrix.shape[0]
        num_new_strokes = 0
    else:
        if old_stroke_matrix.shape[1] != new_stroke_matrix.shape[1]:
            raise ValueError("Mismatch in column dimensions between old and new stroke matrices.")
        combined_stroke_matrix = np.vstack((old_stroke_matrix, new_stroke_matrix))[:, :6]
        num_old_strokes = old_stroke_matrix.shape[0]
        num_new_strokes = new_stroke_matrix.shape[0]

    return combined_stroke_matrix, num_old_strokes, num_new_strokes


def face_aggregate_direct(stroke_matrix):
    """
    This function finds all connected groups of strokes with size 3 or 4
    by directly checking if each point appears exactly twice within each group.

    Parameters:
    stroke_matrix (numpy.ndarray): A matrix of shape (num_strokes, 7) where each row represents a stroke
                                   with start and end points in 3D space.

    Returns:
    list: A list of indices of valid connected groups of strokes, where each group contains either 3 or 4 strokes.
    """
    
    # Ensure input is a numpy array and ignore the last column
    stroke_matrix = np.array(stroke_matrix)[:, :6]
    
    # Get the number of strokes
    num_strokes = stroke_matrix.shape[0]
    
    # List to store valid groups
    valid_groups = []
    
    # Function to check if all points appear exactly twice
    def check_group(group_indices):
        point_count = {}
        for idx in group_indices:
            start_point = tuple(np.round(stroke_matrix[idx, :3], 4))
            end_point = tuple(np.round(stroke_matrix[idx, 3:], 4))
            point_count[start_point] = point_count.get(start_point, 0) + 1
            point_count[end_point] = point_count.get(end_point, 0) + 1
        # Check if all points appear exactly twice
        return all(count == 2 for count in point_count.values())
    
    # Check all combinations of 3 strokes
    for group_indices in combinations(range(num_strokes), 3):
        if check_group(group_indices):
            valid_groups.append(list(group_indices))
    
    # Check all combinations of 4 strokes
    for group_indices in combinations(range(num_strokes), 4):
        if check_group(group_indices):
            valid_groups.append(list(group_indices))
    
    return valid_groups


def face_aggregate_circle(stroke_matrix):

    circle_loops = []
    for i in range(stroke_matrix.shape[0]):

        # Is circle
        if stroke_matrix[i, -1] == 2 or stroke_matrix[i, -1] == 4:
            circle_loops.append(frozenset([i]))

    return circle_loops


def face_aggregate_circle_brep(brep_matrix):

    circle_loops = []
    for i in range(brep_matrix.shape[0]):

        # Is circle
        if brep_matrix[i, 7] != 0 and brep_matrix[i, 6] == 0:
            circle_loops.append(frozenset([i]))

    return circle_loops



#----------------------------------------------------------------------------------#
def reorder_loops(loops):
    """
    Reorder loops based on the average order of their strokes.
    
    Parameters:
    loops (list of list of int): A list where each sublist contains indices representing strokes of a loop.
    
    Returns:
    list of list of int: The reordered loops where the smallest loop order comes first.
    """
    # Calculate the order for each loop (average of stroke indices)
    loop_orders = [(i, sum(loop) / len(loop)) for i, loop in enumerate(loops)]

    # Sort loops by their calculated order
    loop_orders.sort(key=lambda x: x[1])

    # Reorder loops according to the sorted order
    reordered_loops = [loops[i] for i, _ in loop_orders]

    return reordered_loops


def swap_rows_with_probability(matrix_a, matrix_b, swap_range=5, swap_prob=0.3):
    num_rows = matrix_a.shape[0]
    
    for i in range(num_rows):
        # Perform the swap with the given probability
        if random.random() < swap_prob:
            # Determine the range of valid indices to swap with
            lower_bound = max(0, i - swap_range)
            upper_bound = min(num_rows - 1, i + swap_range)
            
            # Choose a random index within the swap range
            swap_idx = random.randint(lower_bound, upper_bound)
            
            # Swap rows in both matrices if the chosen index is different from the current index
            if swap_idx != i:
                matrix_a[[i, swap_idx]] = matrix_a[[swap_idx, i]]
                matrix_b[[i, swap_idx]] = matrix_b[[swap_idx, i]]
    
    return matrix_a, matrix_b


#----------------------------------------------------------------------------------#
def loop_neighboring_simple(loops):
    """
    Determine neighboring loops based on shared edges.
    
    Parameters:
    loops (list of list of int): A list where each sublist contains indices representing edges of a loop.
    
    Returns:
    np.ndarray: A matrix of shape (num_loops, num_loops) where [i, j] is 1 if loops i and j share an edge, otherwise 0.
    """
    num_loops = len(loops)
    # Initialize the neighboring matrix with zeros with dtype float32
    neighboring_matrix = np.zeros((num_loops, num_loops), dtype=np.float32)
    
    # Iterate over each pair of loops to check for shared edges
    for i in range(num_loops):
        for j in range(i + 1, num_loops):
            # Check if loops i and j share any edge
            if set(loops[i]).intersection(set(loops[j])):
                neighboring_matrix[i, j] = 1.0
                neighboring_matrix[j, i] = 1.0  # Since the matrix is symmetric
    
    return neighboring_matrix
    


def loop_neighboring_complex(loops, stroke_node_features, loop_neighboring_all):
    """
    Determine neighboring loops based on shared edges and different normals, and populate the matrix with the shared stroke index.
    
    Parameters:
    loops (list of list of int): A list where each sublist contains indices representing edges of a loop.
    stroke_node_features (np.ndarray): A (num_strokes, 7) matrix where the first 6 columns represent two 3D points forming a line.
    
    Returns:
    np.ndarray: A matrix of shape (num_loops, num_loops) where [i, j] contains the index of the shared stroke if loops i and j are connected, otherwise 0.
    """
    num_loops = len(loops)
    # Initialize the neighboring matrix with zeros, using dtype int to store stroke indices
    neighboring_matrix = np.full((num_loops, num_loops), -1, dtype=np.int32)
    
    # Function to compute the normal of a loop
    def compute_normal(loop_indices):
        if len(loop_indices) < 3:
            return [0,0,0]
        
        # List to store edges, each with 6 useful values
        edges = []
        
        # Extract edges from loop_indices and discard the last value
        for idx in loop_indices:
            edge = stroke_node_features[idx, :6]  # First 6 values represent two 3D points
            edges.append(edge)
        
        # Find two edges that share a common point
        vec1 = None
        vec2 = None
        shared_point_found = False
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                # Extract points from edges
                p1_start, p1_end = edges[i][:3], edges[i][3:6]
                p2_start, p2_end = edges[j][:3], edges[j][3:6]

                # Check if there is a common point between these two edges
                if np.allclose(p1_start, p2_start) or np.allclose(p1_start, p2_end):
                    common_point = p1_start
                    vec1 = p1_end - p1_start
                    vec2 = p2_end - p2_start if np.allclose(p1_start, p2_start) else p2_start - p2_end
                    shared_point_found = True
                    break
                elif np.allclose(p1_end, p2_start) or np.allclose(p1_end, p2_end):
                    common_point = p1_end
                    vec1 = p1_start - p1_end
                    vec2 = p2_end - p2_start if np.allclose(p1_end, p2_start) else p2_start - p2_end
                    shared_point_found = True
                    break
            if shared_point_found:
                break
        
        # Compute the cross product to get the normal
        if vec1 is None or vec2 is None:
            return np.array([0,0,0])
        normal = np.cross(vec1, vec2)
        
        if np.isnan(normal).any() or np.all(normal == 0):
            return np.array([0,0,0])
        normal = normal / np.linalg.norm(normal)
        
        # Ensure the normal is positive in the z-direction
        if normal[2] < 0:
            normal = -normal

        return normal

    # Compute normals for each loop
    loop_normals = [compute_normal(loop) for loop in loops]

    # Iterate over each pair of loops to check for shared edges and different normals
    for i in range(num_loops):

        for j in range(i + 1, num_loops):

            if len(loops[i]) < 3 or len(loops[j]) < 3:
                neighboring_matrix[i, j] = loop_neighboring_all[i, j]
                neighboring_matrix[j, i] = loop_neighboring_all[i, j]
                continue

            # Check if loops i and j share any edge (stroke)
            shared_strokes = set(loops[i]).intersection(set(loops[j]))
            if shared_strokes:
                shared_stroke = list(shared_strokes)[0]  # Take the first shared stroke
                
                # Check if the normals are different
                dot_product = np.dot(loop_normals[i], loop_normals[j])
                if np.isclose(dot_product, 0):
                    neighboring_matrix[i, j] = shared_stroke
                    neighboring_matrix[j, i] = shared_stroke  # Since the matrix is symmetric

    return neighboring_matrix



def coplanr_neighorbing_loop(matrix1, matrix2):
    """
    Compute a matrix indicating coplanar neighboring loops based on the differences between matrix1 and matrix2.
    
    Parameters:
    matrix1 (np.ndarray): A matrix of shape (num_loops, num_loops) with more values of 1 than matrix2.
    matrix2 (np.ndarray): A matrix of shape (num_loops, num_loops) with fewer values of 1 than matrix1.
    
    Returns:
    np.ndarray: A matrix of the same shape where [i, j] is 1 if matrix1[i, j] is 1 and matrix2[i, j] is not 1.
    """
    # Ensure matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same shape.")
    
    # Compute the coplanar neighboring matrix
    result_matrix = np.where((matrix1 == 1) & (matrix2 != 1), 1, 0)
    
    return result_matrix
    


def stroke_relations(stroke_node_features, connected_stroke_nodes):
    num_strokes = stroke_node_features.shape[0]
    
    # Initialize result matrices
    strokes_perpendicular = np.zeros((num_strokes, num_strokes), dtype=int)
    strokes_non_perpendicular = np.zeros((num_strokes, num_strokes), dtype=int)
    
    # Handle the circle case (assumes this modifies strokes_perpendicular in-place)
    strokes_perpendicular = stroke_relations_circle(stroke_node_features, strokes_perpendicular)

    def get_direction_vector(stroke_features):
        point1 = stroke_features[:3]
        point2 = stroke_features[3:6]
        vector = point2 - point1
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector  # Avoid divide by zero

    for i in range(num_strokes):
        for j in range(num_strokes):
            if stroke_node_features[i, -1] != 1 or stroke_node_features[j, -1] != 1:
                continue

            if connected_stroke_nodes[i, j] == 1:
                vec_i = get_direction_vector(stroke_node_features[i])
                vec_j = get_direction_vector(stroke_node_features[j])

                dot_product = np.dot(vec_i, vec_j)
                

                if abs(dot_product) < 0.15 and not np.allclose(vec_i, vec_j, atol=1e-1) and not np.allclose(vec_i, -vec_j, atol=1e-1):
                    strokes_perpendicular[i, j] = 1
                else:
                    strokes_non_perpendicular[i, j] = 1
            else:
                strokes_perpendicular[i, j] = 0
                strokes_non_perpendicular[i, j] = 0

    return strokes_perpendicular, strokes_non_perpendicular
    


def stroke_relations_circle(stroke_node_features, strokes_perpendicular):
    """
    Find all strokes that are perpendicular to a circle (or cylinder base),
    meaning the stroke points lie on the circle and the direction is close to the sketch normal.
    """

    num_strokes = stroke_node_features.shape[0]
    EPSILON = 5e-5
    DEGREE_TOLERANCE = np.cos(np.deg2rad(10))  # cosine of 10 degrees

    for i in range(num_strokes):
        if stroke_node_features[i][-1] == 2:  # Circle
            sketch_center = stroke_node_features[i][0:3]
            sketch_normal = stroke_node_features[i][3:6]
            sketch_radius = stroke_node_features[i][7]

            for j in range(num_strokes):
                if stroke_node_features[j][-1] == 1:  # Line
                    point_1 = stroke_node_features[j][0:3]
                    point_2 = stroke_node_features[j][3:6]

                    # Distances from endpoints to the circle center
                    d1 = np.linalg.norm(point_1 - sketch_center)
                    d2 = np.linalg.norm(point_2 - sketch_center)

                    # Direction vector of the line (normalized)
                    direction_vec = point_2 - point_1
                    direction_vec_norm = np.linalg.norm(direction_vec)
                    if direction_vec_norm == 0:
                        continue  # skip degenerate stroke
                    direction_vec = direction_vec / direction_vec_norm

                    # Check if one end lies on the circle and if the direction is nearly perpendicular to the normal
                    dot_product = abs(np.dot(direction_vec, sketch_normal))  # cosine of angle between vectors

                    if (abs(d1) < sketch_radius * 1.2 or abs(d2) < sketch_radius * 1.2) and dot_product >= DEGREE_TOLERANCE:
                        strokes_perpendicular[i][j] = 1
                        strokes_perpendicular[j][i] = 1

    return strokes_perpendicular



def loop_contained(loops, stroke_node_features):
    """
    Determine if a loop is contained within another loop based on bounding boxes in 3D space.
    
    Parameters:
    loops (list of list of int): A list where each sublist contains indices representing strokes of a loop.
    stroke_node_features (dict): A dictionary where the key is the stroke index, and the value is a list of two 3D points [(x1, y1, z1), (x2, y2, z2)] representing the stroke.
    
    Returns:
    np.ndarray: A matrix of shape (num_loops, num_loops) where [i, j] is 1 if loop i contains loop j, otherwise 0.
    """
    num_loops = len(loops)
    
    # Initialize the contained matrix with zeros
    contained_matrix = np.zeros((num_loops, num_loops), dtype=np.float32)
    
    # Step 1: Calculate the bounding box (min_x, max_x, min_y, max_y, min_z, max_z) for each loop
    bounding_boxes = []
    
    for loop in loops:
        # Initialize min/max values with extreme values
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        

        # If circle
        if len(loop) < 3 and len(loop) > 0:
            circle_id = list(loop)[0]
            center_x, center_y, center_z, normal_x, normal_y, normal_z, alpha_value, radius, _, _, _ = stroke_coords = stroke_node_features[circle_id]
            min_x = max_x = center_x
            min_y = max_y = center_y
            min_z = max_z = center_z
            bounding_boxes.append((min_x, max_x, min_y, max_y, min_z, max_z))
            continue

        # Process each stroke in the loop
        for stroke in loop:
            stroke_coords = stroke_node_features[stroke]  # Each stroke has exactly 11 values: [x1, y1, z1, x2, y2, z2]
            x1, y1, z1, x2, y2, z2, _,  _, _, _, _ = stroke_coords
            
            # Update bounding box for the loop
            min_x, max_x = min(min_x, x1, x2), max(max_x, x1, x2)
            min_y, max_y = min(min_y, y1, y2), max(max_y, y1, y2)
            min_z, max_z = min(min_z, z1, z2), max(max_z, z1, z2)
        
        # Store the bounding box as a tuple (min_x, max_x, min_y, max_y, min_z, max_z)
        bounding_boxes.append((min_x, max_x, min_y, max_y, min_z, max_z))
    
    # Step 2: Check if one loop is contained in another in 3D space
    for i in range(num_loops):
        for j in range(num_loops):
            if i != j:  # Avoid comparing the same loop
                min_x_i, max_x_i, min_y_i, max_y_i, min_z_i, max_z_i = bounding_boxes[i]
                min_x_j, max_x_j, min_y_j, max_y_j, min_z_j, max_z_j = bounding_boxes[j]
                
                # Check containment conditions in 3D space
                if (min_x_i <= min_x_j <= max_x_j <= max_x_i) and \
                   (min_y_i <= min_y_j <= max_y_j <= max_y_i) and \
                   (min_z_i <= min_z_j <= max_z_j <= max_z_i):
                    contained_matrix[i, j] = 1.0  # Loop i contains loop j
    
    return contained_matrix



def loop_coplanar(loops, stroke_node_features):
    """
    Determine which loops are co-planar by comparing common values in the x, y, or z axes.
    
    Parameters:
    loops (list of lists): A list where each sublist contains indices of strokes forming a loop.
    stroke_node_features (np.ndarray): A NumPy array of shape (num_strokes, 7), where the first 6 values represent two 3D points.

    Returns:
    coplanar_matrix (np.ndarray): A binary matrix of shape (num_loops, num_loops), where [i, j] = 1 if loops i and j are co-planar.
    """
    num_loops = len(loops)
    coplanar_matrix = np.zeros((num_loops, num_loops), dtype=np.float32)

    # To store the (axis, value) for each loop
    loop_features = []

    # Step 1: Compute the (axis, value) for each loop
    for loop in loops:
        points = []
        # Gather all the points from the strokes in the loop
        for stroke_idx in loop:
            stroke_points = stroke_node_features[stroke_idx, :6]  # First 6 values represent two 3D points
            points.append(stroke_points[:3])  # Start point
            points.append(stroke_points[3:6])  # End point
        
        points = np.array(points)
        
        # Check if all points share the same value in the x, y, or z axis
        if np.all(points[:, 0] == points[0, 0]):
            loop_features.append(('x', points[0, 0]))
        elif np.all(points[:, 1] == points[0, 1]):
            loop_features.append(('y', points[0, 1]))
        elif np.all(points[:, 2] == points[0, 2]):
            loop_features.append(('z', points[0, 2]))
        else:
            loop_features.append(('none', 0))

    # Step 2: Determine co-planar loops
    for i in range(num_loops):
        for j in range(i + 1, num_loops):
            # If two loops have the same (axis, value), they are co-planar
            if loop_features[i] == loop_features[j]:
                coplanar_matrix[i, j] = 1
                coplanar_matrix[j, i] = 1  # Symmetric for undirected relationship

    return coplanar_matrix
    

def check_validacy(matrix1, matrix2):
    """
    Check the validity between two matrices.
    
    Parameters:
    matrix1 (np.ndarray): The first matrix of shape (num_loops, num_loops).
    matrix2 (np.ndarray): The second matrix of shape (num_loops, num_loops).
    
    Returns:
    bool: True if for every entry [i, j] in matrix2 with value 1, the corresponding entry in matrix1 is also 1.
    """
    # Ensure matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same shape.")
    
    # Check validity
    is_valid = np.all((matrix2 == 1) <= (matrix1 == 1))
    
    return is_valid



def connected_strokes(stroke_node_features):
    """
    Returns a matrix indicating if two strokes share a common point.
    
    Parameters:
    stroke_node_features (np.ndarray): A numpy array of shape (num_strokes, 7), where the first 6 values
                                       represent two 3D points of a stroke and the last value is unused.
    
    Returns:
    connected (np.ndarray): A binary matrix of shape (num_strokes, num_strokes), where [i, j] is 1 if stroke i and stroke j share a common point.
    """
    num_strokes = stroke_node_features.shape[0]
    
    # Initialize the connected matrix with zeros
    connected = np.zeros((num_strokes, num_strokes), dtype=np.float32)
    
    # Iterate over each pair of strokes
    # Iterate over each pair of strokes
    for i in range(num_strokes):
        if stroke_node_features[i, -1] == 2:
            continue
        stroke_i_start = stroke_node_features[i, :3]
        stroke_i_end = stroke_node_features[i, 3:6]

        for j in range(i + 1, num_strokes):
            if stroke_node_features[j, -1] == 2:
                continue
            stroke_j_start = stroke_node_features[j, :3]
            stroke_j_end = stroke_node_features[j, 3:6]

            # Use global thresholding based on max stroke length
            if (Preprocessing.proc_CAD.global_thresholding.point_close(stroke_i_start, stroke_j_start, stroke_node_features[i], stroke_node_features[j]) or
                Preprocessing.proc_CAD.global_thresholding.point_close(stroke_i_start, stroke_j_end,   stroke_node_features[i], stroke_node_features[j]) or
                Preprocessing.proc_CAD.global_thresholding.point_close(stroke_i_end,   stroke_j_start, stroke_node_features[i], stroke_node_features[j]) or
                Preprocessing.proc_CAD.global_thresholding.point_close(stroke_i_end,   stroke_j_end,   stroke_node_features[i], stroke_node_features[j])):
                
                connected[i, j] = 1
                connected[j, i] = 1


    # Now consider the circles
    for i in range(num_strokes):

        # Is circle
        if stroke_node_features[i, -1] == 2:
            center = stroke_node_features[i, :3]
            radius = stroke_node_features[i, 7]

            for j in range(num_strokes):
                if i == j or stroke_node_features[j, -1] != 1:
                    continue

                stroke_j_start = stroke_node_features[j, :3]
                stroke_j_end = stroke_node_features[j, 3:6]

                if dist(center, stroke_j_start) < radius * 1.2 or dist(center, stroke_j_end) < radius * 1.2:
                    connected[i, j] = 1
                    connected[j, i] = 1

    return connected
 


#----------------------------------------------------------------------------------#


def pad_brep_features(final_brep_edges):
    # Target padded_length = 10
    padded_edges = []

    # Iterate through each sublist in final_brep_edges
    for edge in final_brep_edges:
        # Pad each edge list with zeros to ensure it has exactly 10 elements
        padded_edge = edge + [0] * (10 - len(edge))
        padded_edges.append(padded_edge)

    return np.array(padded_edges)


#----------------------------------------------------------------------------------#
def stroke_to_edge(stroke_node_features, final_brep_edges):
    """
    Matches each BREP edge to the closest stroke of the same type (line or arc) in stroke_node_features.
    Marks the closest stroke as used (1). Each BREP edge is guaranteed to match exactly one stroke.

    Parameters:
    - stroke_node_features: np.ndarray of shape (num_strokes, 11)
    - final_brep_edges: np.ndarray of shape (num_brep_edges, 11)

    Returns:
    - stroke_used_matrix: np.ndarray of shape (num_strokes, 1), values in {0, 1}
    """
    import numpy as np

    stroke_used_matrix = np.zeros((len(stroke_node_features), 1), dtype=np.int32)

    for brep_edge in final_brep_edges:
        brep_type = brep_edge[-1]
        brep_pt1 = brep_edge[:3]
        brep_pt2 = brep_edge[3:6]

        # Determine stroke type to compare with
        target_stroke_type = 1 if brep_type == 1 else 3 if brep_type == 4 else None
        if target_stroke_type is None:
            continue

        # Search for the closest stroke of matching type
        min_distance = float('inf')
        best_stroke_idx = None

        for stroke_idx, stroke in enumerate(stroke_node_features):
            if stroke[-1] != target_stroke_type:
                continue

            stroke_pt1 = stroke[:3]
            stroke_pt2 = stroke[3:6]

            d1 = max(np.linalg.norm(stroke_pt1 - brep_pt1), np.linalg.norm(stroke_pt2 - brep_pt2))
            d2 = max(np.linalg.norm(stroke_pt1 - brep_pt2), np.linalg.norm(stroke_pt2 - brep_pt1))

            best_d = min(d1, d2)
            stroke_length = np.linalg.norm(stroke_pt1 - stroke_pt2)

            if brep_type == 4 and stroke[-1] == 3:
                if best_d < min_distance and min(d1, d2) < stroke_length * 0.3:
                    min_distance = best_d
                    best_stroke_idx = stroke_idx

                
            if best_d < min_distance and min(d1, d2) < stroke_length * 0.3:
                min_distance = best_d
                best_stroke_idx = stroke_idx

        if best_stroke_idx is not None:
            stroke_used_matrix[best_stroke_idx] = 1

    return stroke_used_matrix



def ensure_brep_edges(stroke_node_features, edge_features_list):
    """
    Ensures all brep edges of type 1 or 4 are represented in stroke_node_features.
    If a brep edge is not matched by any existing stroke (based on endpoint distance),
    it is added to stroke_node_features using the closest stroke points.

    Parameters:
    - stroke_node_features: np.ndarray of shape (num_strokes, 11)
    - edge_features_list: np.ndarray of shape (num_edges, 11)

    Returns:
    - stroke_node_features: np.ndarray with unmatched brep edges added
    - num_add_edges: int, number of added edges
    - new_edges: np.ndarray of shape (num_added, 11)
    """
    import numpy as np

    threshold = 5e-4
    num_add_edges = 0
    new_edges = []

    # Collect all unique stroke points
    unique_stroke_points = []
    for stroke in stroke_node_features:
        if stroke[-1] == 1 or stroke[-1] == 3:
            pt1 = tuple(np.round(stroke[:3], 6))
            pt2 = tuple(np.round(stroke[3:6], 6))
            if pt1 not in unique_stroke_points:
                unique_stroke_points.append(pt1)
            if pt2 not in unique_stroke_points:
                unique_stroke_points.append(pt2)

    unique_stroke_points = np.array(unique_stroke_points)
    
    for brep_edge in edge_features_list:
        if brep_edge[-1] == 1 or brep_edge[-1] == 4:  # Line or Arc
            brep_pt1 = brep_edge[:3]
            brep_pt2 = brep_edge[3:6]

            no_match = True

            for stroke_idx, stroke in enumerate(stroke_node_features):
                if stroke[-1] == 1 or stroke[-1] == 3:
                    stroke_pt1 = stroke[:3]
                    stroke_pt2 = stroke[3:6]

                    d1 = np.linalg.norm(brep_pt1 - stroke_pt1) + np.linalg.norm(brep_pt2 - stroke_pt2)
                    d2 = np.linalg.norm(brep_pt1 - stroke_pt2) + np.linalg.norm(brep_pt2 - stroke_pt1)
                    stroke_length = np.linalg.norm(stroke_pt1 - stroke_pt2)

                    if min(d1, d2) < stroke_length * 0.25:
                        no_match = False
                        break

            if no_match:
                new_stroke = np.array(list(brep_pt1) + list(brep_pt2) + [0, 0, 0, 0, 1], dtype=np.float32)
                new_edges.append(new_stroke)
                stroke_node_features = np.vstack([stroke_node_features, new_stroke])
                num_add_edges += 1

    return stroke_node_features, num_add_edges, np.array(new_edges)




def ensure_brep_edges_selected(stroke_node_features, edge_features_list, prev_sketch):
    """
    Ensures all brep edges of type 1 or 4 are represented in stroke_node_features.
    If a brep edge is not matched by any existing stroke (based on endpoint distance),
    it is added to stroke_node_features using the closest stroke points.

    Parameters:
    - stroke_node_features: np.ndarray of shape (num_strokes, 11)
    - edge_features_list: np.ndarray of shape (num_edges, 11)

    Returns:
    - stroke_node_features: np.ndarray with unmatched brep edges added
    - num_add_edges: int, number of added edges
    - new_edges: np.ndarray of shape (num_added, 11)
    """
    def is_perpendicular(v1, v2, tol=5e-6):
        """Check if two vectors are perpendicular within a tolerance."""
        return abs(np.dot(v1, v2)) < tol

    threshold = 5e-4
    num_add_edges = 0
    new_edges = []

    # Collect all unique stroke points
    unique_stroke_points = []
    for stroke in stroke_node_features:
        if stroke[-1] == 1 or stroke[-1] == 3:
            pt1 = tuple(np.round(stroke[:3], 6))
            pt2 = tuple(np.round(stroke[3:6], 6))
            if pt1 not in unique_stroke_points:
                unique_stroke_points.append(pt1)
            if pt2 not in unique_stroke_points:
                unique_stroke_points.append(pt2)

    unique_stroke_points = np.array(unique_stroke_points)
    

    for brep_edge in edge_features_list:
        if brep_edge[-1] == 1 or brep_edge[-1] == 4:  # Line or Arc
            brep_pt1 = brep_edge[:3]
            brep_pt2 = brep_edge[3:6]

            no_match = True

            for stroke_idx, stroke in enumerate(stroke_node_features):
                if stroke[-1] == 1 or stroke[-1] == 3:
                    stroke_pt1 = stroke[:3]
                    stroke_pt2 = stroke[3:6]

                    d1 = np.linalg.norm(brep_pt1 - stroke_pt1) + np.linalg.norm(brep_pt2 - stroke_pt2)
                    d2 = np.linalg.norm(brep_pt1 - stroke_pt2) + np.linalg.norm(brep_pt2 - stroke_pt1)
                    stroke_length = np.linalg.norm(stroke_pt1 - stroke_pt2)

                    if min(d1, d2) < stroke_length * 0.3:
                        no_match = False
                        break

        if no_match:
            brep_vec = np.array(brep_pt2) - np.array(brep_pt1)

            all_perpendicular = True
            for prev_sketch_brep in prev_sketch:
                sketch_pt1 = prev_sketch_brep[:3]
                sketch_pt2 = prev_sketch_brep[3:6]
                sketch_vec = np.array(sketch_pt2) - np.array(sketch_pt1)

                if not is_perpendicular(sketch_vec, brep_vec):
                    all_perpendicular = False
                    break

            if all_perpendicular:
                for prev_sketch_brep in prev_sketch:
                    sketch_pt1 = prev_sketch_brep[:3]
                    sketch_pt2 = prev_sketch_brep[3:6]

                    condition1 = (
                        Preprocessing.proc_CAD.global_thresholding.point_close(sketch_pt1, brep_pt1, prev_sketch_brep, brep_edge) and
                        not Preprocessing.proc_CAD.global_thresholding.point_close(sketch_pt2, brep_pt2, prev_sketch_brep, brep_edge)
                    )
                    condition2 = (
                        Preprocessing.proc_CAD.global_thresholding.point_close(sketch_pt1, brep_pt2, prev_sketch_brep, brep_edge) and
                        not Preprocessing.proc_CAD.global_thresholding.point_close(sketch_pt2, brep_pt1, prev_sketch_brep, brep_edge)
                    )

                    if condition1 or condition2:
                        new_stroke = np.array(list(brep_pt1) + list(brep_pt2) + [0, 0, 0, 0, 1], dtype=np.float32)
                        new_edges.append(new_stroke)
                        stroke_node_features = np.vstack([stroke_node_features, new_stroke])
                        num_add_edges += 1
                        

    return stroke_node_features, num_add_edges, np.array(new_edges)





def match(center1, center2, radius):
    """Utility function to check if two centers are approximately equal."""

    return np.linalg.norm(center1 - center2) < radius * 0.3

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def stroke_to_edge_circle(stroke_node_features, final_brep_edges):
    """
    Matches strokes to BREP edges and marks them as used if a match is found.

    Args:
        stroke_node_features: numpy array of shape (N, F) where each row represents a stroke.
                              Last element in each row indicates type: 2 = circle, 1 = straight.
        final_brep_edges: numpy array or list where each item represents a BREP edge.
                          Last element in each row indicates type: 2 = circle, 3 = cylinder.

    Returns:
        stroke_used_matrix: numpy array of shape (N, 1), where 1 indicates stroke is matched/used.
    """
    num_strokes = stroke_node_features.shape[0]
    stroke_used_matrix = np.zeros((num_strokes, 1), dtype=np.float32)

    # First pass: Match circle strokes to circle edges
    for i, stroke in enumerate(stroke_node_features):
        stroke_type = stroke[-1]
        if stroke_type == 2:  # Circle stroke
            stroke_center = np.array(stroke[:3])
            radius = np.array(stroke[7])

            for brep_edge in final_brep_edges:
                if brep_edge[-1] == 2:  # Circle edge
                    brep_center = np.array(brep_edge[:3])
                    if dist(stroke_center, brep_center) < 0.3 * radius:
                        stroke_used_matrix[i] = 1
                        

    # Second pass: Detect paired circle faces (for cylinders)
    circle_pairs = []
    for i, brep_edge1 in enumerate(final_brep_edges):
        if brep_edge1[-1] == 2:
            center1 = np.array(brep_edge1[:3])
            radius1 = brep_edge1[7]  # Assuming radius is at index 6

            for j, brep_edge2 in enumerate(final_brep_edges):
                if i == j:
                    continue
                if brep_edge2[-1] == 2:
                    center2 = np.array(brep_edge2[:3])
                    radius2 = brep_edge2[7]

                    if abs(radius1 - radius2) < 0.25 * abs(radius1) and dist(center1, center2) > 1e-4:
                        circle_pairs.append((center1, center2, radius1))

    # Third pass: Match straight strokes connecting paired circles
    for i, stroke in enumerate(stroke_node_features):
        stroke_type = stroke[-1]
        if stroke_type == 1:  # Straight stroke
            point1 = np.array(stroke[:3])
            point2 = np.array(stroke[3:6])
            stroke_length = np.linalg.norm(point1 - point2)

            for center1, center2, radius in circle_pairs:
                # print("abs(dist(point1, center1) - radius)", abs(dist(point1, center1) - radius))
                if (
                    abs(dist(point1, center1) - radius) < 0.4 * stroke_length and abs(dist(point2, center2) - radius) < 0.4 * stroke_length
                ) or (
                    abs(dist(point2, center1) - radius) < 0.4 * stroke_length and abs(dist(point1, center2) - radius) < 0.4 * stroke_length
                ):
                    stroke_used_matrix[i] = 1
                    break  # Go to next stroke
    
    return stroke_used_matrix




def stroke_to_edge_circle_full(stroke_node_features, final_brep_edges):
    num_strokes = stroke_node_features.shape[0]
    stroke_used_matrix = np.zeros((num_strokes, 1), dtype=np.float32)

    # Step 1: Find paired brep circle faces
    paired_circle_faces = []
    for brep_edge in final_brep_edges:
        if brep_edge[-1] != 2:
            continue

        brep_center = np.array(brep_edge[:3])  # Center of the circle
        radius = brep_edge[7]

        # Check for pairing conditions
        paired = False
        for pair in paired_circle_faces:
            center1, center2, rad = pair
            if rad == radius and np.sum(np.isclose(center1 - center2, 0)) == 2:
                paired = True
                break

        if not paired:
            paired_circle_faces.append([brep_center, brep_center, radius])

    # Step 2: Map strokes to brep edges
    for i, stroke in enumerate(stroke_node_features):
        if stroke[-1] == 2:  # Circle stroke
            stroke_center = np.array(stroke[:3])
            for center1, center2, radius in paired_circle_faces:
                if any(np.linalg.norm(stroke_center - center) < 0.3 * radius for center in [center1, center2]):
                    stroke_used_matrix[i] = 1
                    break

        elif stroke[-1] == 1:  # Straight stroke
            point1 = np.array(stroke[:3])
            point2 = np.array(stroke[3:6])

            for brep_edge in final_brep_edges:
                if brep_edge[-1] == 3:  # Cylinder edge
                    cylinder_center = np.array(brep_edge[:3])
                    cylinder_radius = brep_edge[7]  # Assuming radius is stored at index 7
                    dist1 = np.linalg.norm(point1 - cylinder_center)
                    dist2 = np.linalg.norm(point2 - cylinder_center)

                    if np.isclose(dist1, cylinder_radius, atol=1e-3) or np.isclose(dist2, cylinder_radius, atol=1e-3):
                        stroke_used_matrix[i] = 1
                        break

    
    return stroke_used_matrix



def stroke_to_brep(stroke_cloud_loops, brep_loops, stroke_node_features, final_brep_edges):
    """
    Maps strokes to BRep edges and finds corresponding loops based on matching edges.

    Parameters:
        stroke_cloud_loops (list of lists): Each sublist contains indices of strokes in a stroke loop.
        brep_loops (list of lists): Each sublist contains indices of BRep edges in final_brep_edges forming a BRep loop.
        stroke_node_features (list or np.ndarray): Features of strokes, each defined by its start and end points.
        final_brep_edges (list or np.ndarray): Features of BRep edges, each defined by its start and end points.

    Returns:
        np.ndarray: A correspondence matrix where entry [i, j] is 1 if stroke_cloud_loops[i] corresponds to brep_loops[j].
    """
    import numpy as np

    def points_match(point1, point2, tolerance=5e-5):
        """Check if two points are approximately equal within a tolerance."""
        return all(abs(a - b) <= tolerance for a, b in zip(point1, point2))

    # Step 1: Map BRep edges to strokes as index pairs
    brep_to_stroke_map = {}

    for brep_idx, brep_edge in enumerate(final_brep_edges):
        brep_points = set(map(tuple, [brep_edge[:3], brep_edge[3:6]]))  # Start and end points of the BRep edge

        for stroke_idx, stroke in enumerate(stroke_node_features):
            stroke_points = set(map(tuple, [stroke[:3], stroke[3:6]]))  # Start and end points of the stroke

            # Check if all points match between stroke and BRep edge (bidirectional check)
            stroke_match = all(
                any(points_match(stroke_point, brep_point) for brep_point in brep_points)
                for stroke_point in stroke_points
            )

            brep_match = all(
                any(points_match(brep_point, stroke_point) for stroke_point in stroke_points)
                for brep_point in brep_points
            )

            if stroke_match or brep_match:
                brep_to_stroke_map[brep_idx] = stroke_idx  # Pair BRep edge index with stroke index
                break  # No need to check further for this BRep edge

    # Step 2: Initialize correspondence matrix
    num_stroke_cloud_loops = len(stroke_cloud_loops)
    num_brep_loops = len(brep_loops)
    correspondence_matrix = np.zeros((num_stroke_cloud_loops, num_brep_loops), dtype=np.float32)

    # Step 3: Find corresponding loops
    for stroke_loop_idx, stroke_loop in enumerate(stroke_cloud_loops):
        if len(stroke_loop) == 1:
            continue
        stroke_loop_set = set(stroke_loop)  # Convert stroke loop to a set for fast lookup
        for brep_loop_idx, brep_loop in enumerate(brep_loops):
            if len(brep_loop) == 1:
                continue
            # Check if all strokes in the stroke loop map to the BRep edges in the BRep loop
            if all(
                brep_to_stroke_map.get(brep_edge_idx, None) in stroke_loop_set
                for brep_edge_idx in brep_loop
            ):
                correspondence_matrix[stroke_loop_idx, brep_loop_idx] = 1.0


    brep_loops_used = np.any(correspondence_matrix == 1, axis=0)
    new_loops_mark_off = np.sum(brep_loops_used)
    # print("striaght loops", new_loops_mark_off)

    return correspondence_matrix





def stroke_to_brep_circle(stroke_cloud_loops, brep_loops, stroke_node_features, final_brep_edges):
    
    num_stroke_cloud_loops = len(stroke_cloud_loops)
    num_brep_loops = len(brep_loops)
    correspondence_matrix = np.zeros((num_stroke_cloud_loops, num_brep_loops), dtype=np.float32)
    

    for i, stroke_loop in enumerate(stroke_cloud_loops):
        for j, brep_loop in enumerate(brep_loops):
            
            if not isinstance(brep_loop, list):
                continue

            if len(stroke_loop) == 1 and len(brep_loop) ==1:
                
                stroke_circle_edge = stroke_node_features[stroke_loop[0]]
                brep_circle_edge = final_brep_edges[brep_loop[0]]

                if (stroke_circle_edge[:3] == brep_circle_edge[:3]).all():
                    correspondence_matrix[i, j] = 1.0

                    # print("stroke_circle_edge", stroke_circle_edge[:3])
                    # print("brep_circle_edge", brep_circle_edge[:3])
                    # print("----------")


    brep_loops_used = np.any(correspondence_matrix == 1, axis=0)
    new_loops_mark_off = np.sum(brep_loops_used)
    # print("circle loops", new_loops_mark_off)

    return correspondence_matrix



def union_matrices(stroke_to_loop_lines, stroke_to_loop_circle):
    # Ensure both matrices have the same shape
    assert stroke_to_loop_lines.shape == stroke_to_loop_circle.shape, "Matrices must have the same shape."
    
    # Create a union matrix where each element is 1 if either matrix has a 1 at that position
    union_matrix = np.where((stroke_to_loop_lines == 1) | (stroke_to_loop_circle == 1), 1, 0)
    
    return union_matrix





def vis_specific_loop(loop, strokes):
    """
    Visualize specific loops and strokes.
    
    Parameters:
    loop (list of int): A list containing indices of the strokes to be highlighted.
    strokes (np.ndarray): A matrix of shape (num_strokes, 7), where the first 6 columns represent two 3D points.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all strokes in blue
    for stroke in strokes:
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', alpha=0.5)

    # Plot strokes in the loop in red
    for idx in loop:
        stroke = strokes[idx]
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', linewidth=2)

    # Set labels and show plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def vis_multiple_loops(loops, strokes):
    """
    Visualize multiple loops, one by one.

    Parameters:
    loops (list of lists): A list where each element is a list containing indices of strokes to be highlighted.
    strokes (np.ndarray): A matrix of shape (num_strokes, 7), where the first 6 columns represent two 3D points.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Iterate through the list of loops
    for i, loop in enumerate(loops):
        print(f"Visualizing loop {i+1}/{len(loops)}")

        # Initialize the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all strokes in blue
        for stroke in strokes:
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', alpha=0.5)

        # Plot strokes in the current loop in red
        for idx in loop:
            stroke = strokes[idx]
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='red', linewidth=2)

        # Set labels and show plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot for each loop
        plt.show()


def vis_partial_graph(loops, strokes):
    """
    Visualize multiple loops and strokes in 3D space.

    Parameters:
    loops (list of lists of int): A list of loops, where each loop is a list containing indices of strokes to be highlighted.
    strokes (np.ndarray): A matrix of shape (num_strokes, 7), where the first 6 columns represent two 3D points.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all strokes in blue
    for stroke in strokes:
        start, end = stroke[:3], stroke[3:6]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', alpha=0.5)

    # Plot strokes in each loop with different colors
    colors = plt.cm.jet(np.linspace(0, 1, len(loops)))  # Generate a color map for the loops

    for i, loop in enumerate(loops):
        for idx in loop:
            stroke = strokes[idx]
            start, end = stroke[:3], stroke[3:6]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=colors[i], linewidth=2)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()
    


def vis_brep(brep):
    """
    Visualize the brep strokes in 3D space if brep is not empty.
    
    Parameters:
    brep (np.ndarray): A matrix with shape (num_strokes, 6) representing strokes.
                       Each row contains two 3D points representing the start and end of a stroke.
                       If brep.shape[0] == 0, the function returns without plotting.
    """
    # Check if brep is empty
    if brep.shape[0] == 0:
        return

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot all brep strokes in blue with line width 1
    for stroke in brep:
        start, end = stroke[:3], stroke[3:6]
        
        # Update the min and max limits for each axis
        x_min, x_max = min(x_min, start[0], end[0]), max(x_max, start[0], end[0])
        y_min, y_max = min(y_min, start[1], end[1]), max(y_max, start[1], end[1])
        z_min, z_max = min(z_min, start[2], end[2]), max(z_max, start[2], end[2])
        
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='blue', linewidth=1)

    # Compute the center of the shape
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Compute the maximum difference across x, y, z directions
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Set the same limits for x, y, and z axes centered around the computed center
    if max_diff == 0:
        return
    
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()



#----------------------------------------------------------------------------------#


def remove_duplicate_circle_breps(brep_loops, final_brep_edges):
    """
    Removes duplicate circle breps based on their center points.
    
    Args:
        brep_loops (list of lists): List of brep loops, where each loop contains indices to final_brep_edges.
        final_brep_edges (list of lists): List of brep edges, where each edge contains information including center points.
        
    Returns:
        list: The filtered brep_loops after removing duplicates.
    """
    seen_centers = set()  # To track unique center points
    unique_brep_loops = []  # To store the resulting brep_loops

    for brep_loop in brep_loops:
        # Only consider brep_loops with a single edge
        if len(brep_loop) != 1:
            unique_brep_loops.append(brep_loop)
            continue

        if len(brep_loop) == 1:
            edge_index = brep_loop[0]
            brep_circle_edge = final_brep_edges[edge_index]

            # Extract the center point (assuming it's the first three elements of the edge)
            center_point = tuple(brep_circle_edge[:3])  # Convert to tuple for hashing in the set

            # If this center point is unique, add it to the result
            if center_point not in seen_centers:
                seen_centers.add(center_point)
                unique_brep_loops.append(brep_loop)

    return unique_brep_loops



import math
import random

def get_fillet_amount(target_output_edge, brep_edges):
    point1 = target_output_edge[:3]
    point2 = target_output_edge[3:6]

    def midpoint(p1, p2):
        return [(a + b) / 2 for a, b in zip(p1, p2)]

    def distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    candidate_edges = []
    min_distance = float('inf')
    tolerance = 1e-5

    for edge in brep_edges:
        verts = edge.vertices()
        if len(verts) == 2:
            edge_point1 = [verts[0].X, verts[0].Y, verts[0].Z]
            edge_point2 = [verts[1].X, verts[1].Y, verts[1].Z]
            edge_mid_point = midpoint(edge_point1, edge_point2)

            distance1 = distance(point1, edge_mid_point)
            distance2 = distance(point2, edge_mid_point)

            if abs(distance1 - distance2) < 5e-4:
                if distance1 < min_distance - tolerance:
                    # New minimum found
                    min_distance = distance1
                    candidate_edges = [edge]
                elif abs(distance1 - min_distance) <= tolerance:
                    # Equally good candidate
                    candidate_edges.append(edge)

    if candidate_edges:
        fillet_edge = random.choice(candidate_edges)
        fillet_amount = (math.sqrt(2) / 2) * distance(point1, point2)
        return fillet_edge, fillet_amount

    return None, 0
