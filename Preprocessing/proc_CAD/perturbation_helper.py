import numpy as np
import copy
import random



def points_are_close(p1, p2, tol=1e-6):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < tol

def point_on_segment(pt, seg_start, seg_end, tol=1e-5):
    """Check if pt lies on the segment [seg_start, seg_end] within tolerance."""
    pt = np.array(pt)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    seg_vec = seg_end - seg_start
    pt_vec = pt - seg_start

    proj_len = np.dot(pt_vec, seg_vec) / (np.linalg.norm(seg_vec)**2 + 1e-8)
    if proj_len < -tol or proj_len > 1 + tol:
        return False

    closest_point = seg_start + proj_len * seg_vec
    return np.linalg.norm(pt - closest_point) < tol

def stroke_is_contained(small, big, tol=1e-5):
    """Check if stroke 'small' is fully contained within stroke 'big'."""
    a1, a2 = small[:3], small[3:6]
    b1, b2 = big[:3], big[3:6]

    return point_on_segment(a1, b1, b2, tol) and point_on_segment(a2, b1, b2, tol)



def remove_contained_lines(all_lines, stroke_node_features):
    """
    Remove feature strokes (type == 1) if:
    - They are contained in another stroke, OR
    - With 20% probability even if not contained.

    Returns:
    - new_all_lines: filtered list of line dicts
    - new_stroke_node_features: filtered array of features
    - new_line_types: filtered list of types
    """
    removed_idx = []
    num_strokes = len(stroke_node_features)

    for i in range(num_strokes):
        stroke_i = stroke_node_features[i]

        if stroke_i[-1] == 1:
            contained = False
            for j in range(num_strokes):
                if i == j:
                    continue
                stroke_j = stroke_node_features[j]
                if stroke_is_contained(stroke_i, stroke_j):
                    contained = True
                    break

            # Remove if contained or randomly with 20% chance
            if contained:
                removed_idx.append(i)

    # Filter all outputs
    new_all_lines = [line for idx, line in enumerate(all_lines) if idx not in removed_idx]
    new_stroke_node_features = np.array([
        stroke for idx, stroke in enumerate(stroke_node_features) if idx not in removed_idx
    ])

    return new_all_lines, new_stroke_node_features



def remove_contained_lines_opacity(all_lines, stroke_node_features, is_feature_lines):
    """
    For NON-feature strokes (type == 0):
    - If contained in another stroke, set its opacity to 0
    - Otherwise, leave it unchanged.

    Returns:
    - updated_all_lines: all lines (opacity possibly modified)
    - updated_stroke_node_features: same (no removal)
    """
    num_strokes = len(stroke_node_features)

    for i in range(num_strokes):
        stroke_i = stroke_node_features[i]

        # Check if NOT a feature line
        if is_feature_lines[i][0] == 0:  
            contained = False
            for j in range(num_strokes):
                if i == j:
                    continue
                stroke_j = stroke_node_features[j]
                if stroke_is_contained(stroke_i, stroke_j):
                    contained = True
                    break

            if contained:
                # Set opacity to 0 in all_lines
                if i < len(all_lines):
                    all_lines[i]["opacity"] = 0.0
                # Also set opacity to 0 in stroke_node_features
                stroke_node_features[i][6] = 0.0

    return all_lines, stroke_node_features


def remove_random_lines(all_lines, stroke_node_features, is_feature_lines):
    """
    Randomly removes (sets opacity = 0) for non-feature lines with 50% probability.

    Args:
        all_lines: list of stroke objects.
        stroke_node_features: array containing node features, where opacity is at index 6.
        is_feature_lines: array of shape (num_strokes, 1), 1 if feature line else 0.
    """
    for idx, line in enumerate(all_lines):
        if is_feature_lines[idx][0] == 0:  # If not a feature line
            if np.random.rand() < 0.5:  # 50% chance
                line["opacity"] = 0.1
                stroke_node_features[idx][6] = 0.1  # Also update in node features if needed

    return all_lines, stroke_node_features


def duplicate_lines(all_lines, stroke_node_features):
    """
    Duplicate strokes based on their type:
    - Arc (3): 50% chance
    - Circle (2): 80% chance
    - Other: 30% chance

    Duplicated lines will have their type set to 'duplicate'.

    Parameters:
    - all_lines: list of stroke dicts
    - stroke_node_features: np.ndarray of shape (N, F)

    Returns:
    - new_all_lines: extended list with duplicates (with type='duplicate')
    - new_stroke_node_features: extended np.ndarray with duplicates
    """
    new_all_lines = copy.deepcopy(all_lines)
    new_stroke_node_features = np.copy(stroke_node_features)

    for i, line in enumerate(all_lines):
        stroke_type = stroke_node_features[i][-1]

        # Determine duplication probability
        if stroke_type == 3:
            prob = 0.5
        elif stroke_type == 2:
            prob = 0.5
        else:
            prob = 0.1

        if random.random() < prob:
            duplicated_line = copy.deepcopy(line)
            duplicated_features = stroke_node_features[i]

            # Set the line's type to 'duplicate'
            duplicated_line["type"] = "duplicate"

            new_all_lines.append(duplicated_line)
            new_stroke_node_features = np.vstack([new_stroke_node_features, duplicated_features])

    return new_all_lines, new_stroke_node_features
    


# --------------------------------------------------------------------------------- #


def compute_opacity(all_lines):
    """
    Adds an 'opacity' field to each line in all_lines based on its type.

    - 'feature_line': opacity ∈ [0.8, 1.0]
    - 'construction_line': opacity ∈ [0.2, 0.5]
    """
    for line in all_lines:
        line_type = line.get("type", "construction_line")

        if line_type in ['feature_line', 'extrude_line', 'fillet_line', 'extrude_face']:
            line["opacity"] = random.uniform(0.6, 0.8)
        elif line_type in ['duplicate']:
            line["opacity"] = random.uniform(0.2, 0.4)
        else:
            line["opacity"] = random.uniform(0.2, 0.4)

    return all_lines



# --------------------------------------------------------------------------------- #



def do_perturb(all_lines, stroke_node_features):
    """
    Perturbs only feature strokes in all_lines (1: straight, 3: arc, 2: circle).
    Returns new_all_lines with perturbed geometry.
    """
    new_all_lines = copy.deepcopy(all_lines)

    for i, stroke in enumerate(new_all_lines):
        geometry = stroke["geometry"]
        stroke_type = stroke_node_features[i][-1]

        # Straight stroke
        if stroke_type == 1:
            stroke["geometry"] = perturb_straight_line(np.array(geometry)).tolist()

        # Arc
        elif stroke_type == 3:
            if len(geometry) < 3:
                continue
            perturbed = perturb_arc_by_interpolation(
                geometry,
                arc_fraction=None,
                noise_scale_ratio=0.0005
            )
            stroke["geometry"] = perturbed.tolist()

        # Circle
        elif stroke_type == 2:
            if len(geometry) < 6:
                continue
            perturbed = perturb_circle_geometry(np.array(geometry))
            stroke["geometry"] = perturbed.tolist()

    return new_all_lines



def perturb_straight_line(pts):
    """
    Perturb a straight line stroke to resemble a hand-drawn stroke.
    Original geometry is perturbed first.
    Then 5 evenly spaced points are sampled from the perturbed endpoints.
    Jitter is added only to the interior points.

    Parameters:
    - pts: np.ndarray of shape (N, 3)

    Returns:
    - np.ndarray of perturbed points
    """
    pts = np.array(pts)
    if len(pts) < 2 :
        return pts

    stroke_length = np.linalg.norm(pts[0] - pts[-1])
    if stroke_length < 1e-8:
        return pts

    # Randomize perturbation strengths
    point_jitter_ratio = np.random.uniform(0.001, 0.003)
    endpoint_shift_ratio = np.random.uniform(0.01, 0.03)
    underdraw_ratio = np.random.uniform(0.05, 0.1)  # Underdraw: shorten stroke

    point_jitter = point_jitter_ratio * stroke_length
    endpoint_shift = endpoint_shift_ratio * stroke_length
    underdraw = underdraw_ratio * stroke_length

    # Perturb original geometry
    for j in range(len(pts)):
        if j == 0 or j == len(pts) - 1:
            shift = np.random.uniform(-endpoint_shift, endpoint_shift, size=3)
        else:
            shift = np.random.normal(scale=point_jitter, size=3)
        pts[j] += shift


    # Underdraw (shorten stroke slightly at both ends)
    vec_start = pts[1] - pts[0]
    vec_end = pts[-2] - pts[-1]
    vec_start /= np.linalg.norm(vec_start) + 1e-8
    vec_end /= np.linalg.norm(vec_end) + 1e-8
    pts[0] += underdraw * vec_start
    pts[-1] += underdraw * vec_end

    # Resample with jitter
    start, end = pts[0], pts[-1]
    t_vals = np.linspace(0, 1, 5)
    resampled_pts = np.array([(1 - t) * start + t * end for t in t_vals])

    for i in range(1, len(resampled_pts) - 1):
        resampled_pts[i] += np.random.normal(scale=point_jitter, size=3)

    return resampled_pts




def perturb_arc_by_interpolation(pts, arc_fraction=None,
                                  noise_scale_ratio=0.0001,
                                  endpoint_shift_ratio=0.002):
    """
    Simulate a hand-drawn arc by perturbing the original arc,
    blending with the straight line, and adding jitter.

    Parameters:
    - pts: (N, 3) numpy array of original arc points
    - arc_fraction: How close to the arc shape (1.0 = pure arc, <1 = more straight)
    - noise_scale_ratio: interior jitter relative to arc size
    - endpoint_shift_ratio: how much to shift start/end

    Returns:
    - np.ndarray of perturbed arc
    """
    import numpy as np

    pts = np.array(pts)
    num_points = len(pts)

    if num_points < 3:
        return pts

    start = pts[0].copy()
    end = pts[-1].copy()
    arc_points = pts.copy()

    # Estimate length and radius for scaling
    chord_len = np.linalg.norm(end - start)
    R = chord_len / np.sqrt(2)

    # Randomize strength
    if arc_fraction is None:
        if np.random.rand() < 0.5:
            arc_fraction = np.random.uniform(0.4, 0.7)
        else:
            arc_fraction = np.random.uniform(1.2, 1.5)
    noise_scale = R * noise_scale_ratio
    endpoint_shift = R * endpoint_shift_ratio

    # Perturb endpoints
    start += np.random.normal(scale=endpoint_shift, size=3)
    end += np.random.normal(scale=endpoint_shift, size=3)

    # Interpolate from straight line to original arc
    blended_points = []
    for j in range(num_points):
        t = j / (num_points - 1)
        line_pt = (1 - t) * start + t * end
        arc_pt = arc_points[j]

        blended = (1 - arc_fraction) * line_pt + arc_fraction * arc_pt

        if 0 < j < num_points - 1:
            blended += np.random.normal(scale=noise_scale, size=3)

        blended_points.append(blended)

    return np.array(blended_points)


def perturb_circle_geometry(pts):
    """
    Perturb a clean circle to resemble a human-drawn ellipse, with smooth imperfections.
    """
    pts = np.array(pts)
    N = len(pts)
    if N < 6:
        return pts

    # Estimate best-fit plane
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vh = np.linalg.svd(centered)
    u, v, normal = vh[0], vh[1], vh[2]

    # Project to 2D and get radius
    coords_2d = np.array([[np.dot(p - center, u), np.dot(p - center, v)] for p in pts])
    radius = np.mean(np.linalg.norm(coords_2d, axis=1))

    # === Randomized parameters ===
    rx = radius * np.random.uniform(0.8, 1.2)
    ry = radius * np.random.uniform(0.8, 1.2)
    theta = np.random.uniform(0, 2 * np.pi)
    noise_scale = np.random.uniform(0.001, 0.005) * radius
    shift_last_point = np.random.uniform(0.4, 0.8) * radius

    # Rotation matrix
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Generate ellipse
    t_vals = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ellipse = np.stack([rx * np.cos(t_vals), ry * np.sin(t_vals)], axis=1)
    ellipse = ellipse @ rot.T
    ellipse += np.random.normal(scale=noise_scale, size=ellipse.shape)

    # Back to 3D
    new_pts = np.array([center + x * u + y * v for x, y in ellipse])

    # Spread final distortion across last ~5 points smoothly
    distortion = np.random.normal(scale=shift_last_point, size=3)
    decay_weights = np.linspace(1.0, 0.9, 5)
    for k, w in enumerate(decay_weights):
        idx = -2 - k
        if idx >= 0:
            new_pts[idx] += w * distortion


    # === Add extension line beyond the circle ===
    num_extra_points = np.random.randint(1, 4)  # 1 to 3 extra points
    extension_spacing = np.random.uniform(0.05, 0.1) * radius

    # Tangent direction at end
    tangent = new_pts[-1] - new_pts[-2]
    tangent /= np.linalg.norm(tangent) + 1e-8

    for i in range(1, num_extra_points + 1):
        extension_point = new_pts[-1] + i * extension_spacing * tangent
        new_pts = np.vstack([new_pts, extension_point])

    return new_pts
