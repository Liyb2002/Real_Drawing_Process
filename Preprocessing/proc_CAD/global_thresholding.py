import numpy as np

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def point_close(point_1, point_2, stroke_1, stroke_2):
    """
    Check if two points are close based on the scale of their strokes.

    Parameters:
    - point_1: np.ndarray or list of shape (3,)
    - point_2: np.ndarray or list of shape (3,)
    - stroke_1: np.ndarray or list of shape (6,) representing (x1, y1, z1, x2, y2, z2)
    - stroke_2: np.ndarray or list of shape (6,) representing (x1, y1, z1, x2, y2, z2)

    Returns:
    - bool: True if distance between points is < 0.15 * max(stroke lengths), else False
    """
    length_1 = dist(stroke_1[:3], stroke_1[3:6])
    length_2 = dist(stroke_2[:3], stroke_2[3:6])
    threshold = 0.15 * max(length_1, length_2)

    return dist(point_1, point_2) < threshold


def circle_radius_close(center, radius, point):
    """
    Checks if a point lies within 0.15 * radius distance from the circle center.

    Parameters:
    - center: np.ndarray or list of shape (3,)
    - radius: float
    - point: np.ndarray or list of shape (3,)

    Returns:
    - bool: True if point is close to the circle's boundary
    """
    return abs(dist(center, point) - abs(radius)) < 0.15 * abs(radius)



def stroke_match(brep, stroke):
    # Compares brep line with stroke line, returns True if both ends match (order doesn't matter)
    d1 = np.linalg.norm(brep[:3] - stroke[:3]) + np.linalg.norm(brep[3:6] - stroke[3:6])
    d2 = np.linalg.norm(brep[:3] - stroke[3:6]) + np.linalg.norm(brep[3:6] - stroke[:3])
    stroke_len = np.linalg.norm(stroke[:3] - stroke[3:6])
    return min(d1, d2) < stroke_len * 0.3
