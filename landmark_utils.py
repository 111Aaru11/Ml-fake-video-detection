import numpy as np
import math

# Typical mediapipe face_mesh landmark indices for eyes and lips (common mapping)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # landmarks used to compute EAR
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
UPPER_LIP = 13
LOWER_LIP = 14
# face width used for normalization - use left-most and right-most landmarks
LEFT_FACE = 234   # approximate left cheek
RIGHT_FACE = 454  # approximate right cheek

def landmark_list_to_np(landmarks, image_shape):
    """
    Convert mediapipe normalized landmarks list to Nx2 numpy pixel coords.
    landmarks: list of objects with x,y attributes (normalized)
    image_shape: (h, w)
    """
    h, w = image_shape[:2]
    pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])
    return pts

def eye_aspect_ratio(eye_points):
    # eye_points: 6x2 array
    # formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    p1, p2, p3, p4, p5, p6 = eye_points
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_opening_ratio(upper_point, lower_point, face_width):
    # vertical distance normalized by face width
    dist = np.linalg.norm(np.array(upper_point) - np.array(lower_point))
    if face_width == 0:
        return 0.0
    return dist / float(face_width)

def normalized_face_width(pts):
    # pts is Nx2; use approximate left-most and right-most
    xs = pts[:,0]
    return np.max(xs) - np.min(xs)
