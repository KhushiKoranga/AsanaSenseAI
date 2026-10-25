"""Geometry utilities for visibility-aware yoga pose analysis."""

import math
import numpy as np


JOINT_DEFINITIONS = {
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
    "left_ankle": (25, 27, 31),
    "right_ankle": (26, 28, 32),
}


def calculate_angle(a, b, c):
    """Calculate the smaller 2D angle ABC in degrees."""

    a = np.asarray(a, dtype=np.float64)[:2]
    b = np.asarray(b, dtype=np.float64)[:2]
    c = np.asarray(c, dtype=np.float64)[:2]

    if not (
        np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.isfinite(c).all()
    ):
        return None

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-9:
        return None

    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)

    return float(math.degrees(math.acos(cosine)))


def landmarks_to_xy(landmarks, frame_shape=None):
    """
    Convert 33 pose landmarks to coordinates.

    If frame_shape=(height, width) is supplied, convert normalized
    coordinates into pixel coordinates so angle calculations account
    for the camera frame's aspect ratio.
    """

    if landmarks is None or len(landmarks) < 33:
        raise ValueError("Expected 33 pose landmarks.")

    height, width = None, None

    if frame_shape is not None:
        height, width = frame_shape[:2]

    points = []

    for landmark in landmarks[:33]:
        if hasattr(landmark, "x") and hasattr(landmark, "y"):
            x, y = float(landmark.x), float(landmark.y)
        else:
            x, y = float(landmark[0]), float(landmark[1])

        if width is not None:
            x *= width
            y *= height

        points.append((x, y))

    return points


def calculate_joint_angles(landmarks, frame_shape=None, visibility_threshold=0.65):
    """
    Calculate joint angles only when all three required landmarks
    are sufficiently visible.

    Unreliable angles are omitted instead of being reported as zero.
    """

    points = landmarks_to_xy(landmarks, frame_shape)
    angles = {}

    for name, (a, b, c) in JOINT_DEFINITIONS.items():
        required = (a, b, c)
        valid = True

        for index in required:
            landmark = landmarks[index]

            visibility = getattr(landmark, "visibility", None)

            if visibility is not None and float(visibility) < visibility_threshold:
                valid = False
                break

            presence = getattr(landmark, "presence", None)

            if presence is not None and float(presence) < 0.5:
                valid = False
                break

            x, y = points[index]

            if not (math.isfinite(x) and math.isfinite(y)):
                valid = False
                break

            if hasattr(landmark, "x") and hasattr(landmark, "y"):
                if not (0.0 <= float(landmark.x) <= 1.0):
                    valid = False
                    break

                if not (0.0 <= float(landmark.y) <= 1.0):
                    valid = False
                    break

        if not valid:
            continue

        angle = calculate_angle(points[a], points[b], points[c])

        if angle is not None:
            angles[name] = round(angle, 2)

    return angles
