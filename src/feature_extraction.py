"""Landmark feature extraction shared by training and inference."""
import numpy as np

LANDMARK_COUNT = 33
FEATURE_COLUMNS = [f"{i}_{axis}" for i in range(LANDMARK_COUNT) for axis in ("x", "y")]


def landmarks_to_features(landmarks):
    """Return 66 normalized x/y values from MediaPipe landmarks or row-like pairs."""
    if landmarks is None or len(landmarks) < LANDMARK_COUNT:
        raise ValueError("Expected 33 pose landmarks.")
    values = []
    for lm in landmarks[:LANDMARK_COUNT]:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            values.extend([float(lm.x), float(lm.y)])
        else:
            values.extend([float(lm[0]), float(lm[1])])
    arr = np.asarray(values, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Landmark features contain non-finite values.")
    return arr
