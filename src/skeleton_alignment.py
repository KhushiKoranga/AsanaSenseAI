
"""Align a reference yoga skeleton with the live body."""

import numpy as np
ANCHOR_IDS = (11, 12, 23, 24)

def align_reference(reference_points, live_landmarks):
    """
    Align a reference skeleton using shoulder and hip anchors.

    The reference pose's joint proportions are preserved while its
    position, orientation and overall scale adapt to the live body.
    """

    ref = np.asarray(reference_points, dtype=np.float64)

    if ref.ndim != 2 or ref.shape[0] < 33 or ref.shape[1] < 2:
        raise ValueError("Reference pose must contain at least 33 x/y points.")

    ref = ref[:33, :2]

    live = np.asarray(
        [
            [float(live_landmarks[i].x), float(live_landmarks[i].y)]
            for i in ANCHOR_IDS
        ],
        dtype=np.float64,
    )

    source = ref[list(ANCHOR_IDS)]

    if not np.isfinite(source).all() or not np.isfinite(live).all():
        raise ValueError("Invalid reference or live anchor coordinates.")

    source_center = source.mean(axis=0)
    live_center = live.mean(axis=0)

    source_centered = source - source_center
    live_centered = live - live_center

    covariance = source_centered.T @ live_centered
    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(2)

    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1

    rotation = u @ correction @ vt

    denominator = float(np.sum(source_centered ** 2))

    if denominator < 1e-9:
        raise ValueError("Reference anchors are too close together.")

    scale = float(
        np.sum(singular_values * np.diag(correction)) / denominator
    )

    # Avoid extreme scale changes caused by unreliable anchors.
    scale = float(np.clip(scale, 0.5, 2.5))

    aligned = (ref - source_center) @ rotation * scale + live_center

    if not np.isfinite(aligned).all():
        raise ValueError("Alignment produced invalid coordinates.")

    return aligned

