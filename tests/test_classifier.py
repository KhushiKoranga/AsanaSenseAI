import numpy as np
import pytest
from src.feature_extraction import landmarks_to_features


class Landmark:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_33_landmarks_make_66_features():
    points = [Landmark(i / 33, i / 66) for i in range(33)]
    values = landmarks_to_features(points)
    assert values.shape == (66,)
    assert np.isfinite(values).all()


def test_missing_landmarks_raise():
    with pytest.raises(ValueError):
        landmarks_to_features([Landmark(0, 0)])
