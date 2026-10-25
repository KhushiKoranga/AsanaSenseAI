import math
from src.angle_analyzer import calculate_angle


def test_right_angle():
    assert math.isclose(calculate_angle((1, 0), (0, 0), (0, 1)), 90.0, abs_tol=1e-6)


def test_straight_angle():
    assert math.isclose(calculate_angle((-1, 0), (0, 0), (1, 0)), 180.0, abs_tol=1e-6)


def test_zero_length_vector_is_safe():
    assert calculate_angle((0, 0), (0, 0), (1, 1)) == 0.0
