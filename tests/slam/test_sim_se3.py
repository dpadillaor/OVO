"""se3: rotations, transforms and the camera-axes convention (x right, y down, z forward)."""

import math

import pytest
import torch

from ovo.slam.simulated import se3

X = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
Y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
Z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)


def assert_close(a, b, atol=1e-12):
    assert torch.allclose(a, b, atol=atol), f"{a} != {b}"


def test_transform_layout():
    R = se3.axis_angle(torch.tensor([0.0, 0.0, math.pi / 2]))
    T = se3.transform(R, torch.tensor([1.0, 2.0, 3.0]))
    assert T.shape == (4, 4) and T.dtype == torch.float64
    assert_close(T[:3, :3], R)
    assert_close(T[:3, 3], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert_close(T[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))


def test_axis_angle_zero_is_identity():
    R = se3.axis_angle(torch.zeros(3))
    assert torch.equal(R, torch.eye(3, dtype=torch.float64))


@pytest.mark.parametrize("axis, before, after", [
    (Z, X, Y),   # 90 deg about z: x -> y
    (X, Y, Z),   # 90 deg about x: y -> z
    (Y, Z, X),   # 90 deg about y: z -> x
])
def test_axis_angle_quarter_turns(axis, before, after):
    R = se3.axis_angle(axis * math.pi / 2)
    assert_close(R @ before, after)
    assert_close(R @ axis, axis)  # the axis itself does not move


def test_axis_angle_half_turn():
    R = se3.axis_angle(Z * math.pi)
    assert_close(R @ X, -X)
    assert_close(R @ Y, -Y)


def test_axis_angle_accepts_float32_and_returns_float64():
    R = se3.axis_angle(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    assert R.dtype == torch.float64


@pytest.mark.parametrize("rotvec", [[0.1, -0.2, 0.3], [2.0, 1.0, -0.5], [1e-9, 0.0, 0.0]])
def test_axis_angle_is_orthonormal(rotvec):
    R = se3.axis_angle(torch.tensor(rotvec))
    assert_close(R @ R.T, torch.eye(3, dtype=torch.float64))
    assert torch.linalg.det(R).item() == pytest.approx(1.0, abs=1e-12)


def test_yaw_is_about_camera_y():
    R = se3.yaw_pitch_roll(90, 0, 0)
    assert_close(R @ Y, Y)
    assert_close(R @ Z, X)  # looking forward turns to the right


def test_pitch_is_about_camera_x():
    R = se3.yaw_pitch_roll(0, 90, 0)
    assert_close(R @ X, X)
    assert_close(R @ Y, Z)


def test_roll_is_about_camera_z():
    R = se3.yaw_pitch_roll(0, 0, 90)
    assert_close(R @ Z, Z)
    assert_close(R @ X, Y)


def test_yaw_pitch_roll_order():
    expected = (se3.yaw_pitch_roll(30, 0, 0) @ se3.yaw_pitch_roll(0, 20, 0)
                @ se3.yaw_pitch_roll(0, 0, 10))
    assert_close(se3.yaw_pitch_roll(30, 20, 10), expected)
    assert not torch.allclose(se3.yaw_pitch_roll(30, 20, 10),
                              se3.yaw_pitch_roll(0, 0, 10) @ se3.yaw_pitch_roll(0, 20, 0)
                              @ se3.yaw_pitch_roll(30, 0, 0))


@pytest.mark.parametrize("deg", [0.0, 0.5, 30.0, 90.0, 179.0, 180.0])
def test_angle_deg(deg):
    axis = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    R = se3.axis_angle(axis / torch.linalg.norm(axis) * math.radians(deg))
    assert se3.angle_deg(R) == pytest.approx(deg, abs=1e-5)


def test_angle_deg_of_yaw():
    assert se3.angle_deg(se3.yaw_pitch_roll(5, 0, 0)) == pytest.approx(5.0, abs=1e-9)


def test_random_unit_is_unit_and_seeded():
    a = se3.random_unit(torch.Generator().manual_seed(3))
    b = se3.random_unit(torch.Generator().manual_seed(3))
    c = se3.random_unit(torch.Generator().manual_seed(4))
    assert torch.linalg.norm(a).item() == pytest.approx(1.0, abs=1e-12)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
