"""
Unit tests for the local-frame rotation jump helpers added to geometry_utils:
sample_truncated_normal_deg and rotation_matrix_from_ypr.

Axis convention (OpenCV: X right, Y down, Z forward):
    pitch -> X axis, yaw -> Y axis, roll -> Z axis
"""
import math
import torch

from ovo.utils import geometry_utils


# ---------------------------------------------------------------------------
# sample_truncated_normal_deg
# ---------------------------------------------------------------------------

class TestSampleTruncatedNormalDegZeroStd:
    def test_zero_std_always_returns_zero(self):
        gen = torch.Generator(device="cpu").manual_seed(0)
        for _ in range(5):
            val = geometry_utils.sample_truncated_normal_deg(0.0, 90.0, gen, "cpu")
            assert val == 0.0


class TestSampleTruncatedNormalDegRespectsMax:
    def test_resample_respects_max_deg_across_seeds(self):
        # Huge std relative to max forces the resample loop to actually matter.
        for seed in range(20):
            gen = torch.Generator(device="cpu").manual_seed(seed)
            val = geometry_utils.sample_truncated_normal_deg(1000.0, 10.0, gen, "cpu")
            assert abs(val) <= 10.0 + 1e-6


class TestSampleTruncatedNormalDegReproducible:
    def test_same_seed_same_value(self):
        gen1 = torch.Generator(device="cpu").manual_seed(7)
        gen2 = torch.Generator(device="cpu").manual_seed(7)
        v1 = geometry_utils.sample_truncated_normal_deg(40.0, 150.0, gen1, "cpu")
        v2 = geometry_utils.sample_truncated_normal_deg(40.0, 150.0, gen2, "cpu")
        assert v1 == v2


# ---------------------------------------------------------------------------
# rotation_matrix_from_ypr
# ---------------------------------------------------------------------------

class TestRotationMatrixFromYprYawOnly:
    def test_yaw_90_matches_y_axis_rotation(self):
        R = geometry_utils.rotation_matrix_from_ypr(
            yaw_rad=math.pi / 2, pitch_rad=0.0, roll_rad=0.0
        )
        expected = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        assert torch.allclose(R, expected, atol=1e-6)


class TestRotationMatrixFromYprPitchOnly:
    def test_pitch_90_matches_x_axis_rotation(self):
        R = geometry_utils.rotation_matrix_from_ypr(
            yaw_rad=0.0, pitch_rad=math.pi / 2, roll_rad=0.0
        )
        expected = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        assert torch.allclose(R, expected, atol=1e-6)


class TestRotationMatrixFromYprRollOnly:
    def test_roll_90_matches_z_axis_rotation(self):
        R = geometry_utils.rotation_matrix_from_ypr(
            yaw_rad=0.0, pitch_rad=0.0, roll_rad=math.pi / 2
        )
        expected = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert torch.allclose(R, expected, atol=1e-6)


class TestRotationMatrixFromYprComposeOrder:
    def test_yaw_then_pitch_matches_rz_rx_ry_order(self):
        yaw, pitch, roll = math.pi / 6, math.pi / 9, 0.0

        def Rx(a):
            c, s = math.cos(a), math.sin(a)
            return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])

        def Ry(a):
            c, s = math.cos(a), math.sin(a)
            return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        expected = Rx(pitch) @ Ry(yaw)  # roll=0 -> Rz(roll) drops out
        R = geometry_utils.rotation_matrix_from_ypr(yaw, pitch, roll)
        assert torch.allclose(R, expected, atol=1e-6)
