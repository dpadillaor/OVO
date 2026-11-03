"""jumps: config validation, drawing the random parts, and applying a jump to a pose."""

import math

import pytest
import torch

from ovo.slam.simulated import se3
from ovo.slam.simulated.jumps import Jump, JumpCfg, resolve

F64 = torch.float64


def test_jumpcfg_defaults():
    cfg = JumpCfg(at_frame=3)
    assert (cfg.translation, cfg.rotation, cfg.axes) == (0.0, 0.0, "camera")


@pytest.mark.parametrize("kwargs, match", [
    ({"at_frame": 1, "axes": "body"}, "axes"),
    ({"at_frame": -1}, "at_frame"),
    ({"at_frame": 1, "translation": [0, 1]}, "translation"),
    ({"at_frame": 1, "rotation": [0, 1, 2, 3]}, "rotation"),
])
def test_jumpcfg_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JumpCfg(**kwargs)


def test_jumpcfg_frame_zero_and_world_axes_ok():
    assert JumpCfg(at_frame=0, axes="world").at_frame == 0


def test_resolve_explicit_values_are_exact():
    (jump,) = resolve([JumpCfg(at_frame=5, translation=[0, 0, -0.1], rotation=[5, 2, 1], axes="world")], seed=0)
    assert jump.at_frame == 5 and jump.axes == "world"
    assert torch.equal(jump.translation, torch.tensor([0, 0, -0.1], dtype=F64))
    assert torch.equal(jump.rotation, se3.yaw_pitch_roll(5, 2, 1))


def test_resolve_defaults_are_no_change():
    (jump,) = resolve([JumpCfg(at_frame=0)], seed=0)
    assert torch.equal(jump.translation, torch.zeros(3, dtype=F64))
    assert torch.equal(jump.rotation, torch.eye(3, dtype=F64))


def test_resolve_scalar_is_magnitude_in_random_direction():
    (jump,) = resolve([JumpCfg(at_frame=0, translation=0.3, rotation=7)], seed=1)
    assert torch.linalg.norm(jump.translation).item() == pytest.approx(0.3, abs=1e-12)
    assert se3.angle_deg(jump.rotation) == pytest.approx(7.0, abs=1e-6)
    # a random direction, not along an axis
    assert (jump.translation.abs() > 1e-6).sum() >= 2


def test_resolve_seeded():
    cfgs = [JumpCfg(at_frame=0, translation=0.3, rotation=7)]
    a, b, c = resolve(cfgs, seed=1)[0], resolve(cfgs, seed=1)[0], resolve(cfgs, seed=2)[0]
    assert torch.equal(a.translation, b.translation) and torch.equal(a.rotation, b.rotation)
    assert not torch.allclose(a.translation, c.translation)


def test_resolve_sorted_by_frame_and_drawn_in_config_order():
    late, early = JumpCfg(at_frame=50, translation=0.2), JumpCfg(at_frame=10, translation=0.4)
    jumps = resolve([late, early], seed=3)
    assert [j.at_frame for j in jumps] == [10, 50]
    # draws follow config order: the first config entry gets the first draw
    alone = resolve([late], seed=3)[0]
    assert torch.equal(jumps[1].translation, alone.translation)


def test_resolve_empty():
    assert resolve([], seed=0) == []


def yawed_pose(yaw_deg, t):
    return se3.transform(se3.yaw_pitch_roll(yaw_deg, 0, 0), torch.tensor(t, dtype=F64))


def test_apply_camera_axes_moves_along_the_view():
    pose = yawed_pose(90, [1.0, 2.0, 3.0])  # camera z (forward) points along world x
    jump = Jump(0, "camera", torch.tensor([0.0, 0.0, 0.5], dtype=F64), torch.eye(3, dtype=F64))
    out = jump.apply(pose)
    assert torch.allclose(out[:3, 3], torch.tensor([1.5, 2.0, 3.0], dtype=F64), atol=1e-12)
    assert torch.equal(out[:3, :3], pose[:3, :3])


def test_apply_world_axes_moves_along_world():
    pose = yawed_pose(90, [1.0, 2.0, 3.0])
    jump = Jump(0, "world", torch.tensor([0.0, 0.0, 0.5], dtype=F64), torch.eye(3, dtype=F64))
    out = jump.apply(pose)
    assert torch.allclose(out[:3, 3], torch.tensor([1.0, 2.0, 3.5], dtype=F64), atol=1e-12)


def test_apply_rotation_keeps_camera_centre_and_turns_in_camera_axes():
    pose = yawed_pose(30, [1.0, 2.0, 3.0])
    rot = se3.yaw_pitch_roll(0, 10, 0)
    out = Jump(0, "camera", torch.zeros(3, dtype=F64), rot).apply(pose)
    assert torch.equal(out[:3, 3], pose[:3, 3])
    assert torch.allclose(out[:3, :3], pose[:3, :3] @ rot, atol=1e-12)
    assert se3.angle_deg(out[:3, :3] @ pose[:3, :3].T) == pytest.approx(10.0, abs=1e-9)


def test_apply_does_not_modify_the_input():
    pose = yawed_pose(0, [0.0, 0.0, 0.0])
    before = pose.clone()
    Jump(0, "camera", torch.ones(3, dtype=F64), se3.yaw_pitch_roll(5, 0, 0)).apply(pose)
    assert torch.equal(pose, before)


def test_to_json():
    jump = Jump(4, "world", torch.tensor([0.0, 0.0, -0.1], dtype=F64), se3.yaw_pitch_roll(5, 0, 0))
    assert jump.to_json() == {"at_frame": 4, "axes": "world", "translation": [0.0, 0.0, -0.1],
                              "rotation_deg": pytest.approx(5.0, abs=1e-6)}
    assert math.isclose(jump.to_json()["rotation_deg"], 5.0, abs_tol=1e-6)
