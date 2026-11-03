"""keyframes: a mapped frame is a keyframe when it moved or turned enough."""

import torch

from ovo.slam.simulated import se3
from ovo.slam.simulated.keyframes import KeyframeCfg, is_new_keyframe

CFG = KeyframeCfg(dist=0.1, rot_deg=5.0)


def pose(x=0.0, yaw=0.0):
    return se3.transform(se3.yaw_pitch_roll(yaw, 0, 0), torch.tensor([x, 0.0, 0.0]))


def test_defaults():
    assert KeyframeCfg() == CFG


def test_first_is_always_a_keyframe():
    assert is_new_keyframe(CFG, pose(), None)


def test_same_pose_is_not_a_keyframe():
    assert not is_new_keyframe(CFG, pose(), pose())


def test_distance_threshold():
    assert not is_new_keyframe(CFG, pose(x=0.09), pose())
    assert not is_new_keyframe(KeyframeCfg(dist=0.5), pose(x=0.5), pose())  # strictly greater
    assert is_new_keyframe(CFG, pose(x=0.11), pose())
    assert is_new_keyframe(CFG, pose(x=-0.11), pose())


def test_rotation_threshold():
    assert not is_new_keyframe(CFG, pose(yaw=4.0), pose())
    assert is_new_keyframe(CFG, pose(yaw=6.0), pose())
    assert is_new_keyframe(CFG, pose(yaw=-6.0), pose())


def test_relative_to_the_last_keyframe():
    last = pose(x=1.0, yaw=30.0)
    assert not is_new_keyframe(CFG, pose(x=1.05, yaw=33.0), last)
    assert is_new_keyframe(CFG, pose(x=1.05, yaw=36.0), last)


def test_works_with_float32_poses():
    assert is_new_keyframe(CFG, pose(x=0.2).float(), pose().float())
