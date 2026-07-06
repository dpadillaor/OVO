"""
Unit tests for simulated SLAM jump drift.
Task 18: GT SLAM Jump Drift Simulation

All tests use a mocked SimulatedSLAM — no filesystem access, no trajectory file.
Synthetic trajectory: 20 frames of linear motion along X axis (step 0.1 m), identity rotations.
"""
import pytest
import torch
import math
from unittest.mock import patch, MagicMock

from ovo.utils import geometry_utils


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def make_linear_trajectory(n_frames=20, step=0.1, device="cpu"):
    """Create a simple linear trajectory along the X axis."""
    traj = []
    for i in range(n_frames):
        pose = torch.eye(4, dtype=torch.float32)
        pose[0, 3] = i * step
        traj.append(pose.to(device))
    return traj


def make_minimal_config(noise_cfg=None, device="cpu"):
    """Build the minimal config dict needed to instantiate SimulatedSLAM."""
    cfg = {
        "dataset_name": "replica",
        "data": {"scene_name": "room0"},
        "device": device,
        "mapping": {
            "map_every": 10,
            "max_frame_points": 1e5,
            "k_pooling": 1,
            "downscale_res": 1,
        },
        "noise": noise_cfg or {},
        "slam": {"close_loops": False},
        "kf_dist_thresh": 0.09,
        "kf_rot_thresh": 5.0,
    }
    return cfg


def make_slam(config, trajectory=None, device="cpu"):
    """
    Construct a SimulatedSLAM without touching the filesystem.
    Patches `open` so __init__ never tries to read traj.txt, and wires the
    collaborators manually (mirroring SimulatedSLAM.__init__ minus trajectory I/O).
    """
    from ovo.slam.simulated import SimulatedSLAM, JumpDriftController, KeyframeSelector
    from ovo.slam.vanilla_mapper import VanillaMapper

    if trajectory is None:
        trajectory = make_linear_trajectory(device=device)

    cam_intrinsics = torch.eye(3, dtype=torch.float32)

    with patch("builtins.open", MagicMock()):
        slam = SimulatedSLAM.__new__(SimulatedSLAM)
        # Bootstrap parent without filesystem
        VanillaMapper.__init__(slam, config, cam_intrinsics)
        slam.trajectory = trajectory
        slam.c2w = torch.eye(4, dtype=torch.float32, device=device)

        # Wire collaborators (same as SimulatedSLAM.__init__)
        noise_cfg = config.get("noise", {})
        slam.noise_enabled = noise_cfg.get("noise_enabled", False)
        slam.jump_controller = JumpDriftController(noise_cfg, device)
        slam.jump_drift_enabled = slam.jump_controller.enabled
        slam.pending_jump_events = []
        slam.keyframe_selector = KeyframeSelector(
            config.get("kf_dist_thresh", 0.1),
            config.get("kf_rot_thresh", 5.0),
        )
        slam.tracking_strategy = slam._build_tracking_strategy(noise_cfg)

        slam.close_loops = config.get("slam", {}).get("close_loops", True)
        slam.map_every = config.get("mapping", {}).get("map_every", 10)
        slam.correction_done = False
        slam._lc_pcd_before = None
        slam._lc_traj_before = None
        slam.last_big_change_id = -1
        slam._dedup_min_idx = 0

    return slam


# ---------------------------------------------------------------------------
# UT-01: Explicit translation and rotation parsing
# ---------------------------------------------------------------------------

class TestJumpParamsParsedExplicit:
    def test_jump_params_parsed_explicit(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {
                    "kf_index": 10,
                    "translation": [0.5, 0.0, 0.0],
                    "rotation": [0.0, 30.0, 0.0],
                }
            ],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        assert len(slam.jump_controller.jump_configs) == 1
        jc = slam.jump_controller.jump_configs[0]
        assert jc["kf_index"] == 10

        # translation
        assert torch.allclose(jc["translation"], torch.tensor([0.5, 0.0, 0.0]), atol=1e-5)

        # rotation: Y-axis 30 deg rotation matrix
        angle_rad = 30.0 * math.pi / 180.0
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        expected_R = torch.tensor([
            [c,  0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ], dtype=torch.float32)
        # rodrigues uses axis-angle; for pure Y rotation, axis=[0,1,0], angle=30 deg
        assert torch.allclose(jc["rotation_matrix"], expected_R, atol=1e-4)


# ---------------------------------------------------------------------------
# UT-02: Magnitude-based translation parsing
# ---------------------------------------------------------------------------

class TestJumpParamsParsedMagnitude:
    def test_jump_params_parsed_magnitude(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {
                    "kf_index": 5,
                    "translation_magnitude": 0.5,
                    "rotation_magnitude": 20.0,
                }
            ],
        }
        slam1 = make_slam(make_minimal_config(noise_cfg))
        slam2 = make_slam(make_minimal_config(noise_cfg))

        t1 = slam1.jump_controller.jump_configs[0]["translation"]
        t2 = slam2.jump_controller.jump_configs[0]["translation"]

        assert abs(torch.norm(t1).item() - 0.5) < 1e-5
        assert torch.allclose(t1, t2, atol=1e-6)


# ---------------------------------------------------------------------------
# UT-03: Jump disabled by default
# ---------------------------------------------------------------------------

class TestJumpDisabledByDefault:
    def test_jump_disabled_by_default(self):
        slam = make_slam(make_minimal_config())  # no noise_cfg

        assert slam.jump_drift_enabled is False
        assert torch.allclose(slam.jump_controller.offset, torch.eye(4))


# ---------------------------------------------------------------------------
# UT-04: Jump offset stays identity before KF trigger
# ---------------------------------------------------------------------------

class TestJumpOffsetIdentityBeforeTrigger:
    def test_jump_offset_identity_before_trigger(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [{"kf_index": 2, "translation": [1.0, 0.0, 0.0]}],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        # Run track_camera for frames 0-9 (no KF creation)
        for i in range(10):
            frame_data = [i, None, None, None]
            slam.track_camera(frame_data)

        # No KFs created yet, so jump_offset should be identity
        assert torch.allclose(slam.jump_controller.offset, torch.eye(4), atol=1e-6)


# ---------------------------------------------------------------------------
# UT-05: Jump applied at trigger KF
# ---------------------------------------------------------------------------

class TestJumpAppliedAtTriggerKf:
    def test_jump_applied_at_trigger_kf(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [{"kf_index": 1, "translation": [1.0, 0.0, 0.0]}],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        # Manually create KF 0 (index 0) so the next new KF has index 1
        dummy_c2w = torch.eye(4, dtype=torch.float32)
        slam.kfs[0] = {"id": 0, "pcd_idxs": (0, 0), "pose": dummy_c2w}

        # The keyframe about to be created has index N = len(kfs) == 1 → jump fires
        events = slam.jump_controller.maybe_trigger(len(slam.kfs), frame_id=1)

        assert len(events) == 1
        assert torch.allclose(slam.jump_controller.offset[:3, 3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
        assert 1 in slam.jump_controller._applied_triggers


# ---------------------------------------------------------------------------
# UT-06: Jump not applied twice
# ---------------------------------------------------------------------------

class TestJumpNotAppliedTwice:
    def test_jump_not_applied_twice(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [{"kf_index": 1, "translation": [1.0, 0.0, 0.0]}],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        # Apply the jump once
        events = slam.jump_controller.maybe_trigger(1, frame_id=1)
        assert len(events) == 1
        first_offset = slam.jump_controller.offset.clone()

        # Try to apply again — should be skipped (no event, offset unchanged)
        events_again = slam.jump_controller.maybe_trigger(1, frame_id=1)
        assert events_again == []
        assert torch.allclose(slam.jump_controller.offset, first_offset, atol=1e-6)


# ---------------------------------------------------------------------------
# UT-07: Multiple jumps accumulate
# ---------------------------------------------------------------------------

class TestMultipleJumpsAccumulate:
    def test_multiple_jumps_accumulate(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {"kf_index": 1, "translation": [1.0, 0.0, 0.0]},
                {"kf_index": 2, "translation": [0.0, 1.0, 0.0]},
            ],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        # Trigger both jumps at their respective KF indices
        slam.jump_controller.maybe_trigger(1, frame_id=1)
        slam.jump_controller.maybe_trigger(2, frame_id=2)

        # Both jumps applied; combined translation should be [1, 1, 0]
        # (pure translations compose additively when rotations are identity)
        assert torch.allclose(slam.jump_controller.offset[:3, 3], torch.tensor([1.0, 1.0, 0.0]), atol=1e-4)


# ---------------------------------------------------------------------------
# UT-08: Jump tracking returns offset pose
# ---------------------------------------------------------------------------

class TestJumpTrackingReturnsOffsetPose:
    def test_jump_tracking_returns_offset_pose(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        # Manually set a known jump offset: translate [2,0,0]
        offset = torch.eye(4, dtype=torch.float32)
        offset[0, 3] = 2.0
        slam.jump_controller._offset = offset

        # Frame 0 has GT identity pose (position = [0,0,0]); tracking applies the offset
        frame_data = [0, None, None, None]
        slam.track_camera(frame_data)

        assert torch.allclose(slam.c2w[:3, 3], torch.tensor([2.0, 0.0, 0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# UT-09: Explicit 90-degree Y rotation
# ---------------------------------------------------------------------------

class TestExplicitRotationExact:
    def test_explicit_rotation_exact(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {
                    "kf_index": 0,
                    "rotation": [0.0, 90.0, 0.0],  # 90 deg around Y
                }
            ],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        R = slam.jump_controller.jump_configs[0]["rotation_matrix"]
        # Standard Y-90° rotation: [[0,0,1],[0,1,0],[-1,0,0]]
        expected = torch.tensor([
            [0.0,  0.0, 1.0],
            [0.0,  1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        assert torch.allclose(R, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# UT-10: Seed reproducibility
# ---------------------------------------------------------------------------

class TestJumpSeedReproducibility:
    def test_jump_seed_reproducibility(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 7,
            "jumps": [{"kf_index": 0, "translation_magnitude": 0.3}],
        }
        slam1 = make_slam(make_minimal_config(noise_cfg))
        slam2 = make_slam(make_minimal_config(noise_cfg))

        t1 = slam1.jump_controller.jump_configs[0]["translation"]
        t2 = slam2.jump_controller.jump_configs[0]["translation"]
        assert torch.allclose(t1, t2, atol=1e-6)


# ---------------------------------------------------------------------------
# UT-11: rotation_jump — disabled axes contribute exactly zero
# ---------------------------------------------------------------------------

class TestRotationJumpDisabledAxesContributeZero:
    def test_disabled_axes_ignored_even_with_large_std(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {
                    "kf_index": 0,
                    "rotation_jump": {
                        "yaw": {"enabled": True, "std_deg": 0.0, "max_deg": 90.0},
                        # disabled but with a huge std_deg: must be skipped entirely,
                        # not sampled-then-zeroed (that would still burn RNG state).
                        "pitch": {"enabled": False, "std_deg": 999.0, "max_deg": 999.0},
                        "roll": {"enabled": False, "std_deg": 999.0, "max_deg": 999.0},
                    },
                }
            ],
        }
        slam = make_slam(make_minimal_config(noise_cfg))
        jc = slam.jump_controller.jump_configs[0]

        assert jc["is_local_rotation"] is True
        assert torch.allclose(jc["rotation_matrix"], torch.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# UT-12: rotation_jump — precedence over legacy rotation keys
# ---------------------------------------------------------------------------

class TestRotationJumpTakesPrecedenceOverLegacy:
    def test_rotation_jump_wins_over_legacy_rotation(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {
                    "kf_index": 0,
                    "rotation": [0.0, 90.0, 0.0],  # legacy explicit vector
                    "rotation_jump": {
                        "yaw": {"enabled": True, "std_deg": 0.0, "max_deg": 90.0},
                    },
                }
            ],
        }
        slam = make_slam(make_minimal_config(noise_cfg))
        jc = slam.jump_controller.jump_configs[0]

        assert jc["is_local_rotation"] is True
        # yaw sampled with std_deg=0 -> 0 rad -> identity, NOT the legacy 90 deg vector
        assert torch.allclose(jc["rotation_matrix"], torch.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# UT-13: legacy rotation entries are flagged as world-frame (not local)
# ---------------------------------------------------------------------------

class TestLegacyRotationIsNotLocal:
    def test_legacy_explicit_rotation_is_not_local(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [{"kf_index": 0, "rotation": [0.0, 30.0, 0.0]}],
        }
        slam = make_slam(make_minimal_config(noise_cfg))
        jc = slam.jump_controller.jump_configs[0]
        assert jc.get("is_local_rotation", False) is False


# ---------------------------------------------------------------------------
# UT-14: a single local rotation jump is conjugated by the current pose
# ---------------------------------------------------------------------------

class TestMaybeTriggerConjugatesLocalRotation:
    def test_local_rotation_conjugated_by_current_pose_not_applied_raw(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {
                    "frame_id": 0,
                    "rotation_jump": {"yaw": {"enabled": True, "std_deg": 0.0, "max_deg": 90.0}},
                }
            ],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        # Force a fixed, non-trivial local yaw angle (bypassing the sampler) so the
        # test asserts on the conjugation formula, not on RNG behaviour.
        fixed_yaw_rad = math.pi / 6
        slam.jump_controller.jump_configs[0]["rotation_matrix"] = geometry_utils.rotation_matrix_from_ypr(
            fixed_yaw_rad, 0.0, 0.0
        )

        # Camera pre-rotated 90 deg about Z (world) at the trigger frame - i.e. it is
        # NOT axis-aligned with the world when the jump fires.
        R_gt = geometry_utils.rotation_matrix_from_ypr(0.0, 0.0, math.pi / 2)
        gt_pose = torch.eye(4)
        gt_pose[:3, :3] = R_gt

        slam.jump_controller.maybe_trigger(0, frame_id=0, gt_pose=gt_pose)

        R_jump_local = geometry_utils.rotation_matrix_from_ypr(fixed_yaw_rad, 0.0, 0.0)
        expected_offset_rot = R_gt @ R_jump_local @ R_gt.T
        naive_offset_rot = R_jump_local  # what a (wrong) non-conjugated apply would give

        assert torch.allclose(slam.jump_controller.offset[:3, :3], expected_offset_rot, atol=1e-5)
        assert not torch.allclose(slam.jump_controller.offset[:3, :3], naive_offset_rot, atol=1e-3)


# ---------------------------------------------------------------------------
# UT-15: chained local jumps stay coherent — second jump uses the ALREADY
# DRIFTED pose (offset @ gt_pose), not the raw GT pose
# ---------------------------------------------------------------------------

class TestChainedLocalJumpsUseDriftedPose:
    def test_second_jump_conjugates_by_drifted_pose(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {
                    "frame_id": 0,
                    "rotation_jump": {"yaw": {"enabled": True, "std_deg": 0.0, "max_deg": 90.0}},
                },
                {
                    "frame_id": 1,
                    "rotation_jump": {"yaw": {"enabled": True, "std_deg": 0.0, "max_deg": 90.0}},
                },
            ],
        }
        slam = make_slam(make_minimal_config(noise_cfg))

        yaw1, yaw2 = math.pi / 6, math.pi / 4
        R_local_1 = geometry_utils.rotation_matrix_from_ypr(yaw1, 0.0, 0.0)
        R_local_2 = geometry_utils.rotation_matrix_from_ypr(yaw2, 0.0, 0.0)
        slam.jump_controller.jump_configs[0]["rotation_matrix"] = R_local_1
        slam.jump_controller.jump_configs[1]["rotation_matrix"] = R_local_2

        # Two different true camera orientations at the two trigger frames.
        R_gt_0 = geometry_utils.rotation_matrix_from_ypr(0.0, 0.0, math.pi / 2)
        R_gt_1 = geometry_utils.rotation_matrix_from_ypr(math.pi / 3, 0.0, 0.0)
        gt_pose_0 = torch.eye(4)
        gt_pose_0[:3, :3] = R_gt_0
        gt_pose_1 = torch.eye(4)
        gt_pose_1[:3, :3] = R_gt_1

        slam.jump_controller.maybe_trigger(0, frame_id=0, gt_pose=gt_pose_0)
        offset_after_first = slam.jump_controller.offset.clone()

        slam.jump_controller.maybe_trigger(1, frame_id=1, gt_pose=gt_pose_1)

        # Correct: second jump's local rotation must be conjugated by the pose the
        # camera *currently believes* it has, i.e. offset_after_first @ gt_pose_1.
        current_pose_rot = (offset_after_first @ gt_pose_1)[:3, :3]
        R_jump2_world = current_pose_rot @ R_local_2 @ current_pose_rot.T
        expected_offset_rot = R_jump2_world @ offset_after_first[:3, :3]

        # Wrong (regression to catch): conjugating with the raw GT pose instead of
        # the drifted one.
        wrong_current_pose_rot = R_gt_1
        wrong_R_jump2_world = wrong_current_pose_rot @ R_local_2 @ wrong_current_pose_rot.T
        wrong_offset_rot = wrong_R_jump2_world @ offset_after_first[:3, :3]

        assert torch.allclose(slam.jump_controller.offset[:3, :3], expected_offset_rot, atol=1e-5)
        assert not torch.allclose(slam.jump_controller.offset[:3, :3], wrong_offset_rot, atol=1e-3)
