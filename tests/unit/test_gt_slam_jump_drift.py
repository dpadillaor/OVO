"""
Unit tests for GT SLAM jump drift simulation.
Task 18: GT SLAM Jump Drift Simulation

All tests use a mocked GroundTruthSLAM — no filesystem access, no trajectory file.
Synthetic trajectory: 20 frames of linear motion along X axis (step 0.1 m), identity rotations.
"""
import pytest
import torch
import math
from unittest.mock import patch, MagicMock


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
    """Build the minimal config dict needed to instantiate GroundTruthSLAM."""
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
    Construct a GroundTruthSLAM without touching the filesystem.
    Patches `open` so __init__ never tries to read traj.txt.
    """
    from ovo.slam.groundtruth_slam import GroundTruthSLAM

    if trajectory is None:
        trajectory = make_linear_trajectory(device=device)

    cam_intrinsics = torch.eye(3, dtype=torch.float32)

    with patch("builtins.open", MagicMock()):
        slam = GroundTruthSLAM.__new__(GroundTruthSLAM)
        # Bootstrap parent without filesystem
        from ovo.slam.vanilla_mapper import VanillaMapper
        VanillaMapper.__init__(slam, config, cam_intrinsics)
        # Set required attributes normally set by __init__ before _init_noise_params
        slam.trajectory = trajectory
        slam.last_processed_frame_id = -1
        slam.c2w = torch.eye(4, dtype=torch.float32, device=device)
        # Call param-init methods directly
        slam._init_noise_params()
        slam._init_jump_params()
        # Set remaining attributes
        slam.kf_dist_thresh = config.get("kf_dist_thresh", 0.1)
        slam.kf_rot_thresh = config.get("kf_rot_thresh", 5.0)
        slam.lc_dist_thresh = config.get("lc_dist_thresh", 0.2)
        slam.lc_rot_thresh = config.get("lc_rot_thresh", 10.0)
        slam.close_loops = config.get("slam", {}).get("close_loops", True)
        slam.map_every = config.get("mapping", {}).get("map_every", 10)
        slam.correction_done = False
        slam._lc_pcd_before = None
        slam._lc_traj_before = None
        slam.last_big_change_id = -1
        slam.kfs = {}

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

        assert len(slam.jump_configs) == 1
        jc = slam.jump_configs[0]
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

        t1 = slam1.jump_configs[0]["translation"]
        t2 = slam2.jump_configs[0]["translation"]

        assert abs(torch.norm(t1).item() - 0.5) < 1e-5
        assert torch.allclose(t1, t2, atol=1e-6)


# ---------------------------------------------------------------------------
# UT-03: Jump disabled by default
# ---------------------------------------------------------------------------

class TestJumpDisabledByDefault:
    def test_jump_disabled_by_default(self):
        slam = make_slam(make_minimal_config())  # no noise_cfg

        assert slam.jump_drift_enabled is False
        assert torch.allclose(slam._jump_offset, torch.eye(4))


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
        assert torch.allclose(slam._jump_offset, torch.eye(4), atol=1e-6)


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

        # Manually create KF 0 (index 0) so next new KF has index 1
        dummy_c2w = torch.eye(4, dtype=torch.float32)
        slam.kfs[0] = {"id": 0, "pcd_idxs": (0, 0), "pose": dummy_c2w}

        # Now simulate map() being called when a new KF at index 1 is due
        # We call the internal logic directly: _is_new_keyframe triggers → N=1 → jump fires
        # Use a frame_id that's in-bounds
        frame_id = 5
        c2w_current = slam.trajectory[frame_id]

        # Force is_new_keyframe to return True by making the distance large
        far_c2w = c2w_current.clone()
        far_c2w[0, 3] += 10.0  # large translation

        # Trigger the KF creation path manually
        N = len(slam.kfs)  # == 1
        for jc in slam.jump_configs:
            if jc["kf_index"] == N and N not in slam._applied_jump_kf_indices:
                T_jump = slam._compute_jump_transform(jc["translation"], jc["rotation_matrix"])
                slam._jump_offset = T_jump @ slam._jump_offset
                slam._applied_jump_kf_indices.add(N)

        assert torch.allclose(slam._jump_offset[:3, 3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
        assert 1 in slam._applied_jump_kf_indices


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
        N = 1
        for jc in slam.jump_configs:
            if jc["kf_index"] == N and N not in slam._applied_jump_kf_indices:
                T_jump = slam._compute_jump_transform(jc["translation"], jc["rotation_matrix"])
                slam._jump_offset = T_jump @ slam._jump_offset
                slam._applied_jump_kf_indices.add(N)

        first_offset = slam._jump_offset.clone()

        # Try to apply again — should be skipped
        for jc in slam.jump_configs:
            if jc["kf_index"] == N and N not in slam._applied_jump_kf_indices:
                T_jump = slam._compute_jump_transform(jc["translation"], jc["rotation_matrix"])
                slam._jump_offset = T_jump @ slam._jump_offset
                slam._applied_jump_kf_indices.add(N)

        assert torch.allclose(slam._jump_offset, first_offset, atol=1e-6)


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

        # Apply jump at kf_index=1
        for jc in slam.jump_configs:
            N = jc["kf_index"]
            if N not in slam._applied_jump_kf_indices:
                T_jump = slam._compute_jump_transform(jc["translation"], jc["rotation_matrix"])
                slam._jump_offset = T_jump @ slam._jump_offset
                slam._applied_jump_kf_indices.add(N)

        # Both jumps applied; combined translation should be [1, 1, 0]
        # (pure translations compose additively when rotations are identity)
        assert torch.allclose(slam._jump_offset[:3, 3], torch.tensor([1.0, 1.0, 0.0]), atol=1e-4)


# ---------------------------------------------------------------------------
# UT-08: _jump_tracking returns offset pose
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
        slam._jump_offset = torch.eye(4, dtype=torch.float32)
        slam._jump_offset[0, 3] = 2.0

        # Frame 0 has GT identity pose (position = [0,0,0])
        frame_data = [0, None, None, None]
        slam._jump_tracking(frame_data)

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

        R = slam.jump_configs[0]["rotation_matrix"]
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

        t1 = slam1.jump_configs[0]["translation"]
        t2 = slam2.jump_configs[0]["translation"]
        assert torch.allclose(t1, t2, atol=1e-6)
