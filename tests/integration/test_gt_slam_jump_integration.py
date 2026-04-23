"""
Integration tests for GT SLAM jump drift simulation.
Task 18: GT SLAM Jump Drift Simulation

Synthetic trajectory: 20 frames, linear motion along X axis (step 0.1 m/frame).
KF threshold set low (0.09 m) so a KF is created roughly every frame that moves enough.
With step=0.1 and thresh=0.09, we get a KF every frame → 20 KFs total (indices 0..19).
The tests exercise jump triggering, pose offset, correction, and independence from noise.
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_linear_trajectory(n_frames=20, step=0.1, device="cpu"):
    traj = []
    for i in range(n_frames):
        pose = torch.eye(4, dtype=torch.float32)
        pose[0, 3] = i * step
        traj.append(pose.to(device))
    return traj


def make_config(noise_cfg=None, close_loops=False, device="cpu"):
    return {
        "dataset_name": "replica",
        "data": {"scene_name": "room0"},
        "device": device,
        "mapping": {
            "map_every": 1,
            "max_frame_points": 1e5,
            "k_pooling": 1,
            "downscale_res": 1,
        },
        "noise": noise_cfg or {},
        "slam": {"close_loops": close_loops},
        # Low threshold → KF every frame that moves 0.1 m (which is all of them)
        "kf_dist_thresh": 0.09,
        "kf_rot_thresh": 5.0,
    }


def make_slam(config, trajectory=None, device="cpu"):
    """Construct GroundTruthSLAM without filesystem access."""
    from ovo.slam.groundtruth_slam import GroundTruthSLAM
    from ovo.slam.vanilla_mapper import VanillaMapper

    if trajectory is None:
        trajectory = make_linear_trajectory(device=device)

    cam_intrinsics = torch.eye(3, dtype=torch.float32)
    cam_intrinsics[0, 0] = 320.0
    cam_intrinsics[1, 1] = 320.0
    cam_intrinsics[0, 2] = 160.0
    cam_intrinsics[1, 2] = 120.0

    with patch("builtins.open", MagicMock()):
        slam = GroundTruthSLAM.__new__(GroundTruthSLAM)
        VanillaMapper.__init__(slam, config, cam_intrinsics)
        slam.trajectory = trajectory
        slam.last_processed_frame_id = -1
        slam.c2w = torch.eye(4, dtype=torch.float32, device=device)
        slam._init_noise_params()
        slam._init_jump_params()
        slam.kf_dist_thresh = config.get("kf_dist_thresh", 0.1)
        slam.kf_rot_thresh = config.get("kf_rot_thresh", 5.0)
        slam.lc_dist_thresh = config.get("lc_dist_thresh", 0.2)
        slam.lc_rot_thresh = config.get("lc_rot_thresh", 10.0)
        slam.close_loops = config.get("slam", {}).get("close_loops", False)
        slam.map_every = config.get("mapping", {}).get("map_every", 10)
        slam.correction_done = False
        slam._lc_pcd_before = None
        slam._lc_traj_before = None
        slam.last_big_change_id = -1
        slam.kfs = {}

    return slam


def make_frame_data(frame_id, h=10, w=10):
    """Create synthetic frame data: (frame_id, image, depth, pose_as_numpy)."""
    image = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32) * 2.0  # uniform 2m depth
    pose = np.eye(4, dtype=np.float32)  # unused in GT tracking
    return [frame_id, image, depth, pose]


def run_loop(slam, n_frames=20):
    """Run the full track+map loop for n_frames."""
    for i in range(n_frames):
        frame_data = make_frame_data(i)
        slam.track_camera(frame_data)
        c2w = slam.c2w
        slam.map(frame_data, c2w)


# ---------------------------------------------------------------------------
# IT-01: Single jump — poses split before/after
# ---------------------------------------------------------------------------

class TestSingleJumpPosesSplit:
    def test_single_jump_poses_split(self):
        # Jump at KF index 1 (first KF created after the initial one)
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [{"kf_index": 1, "translation": [0.5, 0.0, 0.0]}],
        }
        slam = make_slam(make_config(noise_cfg))
        traj = slam.trajectory

        run_loop(slam, n_frames=20)

        # KF 0 is created at frame 0 (before any jump)
        # KF 1 is created at frame 1 — jump fires when N=len(kfs)=1
        # => all frames AFTER KF 1 should have +0.5 offset on X

        # Find the frame_id at which KF 1 was created
        kf_ids = list(slam.kfs.keys())
        kf1_frame = kf_ids[1]  # frame_id of the 2nd KF

        # Frames before the jump: estimated == GT
        frame_before = kf_ids[0]
        gt_before = traj[frame_before]
        est_before = slam.estimated_c2ws[frame_before]
        assert torch.allclose(est_before[:3, 3], gt_before[:3, 3], atol=1e-3)

        # Frames after the jump: estimated offset from GT by [0.5, 0, 0]
        # Use last frame to verify
        last_frame = kf_ids[-1]
        gt_last = traj[last_frame]
        est_last = slam.estimated_c2ws[last_frame]
        diff = est_last[:3, 3] - gt_last[:3, 3]
        assert torch.allclose(diff, torch.tensor([0.5, 0.0, 0.0]), atol=0.05)


# ---------------------------------------------------------------------------
# IT-02: Correction resets to GT
# ---------------------------------------------------------------------------

class TestCorrectionResetsToGT:
    def test_correction_resets_to_gt(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [{"kf_index": 1, "translation": [0.5, 0.0, 0.0]}],
        }
        # close_loops=True triggers correct_map_globally at end
        slam = make_slam(make_config(noise_cfg, close_loops=True))
        traj = slam.trajectory

        run_loop(slam, n_frames=20)

        # After global correction, all poses should match GT
        for frame_id, est_pose in slam.estimated_c2ws.items():
            if frame_id < len(traj):
                gt_pose = traj[frame_id]
                assert torch.allclose(est_pose[:3, 3], gt_pose[:3, 3], atol=1e-3), \
                    f"Frame {frame_id}: est {est_pose[:3,3]} != gt {gt_pose[:3,3]}"

        # KF poses should also match GT
        for kf_id, kf_data in slam.kfs.items():
            if kf_id < len(traj):
                gt_pose = traj[kf_id]
                assert torch.allclose(kf_data["pose"][:3, 3], gt_pose[:3, 3], atol=1e-3), \
                    f"KF {kf_id}: pose {kf_data['pose'][:3,3]} != gt {gt_pose[:3,3]}"


# ---------------------------------------------------------------------------
# IT-03: Multiple jumps compound offset
# ---------------------------------------------------------------------------

class TestMultipleJumpsCompoundOffset:
    def test_multiple_jumps_compound_offset(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {"kf_index": 1, "translation": [0.5, 0.0, 0.0]},
                {"kf_index": 2, "translation": [0.0, 0.5, 0.0]},
            ],
        }
        slam = make_slam(make_config(noise_cfg))
        traj = slam.trajectory

        run_loop(slam, n_frames=20)

        kf_ids = list(slam.kfs.keys())
        # After KF 2, the accumulated offset should be [0.5, 0.5, 0]
        last_frame = kf_ids[-1]
        gt_last = traj[last_frame]
        est_last = slam.estimated_c2ws[last_frame]
        diff = est_last[:3, 3] - gt_last[:3, 3]
        assert torch.allclose(diff, torch.tensor([0.5, 0.5, 0.0]), atol=0.05)


# ---------------------------------------------------------------------------
# IT-04: YAML config loaded correctly
# ---------------------------------------------------------------------------

class TestYamlConfigLoadedCorrectly:
    def test_yaml_config_loaded_correctly(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [
                {"kf_index": 5, "translation": [0.3, 0.0, 0.0]},
                {"kf_index": 10, "translation": [0.0, 0.3, 0.0]},
            ],
        }
        slam = make_slam(make_config(noise_cfg))

        assert slam.jump_drift_enabled is True
        assert len(slam.jump_configs) == 2
        assert slam.jump_configs[0]["kf_index"] == 5
        assert slam.jump_configs[1]["kf_index"] == 10


# ---------------------------------------------------------------------------
# IT-05: Jump and continuous drift are independent
# ---------------------------------------------------------------------------

class TestJumpAndContinuousDriftIndependent:
    def test_noise_enabled_no_jump_offset(self):
        """Case A: noise_enabled=True, jump_drift_enabled=False → _jump_offset stays identity."""
        noise_cfg = {
            "noise_enabled": True,
            "translation_noise_std": 0.01,
            "rotation_noise_std": 0.5,
            "noise_seed": 42,
            "jump_drift_enabled": False,
        }
        slam = make_slam(make_config(noise_cfg))

        run_loop(slam, n_frames=20)

        assert torch.allclose(slam._jump_offset, torch.eye(4), atol=1e-6)

    def test_jump_enabled_no_noise_applied(self):
        """Case B: jump_drift_enabled=True, no jumps configured → poses identical to GT."""
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [],
        }
        slam = make_slam(make_config(noise_cfg))
        traj = slam.trajectory

        run_loop(slam, n_frames=20)

        # With no jumps, all poses should match GT exactly
        for frame_id, est_pose in slam.estimated_c2ws.items():
            if frame_id < len(traj):
                gt_pose = traj[frame_id]
                assert torch.allclose(est_pose[:3, 3], gt_pose[:3, 3], atol=1e-4), \
                    f"Frame {frame_id}: est {est_pose[:3,3]} != gt {gt_pose[:3,3]}"


# ---------------------------------------------------------------------------
# IT-06: Point cloud offset after jump
# ---------------------------------------------------------------------------

class TestPointCloudOffsetAfterJump:
    def test_point_cloud_offset_after_jump(self):
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [{"kf_index": 1, "translation": [1.0, 0.0, 0.0]}],
        }
        slam = make_slam(make_config(noise_cfg))
        traj = slam.trajectory

        run_loop(slam, n_frames=20)

        kf_ids = list(slam.kfs.keys())

        # KF 0: pcd points added with GT pose (no jump yet)
        kf0_id = kf_ids[0]
        kf0_data = slam.kfs[kf0_id]
        start0, end0 = kf0_data["pcd_idxs"]

        # KF after jump (index >= 1): pcd points added with offset pose
        # Use the last KF to check
        kf_last_id = kf_ids[-1]
        kf_last_data = slam.kfs[kf_last_id]
        start_last, end_last = kf_last_data["pcd_idxs"]

        if end0 > start0 and end_last > start_last:
            pcd_kf0 = slam.pcd[start0:end0]
            pcd_last = slam.pcd[start_last:end_last]

            centroid_x_kf0 = pcd_kf0[:, 0].mean().item()
            centroid_x_last = pcd_last[:, 0].mean().item()
            gt_x_kf0 = traj[kf0_id][0, 3].item()
            gt_x_last = traj[kf_last_id][0, 3].item()

            # The expected centroid X for a KF should be roughly:
            #   gt_camera_x + jump_offset_x + local_projection_offset
            # KF0: no jump → centroid_x_kf0 ≈ gt_x_kf0 + local_projection_offset
            # Last KF (after jump): centroid_x_last ≈ gt_x_last + 1.0 + local_projection_offset
            #
            # So the difference between (centroid_last - gt_last) and (centroid_kf0 - gt_kf0)
            # should be approximately +1.0 (the jump magnitude).
            diff_last = centroid_x_last - gt_x_last
            diff_kf0 = centroid_x_kf0 - gt_x_kf0
            jump_in_pcd = diff_last - diff_kf0
            assert abs(jump_in_pcd - 1.0) < 0.5, \
                f"Expected jump of ~1.0 in point cloud X, got {jump_in_pcd:.3f}"


# ---------------------------------------------------------------------------
# IT-07: Jump config with close_loops triggers correction event
# ---------------------------------------------------------------------------

class TestJumpConfigWithLoopClosure:
    def test_extract_slam_baseline_with_jump_config(self):
        """
        Verify that with jump_drift_enabled and close_loops=True, the correction
        is triggered and the final state has GT poses.
        """
        noise_cfg = {
            "jump_drift_enabled": True,
            "jump_seed": 42,
            "jumps": [{"kf_index": 2, "translation": [0.5, 0.0, 0.0]}],
        }
        slam = make_slam(make_config(noise_cfg, close_loops=True))
        traj = slam.trajectory

        run_loop(slam, n_frames=20)

        # correction_done should be True (triggered near end)
        assert slam.correction_done is True

        # After correction, poses should match GT
        for frame_id, est_pose in slam.estimated_c2ws.items():
            if frame_id < len(traj):
                gt_pose = traj[frame_id]
                assert torch.allclose(est_pose[:3, 3], gt_pose[:3, 3], atol=1e-3), \
                    f"Frame {frame_id}: after correction est {est_pose[:3,3]} != gt {gt_pose[:3,3]}"
