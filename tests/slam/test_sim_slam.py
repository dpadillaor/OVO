"""SimulatedSLAM end to end on a synthetic scene: a flat wall at z=2 seen by a camera that
looks along +z and slides along x. Every mapped frame moves 0.15 m, so it is a keyframe."""

import json

import numpy as np
import pytest
import torch

from ovo.slam.simulated import SimulatedSLAM

H = W = 8
STEP = 0.15
K = torch.tensor([[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]])


def gt_pose(i):
    pose = np.eye(4)
    pose[0, 3] = STEP * i
    return pose


def frame(i, pose=None):
    image = np.full((H, W, 3), 100, dtype=np.uint8)
    depth = np.full((H, W), 2.0, dtype=np.float32)
    return [i, image, depth, gt_pose(i) if pose is None else pose]


def make(simulated=None, n_frames=10, close_loops=True, output_path=None, map_every=1):
    config = {"device": "cpu",
              "mapping": {"k_pooling": 1, "downscale_res": 1, "map_every": map_every},
              "slam": {"close_loops": close_loops, "simulated": simulated or {}}}
    if output_path is not None:
        config["output_path"] = str(output_path)
    return SimulatedSLAM(config, K, n_frames=n_frames)


def run(slam, frames, map_every=1, stop_before=None):
    """The tracking/mapping part of OVOSemMap.run. Returns the frames that set map_updated."""
    updates = []
    for fd in frames:
        if stop_before is not None and fd[0] >= stop_before:
            break
        slam.track_camera(fd)
        c2w = slam.get_c2w(fd[0])
        if c2w is None:
            continue
        if fd[0] % map_every == 0:
            slam.map(fd, c2w)
            if slam.map_updated:
                updates.append(fd[0])
                slam.map_updated = False
    return updates


def frames(n):
    return [frame(i) for i in range(n)]


def z_values(slam):
    return slam.pcd[:, 2]


def assert_poses_are_gt(slam):
    for f, c2w in slam.estimated_c2ws.items():
        assert torch.equal(c2w, torch.from_numpy(gt_pose(f)).float()), f"frame {f}"


def test_clean_run_is_exact_gt():
    slam = make(n_frames=10)
    assert run(slam, frames(10)) == []
    assert len(slam.estimated_c2ws) == 10
    assert_poses_are_gt(slam)
    assert len(slam.kfs) == 10
    assert slam.pcd.shape[0] > H * W
    assert torch.allclose(z_values(slam), torch.full_like(z_values(slam), 2.0), atol=1e-6)
    assert slam.events == {"jumps": [], "corrections": []}
    assert not slam.poses.has_error


def test_keyframes_follow_the_thresholds():
    slam = make({"keyframes": {"dist": 0.4}}, n_frames=10)
    run(slam, frames(10))
    assert list(slam.kfs) == [0, 3, 6, 9]  # 0.45 m > 0.4 m every third frame


def test_jump_duplicates_then_final_correction_snaps_back():
    n = 10
    slam = make({"jumps": [{"at_frame": 5, "translation": [0, 0, -0.1]}]}, n_frames=n)
    run(slam, frames(n), stop_before=n - 1)
    z = z_values(slam)
    assert (z - 1.9).abs().lt(1e-5).sum() > 0, "the jump maps a displaced copy"
    assert (z - 2.0).abs().lt(1e-5).sum() > 0
    assert slam.poses.has_error

    # the last mapped frame (n - map_every) runs the correction
    slam.track_camera(frame(n - 1))
    held = slam.get_c2w(n - 1)
    assert not torch.equal(held, torch.from_numpy(gt_pose(n - 1)).float())
    slam.map(frame(n - 1), held)
    assert slam.map_updated
    slam.map_updated = False

    assert torch.allclose(z_values(slam), torch.full_like(z_values(slam), 2.0), atol=1e-5)
    assert_poses_are_gt(slam)
    # the pose the caller holds for this frame is the corrected one (used for semantics after map)
    assert torch.equal(held, torch.from_numpy(gt_pose(n - 1)).float())
    for kf_id, kf in slam.kfs.items():
        assert torch.equal(kf["pose"], torch.from_numpy(gt_pose(kf_id)).float())
    assert not slam.poses.has_error
    assert slam.events["corrections"] == [{"frame": n - 1, "keyframes_moved": n - 5}]
    assert [j["frame"] for j in slam.events["jumps"]] == [5]


def test_jump_map_updated_set_exactly_once():
    n = 10
    slam = make({"jumps": [{"at_frame": 5, "translation": [0, 0, -0.1]}]}, n_frames=n)
    assert run(slam, frames(n)) == [n - 1]


def test_jump_at_frame_zero():
    n = 6
    slam = make({"jumps": [{"at_frame": 0, "translation": [0, 0, -0.1]}]}, n_frames=n)
    run(slam, frames(n), stop_before=n - 1)
    assert torch.allclose(z_values(slam), torch.full_like(z_values(slam), 1.9), atol=1e-5)
    assert run(slam, [frame(n - 1)]) == [n - 1]
    assert torch.allclose(z_values(slam), torch.full_like(z_values(slam), 2.0), atol=1e-5)
    assert slam.events["corrections"][0]["keyframes_moved"] == n


def test_rotation_jump_is_corrected():
    n = 8
    slam = make({"jumps": [{"at_frame": 3, "rotation": [3, 2, 1], "translation": 0.05}]}, n_frames=n)
    assert run(slam, frames(n)) == [n - 1]
    assert torch.allclose(z_values(slam), torch.full_like(z_values(slam), 2.0), atol=1e-5)


def test_drift_is_corrected_at_the_end():
    n = 10
    slam = make({"drift": {"translation_std": 0.01, "rotation_std_deg": 0.2}}, n_frames=n)
    assert run(slam, frames(n)) == [n - 1]
    assert torch.allclose(z_values(slam), torch.full_like(z_values(slam), 2.0), atol=1e-5)
    assert_poses_are_gt(slam)


def test_mid_sequence_correction_resets_error():
    n = 12
    slam = make({"jumps": [{"at_frame": 3, "translation": [0, 0, -0.1]}],
                 "correction": {"at_frames": [7]}}, n_frames=n)
    updates = run(slam, frames(n))
    assert updates == [7]  # the final correction finds no error: nothing to do
    assert [c["frame"] for c in slam.events["corrections"]] == [7]
    assert not slam.poses.has_error
    assert_poses_are_gt(slam)
    assert torch.allclose(z_values(slam), torch.full_like(z_values(slam), 2.0), atol=1e-5)


def test_correction_waits_for_the_next_mapped_frame():
    n = 20
    slam = make({"jumps": [{"at_frame": 2, "translation": [0, 0, -0.1]}],
                 "correction": {"at_frames": [7]}}, n_frames=n, map_every=5)
    # due at 7 -> runs at 10; the final one (15) finds no error left
    assert run(slam, frames(n), map_every=5) == [10]
    assert [c["frame"] for c in slam.events["corrections"]] == [10]


def test_close_loops_false_never_corrects():
    n = 10
    slam = make({"jumps": [{"at_frame": 5, "translation": [0, 0, -0.1]}],
                 "correction": {"at_frames": [7]}}, n_frames=n, close_loops=False)
    assert run(slam, frames(n)) == []
    assert slam.events["corrections"] == []
    assert slam.poses.has_error
    assert (z_values(slam) - 1.9).abs().lt(1e-5).sum() > 0


def test_invalid_gt_pose_is_skipped():
    n = 6
    bad = np.full((4, 4), np.inf)
    nan = gt_pose(3).copy()
    nan[0, 0] = np.nan
    seq = [frame(0, bad)] + [frame(i) for i in (1, 2)] + [frame(3, nan)] + [frame(i) for i in (4, 5)]
    slam = make({"drift": {"translation_std": 0.01}}, n_frames=n)
    run(slam, seq, stop_before=2)
    assert slam.get_c2w(0) is None
    # frame 1 is the first tracked frame: exact even with drift
    assert torch.equal(slam.get_c2w(1), torch.from_numpy(gt_pose(1)).float())
    run(slam, seq[2:])
    assert slam.get_c2w(3) is None and 3 not in slam.kfs
    assert set(slam.estimated_c2ws) == {1, 2, 4, 5}


def test_simulation_json_written_with_output_path(tmp_path):
    n = 10
    slam = make({"jumps": [{"at_frame": 5, "translation": [0, 0, -0.1]}]}, n_frames=n,
                output_path=tmp_path / "scene")
    path = tmp_path / "scene" / "simulation.json"
    assert json.loads(path.read_text()) == {"jumps": [], "corrections": []}
    run(slam, frames(n))
    events = json.loads(path.read_text())
    assert events["jumps"] == [{"frame": 5, "at_frame": 5, "axes": "camera",
                                "translation": [0.0, 0.0, -0.1], "rotation_deg": 0.0}]
    assert events["corrections"] == [{"frame": n - 1, "keyframes_moved": n - 5}]


def test_simulation_json_not_written_without_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    slam = make({"jumps": [{"at_frame": 2, "translation": [0, 0, -0.1]}]}, n_frames=5)
    run(slam, frames(5))
    assert slam.events["corrections"]  # events happened, they just are not written
    assert list(tmp_path.iterdir()) == []


def test_unknown_config_key_raises_at_init():
    with pytest.raises(ValueError, match="slam.simulated"):
        make({"jumpz": []})


def test_after_the_final_correction_poses_are_exact_gt():
    # n=20, map_every=5: the final correction runs at frame 15; frames 16..19 would drift
    slam = make({"drift": {"seed": 3}}, n_frames=20, map_every=5)
    run(slam, frames(20), map_every=5)
    assert slam.events["corrections"][-1]["frame"] == 15
    assert_poses_are_gt(slam)


def test_jump_after_the_final_correction_never_fires():
    slam = make({"jumps": [{"at_frame": 18, "translation": [0, 0, -0.1]}]}, n_frames=20, map_every=5)
    run(slam, frames(20), map_every=5)
    assert slam.events == {"jumps": [], "corrections": []}
    assert_poses_are_gt(slam)


def test_correction_at_or_after_the_final_one_is_the_final_one():
    slam = make({"drift": {"seed": 3}, "correction": {"at_frames": [15, 17]}}, n_frames=20, map_every=5)
    run(slam, frames(20), map_every=5)
    assert [c["frame"] for c in slam.events["corrections"]] == [15]
    assert_poses_are_gt(slam)
