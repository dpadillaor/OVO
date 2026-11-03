"""poses: PoseSimulator, estimated = error . GT."""

import pytest
import torch

from ovo.slam.simulated import se3
from ovo.slam.simulated.drift import Drift, DriftCfg
from ovo.slam.simulated.jumps import Jump
from ovo.slam.simulated.poses import PoseSimulator

F64 = torch.float64


def gt(i):
    """Camera sliding along x, slowly turning."""
    return se3.transform(se3.yaw_pitch_roll(i, 0, 0), torch.tensor([0.1 * i, 0.0, 0.0], dtype=F64))


def jump(at_frame, t=(0.0, 0.0, -0.1)):
    return Jump(at_frame, "camera", torch.tensor(t, dtype=F64), torch.eye(3, dtype=F64))


def test_no_drift_no_jumps_returns_the_gt_object():
    sim = PoseSimulator(None, [])
    for i in range(4):
        g = gt(i)
        estimated, fired = sim.track(i, g)
        assert estimated is g and fired == []
    assert not sim.has_error


def test_drift_first_frame_exact_then_drifts():
    sim = PoseSimulator(Drift(DriftCfg(seed=1)), [])
    g0 = gt(0)
    e0, _ = sim.track(0, g0)
    assert e0 is g0 and not sim.has_error
    e1, _ = sim.track(1, gt(1))
    assert not torch.allclose(e1, gt(1))
    assert sim.has_error


def test_drift_accumulates():
    sim = PoseSimulator(Drift(DriftCfg(translation_std=0.01, rotation_std_deg=0.0, seed=1)), [])
    errors = []
    for i in range(30):
        e, _ = sim.track(i, gt(i))
        errors.append(torch.linalg.norm(e[:3, 3] - gt(i)[:3, 3]).item())
    assert errors[0] == 0.0
    assert errors[-1] > errors[1]


def test_drift_seeded():
    def run(seed):
        sim = PoseSimulator(Drift(DriftCfg(seed=seed)), [])
        return [sim.track(i, gt(i))[0] for i in range(5)]
    assert all(torch.equal(a, b) for a, b in zip(run(3), run(3)))
    assert not torch.allclose(run(3)[-1], run(4)[-1])


def test_jump_fires_once_at_first_tracked_frame_after_at_frame():
    sim = PoseSimulator(None, [jump(3)])
    fired_at = {}
    for i in (0, 2, 4, 6, 8):
        estimated, fired = sim.track(i, gt(i))
        fired_at[i] = fired
        offset = estimated[:3, 3] - gt(i)[:3, 3]
        if i < 4:
            assert torch.equal(estimated, gt(i))
        else:
            # camera z of gt(i) is R @ [0,0,1]; the jump moved the pose 0.1 back along it
            # at frame 4, and the same world error is carried to every later frame
            expected = gt(4)[:3, :3] @ torch.tensor([0.0, 0.0, -0.1], dtype=F64)
            assert torch.allclose(offset, expected, atol=1e-12)
    assert len(fired_at[4]) == 1 and fired_at[4][0].at_frame == 3
    assert all(fired_at[i] == [] for i in (0, 2, 6, 8))


def test_jump_at_frame_zero_fires_on_the_first_frame():
    sim = PoseSimulator(Drift(DriftCfg(seed=1)), [jump(0)])
    g0 = gt(0)
    e0, fired = sim.track(0, g0)
    assert len(fired) == 1 and sim.has_error
    assert torch.allclose(e0[:3, 3], torch.tensor([0.0, 0.0, -0.1], dtype=F64), atol=1e-12)


def test_several_jumps_same_and_different_frames():
    j_a, j_b, j_c = jump(2, (0.1, 0, 0)), jump(2, (0, 0.2, 0)), jump(5, (0, 0, 0.3))
    sim = PoseSimulator(None, [j_c, j_a, j_b])
    ident = se3.transform(torch.eye(3, dtype=F64), torch.zeros(3))
    fired = {i: sim.track(i, ident)[1] for i in range(7)}
    assert fired[2] == [j_a, j_b] and fired[5] == [j_c]
    assert sum(len(f) for f in fired.values()) == 3
    e6, _ = sim.track(6, ident)
    assert torch.allclose(e6[:3, 3], torch.tensor([0.1, 0.2, 0.3], dtype=F64), atol=1e-12)


def test_correct_resets_the_error():
    sim = PoseSimulator(None, [jump(1)])
    sim.track(0, gt(0))
    sim.track(1, gt(1))
    assert sim.has_error
    sim.correct()
    assert not sim.has_error
    g2 = gt(2)
    e2, _ = sim.track(2, g2)
    assert e2 is g2 and not sim.has_error


def test_correct_with_drift_starts_over_from_zero():
    sim = PoseSimulator(Drift(DriftCfg(translation_std=0.01, rotation_std_deg=0.1, seed=1)), [])
    for i in range(20):
        sim.track(i, gt(i))
    before = torch.linalg.norm(sim.track(20, gt(20))[0][:3, 3] - gt(20)[:3, 3]).item()
    sim.correct()
    after = torch.linalg.norm(sim.track(21, gt(21))[0][:3, 3] - gt(21)[:3, 3]).item()
    assert after < before
    assert sim.has_error  # drift keeps adding error after the correction


def test_pending_jumps_survive_a_correction():
    sim = PoseSimulator(None, [jump(1), jump(5)])
    for i in range(3):
        sim.track(i, gt(i))
    sim.correct()
    assert sim.track(4, gt(4))[1] == []
    assert len(sim.track(5, gt(5))[1]) == 1 and sim.has_error


def test_has_error_false_initially():
    assert not PoseSimulator(Drift(DriftCfg()), [jump(3)]).has_error


def test_stop_returns_gt_and_drops_pending_jumps():
    from ovo.slam.simulated.jumps import JumpCfg, resolve
    sim = PoseSimulator(Drift(DriftCfg(seed=1)), resolve([JumpCfg(at_frame=5, translation=[1, 0, 0])], 0))
    for i in range(3):
        sim.track(i, torch.eye(4, dtype=torch.float64))
    assert sim.has_error
    sim.stop()
    assert not sim.has_error
    gt = torch.eye(4, dtype=torch.float64)
    for i in range(3, 8):
        estimated, fired = sim.track(i, gt)
        assert estimated is gt and fired == []
