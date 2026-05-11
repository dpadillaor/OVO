"""Unit tests for FrustumOverlapEstimator."""
import torch

from ovo.covisibility.dataclasses import KeyFrameData
from ovo.covisibility.estimator import FrustumOverlapEstimator


def _make_intrinsics(fx: float = 100.0, cx: float = 10.0) -> torch.Tensor:
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fx, cx], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


def _make_depth(value: float = 2.0, h: int = 20, w: int = 20) -> torch.Tensor:
    return torch.full((h, w), value, dtype=torch.float32)


def _make_kf(kf_id: int, c2w: torch.Tensor) -> KeyFrameData:
    return KeyFrameData(
        kf_id=kf_id,
        frame_id=kf_id,
        c2w=c2w.float(),
        depth=_make_depth(),
        cam_intrinsics=_make_intrinsics(),
    )


def _identity_pose() -> torch.Tensor:
    return torch.eye(4)


def _translated_pose(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = dx
    pose[1, 3] = dy
    pose[2, 3] = dz
    return pose


def test_identical_keyframes_overlap_full():
    est = FrustumOverlapEstimator(min_overlap_ratio=0.0, max_distance=5.0)
    kf_a = _make_kf(0, _identity_pose())
    kf_b = _make_kf(1, _identity_pose())
    assert est.estimate_pair(kf_a, kf_b) == 1.0


def test_distance_filter_short_circuits():
    est = FrustumOverlapEstimator(min_overlap_ratio=0.0, max_distance=1.0)
    kf_a = _make_kf(0, _identity_pose())
    kf_b = _make_kf(1, _translated_pose(dz=10.0))
    assert est.estimate_pair(kf_a, kf_b) == 0.0


def test_far_apart_inside_distance_threshold_no_overlap():
    est = FrustumOverlapEstimator(min_overlap_ratio=0.0, max_distance=100.0)
    kf_a = _make_kf(0, _identity_pose())
    # Camera B looks the same way but is translated way back along -Z, so
    # corners of B fall outside the frustum of A.
    kf_b = _make_kf(1, _translated_pose(dz=-50.0))
    overlap = est.estimate_pair(kf_a, kf_b)
    assert overlap < 0.5


def test_estimate_candidates_filters_by_min_overlap():
    est = FrustumOverlapEstimator(min_overlap_ratio=0.5, max_distance=100.0)
    query = _make_kf(0, _identity_pose())
    candidates = [
        _make_kf(1, _identity_pose()),               # overlap 1.0 → kept
        _make_kf(2, _translated_pose(dz=-50.0)),     # overlap low → dropped
    ]
    edges = est.estimate_candidates(query, candidates)
    assert len(edges) == 1
    assert edges[0].kf_id_b == 1
    assert edges[0].overlap_ratio >= 0.5


def test_estimate_candidates_skips_self():
    est = FrustumOverlapEstimator(min_overlap_ratio=0.0, max_distance=5.0)
    query = _make_kf(0, _identity_pose())
    edges = est.estimate_candidates(query, [query])
    assert edges == []
