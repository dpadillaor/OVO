"""Gates (ovo.entities.fusion.gates): centroid and aabb vetoes, and gate construction."""

import pytest
import torch

from ovo.entities.fusion.gates import GATES, AabbGate, CentroidGate, build_gate


def _gate(max_dist, pcds):
    gate = build_gate("centroid", {"max_dist": max_dist})
    gate.prepare([], pcds)
    return gate


def _at(*centre, n=4):
    """A cloud of n points symmetric around `centre`, so its mean is exactly `centre`."""
    c = torch.tensor(centre, dtype=torch.float64)
    offsets = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float64)[:n]
    return c + 0.01 * offsets


def test_registry_and_defaults():
    assert GATES == {"centroid": CentroidGate, "aabb": AabbGate}
    assert build_gate("centroid", {}).params == CentroidGate.Params(max_dist=1.5)
    assert build_gate("aabb", {}).params == AabbGate.Params(max_dist=0.3)


def test_check_returns_the_centroid_distance():
    gate = _gate(1.5, {1: _at(0, 0, 0), 2: _at(0.3, 0.4, 0)})
    ok, dist = gate.check(1, 2)
    assert ok
    assert dist == pytest.approx(0.5)


def test_vetoes_above_max_dist():
    gate = _gate(1.0, {1: _at(0, 0, 0), 2: _at(2, 0, 0)})
    assert gate.check(1, 2) == (False, pytest.approx(2.0))
    assert gate.check(2, 1)[0] is False


def test_distance_exactly_max_dist_passes():
    gate = _gate(1.5, {1: _at(0, 0, 0), 2: _at(1.5, 0, 0)})
    ok, dist = gate.check(1, 2)
    assert dist == 1.5
    assert ok


def test_unknown_gate():
    with pytest.raises(ValueError, match=r"instance_fusion\.gates: unknown gate 'centroidd'"):
        build_gate("centroidd", {})


def test_bad_gate_param():
    with pytest.raises(ValueError, match=r"instance_fusion\.gates\.centroid:.*max"):
        build_gate("centroid", {"max": 1.0})


# --- aabb -------------------------------------------------------------------------------------


def _box(lo, hi):
    """A cloud whose bounding box is exactly [lo, hi] (two opposite corners)."""
    return torch.tensor([lo, hi], dtype=torch.float64)


def _aabb(max_dist, pcds):
    gate = build_gate("aabb", {"max_dist": max_dist})
    gate.prepare([], pcds)
    return gate


def test_aabb_gap_between_separated_boxes():
    gate = _aabb(1.0, {1: _box([0, 0, 0], [1, 1, 1]), 2: _box([1.3, 1.4, 0], [2, 2, 1])})
    ok, dist = gate.check(1, 2)
    assert ok
    assert dist == pytest.approx(0.5)  # gap (0.3, 0.4, 0)
    assert gate.check(2, 1)[1] == pytest.approx(0.5)  # symmetric


def test_aabb_touching_or_intersecting_boxes_have_zero_gap():
    gate = _aabb(0.0, {1: _box([0, 0, 0], [1, 1, 1]), 2: _box([1, 0, 0], [2, 1, 1]), 3: _box([0.5, 0.5, 0.5], [3, 3, 3])})
    assert gate.check(1, 2) == (True, 0.0)
    assert gate.check(1, 3) == (True, 0.0)


def test_aabb_fair_to_elongated_objects():
    """A long table and a chair at its end: far centroids, touching boxes."""
    pcds = {1: _box([0, 0, 0], [4, 1, 1]), 2: _box([4.1, 0, 0], [4.6, 0.5, 1])}
    assert _gate(1.5, pcds).check(1, 2)[0] is False  # centroid vetoes
    assert _aabb(0.3, pcds).check(1, 2) == (True, pytest.approx(0.1))


def test_aabb_vetoes_above_max_dist_and_border_passes():
    pcds = {1: _box([0, 0, 0], [1, 1, 1]), 2: _box([1.5, 0, 0], [2, 1, 1])}
    assert _aabb(0.4, pcds).check(1, 2) == (False, pytest.approx(0.5))
    assert _aabb(0.5, pcds).check(1, 2) == (True, 0.5)


def test_aabb_empty_instance_is_vetoed():
    gate = _aabb(10.0, {1: _box([0, 0, 0], [1, 1, 1]), 2: torch.empty(0, 3, dtype=torch.float64)})
    assert gate.check(1, 2) == (False, float("inf"))
    assert gate.check(2, 1)[0] is False
